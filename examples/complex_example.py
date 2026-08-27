"""Run a richer catsy experiment spanning Gaussian and Fock-space optics."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import qutip as qt

from catsy import (
    Circuit,
    GaussianState,
    MachZehnderInterferometer,
    SimulationJournal,
)

try:
    from .config import RunConfig
except ImportError:  # running as a script (`python examples/complex_example.py`)
    from config import RunConfig  # type: ignore[no-redef]

LOGGER = logging.getLogger(__name__)
_DEFAULT_CONFIG_PATH = Path(__file__).parent / "config.toml"


def build_circuit(config: RunConfig, rng: np.random.Generator) -> Circuit:
    """Construct a three-mode Gaussian preparation and readout circuit."""
    circuit = Circuit(name=config.circuit_name)
    signal = circuit.mode("signal")
    idler = circuit.mode("idler")
    reference = circuit.mode("reference")

    alpha = complex(rng.normal(scale=0.3), rng.normal(scale=0.3))
    return (
        circuit.squeeze(signal, r=config.signal_squeezing, theta=0.0)
        .displace(signal, alpha=alpha)
        .beam_splitter(signal, idler, eta=config.signal_idler_transmissivity)
        .rotate(idler, phi=0.35)
        .thermal_loss(idler, eta=0.9, n_thermal=0.15)
        .squeeze(reference, r=0.35, theta=np.pi / 4)
        .beam_splitter(idler, reference, eta=0.5)
        .loss(signal, eta=0.92)
    )


def make_cat_state(N_cutoff: int, alpha: complex) -> qt.Qobj:
    """Return an even Schrödinger-cat state in a truncated Fock basis."""
    plus = qt.coherent(N_cutoff, alpha)
    minus = qt.coherent(N_cutoff, -alpha)
    cat = plus + minus
    return cat.unit()


def run_mach_zehnder() -> dict[str, np.ndarray]:
    """Scan a lossy Mach-Zehnder interferometer with a non-Gaussian input."""
    theta = np.linspace(0.0, 2.0 * np.pi, 25)
    interferometer = MachZehnderInterferometer(
        kappa=0.08,
        N_cutoff=18,
        loss_time=0.75,
    )
    cat = make_cat_state(N_cutoff=18, alpha=1.1 + 0.15j)
    scan = interferometer.scan(cat, theta)
    return {key: np.asarray(value) for key, value in scan.items()}


def main(config_path: str | Path = _DEFAULT_CONFIG_PATH) -> Path:
    """Run the combined Gaussian/Fock experiment and save its journal entry."""
    config = RunConfig.from_toml(config_path)
    rng = np.random.default_rng(config.seed)

    logging.basicConfig(
        level=getattr(logging, config.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    journal = SimulationJournal(config.output_dir)
    circuit = build_circuit(config, rng)
    initial = GaussianState.vacuum(circuit.modes)
    final_state = circuit.run(initial)

    LOGGER.info("Running Gaussian circuit %r on modes %s", circuit.name, circuit.modes)
    mzi_scan = run_mach_zehnder()
    LOGGER.info(
        "MZI phase scan complete: %d points, max output photon number %.3f",
        len(mzi_scan["theta"]),
        float(np.max(mzi_scan["n1"])),
    )

    entry = journal.new_entry(
        "Gaussian circuit with Mach-Zehnder interferometry",
        tags=["example", "gaussian", "fock", "interferometer"],
        notes=(
            "Combines a three-mode Gaussian circuit with a lossy Mach-Zehnder "
            "scan driven by an even cat state."
        ),
        metadata={
            "circuit_name": circuit.name,
            "output_dir": str(config.output_dir),
        },
    )
    entry.log_run(
        "gaussian_circuit",
        circuit=circuit,
        final_state=final_state,
        metrics={
            "num_modes": len(final_state.modes),
            "mean_abs_displacement": float(np.mean(np.abs(final_state.displacement))),
            "covariance_trace": float(np.trace(final_state.covariance)),
        },
        arrays={
            "displacement": {
                "data": final_state.displacement,
                "unit": "quadrature",
                "dimensions": ["quadrature"],
                "description": "Final first moments.",
            },
            "covariance": {
                "data": final_state.covariance,
                "unit": "quadrature^2",
                "dimensions": ["quadrature", "quadrature"],
                "description": "Final covariance matrix.",
            },
        },
    )
    entry.log_run(
        "mach_zehnder_scan",
        metrics={
            "max_output_n1": float(np.max(mzi_scan["n1"])),
            "max_output_n2": float(np.max(mzi_scan["n2"])),
            "max_parity1": float(np.max(mzi_scan["parity1"])),
        },
        arrays={
            "theta": {
                "data": mzi_scan["theta"],
                "unit": "radians",
                "dimensions": ["phase"],
                "description": "Scanned phase in the lossy MZI arm.",
            },
            "n1": {
                "data": mzi_scan["n1"],
                "unit": "photons",
                "dimensions": ["phase"],
                "description": "Mean photon number at MZI output port 1.",
            },
            "n2": {
                "data": mzi_scan["n2"],
                "unit": "photons",
                "dimensions": ["phase"],
                "description": "Mean photon number at MZI output port 2.",
            },
            "parity1": {
                "data": mzi_scan["parity1"],
                "unit": "dimensionless",
                "dimensions": ["phase"],
                "description": "Parity signal at MZI output port 1.",
            },
        },
    )

    saved_path = entry.save(config.output_dir)
    LOGGER.info("Saved journal entry to %s", saved_path)
    return saved_path


if __name__ == "__main__":
    main()
