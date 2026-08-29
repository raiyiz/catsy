"""Run a richer catsy experiment spanning Gaussian and Fock-space optics."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import qutip as qt

from catsy import (
    Circuit,
    GaussianMeasurements,
    GaussianState,
    MachZehnderInterferometer,
    SimulationJournal,
)
from catsy.fock import realistic_photon_addition, realistic_photon_subtraction
from catsy.fock.visualization import plot_fock_dashboard, plot_wigner
from catsy.gaussian.visualization import (
    plot_covariance_matrix,
    plot_mode_correlation_map,
    plot_mzi_scan,
    plot_phase_space,
    plot_phase_space_trajectory,
)

try:
    from .config import RunConfig
except ImportError:
    from config import RunConfig  # type: ignore[no-redef]

LOGGER = logging.getLogger(__name__)
_DEFAULT_CONFIG_PATH = Path(__file__).parent / "config.toml"


def build_circuit(config: RunConfig, rng: np.random.Generator) -> Circuit:
    """Construct a three-mode Gaussian state-preparation and readout circuit."""
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
    return qt.ket2dm((plus + minus).unit())


def run_fock_chain() -> tuple[qt.Qobj, qt.Qobj, qt.Qobj, dict[str, np.ndarray]]:
    """Prepare a cat, herald photon subtraction, then herald photon addition."""
    cat = make_cat_state(N_cutoff=18, alpha=1.1 + 0.15j)
    subtracted = realistic_photon_subtraction(
        cat, tap_reflectivity=0.08, detector_efficiency=0.75, ancilla_cutoff=6
    )
    added = realistic_photon_addition(
        subtracted,
        coupling_strength=0.045,
        detector_efficiency=0.75,
        ancilla_cutoff=6,
    )
    theta = np.linspace(0.0, 2.0 * np.pi, 33)
    interferometer = MachZehnderInterferometer(kappa=0.08, N_cutoff=18, loss_time=0.75)
    scan = interferometer.scan(added, theta)
    return cat, subtracted, added, {key: np.asarray(value) for key, value in scan.items()}


def run_homodyne(
    state: GaussianState, rng: np.random.Generator
) -> tuple[float, GaussianState]:
    """Measure the signal mode's quadrature and return the conditioned state."""
    return GaussianMeasurements.homodyne_measurement(
        state, measured_mode="signal", phi=np.pi / 6, rng=rng
    )


def run_heterodyne(
    state: GaussianState, rng: np.random.Generator
) -> tuple[np.ndarray, GaussianState]:
    """Measure both signal quadratures and return the conditioned state."""
    outcome, conditioned = GaussianMeasurements.heterodyne_measurement(
        state, measured_mode="signal", rng=rng
    )
    return np.asarray(outcome), conditioned


def plot_experiment(
    final_state: GaussianState,
    homodyne_state: GaussianState,
    heterodyne_state: GaussianState,
    cat: qt.Qobj,
    subtracted: qt.Qobj,
    added: qt.Qobj,
    mzi_scan: dict[str, np.ndarray],
    output_dir: Path,
) -> None:
    """Create diagnostics using only Catsy's public plotting helpers."""
    output_dir.mkdir(parents=True, exist_ok=True)

    remaining_modes = tuple(mode for mode in final_state.modes if mode != "signal")
    homodyne_reordered = homodyne_state.reorder_modes(remaining_modes)
    heterodyne_reordered = heterodyne_state.reorder_modes(remaining_modes)

    figures = {
        "01_final_signal_phase_space": plot_phase_space(
            final_state, "signal", show=False
        ),
        "02_final_covariance_matrix": plot_covariance_matrix(final_state, show=False),
        "03_final_mode_correlations": plot_mode_correlation_map(final_state, show=False),
        "04_even_cat_wigner": plot_wigner(
            cat, xlim=(-4.5, 4.5), resolution=150, show=False
        ),
        "05_even_cat_state": plot_fock_dashboard(
            cat, xlim=(-4.5, 4.5), resolution=120, show=False
        ),
        "06_after_photon_subtraction": plot_fock_dashboard(
            subtracted, xlim=(-4.5, 4.5), resolution=120, show=False
        ),
        "07_after_photon_addition": plot_fock_dashboard(
            added, xlim=(-4.5, 4.5), resolution=120, show=False
        ),
        "08_after_homodyne_idler": plot_phase_space(
            homodyne_reordered, "idler", show=False
        ),
        "09_after_heterodyne_idler": plot_phase_space(
            heterodyne_reordered, "idler", show=False
        ),
        "10_measurement_conditioning": plot_phase_space_trajectory(
            [homodyne_reordered, heterodyne_reordered], "idler", show=False
        ),
        "11_mach_zehnder_scan": plot_mzi_scan(mzi_scan, show=False),
    }

    for name, figure in figures.items():
        figure.savefig(output_dir / f"{name}.png", dpi=150)

    LOGGER.info(
        "Saved %d Catsy diagnostic plots to %s (MZI scan has %d phase points)",
        len(figures),
        output_dir,
        len(mzi_scan["theta"]),
    )


def main(config_path: str | Path = _DEFAULT_CONFIG_PATH) -> Path:
    """Run the combined Gaussian/Fock experiment and save its journal entry."""
    config = RunConfig.from_toml(config_path)
    rng = np.random.default_rng(config.seed)
    output_dir = Path(config.output_dir)

    logging.basicConfig(
        level=getattr(logging, config.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    journal = SimulationJournal(output_dir)
    circuit = build_circuit(config, rng)
    initial = GaussianState.vacuum(circuit.modes)
    final_state = circuit.run(initial)

    LOGGER.info("Running Gaussian circuit %r on modes %s", circuit.name, circuit.modes)
    cat, subtracted, added, mzi_scan = run_fock_chain()
    LOGGER.info(
        "Fock chain complete: even cat -> photon subtraction -> photon addition -> MZI; "
        "max output photon number %.3f",
        float(np.max(mzi_scan["n1"])),
    )

    homodyne_outcome, homodyne_state = run_homodyne(final_state, rng)
    heterodyne_outcome, heterodyne_state = run_heterodyne(final_state, rng)
    LOGGER.info(
        "Signal readout: homodyne x_phi=%.4f; heterodyne (x,p)=(%.4f, %.4f)",
        homodyne_outcome,
        heterodyne_outcome[0],
        heterodyne_outcome[1],
    )

    plot_experiment(
        final_state,
        homodyne_state,
        heterodyne_state,
        cat,
        subtracted,
        added,
        mzi_scan,
        output_dir / "plots",
    )

    entry = journal.new_entry(
        "Three-mode Gaussian preparation, heralded Fock processing, and interferometric readout",
        tags=["example", "gaussian", "fock", "interferometer", "measurement", "plotting"],
        notes=(
            "Prepares a three-mode Gaussian state, independently explores an even cat "
            "through heralded photon subtraction and addition, scans a lossy Mach-Zehnder "
            "interferometer, and compares homodyne with heterodyne conditioning of the "
            "Gaussian signal mode."
        ),
        metadata={"circuit_name": circuit.name, "output_dir": str(output_dir)},
    )
    entry.log_run(
        "gaussian_state_preparation",
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
        "heralded_fock_processing",
        metrics={
            "cat_trace": float(cat.tr()),
            "subtracted_trace": float(subtracted.tr()),
            "added_trace": float(added.tr()),
        },
    )
    entry.log_run(
        "lossy_mach_zehnder_scan",
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
    entry.log_run(
        "homodyne_signal_readout",
        final_state=homodyne_state,
        metrics={
            "outcome": float(homodyne_outcome),
            "phase": float(np.pi / 6),
            "remaining_modes": len(homodyne_state.modes),
        },
    )
    entry.log_run(
        "heterodyne_signal_readout",
        final_state=heterodyne_state,
        metrics={
            "outcome_x": float(heterodyne_outcome[0]),
            "outcome_p": float(heterodyne_outcome[1]),
            "remaining_modes": len(heterodyne_state.modes),
        },
    )

    saved_path = entry.save(output_dir)
    LOGGER.info("Saved journal entry to %s", saved_path)
    return saved_path


if __name__ == "__main__":
    main()
