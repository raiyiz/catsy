"""Run a representative multi-mode catsy experiment and persist its results."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from catsy import Circuit, GaussianState, SimulationJournal

LOGGER = logging.getLogger(__name__)


def build_circuit() -> Circuit:
    """Construct a small but representative multi-mode Gaussian circuit."""
    circuit = Circuit(name="Three-mode demo")
    signal = circuit.mode("signal")
    idler = circuit.mode("idler")
    reference = circuit.mode("reference")

    return (
        circuit
        .squeeze(signal, r=0.7, theta=0.0)
        .displace(signal, alpha=0.4 + 0.1j)
        .beam_splitter(signal, idler, eta=0.65)
        .rotate(idler, phi=0.35)
        .thermal_loss(idler, eta=0.9, nbar=0.15)
        .squeeze(reference, r=0.35, theta=np.pi / 4)
        .beam_splitter(idler, reference, eta=0.5)
        .loss(signal, eta=0.92)
    )


def main(output_dir: Path | None = None) -> Path:
    """Run the demo circuit and return the saved journal entry path."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    root = Path(output_dir) if output_dir is not None else Path("runs") / "complex_circuit"
    journal = SimulationJournal(root)
    circuit = build_circuit()
    initial = GaussianState.tmsv("signal", "idler", r=0.5)
    initial = initial.add_mode("reference") if hasattr(initial, "add_mode") else initial

    LOGGER.info("Running circuit %r on modes %s", circuit.name, circuit.modes)
    final_state = circuit.run(initial)

    entry = journal.new_entry(
        "Three-mode Gaussian circuit",
        tags=["example", "gaussian", "three-mode"],
        notes="Representative circuit showing persistence and structured output paths.",
        metadata={"circuit_name": circuit.name, "output_dir": str(root)},
    )
    run = entry.log_run(
        "demo",
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

    saved_path = entry.save(root)
    LOGGER.info("Saved journal entry %s (%s)", run.run_id, saved_path)
    return saved_path


if __name__ == "__main__":
    main()
