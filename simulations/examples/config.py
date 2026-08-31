"""Configuration loading for the executable examples."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RunConfig:
    """Validated settings for an example run."""

    output_dir: Path
    circuit_name: str = "Untitled Circuit"
    signal_squeezing: float = 0.5
    signal_idler_transmissivity: float = 0.5
    log_level: str = "INFO"
    seed: int | None = None

    @classmethod
    def from_toml(cls, path: str | Path) -> RunConfig:
        """Load a run configuration from a TOML file."""
        path = Path(path)
        with path.open("rb") as file:
            data = tomllib.load(file)

        run = data.get("run", {})
        if not isinstance(run, dict):
            raise ValueError("[run] must be a TOML table.")

        output_dir = run.get("output_dir", "runs")
        if not isinstance(output_dir, str) or not output_dir.strip():
            raise ValueError("run.output_dir must be a non-empty string.")

        circuit_name = run.get("circuit_name", cls.circuit_name)
        if not isinstance(circuit_name, str) or not circuit_name.strip():
            raise ValueError("run.circuit_name must be a non-empty string.")

        signal_squeezing = run.get("signal_squeezing", cls.signal_squeezing)
        if not isinstance(signal_squeezing, int | float) or isinstance(
            signal_squeezing, bool
        ):
            raise ValueError("run.signal_squeezing must be a number.")

        signal_idler_transmissivity = run.get(
            "signal_idler_transmissivity", cls.signal_idler_transmissivity
        )
        if not isinstance(signal_idler_transmissivity, int | float) or isinstance(
            signal_idler_transmissivity, bool
        ):
            raise ValueError("run.signal_idler_transmissivity must be a number.")

        log_level = run.get("log_level", cls.log_level)
        if not isinstance(log_level, str) or not log_level.strip():
            raise ValueError("run.log_level must be a non-empty string.")

        seed = run.get("seed")
        if seed is not None and (not isinstance(seed, int) or isinstance(seed, bool)):
            raise ValueError("run.seed must be an integer or omitted.")

        return cls(
            output_dir=Path(output_dir),
            circuit_name=circuit_name,
            signal_squeezing=float(signal_squeezing),
            signal_idler_transmissivity=float(signal_idler_transmissivity),
            log_level=log_level,
            seed=seed,
        )
