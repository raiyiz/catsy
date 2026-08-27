"""Configuration loading for the executable examples."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RunConfig:
    """Validated settings for an example run."""

    output_dir: Path
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

        seed = run.get("seed")
        if seed is not None and (not isinstance(seed, int) or isinstance(seed, bool)):
            raise ValueError("run.seed must be an integer or omitted.")

        return cls(output_dir=Path(output_dir), seed=seed)
