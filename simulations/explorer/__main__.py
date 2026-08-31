"""Run the static Simulation Explorer builder from the repository root."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from . import build

REPO_ROOT = Path(__file__).resolve().parents[2]
build.REPO_ROOT = REPO_ROOT
build.EXAMPLES = tuple(
    replace(
        spec,
        config_path=REPO_ROOT / "simulations" / "examples" / spec.config_path.name,
        run_root=REPO_ROOT / "simulations" / spec.run_root,
    )
    for spec in build.EXAMPLES
)

if __name__ == "__main__":
    build.main()
