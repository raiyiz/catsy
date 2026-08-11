"""Persistent experiment log for catst simulations.

A `JournalEntry` records one experiment: its metadata (title, tags, notes)
plus one or more logged simulation runs. Each run pairs a hardware
description -- either an inline `GaussianCircuit` (full gate sequence
embedded) or a reference to a saved `OpticalSetup` layout file (referenced,
not duplicated) -- with its scalar results and any heavy array payloads.
`SimulationJournal` indexes a directory of saved entries.

Design notes (mirrors states.py):
  - Entries are JSON-serializable dataclasses; callers pass numpy arrays
    into `JournalEntry.log_run` and get plain nested lists back out of
    `to_dict` / the saved file.
  - All public entry points validate their inputs and raise ValueError with
    a specific message; nothing relies on `assert`.
"""

from __future__ import annotations

import datetime
import json
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .states import GaussianCircuit, GaussianState

SCHEMA_VERSION = "1.0.0"


def _normalize_array_payload(payload: Any) -> dict[str, Any]:
    """Converts one array-ish `log_run` payload into its canonical,
    JSON-safe form. Accepts either a raw array/list or an already-annotated
    ``{"data": ..., "unit": ..., "dimensions": ...}`` mapping."""
    if isinstance(payload, dict):
        if "data" not in payload:
            raise ValueError("Array payload dict must contain a 'data' key.")
        data = payload["data"]
        if isinstance(data, np.ndarray):
            data = data.tolist()
        return {
            "values": data,
            "unit": payload.get("unit", "arbitrary_units"),
            "dimensions": payload.get("dimensions", []),
        }
    if isinstance(payload, np.ndarray):
        payload = payload.tolist()
    return {"values": payload, "unit": "arbitrary_units", "dimensions": []}


# ---------------------------------------------------------------------------
# Journal entry
# ---------------------------------------------------------------------------


@dataclass
class SimulationRun:
    """One logged execution within a `JournalEntry`."""

    run_name: str
    run_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(default_factory=lambda: datetime.datetime.now().isoformat())
    circuit: dict[str, Any] | None = None
    hardware_layout_reference: str | None = None
    final_state_cv: dict[str, Any] | None = None
    scalar_results: dict[str, Any] = field(default_factory=dict)
    data_payloads: dict[str, dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "run_name": self.run_name,
            "timestamp": self.timestamp,
            "circuit": self.circuit,
            "hardware_layout_reference": self.hardware_layout_reference,
            "final_state_cv": self.final_state_cv,
            "scalar_results": self.scalar_results,
            "data_payloads": self.data_payloads,
        }


@dataclass
class JournalEntry:
    """A single experiment record: metadata plus a sequence of logged runs."""

    title: str = "Untitled Simulation"
    entry_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(default_factory=lambda: datetime.datetime.now().isoformat())
    tags: list[str] = field(default_factory=list)
    notes: str = ""
    runs: list[SimulationRun] = field(default_factory=list)

    def log_run(
        self,
        run_name: str,
        *,
        circuit: GaussianCircuit | None = None,
        setup_layout_file: str | Path | None = None,
        final_state: GaussianState | None = None,
        metrics: dict[str, Any] | None = None,
        arrays: dict[str, Any] | None = None,
    ) -> SimulationRun:
        """Logs one execution of a hardware setup.

        Give either `circuit` (a `GaussianCircuit`; its full gate sequence
        is embedded inline) or `setup_layout_file` (a path to a saved
        `OpticalSetup` layout; referenced, not duplicated) -- not both.
        `final_state` optionally records the resulting `GaussianState`.
        `metrics` holds single-value results (e.g. ``{"purity": 0.98}``).
        `arrays` holds heavy numeric payloads keyed by name, each either a
        raw array or an annotated
        ``{"data": ..., "unit": ..., "dimensions": ...}`` mapping.
        """
        if circuit is not None and setup_layout_file is not None:
            raise ValueError("Pass either `circuit` or `setup_layout_file`, not both.")
        if circuit is None and setup_layout_file is None:
            raise ValueError("Must supply either `circuit` or `setup_layout_file`.")

        run = SimulationRun(
            run_name=run_name,
            circuit=circuit.to_dict() if circuit is not None else None,
            hardware_layout_reference=(
                str(setup_layout_file) if setup_layout_file is not None else None
            ),
            final_state_cv=final_state.to_dict() if final_state is not None else None,
            scalar_results=dict(metrics or {}),
            data_payloads={
                key: _normalize_array_payload(payload)
                for key, payload in (arrays or {}).items()
            },
        )
        self.runs.append(run)
        return run

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "metadata": {
                "entry_id": self.entry_id,
                "title": self.title,
                "timestamp": self.timestamp,
                "tags": self.tags,
                "notes": self.notes,
            },
            "runs": [run.to_dict() for run in self.runs],
        }

    def save(self, directory: str | Path) -> Path:
        dir_path = Path(directory)
        dir_path.mkdir(parents=True, exist_ok=True)
        file_path = dir_path / f"entry_{self.entry_id}.json"
        file_path.write_text(json.dumps(self.to_dict(), indent=2))
        return file_path


# ---------------------------------------------------------------------------
# Journal index
# ---------------------------------------------------------------------------


class SimulationJournal:
    """Manages a directory of serialized `JournalEntry` files."""

    def __init__(self, storage_path: str | Path):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

    def new_entry(
        self, title: str, tags: list[str] | None = None, notes: str = ""
    ) -> JournalEntry:
        return JournalEntry(title=title, tags=list(tags or []), notes=notes)

    def fetch_history_summary(self) -> list[dict[str, Any]]:
        """Scans the storage directory's entries for their index metadata,
        without hydrating each entry's full run/array payloads into memory."""
        summaries = []
        for file in self.storage_path.glob("entry_*.json"):
            with open(file, "r") as f:
                meta = json.load(f)["metadata"]
            summaries.append(
                {
                    "entry_id": meta["entry_id"],
                    "title": meta["title"],
                    "timestamp": meta["timestamp"],
                    "tags": meta["tags"],
                    "file_path": str(file),
                }
            )
        return sorted(summaries, key=lambda s: s["timestamp"], reverse=True)
