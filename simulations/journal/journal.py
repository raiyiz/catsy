"""Persistent experiment log for reproducible catsy simulations."""

from __future__ import annotations

import datetime
import json
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np

if TYPE_CHECKING:
    from catsy.circuits import Circuit
    from catsy.gaussian import GaussianState

from catsy.types import CircuitData, JsonObject

SCHEMA_VERSION = "2.1.0"


def _make_entry_id() -> str:
    stamp = datetime.datetime.now(datetime.UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}_{uuid.uuid4().hex[:8]}"


def _atomic_write_text(path: Path, text: str) -> None:
    tmp_path = path.with_name(path.name + ".tmp")
    tmp_path.write_text(text)
    tmp_path.replace(path)


def _atomic_write_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    tmp_path = path.with_name(path.name + ".tmp")
    with open(tmp_path, "wb") as f:
        np.savez_compressed(f, **arrays)  # type: ignore[arg-type]
    tmp_path.replace(path)


def _split_array_payload(payload: object) -> tuple[np.ndarray, dict[str, object]]:
    if isinstance(payload, dict):
        if "data" not in payload:
            raise ValueError("Array payload dict must contain a 'data' key.")
        data = np.asarray(payload["data"])
        unit = cast(str, payload.get("unit", "arbitrary_units"))
        dimensions = cast(list[str], payload.get("dimensions", []))
        description = cast(str, payload.get("description", ""))
    else:
        data = np.asarray(payload)
        unit = "arbitrary_units"
        dimensions = []
        description = ""
    return data, {
        "description": description,
        "unit": unit,
        "dimensions": dimensions,
        "shape": list(data.shape),
        "dtype": str(data.dtype),
    }


@dataclass
class SimulationRun:
    run_name: str
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    timestamp: str = field(default_factory=lambda: datetime.datetime.now().isoformat())
    circuit: CircuitData | None = None
    final_state_cv: dict[str, object] | None = None
    scalar_results: JsonObject = field(default_factory=dict)
    data_payloads: dict[str, dict[str, object]] = field(default_factory=dict)

    @property
    def results(self) -> JsonObject:
        return self.scalar_results

    @property
    def arrays(self) -> dict[str, dict[str, object]]:
        return self.data_payloads

    def to_dict(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "run_name": self.run_name,
            "timestamp": self.timestamp,
            "circuit": self.circuit,
            "final_state_cv": self.final_state_cv,
            "scalar_results": self.scalar_results,
            "data_payloads": self.data_payloads,
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "SimulationRun":
        return cls(
            run_id=cast(str, data["run_id"]),
            run_name=cast(str, data["run_name"]),
            timestamp=cast(str, data["timestamp"]),
            circuit=cast("CircuitData | None", data.get("circuit")),
            final_state_cv=cast("dict[str, object] | None", data.get("final_state_cv")),
            scalar_results=cast(JsonObject, data.get("scalar_results", {})),
            data_payloads=cast(dict[str, dict[str, object]], data.get("data_payloads", {})),
        )


@dataclass
class JournalEntry:
    title: str = "Untitled Simulation"
    entry_id: str = field(default_factory=_make_entry_id)
    timestamp: str = field(default_factory=lambda: datetime.datetime.now().isoformat())
    tags: list[str] = field(default_factory=list)
    notes: str = ""
    metadata: JsonObject = field(default_factory=dict)
    runs: list[SimulationRun] = field(default_factory=list)
    _pending_arrays: dict[str, np.ndarray] = field(default_factory=dict, init=False, repr=False, compare=False)
    _npz_file: np.lib.npyio.NpzFile | None = field(default=None, init=False, repr=False, compare=False)

    def _store_array(self, key: str, payload: object) -> dict[str, object]:
        data, meta = _split_array_payload(payload)
        self._pending_arrays[key] = data
        return {"npz_key": key, **meta}

    def log_run(
        self,
        run_name: str,
        *,
        circuit: "Circuit" | None = None,
        final_state: "GaussianState" | None = None,
        metrics: JsonObject | None = None,
        arrays: dict[str, object] | None = None,
    ) -> SimulationRun:
        run = SimulationRun(
            run_name=run_name,
            circuit=circuit.to_dict() if circuit is not None else None,
            scalar_results=dict(metrics or {}),
        )
        if final_state is not None:
            d_key = f"{run.run_id}__final_state__displacement"
            v_key = f"{run.run_id}__final_state__covariance"
            self._pending_arrays[d_key] = np.asarray(final_state.displacement)
            self._pending_arrays[v_key] = np.asarray(final_state.covariance)
            run.final_state_cv = {
                "modes": list(final_state.modes),
                "displacement_npz_key": d_key,
                "covariance_npz_key": v_key,
            }
        for name, payload in (arrays or {}).items():
            run.data_payloads[name] = self._store_array(f"{run.run_id}__{name}", payload)
        self.runs.append(run)
        return run

    def get_array(self, npz_key: str) -> np.ndarray:
        if npz_key in self._pending_arrays:
            return self._pending_arrays[npz_key]
        if self._npz_file is not None and npz_key in self._npz_file.files:
            return self._npz_file[npz_key]
        raise KeyError(f"No array logged under key {npz_key!r}.")

    def get_final_state(self, run: SimulationRun) -> "GaussianState":
        if run.final_state_cv is None:
            raise ValueError(f"Run '{run.run_name}' has no logged final state.")
        cv = run.final_state_cv
        from catsy.gaussian import GaussianState
        return GaussianState(
            modes=tuple(cast(list[str], cv["modes"])),
            displacement=self.get_array(cast(str, cv["displacement_npz_key"])),
            covariance=self.get_array(cast(str, cv["covariance_npz_key"])),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "metadata": {
                "entry_id": self.entry_id,
                "title": self.title,
                "timestamp": self.timestamp,
                "tags": self.tags,
                "notes": self.notes,
                "custom": self.metadata,
            },
            "runs": [run.to_dict() for run in self.runs],
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "JournalEntry":
        meta = cast(dict[str, object], data["metadata"])
        entry = cls(
            title=cast(str, meta["title"]),
            entry_id=cast(str, meta["entry_id"]),
            timestamp=cast(str, meta["timestamp"]),
            tags=cast("list[str]", list(cast("list[object]", meta.get("tags", [])))),
            notes=cast(str, meta.get("notes", "")),
            metadata=dict(cast("dict[str, object]", meta.get("custom", meta.get("metadata", {})))),
        )
        entry.runs = [SimulationRun.from_dict(cast(dict[str, object], r)) for r in cast("list[object]", data["runs"])]
        return entry

    def save(self, directory: str | Path) -> Path:
        dir_path = Path(directory)
        dir_path.mkdir(parents=True, exist_ok=True)
        json_path = dir_path / f"entry_{self.entry_id}.json"
        npz_path = dir_path / f"entry_{self.entry_id}.npz"
        _atomic_write_text(json_path, json.dumps(self.to_dict(), indent=2))
        if self._pending_arrays or self._npz_file is not None:
            combined: dict[str, np.ndarray] = {}
            if self._npz_file is not None:
                combined.update({k: self._npz_file[k] for k in self._npz_file.files})
                self._npz_file.close()
            combined.update(self._pending_arrays)
            _atomic_write_npz(npz_path, combined)
            self._npz_file = np.load(npz_path)
            self._pending_arrays = {}
        return json_path

    @classmethod
    def load(cls, json_path: str | Path) -> "JournalEntry":
        json_path = Path(json_path)
        entry = cls.from_dict(cast(dict[str, object], json.loads(json_path.read_text())))
        npz_path = json_path.with_suffix(".npz")
        if npz_path.exists():
            entry._npz_file = np.load(npz_path)
        return entry

    def close(self) -> None:
        if self._npz_file is not None:
            self._npz_file.close()
            self._npz_file = None


class SimulationJournal:
    def __init__(self, storage_path: str | Path):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

    def new_entry(self, title: str, tags: list[str] | None = None, notes: str = "", metadata: JsonObject | None = None) -> JournalEntry:
        return JournalEntry(title=title, tags=list(tags or []), notes=notes, metadata=dict(metadata or {}))

    def load_entry(self, entry_id: str) -> JournalEntry:
        return JournalEntry.load(self.storage_path / f"entry_{entry_id}.json")

    def list_entries(self) -> list[JsonObject]:
        return self.fetch_history_summary()

    def get_entry(self, entry_id: str) -> JournalEntry:
        return self.load_entry(entry_id)

    def find(self, *, tag: str | None = None, title: str | None = None) -> list[JsonObject]:
        summaries = self.fetch_history_summary()
        if tag is not None:
            summaries = [s for s in summaries if tag in cast(list[str], s["tags"])]
        if title is not None:
            needle = title.casefold()
            summaries = [s for s in summaries if needle in cast(str, s["title"]).casefold()]
        return summaries

    def fetch_history_summary(self) -> list[JsonObject]:
        summaries = []
        for file in self.storage_path.glob("entry_*.json"):
            with open(file, encoding="utf-8") as f:
                meta = cast(dict[str, object], json.load(f)["metadata"])
            summaries.append({
                "entry_id": meta["entry_id"],
                "title": meta["title"],
                "timestamp": meta["timestamp"],
                "tags": meta["tags"],
                "file_path": str(file),
            })
        return sorted(summaries, key=lambda s: cast(str, s["timestamp"]), reverse=True)
