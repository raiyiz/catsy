#import "@preview/physica:0.9.8": *

// ==========================================
// CHAPTER 9
// ==========================================
= Chapter 9: Experiment Persistence (`journal.py`)

Simulation results are worthless if they only exist in interpreter memory. `journal.py` provides a lightweight, file-based lab notebook: `JournalEntry` logs an experiment (title, tags, notes, any number of `SimulationRun` runs with circuit, final state, scalar results, and array data); `SimulationJournal` indexes a directory of such entries.

== Why two files per entry?

An entry is consistently split into two files according to access pattern:

- *`entry_<id>.json`* — metadata, scalar results, and small array *annotations* (description, unit, dimensions, shape, dtype). Always small; this is the only file `fetch_history_summary` reads when merely browsing entries.
- *`entry_<id>.npz`* — every actual NumPy array logged (run results, final-state covariances/displacements), compressed. Only written if the entry has any array data at all.

The reasoning is purely pragmatic: a numeric grid serialized as decimal-text JSON balloons to a multiple of its binary size, and `json.load` has to parse the *entire* file just to read a title. More broadly, separating human-readable metadata from bulk numerical arrays supports the reproducibility and data-management practices recommended in #link("https://doi.org/10.1038/sdata.2016.18")[Wilkinson et al. (2016)] and #link("https://doi.org/10.1371/journal.pcbi.1005510")[Wilson et al. (2017)]. By splitting the two, listing/searching entries stays cheap regardless of how much array data is attached to them — and `numpy.load` on an `.npz` only decompresses an individual array upon actual index access anyway, so `get_array` only pays for the arrays that are actually requested.

== Logging a run (`log_run`)

```python
def log_run(
    self, run_name: str, *, circuit=None, final_state=None, metrics=None, arrays=None,
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
```

`circuit` directly reuses the serialization from Chapter 3 (`GaussianCircuit.to_dict`), so a logged run carries the exact, reproducible circuit along with it. `final_state` is not stored as a full `GaussianState` JSON blob, but as a displacement-vector/covariance-matrix array pair under generated, run-unique keys — consistently taking the same array path as any other `arrays`. Array data is initially held only *in memory* (`_pending_arrays`) and only actually written to disk on `save`.

== Atomic writes

Both JSON and NPZ writes follow the same pattern: write to a `.tmp` file, then atomically move it to the target name via `Path.replace`.

```python
def _atomic_write_text(path: Path, text: str) -> None:
    tmp_path = path.with_name(path.name + ".tmp")
    tmp_path.write_text(text)
    tmp_path.replace(path)
```

This means a crash *during* the write can never corrupt an already-existing, valid entry — the end state is always either the old or the fully new version, never an intermediate one. `save` also merges newly written array data with data already on disk (`combined`) rather than overwriting the `.npz`, so repeated `log_run` + `save` calls on the same entry accumulate losslessly.

== Directory index (`SimulationJournal`)

`SimulationJournal` itself holds no state beyond the storage path and reads freshly from the filesystem on every query:

```python
def fetch_history_summary(self) -> list[dict[str, Any]]:
    summaries = []
    for file in self.storage_path.glob("entry_*.json"):
        with open(file, "r") as f:
            meta = json.load(f)["metadata"]
        summaries.append({
            "entry_id": meta["entry_id"], "title": meta["title"],
            "timestamp": meta["timestamp"], "tags": meta["tags"],
            "file_path": str(file),
        })
    return sorted(summaries, key=lambda s: s["timestamp"], reverse=True)
```

Since only the small `.json` files are opened here, browsing (`find`, filterable by tag and/or title substring) and listing (`list_entries`) stay cheap even with very many or very large entries; the corresponding `.npz` files are only touched by an explicit `load_entry`/`get_entry` call followed by `get_array`/`get_final_state` access. Entry IDs (`_make_entry_id`) are UTC timestamps with a short random suffix — so even a plain directory `ls`/`glob` already sorts in creation order, while concurrent entries never collide.

---


== Scientific literature
This chapter is principally about software and data management rather than quantum-optical theory. Its reproducibility and persistence goals are aligned with:

- #link("https://doi.org/10.1038/sdata.2016.18")[M. D. Wilkinson et al. et al., “The FAIR Guiding Principles for scientific data management and stewardship,” *Scientific Data* 3, 160018 (2016).]
- #link("https://doi.org/10.1371/journal.pcbi.1005510")[G. Wilson et al., “Good enough practices in scientific computing,” *PLoS Computational Biology* 13, e1005510 (2017).]

These references motivate the emphasis on machine-readable metadata, explicit provenance, reproducible storage, and separation of data from descriptive metadata. They do not prescribe the exact JSON/NPZ format used by `catsy`.
