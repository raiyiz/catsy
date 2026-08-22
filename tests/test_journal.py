import json

import numpy as np
import pytest

from catsy.core import Circuit, Gate
from catsy.gaussian import (
    GaussianState,
    beam_splitter,
    compute_duan_inseparability,
    compute_joint_correlation,
    squeeze,
)
from catsy.journal import JournalEntry, SimulationJournal


def test_log_run_can_record_a_run_without_a_circuit():
    entry = JournalEntry(title="Fock calculation")
    run = entry.log_run("Photon subtraction", metrics={"success": True})

    assert run.circuit is None
    assert run.results == {"success": True}


def test_log_run_with_inline_circuit_embeds_full_definition():
    circuit = Circuit().add_mode("a").add_mode("b")
    circuit.add_gate(
        Gate(
            name="Squeezer",
            transform=squeeze,
            modes=("a",),
            kwargs={"r": 0.8, "theta": 0.0},
        )
    )
    circuit.add_gate(
        Gate(
            name="Squeezer",
            transform=squeeze,
            modes=("b",),
            kwargs={"r": 0.8, "theta": np.pi / 2},
        )
    )
    circuit.add_gate(
        Gate(
            name="BeamSplitter",
            transform=beam_splitter,
            modes=("a", "b"),
            kwargs={"eta": 0.5},
        )
    )
    final_state = circuit.run(GaussianState.vacuum(("a", "b")))

    P, X_a, X_b, _, _ = compute_joint_correlation(final_state, "a", "b", quadrature="x")

    entry = JournalEntry(title="EPR BeamSplitter Correlation Scan")
    run = entry.log_run(
        "Correlation scan",
        circuit=circuit,
        final_state=final_state,
        arrays={"joint_prob_x": P, "grid_coords_xa": X_a, "grid_coords_xb": X_b},
    )

    assert run.circuit["modes"] == ["a", "b"]
    assert run.final_state_cv["modes"] == ["a", "b"]

    payload_meta = run.arrays["joint_prob_x"]
    assert payload_meta["shape"] == list(P.shape)
    assert payload_meta["unit"] == "arbitrary_units"
    np.testing.assert_array_equal(entry.get_array(payload_meta["npz_key"]), P)

    reconstructed = entry.get_final_state(run)
    np.testing.assert_array_equal(reconstructed.displacement, final_state.displacement)
    np.testing.assert_array_equal(reconstructed.covariance, final_state.covariance)
    assert reconstructed.modes == final_state.modes


def test_log_run_accepts_raw_array_or_annotated_dict_payloads():
    entry = JournalEntry(title="Payload shapes")
    run = entry.log_run(
        "run",
        circuit=Circuit(),
        arrays={
            "raw": np.array([1.0, 2.0]),
            "annotated": {
                "data": [3.0, 4.0],
                "unit": "V",
                "dimensions": ["t"],
                "description": "test signal",
            },
        },
    )
    raw_meta = run.data_payloads["raw"]
    assert raw_meta["unit"] == "arbitrary_units"
    assert raw_meta["dimensions"] == []
    np.testing.assert_array_equal(entry.get_array(raw_meta["npz_key"]), [1.0, 2.0])

    annotated_meta = run.data_payloads["annotated"]
    assert annotated_meta["unit"] == "V"
    assert annotated_meta["dimensions"] == ["t"]
    assert annotated_meta["description"] == "test signal"
    np.testing.assert_array_equal(entry.get_array(annotated_meta["npz_key"]), [3.0, 4.0])


def test_log_run_rejects_array_dict_payload_missing_data_key():
    entry = JournalEntry(title="Bad payload")
    with pytest.raises(ValueError):
        entry.log_run("run", circuit=Circuit(), arrays={"x": {"unit": "V"}})


def test_get_array_raises_for_unknown_key():
    entry = JournalEntry(title="No arrays logged")
    with pytest.raises(KeyError):
        entry.get_array("nonexistent")


def test_get_final_state_raises_when_run_has_none_logged():
    entry = JournalEntry(title="No final state")
    run = entry.log_run("run", circuit=Circuit())
    with pytest.raises(ValueError):
        entry.get_final_state(run)


# JournalEntry serialization and persistence


def test_journal_entry_initialization_generates_metadata():
    entry = JournalEntry(
        title="Entanglement Test Log",
        tags=["quantum", "unit-test"],
        metadata={"purpose": "test"},
    )
    assert entry.title == "Entanglement Test Log"
    assert isinstance(entry.entry_id, str) and entry.entry_id
    assert entry.runs == []
    assert "quantum" in entry.tags
    assert entry.metadata["purpose"] == "test"


def test_to_dict_matches_schema_and_excludes_array_data():
    entry = JournalEntry(
        title="Schema Test",
        metadata={"purpose": "regression", "temperature": 4.2},
    )
    entry.log_run("run", circuit=Circuit(), metrics={"purity": 1.0}, arrays={"x": [1, 2]})

    serialized = entry.to_dict()
    assert serialized["schema_version"] == "2.1.0"
    assert serialized["metadata"]["title"] == "Schema Test"
    assert serialized["metadata"]["custom"]["purpose"] == "regression"
    assert len(serialized["runs"]) == 1
    assert serialized["runs"][0]["scalar_results"] == {"purity": 1.0}
    # The array metadata (shape/unit/npz_key) is there, but never the values.
    assert "npz_key" in serialized["runs"][0]["data_payloads"]["x"]
    assert "values" not in serialized["runs"][0]["data_payloads"]["x"]
    assert json.dumps(serialized)


def test_save_writes_only_json_when_entry_has_no_array_data(tmp_path):
    entry = JournalEntry(title="No arrays")
    entry.log_run("run", circuit=Circuit(), metrics={"purity": 1.0})

    saved_path = entry.save(tmp_path)

    assert saved_path.exists()
    assert saved_path.suffix == ".json"
    assert not saved_path.with_suffix(".npz").exists()


def test_save_writes_a_companion_npz_when_arrays_are_logged(tmp_path):
    entry = JournalEntry(title="Has arrays")
    entry.log_run("run", circuit=Circuit(), arrays={"x": [1.0, 2.0, 3.0]})

    saved_path = entry.save(tmp_path)

    npz_path = saved_path.with_suffix(".npz")
    assert npz_path.exists()
    # No leftover .tmp files from the atomic-write step.
    assert list(tmp_path.glob("*.tmp")) == []


def test_save_creates_missing_directories(tmp_path):
    deep_dir = tmp_path / "deep" / "nested" / "sub" / "folder"
    entry = JournalEntry(title="Deep Path Test")
    entry.log_run("run", circuit=Circuit())

    saved_path = entry.save(deep_dir)

    assert saved_path.exists()
    assert deep_dir.is_dir()


def test_save_and_load_roundtrip_preserves_runs_and_arrays(tmp_path):
    entry = JournalEntry(title="E2E Integration Test", tags=["e2e"])
    entry.log_run(
        "Sweep Point A",
        circuit=Circuit(),
        metrics={"fidelity": 0.95},
        arrays={"quadratures": {"data": [1.2, 3.4], "unit": "Squeezing Level"}},
    )
    entry.log_run(
        "Sweep Point B",
        circuit=Circuit(),
        metrics={"fidelity": 0.88},
        arrays={"quadratures": {"data": [5.6, 7.8], "unit": "Squeezing Level"}},
    )
    saved_path = entry.save(tmp_path)

    reloaded = JournalEntry.load(saved_path)

    assert reloaded.title == "E2E Integration Test"
    assert reloaded.tags == ["e2e"]
    assert len(reloaded.runs) == 2

    point_b = reloaded.runs[1]
    assert point_b.run_name == "Sweep Point B"
    assert point_b.scalar_results["fidelity"] == 0.88
    quad_meta = point_b.data_payloads["quadratures"]
    assert quad_meta["unit"] == "Squeezing Level"
    np.testing.assert_array_equal(reloaded.get_array(quad_meta["npz_key"]), [5.6, 7.8])


def test_resaving_after_reload_preserves_previously_logged_arrays(tmp_path):
    """Loading an entry back and logging a *new* run shouldn't lose the
    arrays that were already on disk from before the reload."""
    entry = JournalEntry(title="Grows over time")
    entry.log_run("First run", circuit=Circuit(), arrays={"x": [1.0, 2.0]})
    saved_path = entry.save(tmp_path)

    reloaded = JournalEntry.load(saved_path)
    reloaded.log_run("Second run", circuit=Circuit(), arrays={"y": [3.0, 4.0]})
    reloaded.save(tmp_path)

    fully_reloaded = JournalEntry.load(saved_path)
    assert len(fully_reloaded.runs) == 2
    x_key = fully_reloaded.runs[0].data_payloads["x"]["npz_key"]
    y_key = fully_reloaded.runs[1].data_payloads["y"]["npz_key"]
    np.testing.assert_array_equal(fully_reloaded.get_array(x_key), [1.0, 2.0])
    np.testing.assert_array_equal(fully_reloaded.get_array(y_key), [3.0, 4.0])


def test_close_releases_the_companion_npz_handle(tmp_path):
    # load() opens the companion .npz eagerly (so get_array reads are fast
    # and repeatable); close() is the other half of that contract -- it must
    # actually release the handle, not just clear a flag, or long-lived
    # analysis scripts that load many entries in a loop would leak open file
    # descriptors.
    entry = JournalEntry(title="Handle Lifecycle")
    entry.log_run("Run", circuit=Circuit(), arrays={"x": [1.0, 2.0]})
    saved_path = entry.save(tmp_path)

    reloaded = JournalEntry.load(saved_path)
    assert reloaded._npz_file is not None  # opened eagerly by load()
    key = reloaded.runs[0].data_payloads["x"]["npz_key"]
    np.testing.assert_array_equal(reloaded.get_array(key), [1.0, 2.0])

    reloaded.close()
    assert reloaded._npz_file is None

    # Calling close() again (no handle open) must be a no-op, not an error --
    # e.g. a caller wrapping load()/close() in a context manager-style
    # try/finally shouldn't have to track whether close() already ran.
    reloaded.close()
    assert reloaded._npz_file is None


# SimulationJournal


def test_new_entry_is_pre_populated_with_title_tags_and_notes(tmp_path):
    journal = SimulationJournal(tmp_path / "runs")
    entry = journal.new_entry(
        title="EPR BeamSplitter Correlation Scan",
        tags=["entanglement", "epr", "gaussian"],
        notes="Testing phase-space cross-correlations on 50:50 beam splitter.",
    )
    assert entry.title == "EPR BeamSplitter Correlation Scan"
    assert entry.tags == ["entanglement", "epr", "gaussian"]
    assert entry.notes.startswith("Testing phase-space")


def test_new_entry_accepts_custom_metadata(tmp_path):
    journal = SimulationJournal(tmp_path / "runs")
    entry = journal.new_entry(
        title="Metadata test",
        metadata={"sample": "vacuum", "temperature": 4.2},
    )
    assert entry.metadata == {"sample": "vacuum", "temperature": 4.2}


def test_load_entry_resolves_the_id_to_its_saved_file(tmp_path):
    journal = SimulationJournal(tmp_path / "runs")
    entry = journal.new_entry(title="Findable entry")
    entry.log_run("run", circuit=Circuit(), arrays={"x": [1.0]})
    entry.save(journal.storage_path)

    reloaded = journal.load_entry(entry.entry_id)
    assert reloaded.title == "Findable entry"
    np.testing.assert_array_equal(
        reloaded.get_array(reloaded.runs[0].data_payloads["x"]["npz_key"]), [1.0]
    )


def test_fetch_history_summary_orders_entries_most_recent_first(tmp_path):
    journal = SimulationJournal(tmp_path / "runs")

    older = journal.new_entry(title="Older run")
    older.timestamp = "2024-01-01T00:00:00"
    older.log_run("run", circuit=Circuit())
    older.save(journal.storage_path)

    newer = journal.new_entry(title="Newer run")
    newer.timestamp = "2024-06-01T00:00:00"
    newer.log_run("run", circuit=Circuit())
    newer.save(journal.storage_path)

    summaries = journal.fetch_history_summary()
    assert [s["title"] for s in summaries] == ["Newer run", "Older run"]


def test_list_entries_is_an_alias_for_history_summary(tmp_path):
    journal = SimulationJournal(tmp_path / "runs")
    entry = journal.new_entry(title="Listed entry", tags=["cat"])
    entry.save(journal.storage_path)

    assert journal.list_entries()[0]["entry_id"] == entry.entry_id


def test_get_entry_loads_by_id(tmp_path):
    journal = SimulationJournal(tmp_path / "runs")
    entry = journal.new_entry(title="Get me")
    entry.save(journal.storage_path)

    loaded = journal.get_entry(entry.entry_id)
    assert loaded.title == "Get me"


def test_find_filters_by_tag_and_title(tmp_path):
    journal = SimulationJournal(tmp_path / "runs")
    first = journal.new_entry(title="Cat interferometer", tags=["cat", "mzi"])
    first.save(journal.storage_path)
    second = journal.new_entry(title="Gaussian EPR", tags=["epr"])
    second.save(journal.storage_path)

    assert [x["entry_id"] for x in journal.find(tag="cat")] == [first.entry_id]
    assert [x["entry_id"] for x in journal.find(title="ePr")] == [second.entry_id]


def test_fetch_history_summary_does_not_open_companion_npz_files(tmp_path, monkeypatch):
    journal = SimulationJournal(tmp_path / "runs")
    entry = journal.new_entry(title="Has a big array")
    entry.log_run("run", circuit=Circuit(), arrays={"x": np.zeros(1000)})
    entry.save(journal.storage_path)

    original_np_load = np.load

    def fail_if_called(*args, **kwargs):
        raise AssertionError("fetch_history_summary must not open .npz files")

    monkeypatch.setattr(np, "load", fail_if_called)
    try:
        summaries = journal.fetch_history_summary()
    finally:
        monkeypatch.setattr(np, "load", original_np_load)

    assert summaries[0]["title"] == "Has a big array"


# End-to-end integration


def test_journal_records_a_full_circuit_experiment(tmp_path):
    circuit = Circuit().add_mode("a").add_mode("b")
    circuit.add_gate(
        Gate(
            name="Squeezer",
            transform=squeeze,
            modes=("a",),
            kwargs={"r": 0.8, "theta": 0.0},
        )
    )
    circuit.add_gate(
        Gate(
            name="Squeezer",
            transform=squeeze,
            modes=("b",),
            kwargs={"r": 0.8, "theta": np.pi / 2},
        )
    )
    circuit.add_gate(
        Gate(
            name="BeamSplitter",
            transform=beam_splitter,
            modes=("a", "b"),
            kwargs={"eta": 0.5},
        )
    )
    result = circuit.run(GaussianState.vacuum(("a", "b")))
    duan_score = compute_duan_inseparability(result, "a", "b")

    journal = SimulationJournal(tmp_path / "journal_store")
    entry = journal.new_entry(
        title="EPR Witness",
        tags=["entanglement", "gaussian"],
        metadata={"purpose": "integration test"},
    )
    entry.log_run(
        "EPR circuit",
        circuit=circuit,
        final_state=result,
        metrics={"duan_score": duan_score},
    )
    saved_path = entry.save(journal.storage_path)

    reloaded = journal.get_entry(entry.entry_id)
    run = reloaded.runs[0]
    assert reloaded.metadata["purpose"] == "integration test"
    assert run.results["duan_score"] == pytest.approx(duan_score)

    reconstructed_state = reloaded.get_final_state(run)
    np.testing.assert_allclose(reconstructed_state.covariance, result.covariance)
    assert saved_path.exists()
