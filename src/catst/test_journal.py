import json

import numpy as np
import pytest

from .composition import OpticalSetup
from .journal import JournalEntry, SimulationJournal
from .states import (
    GaussianCircuit,
    GaussianOperations,
    compute_duan_inseparability,
    compute_joint_correlation,
)

# ---------------------------------------------------------------------------
# JournalEntry.log_run
# ---------------------------------------------------------------------------


def test_log_run_requires_exactly_one_hardware_reference():
    entry = JournalEntry(title="No hardware given")
    with pytest.raises(ValueError):
        entry.log_run("run")
    with pytest.raises(ValueError):
        entry.log_run(
            "run", circuit=GaussianCircuit(), setup_layout_file="layout.json"
        )


def test_log_run_with_inline_circuit_embeds_full_definition():
    circuit = GaussianCircuit()
    circuit.add_mode("a").add_mode("b")
    circuit.squeeze(mode="a", r=0.8, theta=0.0)
    circuit.squeeze(mode="b", r=0.8, theta=np.pi / 2)
    circuit.beam_splitter(mode_a="a", mode_b="b", eta=0.5)
    final_state = circuit.compile_and_run()

    P, X_a, X_b = compute_joint_correlation(final_state, "a", "b", quadrature="x")

    entry = JournalEntry(title="EPR BeamSplitter Correlation Scan")
    run = entry.log_run(
        "Correlation scan",
        circuit=circuit,
        final_state=final_state,
        arrays={"joint_prob_x": P, "grid_coords_xa": X_a, "grid_coords_xb": X_b},
    )

    assert run.hardware_layout_reference is None
    assert run.circuit["modes"] == ["a", "b"]
    assert run.final_state_cv["modes"] == ["a", "b"]

    # Array metadata sits in the run record; the array data itself is only
    # reachable through get_array (it never touches the JSON side).
    payload_meta = run.data_payloads["joint_prob_x"]
    assert payload_meta["shape"] == list(P.shape)
    assert payload_meta["unit"] == "arbitrary_units"
    np.testing.assert_array_equal(entry.get_array(payload_meta["npz_key"]), P)

    reconstructed = entry.get_final_state(run)
    np.testing.assert_array_equal(reconstructed.displacement, final_state.displacement)
    np.testing.assert_array_equal(reconstructed.covariance, final_state.covariance)
    assert reconstructed.modes == final_state.modes


def test_log_run_with_setup_layout_file_references_rather_than_embeds(tmp_path):
    layout_path = tmp_path / "mzi_node.json"
    OpticalSetup("MZI Node").beam_splitter(
        "BS1", port_a="line_1", port_b="line_2", eta=0.5
    ).save_layout(layout_path)

    entry = JournalEntry(title="MZI sweep")
    run = entry.log_run(
        "Sweep point A",
        setup_layout_file=layout_path,
        metrics={"purity": 0.99, "duan_score": 1.2},
        arrays={
            "wigner": {
                "data": [[0.1, 0.2], [0.3, 0.4]],
                "unit": "Wigner Density",
                "dimensions": ["x", "p"],
            }
        },
    )

    assert run.circuit is None
    assert run.hardware_layout_reference == str(layout_path)
    assert run.scalar_results == {"purity": 0.99, "duan_score": 1.2}

    wigner_meta = run.data_payloads["wigner"]
    assert wigner_meta["unit"] == "Wigner Density"
    assert wigner_meta["dimensions"] == ["x", "p"]
    assert wigner_meta["shape"] == [2, 2]
    np.testing.assert_array_equal(
        entry.get_array(wigner_meta["npz_key"]), [[0.1, 0.2], [0.3, 0.4]]
    )


def test_log_run_accepts_raw_array_or_annotated_dict_payloads():
    entry = JournalEntry(title="Payload shapes")
    run = entry.log_run(
        "run",
        circuit=GaussianCircuit(),
        arrays={
            "raw": np.array([1.0, 2.0]),
            "annotated": {"data": [3.0, 4.0], "unit": "V", "dimensions": ["t"]},
        },
    )
    raw_meta = run.data_payloads["raw"]
    assert raw_meta["unit"] == "arbitrary_units"
    assert raw_meta["dimensions"] == []
    np.testing.assert_array_equal(entry.get_array(raw_meta["npz_key"]), [1.0, 2.0])

    annotated_meta = run.data_payloads["annotated"]
    assert annotated_meta["unit"] == "V"
    assert annotated_meta["dimensions"] == ["t"]
    np.testing.assert_array_equal(entry.get_array(annotated_meta["npz_key"]), [3.0, 4.0])


def test_log_run_rejects_array_dict_payload_missing_data_key():
    entry = JournalEntry(title="Bad payload")
    with pytest.raises(ValueError):
        entry.log_run("run", circuit=GaussianCircuit(), arrays={"x": {"unit": "V"}})


def test_get_array_raises_for_unknown_key():
    entry = JournalEntry(title="No arrays logged")
    with pytest.raises(KeyError):
        entry.get_array("nonexistent")


def test_get_final_state_raises_when_run_has_none_logged():
    entry = JournalEntry(title="No final state")
    run = entry.log_run("run", circuit=GaussianCircuit())
    with pytest.raises(ValueError):
        entry.get_final_state(run)


def test_journal_entry_initialization_generates_metadata():
    entry = JournalEntry(title="Entanglement Test Log", tags=["quantum", "unit-test"])
    assert entry.title == "Entanglement Test Log"
    assert isinstance(entry.entry_id, str) and entry.entry_id
    assert entry.runs == []
    assert "quantum" in entry.tags


# ---------------------------------------------------------------------------
# JournalEntry.to_dict / save / load
# ---------------------------------------------------------------------------


def test_to_dict_matches_schema_and_excludes_array_data():
    entry = JournalEntry(title="Schema Test")
    entry.log_run(
        "run", circuit=GaussianCircuit(), metrics={"purity": 1.0}, arrays={"x": [1, 2]}
    )

    serialized = entry.to_dict()
    assert serialized["schema_version"] == "2.0.0"
    assert serialized["metadata"]["title"] == "Schema Test"
    assert len(serialized["runs"]) == 1
    assert serialized["runs"][0]["scalar_results"] == {"purity": 1.0}
    # The array metadata (shape/unit/npz_key) is there, but never the values.
    assert "npz_key" in serialized["runs"][0]["data_payloads"]["x"]
    assert "values" not in serialized["runs"][0]["data_payloads"]["x"]
    assert json.dumps(serialized)  # whole thing is plain-JSON safe


def test_save_writes_only_json_when_entry_has_no_array_data(tmp_path):
    entry = JournalEntry(title="No arrays")
    entry.log_run("run", circuit=GaussianCircuit(), metrics={"purity": 1.0})

    saved_path = entry.save(tmp_path)

    assert saved_path.exists()
    assert saved_path.suffix == ".json"
    assert not saved_path.with_suffix(".npz").exists()


def test_save_writes_a_companion_npz_when_arrays_are_logged(tmp_path):
    entry = JournalEntry(title="Has arrays")
    entry.log_run("run", circuit=GaussianCircuit(), arrays={"x": [1.0, 2.0, 3.0]})

    saved_path = entry.save(tmp_path)

    npz_path = saved_path.with_suffix(".npz")
    assert npz_path.exists()
    # No leftover .tmp files from the atomic-write step.
    assert list(tmp_path.glob("*.tmp")) == []


def test_save_creates_missing_directories(tmp_path):
    deep_dir = tmp_path / "deep" / "nested" / "sub" / "folder"
    entry = JournalEntry(title="Deep Path Test")
    entry.log_run("run", circuit=GaussianCircuit())

    saved_path = entry.save(deep_dir)

    assert saved_path.exists()
    assert deep_dir.is_dir()


def test_save_and_load_roundtrip_preserves_runs_and_arrays(tmp_path):
    entry = JournalEntry(title="E2E Integration Test", tags=["e2e"])
    entry.log_run(
        "Sweep Point A",
        circuit=GaussianCircuit(),
        metrics={"fidelity": 0.95},
        arrays={"quadratures": {"data": [1.2, 3.4], "unit": "Squeezing Level"}},
    )
    entry.log_run(
        "Sweep Point B",
        circuit=GaussianCircuit(),
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
    entry.log_run("First run", circuit=GaussianCircuit(), arrays={"x": [1.0, 2.0]})
    saved_path = entry.save(tmp_path)

    reloaded = JournalEntry.load(saved_path)
    reloaded.log_run("Second run", circuit=GaussianCircuit(), arrays={"y": [3.0, 4.0]})
    reloaded.save(tmp_path)

    fully_reloaded = JournalEntry.load(saved_path)
    assert len(fully_reloaded.runs) == 2
    x_key = fully_reloaded.runs[0].data_payloads["x"]["npz_key"]
    y_key = fully_reloaded.runs[1].data_payloads["y"]["npz_key"]
    np.testing.assert_array_equal(fully_reloaded.get_array(x_key), [1.0, 2.0])
    np.testing.assert_array_equal(fully_reloaded.get_array(y_key), [3.0, 4.0])


# ---------------------------------------------------------------------------
# SimulationJournal
# ---------------------------------------------------------------------------


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


def test_load_entry_resolves_the_id_to_its_saved_file(tmp_path):
    journal = SimulationJournal(tmp_path / "runs")
    entry = journal.new_entry(title="Findable entry")
    entry.log_run("run", circuit=GaussianCircuit(), arrays={"x": [1.0]})
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
    older.log_run("run", circuit=GaussianCircuit())
    older.save(journal.storage_path)

    newer = journal.new_entry(title="Newer run")
    newer.timestamp = "2024-06-01T00:00:00"
    newer.log_run("run", circuit=GaussianCircuit())
    newer.save(journal.storage_path)

    summaries = journal.fetch_history_summary()
    assert [s["title"] for s in summaries] == ["Newer run", "Older run"]


def test_fetch_history_summary_does_not_open_companion_npz_files(tmp_path, monkeypatch):
    journal = SimulationJournal(tmp_path / "runs")
    entry = journal.new_entry(title="Has a big array")
    entry.log_run("run", circuit=GaussianCircuit(), arrays={"x": np.zeros(1000)})
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


# ---------------------------------------------------------------------------
# Integration: journal + composition + states end to end
# ---------------------------------------------------------------------------


def test_journal_records_a_full_optical_bench_experiment(tmp_path):
    """An EPR pair run through a saved hardware layout, journaled end to
    end: build + save a layout, run a beam scenario through it, compute a
    metric, log the run against the saved layout, and reload it from disk."""
    layout_path = tmp_path / "layouts" / "interferometer_node.json"
    node = OpticalSetup("Interferometer Node")
    node.fiber_loss("Atmospheric Jitter A", port="line_1", eta=0.9)
    node.fiber_loss("Atmospheric Jitter B", port="line_2", eta=1.0)
    node.save_layout(layout_path)

    epr_input = GaussianOperations.create_epr_pair(
        mode_a="line_1", mode_b="line_2", r=1.2
    )
    result_epr = node.process_beam(epr_input)
    duan_score = compute_duan_inseparability(result_epr, "line_1", "line_2")

    journal = SimulationJournal(tmp_path / "journal_store")
    entry = journal.new_entry(title="MZI EPR Witness", tags=["entanglement", "mzi"])
    entry.log_run(
        "EPR through lossy channel",
        setup_layout_file=layout_path,
        final_state=result_epr,
        metrics={"duan_score": duan_score},
    )
    saved_path = entry.save(journal.storage_path)

    reloaded = JournalEntry.load(saved_path)
    run = reloaded.runs[0]
    assert run.scalar_results["duan_score"] == pytest.approx(duan_score)
    assert run.hardware_layout_reference == str(layout_path)

    reconstructed_state = reloaded.get_final_state(run)
    np.testing.assert_allclose(reconstructed_state.covariance, result_epr.covariance)
    # Entanglement survives a 10%-loss arm.
    assert duan_score < 2.0
