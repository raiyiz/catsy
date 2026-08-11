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
    assert run.data_payloads["joint_prob_x"]["values"] == P.tolist()
    assert run.data_payloads["joint_prob_x"]["unit"] == "arbitrary_units"
    assert len(entry.runs) == 1


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
    wigner = run.data_payloads["wigner"]
    assert wigner["values"] == [[0.1, 0.2], [0.3, 0.4]]
    assert wigner["unit"] == "Wigner Density"
    assert wigner["dimensions"] == ["x", "p"]


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
    assert run.data_payloads["raw"] == {
        "values": [1.0, 2.0],
        "unit": "arbitrary_units",
        "dimensions": [],
    }
    assert run.data_payloads["annotated"] == {
        "values": [3.0, 4.0],
        "unit": "V",
        "dimensions": ["t"],
    }


def test_log_run_rejects_array_dict_payload_missing_data_key():
    entry = JournalEntry(title="Bad payload")
    with pytest.raises(ValueError):
        entry.log_run("run", circuit=GaussianCircuit(), arrays={"x": {"unit": "V"}})


def test_journal_entry_initialization_generates_metadata():
    entry = JournalEntry(title="Entanglement Test Log", tags=["quantum", "unit-test"])
    assert entry.title == "Entanglement Test Log"
    assert isinstance(entry.entry_id, str) and entry.entry_id
    assert entry.runs == []
    assert "quantum" in entry.tags


# ---------------------------------------------------------------------------
# JournalEntry.to_dict / save
# ---------------------------------------------------------------------------


def test_to_dict_matches_schema():
    entry = JournalEntry(title="Schema Test")
    entry.log_run("run", circuit=GaussianCircuit(), metrics={"purity": 1.0})

    serialized = entry.to_dict()
    assert serialized["schema_version"] == "1.0.0"
    assert serialized["metadata"]["title"] == "Schema Test"
    assert len(serialized["runs"]) == 1
    assert serialized["runs"][0]["scalar_results"] == {"purity": 1.0}


def test_save_writes_a_json_file_and_creates_missing_directories(tmp_path):
    deep_dir = tmp_path / "deep" / "nested" / "sub" / "folder"
    entry = JournalEntry(title="Deep Path Test")
    entry.log_run("run", circuit=GaussianCircuit())

    saved_path = entry.save(deep_dir)

    assert saved_path.exists()
    assert saved_path.suffix == ".json"
    assert deep_dir.is_dir()


def test_save_and_reload_lifecycle_preserves_run_data(tmp_path):
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
    with open(saved_path) as f:
        loaded_data = json.load(f)

    assert loaded_data["schema_version"] == "1.0.0"
    assert loaded_data["metadata"]["title"] == "E2E Integration Test"
    assert len(loaded_data["runs"]) == 2

    point_b = loaded_data["runs"][1]
    assert point_b["run_name"] == "Sweep Point B"
    assert point_b["scalar_results"]["fidelity"] == 0.88
    assert point_b["data_payloads"]["quadratures"]["values"] == [5.6, 7.8]


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

    with open(saved_path) as f:
        loaded = json.load(f)
    assert loaded["runs"][0]["scalar_results"]["duan_score"] == pytest.approx(
        duan_score
    )
    assert loaded["runs"][0]["hardware_layout_reference"] == str(layout_path)
    assert loaded["runs"][0]["final_state_cv"]["modes"] == ["line_1", "line_2"]
    # Entanglement survives a 10%-loss arm.
    assert duan_score < 2.0
