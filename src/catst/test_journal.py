import json
from pathlib import Path

import pytest

from catst.journal import ImprovedJournalEntry, OpticalSetup, SimulationJournal
from catst.states import GaussianCircuit, compute_joint_correlation


@pytest.fixture
def p():
    return Path(__file__).absolute().parent.parent.parent / "journal" / "tests"


def test_run_recorded_experiment(p):
    # 1. Initialize our Journal workspace
    journal = SimulationJournal(p / "runs")
    entry = journal.new_entry(
        title="EPR BeamSplitter Correlation Scan",
        tags=["entanglement", "epr", "gaussian"],
        notes="Testing phase-space cross-correlations on 50:50 beam splitter.",
    )

    # 2. Build the circuit configuration
    circuit = GaussianCircuit()
    circuit.add_mode("a").add_mode("b")
    circuit.squeeze(mode="a", r=0.8, theta=0.0)
    circuit.squeeze(mode="b", r=0.8, theta=3.14159 / 2)
    circuit.beam_splitter(mode_a="a", mode_b="b", eta=0.5)

    # Attach setup mapping and operation lines to the logger
    entry.attach_circuit(circuit)

    # 3. Execute the simulation
    final_state = circuit.compile_and_run()
    entry.results.final_state_cv = final_state.to_dict()

    # 4. Process secondary metrics and register array arrays
    P, X_a, X_b = compute_joint_correlation(final_state, "a", "b", quadrature="x")
    entry.log_result_array("joint_prob_x", P)
    entry.log_result_array("grid_coords_xa", X_a)
    entry.log_result_array("grid_coords_xb", X_b)

    # 5. Archive entry safely to file system
    saved_file = entry.save(journal.storage_path)
    print(f"Successfully archived experiment ledger to: {saved_file}")


from catst.states import GaussianOperations, compute_duan_inseparability


def test_bench_demo(p):
    layout_p = p / "layouts" / "mzi_node_blueprint.json"
    # Compose a Reusable Optical Bench
    mzi_node = OpticalSetup("Interferometer Node")
    mzi_node.beam_splitter("Input BS", port_a="line_1", port_b="line_2", eta=0.5)
    mzi_node.fiber_loss("Atmospheric Jitter A", port="line_1", eta=0.9)
    mzi_node.phase_shifter("LO Tuning", port="line_2", phi=0.785)
    mzi_node.beam_splitter(
        "Recombination BS", port_a="line_1", port_b="line_2", eta=0.5
    )

    # roundtrip
    mzi_node.save_layout(layout_p)
    loaded_setup = OpticalSetup.load_layout(layout_p)

    # Beam Scenario A: Blasting a standard classical Coherent State down the hardware
    coherent_input = GaussianOperations.create_coherent(
        modes=("line_1", "line_2"), alphas=[1.5 + 0.0j, 2.0j]
    )
    result_coherent = loaded_setup.process_beam(coherent_input)
    print(f"Scenario A (Coherent Output Purity): {result_coherent}")

    # Beam Scenario B: Dropping a highly-entangled EPR Pair into the exact same hardware layout
    epr_input = GaussianOperations.create_epr_pair(
        mode_a="line_1", mode_b="line_2", r=1.2
    )
    result_epr = loaded_setup.process_beam(epr_input)

    # Check if quantum entanglement survived the setup layout using your Duan metric
    duan_score = compute_duan_inseparability(result_epr, "line_1", "line_2")
    print(
        f"Scenario B (EPR Witness, Duan Score): {duan_score:.4f} (Entangled if < 2.0)"
    )


def test_visuals():

    mzi = OpticalSetup("MZI Interferometer Node")
    mzi.beam_splitter("BS1", port_a="line_1", port_b="line_2", eta=0.5)
    mzi.fiber_loss("Loss_A", port="line_1", eta=0.9)
    mzi.phase_shifter("Phase_B", port="line_2", phi=0.785)
    mzi.beam_splitter("BS2", port_a="line_1", port_b="line_2", eta=0.5)

    # Visualisierung mit Angabe der Input-Zustände aufrufen
    mzi.draw(input_states={"line_1": "|α=1.5>", "line_2": "|ξ=0.8>"})

    # another schematic
    mzi = OpticalSetup("Colored MZI Architecture")
    mzi.inline_squeezer("Sqz_Input", port="line_2", r=0.7)
    mzi.beam_splitter("BS1", port_a="line_1", port_b="line_2", eta=0.5)
    mzi.fiber_loss("Loss_A", port="line_1", eta=0.85)
    mzi.phase_shifter("Phase_B", port="line_2", phi=1.57)
    mzi.beam_splitter("BS2", port_a="line_1", port_b="line_2", eta=0.5)

    mzi.draw(input_states={"line_1": "|α=2.0>", "line_2": "|0>"})


# =====================================================================
# FIXTURES (Test-Infrastruktur)
# =====================================================================


@pytest.fixture
def sample_setup_file(p):
    """Erstellt ein temporäres, valides Hardware-Layout als JSON-Referenz."""
    setup = OpticalSetup("TestSetup")
    setup.beam_splitter("BS1", port_a="line_1", port_b="line_2", eta=0.5)
    setup.fiber_loss("Loss_A", port="line_1", eta=0.9)
    setup.phase_shifter("Phase_B", port="line_2", phi=0.785)
    setup.beam_splitter("BS2", port_a="line_1", port_b="line_2", eta=0.5)
    # setup = OpticalSetup("Test Bench")
    # setup.set_input_state("line_1", label="|α>", state_type="coherent")
    # setup.beam_splitter("BS1", port_a="line_1", port_b="line_2", eta=0.5)

    layout_path = p / "layouts" / "test_beam_splitter.json"
    setup.save_layout(layout_path)
    return layout_path


@pytest.fixture
def empty_journal():
    """Liefert einen frischen, leeren Journal-Eintrag."""
    return ImprovedJournalEntry(
        title="Entanglement Test Log",
        tags=["quantum", "unit-test"],
        notes="Automated test entry.",
    )


# =====================================================================
# UNIT TESTS (Isolierte Komponententests)
# =====================================================================


def test_journal_initialization(empty_journal):
    """Prüft, ob die Metadaten und IDs beim Erstellen korrekt generiert werden."""
    assert empty_journal.title == "Entanglement Test Log"
    assert isinstance(empty_journal.entry_id, str)
    assert len(empty_journal.entry_id) > 0
    assert len(empty_journal.runs) == 0
    assert "quantum" in empty_journal.tags


def test_log_simulation_run_datastructure(empty_journal, sample_setup_file):
    """Überprüft, ob die log_simulation_run Methode die Payloads korrekt verpackt."""
    metrics = {"purity": 0.99, "duan_score": 1.2}
    arrays = {
        "wigner": {
            "data": [[0.1, 0.2], [0.3, 0.4]],
            "unit": "Wigner Density",
            "dimensions": ["x", "p"],
        }
    }

    empty_journal.log_simulation_run(
        run_name="Run 1",
        setup_layout_file=sample_setup_file,
        metrics=metrics,
        arrays=arrays,
    )

    assert len(empty_journal.runs) == 1
    logged_run = empty_journal.runs[0]

    # Struktur-Validierung
    assert logged_run["run_name"] == "Run 1"
    assert logged_run["hardware_layout_reference"] == str(sample_setup_file)
    assert logged_run["scalar_results"]["purity"] == 0.99

    # Payload- und Metadaten-Validierung der Arrays
    wigner_payload = logged_run["data_payloads"]["wigner"]
    assert wigner_payload["values"] == [[0.1, 0.2], [0.3, 0.4]]
    assert wigner_payload["unit"] == "Wigner Density"
    assert wigner_payload["dimensions"] == ["x", "p"]


def test_to_dict_serialization_schema(empty_journal, sample_setup_file):
    """Stellt sicher, dass das exportierte Dictionary exakt dem Schema 2.0.0 entspricht."""
    empty_journal.log_simulation_run(
        run_name="Schema Test",
        setup_layout_file=sample_setup_file,
        metrics={"purity": 1.0},
        arrays={},
    )

    serialized = empty_journal.to_dict()

    assert serialized["schema_version"] == "2.0.0"
    assert "metadata" in serialized
    assert serialized["metadata"]["title"] == empty_journal.title
    assert "simulations" in serialized
    assert len(serialized["simulations"]) == 1


# =====================================================================
# INTEGRATION TESTS (Vollständige End-to-End Abläufe)
# =====================================================================


def test_e2e_save_and_reload_lifecycle(p, sample_setup_file):
    """INTEGRATIONSTEST: Erstellt, simuliert, speichert und lädt ein Journal aus dem Dateisystem."""
    journal_dir = p / "journal_store"

    # 1. Erstellen und befüllen
    entry = ImprovedJournalEntry(title="E2E Integration Test", tags=["e2e"])

    # Simuliere zwei aufeinanderfolgende Messpunkte (z.B. einen kleinen Sweep)
    entry.log_simulation_run(
        run_name="Sweep Point A",
        setup_layout_file=sample_setup_file,
        metrics={"fidelity": 0.95},
        arrays={"quadratures": {"data": [1.2, 3.4], "unit": "Squeezing Level"}},
    )
    entry.log_simulation_run(
        run_name="Sweep Point B",
        setup_layout_file=sample_setup_file,
        metrics={"fidelity": 0.88},
        arrays={"quadratures": {"data": [5.6, 7.8], "unit": "Squeezing Level"}},
    )

    # 2. Auf Festplatte schreiben
    saved_path = entry.save(journal_dir)
    assert saved_path.exists()
    assert saved_path.suffix == ".json"

    # 3. Datei manuell einlesen und validieren (Simuliert das spätere Laden im Analyseskript)
    with open(saved_path, "r") as f:
        loaded_data = json.load(f)

    assert loaded_data["schema_version"] == "2.0.0"
    assert loaded_data["metadata"]["title"] == "E2E Integration Test"
    assert len(loaded_data["simulations"]) == 2

    # Verifiziere Datenintegrität von Punkt B
    point_b = loaded_data["simulations"][1]
    assert point_b["run_name"] == "Sweep Point B"
    assert point_b["scalar_results"]["fidelity"] == 0.88
    assert point_b["data_payloads"]["quadratures"]["values"] == [5.6, 7.8]


def test_integration_missing_directory_creation(p, sample_setup_file):
    """INTEGRATIONSTEST: Prüft, ob tiefe, noch nicht existierende Ordnerpfade automatisch erstellt werden."""
    # Ein sehr tiefer, verschachtelter Pfad, der definitiv noch nicht existiert
    deep_host_dir = p / "deep" / "nested" / "sub" / "folder" / "structure"

    entry = ImprovedJournalEntry(title="Deep Path Test")
    entry.log_simulation_run("Path Run", sample_setup_file, {}, {})

    # Sollte ohne FileNotFoundError/NotADirectoryError durchlaufen
    saved_path = entry.save(deep_host_dir)

    assert saved_path.exists()
    assert deep_host_dir.is_dir()
