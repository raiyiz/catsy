"""Tests for the boundary between runtime modes and mode-name data."""

from catsy import Circuit, Mode


def test_runtime_mode_is_identity_stable_when_owner_changes():
    mode = Mode("signal")
    modes = {mode}

    mode.owner = Circuit(name="owner")

    assert mode in modes
    assert mode.name == "signal"


def test_circuit_data_keeps_mode_names_as_strings():
    circuit = Circuit(name="Bench").add_mode("signal")

    data = circuit.to_dict()

    assert data["modes"] == ["signal"]
    assert all(isinstance(name, str) for name in data["modes"])
