import pytest

from catsy import Circuit, Mode
from catsy.core import Gate
from catsy.gaussian import squeeze


def test_circuit_adopts_string_modes_as_runtime_objects():
    circuit = Circuit().add_mode("a").add_mode("b")

    assert [mode.name for mode in circuit.modes] == ["a", "b"]
    assert all(mode.owner is circuit for mode in circuit.modes)
    assert circuit.mode_names == ("a", "b")


def test_circuit_adopts_existing_mode_and_rejects_foreign_owner():
    mode = Mode("a")
    circuit = Circuit().add_mode(mode)
    other = Circuit()

    assert circuit.modes == (mode,)
    with pytest.raises(ValueError, match="already belongs"):
        other.add_mode(mode)


def test_circuit_resolves_gate_names_to_owned_modes():
    circuit = Circuit().add_mode("a")
    circuit.add_gate(Gate("Squeezer", squeeze, ("a",), {"r": 0.2, "theta": 0.0}))

    assert circuit.gates[0].modes == (circuit.modes[0],)


def test_circuit_rejects_unregistered_gate_mode():
    circuit = Circuit().add_mode("a")
    with pytest.raises(ValueError, match="not registered"):
        circuit.add_gate(Gate("Squeezer", squeeze, ("missing",), {"r": 0.2, "theta": 0.0}))


def test_remove_mode_releases_ownership():
    mode = Mode("a")
    circuit = Circuit().add_mode(mode)

    circuit.remove_mode(mode)

    assert circuit.modes == ()
    assert mode.owner is None


def test_remove_mode_rejects_modes_used_by_gates():
    circuit = Circuit().add_mode("a")
    circuit.add_gate(Gate("Squeezer", squeeze, ("a",), {"r": 0.2, "theta": 0.0}))

    with pytest.raises(ValueError, match="still referenced"):
        circuit.remove_mode("a")


def test_circuit_serialization_exposes_names_not_runtime_objects():
    circuit = Circuit().add_mode("a").add_mode("b")
    circuit.add_gate(Gate("Squeezer", squeeze, ("a",), {"r": 0.2, "theta": 0.0}))

    data = circuit.to_dict()

    assert data["modes"] == ["a", "b"]
    assert data["gates"][0]["modes"] == ["a"]

    restored = Circuit.from_dict(data)
    assert restored.mode_names == circuit.mode_names
    assert all(mode.owner is restored for mode in restored.modes)
