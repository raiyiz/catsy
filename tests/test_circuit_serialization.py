import json

import numpy as np

from catsy.gaussian import displace
from catsy.optics import Circuit, Gate


def test_gate_normalizes_alpha_to_quadratures():
    gate = Gate(
        name="Displacer",
        transform=displace,
        modes=("signal",),
        kwargs={"alpha": 0.4 + 0.1j},
    )

    np.testing.assert_allclose(
        [gate.kwargs["x"], gate.kwargs["p"]],
        [np.sqrt(2.0) * 0.4, np.sqrt(2.0) * 0.1],
    )


def test_circuit_json_serialization_uses_quadratures_for_displacement(tmp_path):
    circuit = Circuit(name="Serialization").add_mode("signal")
    circuit.displace("signal", alpha=0.4 + 0.1j)

    serialized = circuit.to_dict()
    np.testing.assert_allclose(
        [serialized["gates"][0]["kwargs"]["x"], serialized["gates"][0]["kwargs"]["p"]],
        [np.sqrt(2.0) * 0.4, np.sqrt(2.0) * 0.1],
    )
    assert "alpha" not in serialized["gates"][0]["kwargs"]
    assert json.dumps(serialized)

    path = tmp_path / "circuit.json"
    circuit.save(path)
    loaded = Circuit.load(path)

    assert loaded.to_dict() == serialized
