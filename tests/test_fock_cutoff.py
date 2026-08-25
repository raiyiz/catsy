import pytest
import qutip as qt

from catsy.fock import (
    FockGates,
    mean_photon_number,
    photon_addition,
    photon_number_measurement,
    photon_subtraction,
    realistic_photon_addition,
    realistic_photon_subtraction,
)


def test_system_cutoff_is_inferred_from_single_mode_state():
    rho = qt.ket2dm(qt.fock(7, 2))
    result = photon_addition(rho)
    expected = qt.ket2dm(qt.fock(7, 3))
    assert qt.fidelity(result, expected) == pytest.approx(1.0, abs=1e-10)


def test_system_cutoff_is_inferred_from_multimode_state():
    rho = qt.tensor(qt.ket2dm(qt.fock(6, 0)), qt.ket2dm(qt.fock(6, 1)))
    result = photon_addition(rho, mode_idx=1)
    expected = qt.tensor(qt.ket2dm(qt.fock(6, 0)), qt.ket2dm(qt.fock(6, 2)))
    assert qt.fidelity(result, expected) == pytest.approx(1.0, abs=1e-10)


def test_explicit_cutoff_remains_a_consistency_check():
    rho = qt.ket2dm(qt.fock(8, 0))
    photon_subtraction(rho, N_cutoff=8)
    with pytest.raises(ValueError, match="N_cutoff"):
        photon_addition(rho, N_cutoff=7)


@pytest.mark.parametrize(
    "operation",
    [
        photon_addition,
        photon_subtraction,
        mean_photon_number,
        photon_number_measurement,
        realistic_photon_addition,
        realistic_photon_subtraction,
    ],
)
def test_fock_operations_accept_inferred_cutoff(operation):
    cutoff = 8
    rho = qt.ket2dm(qt.fock(cutoff, 2))
    kwargs = {}
    if operation is photon_number_measurement:
        kwargs["outcome"] = 2
    if operation is realistic_photon_addition:
        kwargs.update(coupling_strength=1e-3, detector_efficiency=0.999, ancilla_cutoff=4)
    if operation is realistic_photon_subtraction:
        kwargs.update(tap_reflectivity=1e-4, detector_efficiency=0.999, ancilla_cutoff=4)
    result = operation(rho, **kwargs)
    if operation is mean_photon_number:
        assert result == pytest.approx(2.0)
    elif operation is photon_number_measurement:
        assert result[0] == 2
    else:
        assert result.tr() == pytest.approx(1.0, abs=1e-8)


def test_fock_gates_compatibility_namespace_uses_inferred_cutoff():
    rho = qt.ket2dm(qt.fock(6, 0))
    result = FockGates.photon_addition(rho)
    expected = qt.ket2dm(qt.fock(6, 1))
    assert qt.fidelity(result, expected) == pytest.approx(1.0, abs=1e-10)
