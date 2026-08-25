import numpy as np
import pytest
import qutip as qt

from catsy.fock import _apply_kraus_operators, photon_addition, photon_subtraction


def test_single_kraus_application_matches_direct_kraus_formula():
    cutoff = 8
    rho = qt.ket2dm((qt.coherent(cutoff, 0.7) + qt.fock(cutoff, 1)).unit())
    kraus = qt.destroy(cutoff)

    result = _apply_kraus_operators(rho, [kraus], "test")
    expected = kraus * rho * kraus.dag()
    expected = expected / expected.tr()

    assert qt.fidelity(result, expected) == pytest.approx(1.0, abs=1e-8)


def test_multiple_kraus_operators_match_sum_of_kraus_terms():
    cutoff = 6
    rho = qt.ket2dm((qt.fock(cutoff, 0) + 0.7 * qt.fock(cutoff, 2)).unit())
    probability = 0.3
    k0 = qt.Qobj(np.diag([1.0, np.sqrt(1.0 - probability), 1.0, 1.0, 1.0, 1.0]))
    k1 = qt.Qobj(np.diag([0.0, np.sqrt(probability), 0.0, 0.0, 0.0, 0.0]))

    result = _apply_kraus_operators(rho, [k0, k1], "test")
    expected = k0 * rho * k0.dag() + k1 * rho * k1.dag()
    expected = expected / expected.tr()

    assert qt.fidelity(result, expected) == pytest.approx(1.0, abs=1e-8)


def test_kraus_application_rejects_empty_or_mismatched_sets():
    rho = qt.ket2dm(qt.fock(5, 1))

    with pytest.raises(ValueError, match="at least one"):
        _apply_kraus_operators(rho, [], "test")

    with pytest.raises(ValueError, match="same Hilbert space"):
        _apply_kraus_operators(rho, [qt.destroy(4)], "test")


def test_kraus_application_rejects_zero_success_probability():
    vacuum = qt.ket2dm(qt.fock(5, 0))
    annihilation = qt.destroy(5)

    with pytest.raises(ValueError, match="success probability"):
        _apply_kraus_operators(vacuum, [annihilation], "test")


def test_ideal_photon_operations_use_same_kraus_semantics():
    cutoff = 10
    rho = qt.ket2dm(qt.fock(cutoff, 3))

    subtracted = photon_subtraction(rho)
    expected_subtraction = qt.ket2dm(qt.fock(cutoff, 2))
    assert qt.fidelity(subtracted, expected_subtraction) == pytest.approx(1.0)

    added = photon_addition(rho)
    expected_addition = qt.ket2dm(qt.fock(cutoff, 4))
    assert qt.fidelity(added, expected_addition) == pytest.approx(1.0)
