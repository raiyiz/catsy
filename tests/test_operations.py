"""Tests for representation-independent state transformations."""

import numpy as np
import pytest
import qutip as qt

from catsy import displace, rotate, squeeze
from catsy.fock import FockState
from catsy.gaussian import GaussianState


def _fock_vacuum(cutoff: int = 12) -> FockState:
    vacuum = qt.fock_dm(cutoff, 0)
    return FockState(("a",), vacuum, cutoff)


@pytest.mark.parametrize("state", [GaussianState.vacuum(("a",)), _fock_vacuum()])
def test_squeeze_dispatches_from_state_type(state):
    result = squeeze(state, "a", r=0.3)

    assert type(result) is type(state)
    assert result.modes == state.modes


def test_gaussian_operations_keep_fock_and_gaussian_representations():
    gaussian = GaussianState.vacuum(("a",)).displace("a", 0.4 + 0.2j)
    fock = gaussian.to_fock(14)

    gaussian_result = rotate(squeeze(gaussian, "a", r=0.2), "a", phi=0.3)
    fock_result = rotate(squeeze(fock, "a", r=0.2), "a", phi=0.3)

    assert isinstance(gaussian_result, GaussianState)
    assert isinstance(fock_result, FockState)
    assert np.isclose(float(fock_result.rho.tr()), 1.0)


def test_displace_dispatches_to_fock_implementation():
    state = _fock_vacuum()
    result = displace(state, "a", alpha=0.5 + 0.1j)

    assert isinstance(result, FockState)
    assert np.isclose(float(result.rho.tr()), 1.0)
    assert np.isclose(
        float(qt.expect(qt.num(state.N_cutoff), result.rho)), 0.26, atol=1e-3
    )


def test_unsupported_state_has_clear_dispatch_error():
    with pytest.raises(TypeError, match="does not support int"):
        squeeze(1, "a", r=0.2)
