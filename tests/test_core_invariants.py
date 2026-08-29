"""Direct unit tests for the physicality guardrails in core.py.

`_validate_physical_covariance` and `_williamson_decomposition` are private,
but each documents a contract wider than what their one current caller
(`GaussianState`) exercises: `GaussianState._validate` already pins
`covariance.shape` to `(2n, 2n)` before either function ever sees the
array, so the shape/dimension guards inside them are unreachable through
the public API today. They're still worth testing directly, the same way
`test_williamson_decomposition_is_genuinely_symplectic` in test_gaussian.py
already calls `_williamson_decomposition` directly rather than only through
`GaussianState.to_qutip`: these functions are the reusable numerical
primitives the rest of the module is built on, and a future caller (a new
state representation, a batch/bulk API) could easily hand them a malformed
array without going through `GaussianState` first. Testing the guard here
also means it isn't silently deleted as "dead code" by a future refactor.
"""

import numpy as np
import pytest

from catsy.core import (
    _normalize_phase_vector,
    _validate_physical_covariance,
    _williamson_decomposition,
)

# _validate_physical_covariance


def test_validate_physical_covariance_rejects_non_2d_array():
    with pytest.raises(ValueError, match="2D array"):
        _validate_physical_covariance(np.zeros(4))


def test_validate_physical_covariance_rejects_odd_dimension():
    with pytest.raises(ValueError, match="dimension must be even"):
        _validate_physical_covariance(0.5 * np.eye(3))


def test_validate_physical_covariance_accepts_zero_modes():
    # dim == 0 is a legitimate degenerate case (e.g. the fully-collapsed
    # empty state returned by homodyne/heterodyne on a single-mode input;
    # see test_homodyne_single_mode_returns_valid_empty_state) and must
    # return, not raise.
    _validate_physical_covariance(np.zeros((0, 0)))


# _williamson_decomposition


@pytest.mark.parametrize("covariance", [np.zeros((0, 0)), np.eye(3)])
def test_williamson_decomposition_rejects_zero_or_odd_dimension(covariance):
    with pytest.raises(ValueError, match="positive even number"):
        _williamson_decomposition(covariance)


def test_williamson_decomposition_rejects_non_positive_definite_input():
    with pytest.raises(ValueError, match="positive definite"):
        _williamson_decomposition(np.diag([-1.0, 1.0]))


def test_williamson_decomposition_rejects_numerically_singular_symplectic_eigenvalue():
    # For a single mode, the symplectic eigenvalue is sqrt(det V); a matrix
    # that's still strictly (if barely) positive definite but has a near-
    # zero determinant drives it below the tol=1e-10 default threshold.
    # No physical GaussianState reaches this: Heisenberg uncertainty forbids
    # nu below 0.5. This is a purely numerical edge case of the Schur-based
    # construction, tested directly against a hand-built matrix instead of
    # a (nonexistent) physical state.
    covariance = np.diag([1e-25, 1.0])
    with pytest.raises(ValueError, match="numerically singular"):
        _williamson_decomposition(covariance)


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({"alpha": 0.6 - 0.9j}, (0.6 - 0.9j, np.sqrt(2) * 0.6, -np.sqrt(2) * 0.9)),
        ({"x": 1.2, "p": -1.8}, ((1.2 - 1.8j) / np.sqrt(2), 1.2, -1.8)),
    ],
)
def test_normalize_phase_vector_accepts_supported_forms(kwargs, expected):
    assert _normalize_phase_vector(**kwargs) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("kwargs", "exception", "match"),
    [
        # alpha combined with x/p, and x or p alone, are each a single branch
        ({"alpha": 1.0, "x": 1.0, "p": 1.0}, ValueError, "either `alpha` or"),
        ({}, TypeError, "need some input"),
        ({"x": 1.0}, ValueError, "both `x` and `p`"),
        ({"alpha": np.nan}, ValueError, "finite"),
        ({"x": np.nan, "p": 1.0}, ValueError, "x must be finite"),
        ({"alpha": "bad"}, TypeError, "numeric"),
    ],
)
def test_normalize_phase_vector_rejects_invalid_inputs(kwargs, exception, match):
    with pytest.raises(exception, match=match):
        _normalize_phase_vector(**kwargs)
