"""Representation-independent state transformations.

Public transformations dispatch on the incoming state representation. Each
one dispatches to exactly two implementations -- Gaussian and Fock -- and
those implementations are the states' own instance methods (`GaussianState`
and `FockState` in :mod:`catsy.gaussian`/:mod:`catsy.fock`). This module
adds dispatch, not new physics: it never reimplements mode lookups or calls
into representation-internal helpers directly.

`thermal_loss` accepts a single noise keyword, `n_thermal`, on both
branches; there is no separate `nbar` spelling anywhere in the package.
"""

from __future__ import annotations

from functools import singledispatch
from typing import Any

from catsy.fock import FockState
from catsy.gaussian import GaussianState

from .gaussian import thermal_loss as _gaussian_thermal_loss


@singledispatch
def squeeze(state: Any, mode: str, r: float, theta: float = 0.0) -> Any:
    """Return ``state`` after single-mode squeezing."""
    raise TypeError(f"squeeze does not support {type(state).__name__}.")


@squeeze.register
def _(state: GaussianState, mode: str, r: float, theta: float = 0.0) -> GaussianState:
    return state.squeeze(mode, r, theta)


@squeeze.register
def _(state: FockState, mode: str, r: float, theta: float = 0.0) -> FockState:
    return state.squeeze(mode, r, theta)


@singledispatch
def rotate(state: Any, mode: str, phi: float) -> Any:
    """Return ``state`` after a phase-space rotation."""
    raise TypeError(f"rotate does not support {type(state).__name__}.")


@rotate.register
def _(state: GaussianState, mode: str, phi: float) -> GaussianState:
    return state.rotate(mode, phi)


@rotate.register
def _(state: FockState, mode: str, phi: float) -> FockState:
    return state.rotate(mode, phi)


@singledispatch
def displace(
    state: Any,
    mode: str,
    alpha: complex | None = None,
    *,
    x: float | None = None,
    p: float | None = None,
) -> Any:
    """Return ``state`` after a displacement."""
    raise TypeError(f"displace does not support {type(state).__name__}.")


@displace.register
def _(
    state: GaussianState,
    mode: str,
    alpha: complex | None = None,
    *,
    x: float | None = None,
    p: float | None = None,
) -> GaussianState:
    return state.displace(mode, alpha, x=x, p=p)


@displace.register
def _(
    state: FockState,
    mode: str,
    alpha: complex | None = None,
    *,
    x: float | None = None,
    p: float | None = None,
) -> FockState:
    return state.displace(mode, alpha, x=x, p=p)


@singledispatch
def beam_splitter(state: Any, mode_a: str, mode_b: str, eta: float) -> Any:
    """Return ``state`` after a two-mode beam splitter."""
    raise TypeError(f"beam_splitter does not support {type(state).__name__}.")


@beam_splitter.register
def _(state: GaussianState, mode_a: str, mode_b: str, eta: float) -> GaussianState:
    return state.beam_splitter(mode_a, mode_b, eta)


@beam_splitter.register
def _(state: FockState, mode_a: str, mode_b: str, eta: float) -> FockState:
    return state.beam_splitter(mode_a, mode_b, eta)


@singledispatch
def loss(state: Any, mode: str, eta: float, ancilla_cutoff: int | None = None) -> Any:
    """Return ``state`` after vacuum loss on one mode."""
    raise TypeError(f"loss does not support {type(state).__name__}.")


@loss.register
def _(
    state: GaussianState, mode: str, eta: float, ancilla_cutoff: int | None = None
) -> GaussianState:
    # ancilla_cutoff is Fock-only (Gaussian loss has no Fock truncation to
    # size); accepted here so callers get one signature across both branches.
    return state.loss(mode, eta)


@loss.register
def _(
    state: FockState, mode: str, eta: float, ancilla_cutoff: int | None = None
) -> FockState:
    return state.loss(mode, eta, ancilla_cutoff)


@singledispatch
def thermal_loss(
    state: Any,
    mode: str,
    eta: float,
    n_thermal: float = 0.0,
    ancilla_cutoff: int | None = None,
) -> Any:
    """Return ``state`` after thermal loss on one mode."""
    raise TypeError(f"thermal_loss does not support {type(state).__name__}.")


@thermal_loss.register
def _(
    state: GaussianState,
    mode: str,
    eta: float,
    n_thermal: float = 0.0,
    ancilla_cutoff: int | None = None,
) -> GaussianState:
    # ancilla_cutoff is Fock-only; accepted here for the same reason as loss()
    # above. GaussianState has no thermal_loss instance method (thermal loss
    # is expressed as a GaussianChannel), so this is the one branch that
    # still calls into catsy.gaussian rather than a state method.
    return _gaussian_thermal_loss(state, (mode,), eta=eta, n_thermal=n_thermal)


@thermal_loss.register
def _(
    state: FockState,
    mode: str,
    eta: float,
    n_thermal: float = 0.0,
    ancilla_cutoff: int | None = None,
) -> FockState:
    return state.thermal_loss(mode, eta, n_thermal, ancilla_cutoff)
