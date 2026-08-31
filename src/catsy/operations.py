"""Representation-independent state transformations.

Public transformations dispatch on the incoming state representation. Concrete
implementations stay in :mod:`catsy.gaussian` and :mod:`catsy.fock`.
"""

from __future__ import annotations

from functools import singledispatch

from catsy.fock import FockState
from catsy.gaussian import GaussianState

from .fock import (
    beam_splitter as _fock_beam_splitter,
)
from .fock import (
    displace as _fock_displace,
)
from .fock import (
    loss as _fock_loss,
)
from .fock import (
    rotate as _fock_rotate,
)
from .fock import (
    squeeze as _fock_squeeze,
)
from .fock import (
    thermal_loss as _fock_thermal_loss,
)
from .gaussian import (
    beam_splitter as _gaussian_beam_splitter,
)
from .gaussian import (
    displace as _gaussian_displace,
)
from .gaussian import (
    loss as _gaussian_loss,
)
from .gaussian import (
    rotate as _gaussian_rotate,
)
from .gaussian import (
    squeeze as _gaussian_squeeze,
)
from .gaussian import (
    thermal_loss as _gaussian_thermal_loss,
)


@singledispatch
def squeeze(state, mode: str, r: float, theta: float = 0.0):
    """Return ``state`` after single-mode squeezing."""
    raise TypeError(f"squeeze does not support {type(state).__name__}.")


@squeeze.register
def _(state: GaussianState, mode: str, r: float, theta: float = 0.0) -> GaussianState:
    return _gaussian_squeeze(state, (mode,), r=r, theta=theta)


@squeeze.register
def _(state: FockState, mode: str, r: float, theta: float = 0.0) -> FockState:
    idx = state.get_mode_index(mode)
    rho = _fock_squeeze(state.rho, idx, r, theta, N_cutoff=state.N_cutoff)
    return FockState(state.modes, rho, state.N_cutoff)


@singledispatch
def rotate(state, mode: str, phi: float):
    """Return ``state`` after a phase-space rotation."""
    raise TypeError(f"rotate does not support {type(state).__name__}.")


@rotate.register
def _(state: GaussianState, mode: str, phi: float) -> GaussianState:
    return _gaussian_rotate(state, (mode,), phi=phi)


@rotate.register
def _(state: FockState, mode: str, phi: float) -> FockState:
    idx = state.get_mode_index(mode)
    rho = _fock_rotate(state.rho, idx, phi, N_cutoff=state.N_cutoff)
    return FockState(state.modes, rho, state.N_cutoff)


@singledispatch
def displace(
    state,
    mode: str,
    alpha: complex | None = None,
    *,
    x: float | None = None,
    p: float | None = None,
):
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
    return _gaussian_displace(state, (mode,), alpha=alpha, x=x, p=p)


@displace.register
def _(
    state: FockState,
    mode: str,
    alpha: complex | None = None,
    *,
    x: float | None = None,
    p: float | None = None,
) -> FockState:
    idx = state.get_mode_index(mode)
    rho = _fock_displace(
        state.rho,
        idx,
        alpha,
        x=x,
        p=p,
        N_cutoff=state.N_cutoff,
    )
    return FockState(state.modes, rho, state.N_cutoff)


@singledispatch
def beam_splitter(state, mode_a: str, mode_b: str, eta: float):
    """Return ``state`` after a two-mode beam splitter."""
    raise TypeError(f"beam_splitter does not support {type(state).__name__}.")


@beam_splitter.register
def _(state: GaussianState, mode_a: str, mode_b: str, eta: float) -> GaussianState:
    return _gaussian_beam_splitter(state, (mode_a, mode_b), eta=eta)


@beam_splitter.register
def _(state: FockState, mode_a: str, mode_b: str, eta: float) -> FockState:
    idx_a = state.get_mode_index(mode_a)
    idx_b = state.get_mode_index(mode_b)
    rho = _fock_beam_splitter(state.rho, idx_a, idx_b, eta, N_cutoff=state.N_cutoff)
    return FockState(state.modes, rho, state.N_cutoff)


@singledispatch
def loss(state, mode: str, eta: float, ancilla_cutoff: int | None = None):
    """Return ``state`` after vacuum loss on one mode."""
    raise TypeError(f"loss does not support {type(state).__name__}.")


@loss.register
def _(
    state: GaussianState, mode: str, eta: float, ancilla_cutoff: int | None = None
) -> GaussianState:
    return _gaussian_loss(state, (mode,), eta=eta)


@loss.register
def _(
    state: FockState, mode: str, eta: float, ancilla_cutoff: int | None = None
) -> FockState:
    idx = state.get_mode_index(mode)
    rho = _fock_loss(
        state.rho,
        idx,
        eta,
        N_cutoff=state.N_cutoff,
        ancilla_cutoff=ancilla_cutoff,
    )
    return FockState(state.modes, rho, state.N_cutoff)


@singledispatch
def thermal_loss(
    state,
    mode: str,
    eta: float,
    n_thermal: float = 0.0,
    ancilla_cutoff: int | None = None,
):
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
    return _gaussian_thermal_loss(state, (mode,), eta=eta, n_thermal=n_thermal)


@thermal_loss.register
def _(
    state: FockState,
    mode: str,
    eta: float,
    n_thermal: float = 0.0,
    ancilla_cutoff: int | None = None,
) -> FockState:
    idx = state.get_mode_index(mode)
    rho = _fock_thermal_loss(
        state.rho,
        idx,
        eta,
        nbar=n_thermal,
        N_cutoff=state.N_cutoff,
        ancilla_cutoff=ancilla_cutoff,
    )
    return FockState(state.modes, rho, state.N_cutoff)
