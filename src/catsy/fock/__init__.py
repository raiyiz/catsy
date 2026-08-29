"""Low-level operations on QuTiP Fock-space states.

This is the discrete-variable counterpart to :mod:`catsy.gaussian`: where
the Gaussian layer represents a state by its first and second phase-space
moments ``(d, V)`` and stays exact for as long as every transformation is
linear/Gaussian, this module represents a state by an explicit, truncated
QuTiP density matrix and can express operations -- such as heralded photon
addition/subtraction -- that push a state outside the Gaussian family.

The low-level API is functional: every operation acts directly on a QuTiP
density matrix addressed by integer ``mode_idx``. ``FockGates`` remains only
as a backwards-compatible namespace for existing callers.

:class:`FockState` is the named-mode wrapper around that functional core --
the Fock-space counterpart of :class:`~catsy.gaussian.GaussianState`, built
by :meth:`GaussianState.to_fock`. Its gate methods (``squeeze``, ``rotate``,
``displace``, ``beam_splitter``, ``loss``) are the exact Fock-space
implementations of the same physical operations ``GaussianState`` provides
in phase space -- built from the same building blocks
(``catsy.core._qutip_passive_unitary`` for the passive/orthogonal-symplectic
ones, ``qutip.squeeze``/``qutip.displace`` for the rest) so a circuit gets
the same physics regardless of which representation it is currently
computing in.

:meth:`photon_subtraction`/:meth:`photon_addition` are the textbook,
unit-efficiency operators ``a``/``a†``. :meth:`realistic_photon_subtraction`/
:meth:`realistic_photon_addition` model how these are actually implemented
on an optical bench: a weak coupling to an ancilla mode (a beamsplitter tap
for subtraction, a weak parametric two-mode-squeezing interaction for
addition) followed by heralding on an imperfect "click" detector on the
ancilla. In the limit of weak coupling and unit detector efficiency, both
converge to the corresponding ideal operator; at finite coupling and
efficiency they reproduce the impurity real heralded sources have. These,
and photon-number measurement, are genuinely non-Gaussian: they have no
``GaussianState`` counterpart by construction, only a `FockState` one.

Conversion is one-way: a :class:`~catsy.gaussian.GaussianState` embeds
exactly (up to Fock-space truncation) into a `FockState` via
``GaussianState.to_fock()``, but not every `FockState` has a Gaussian
description, so there is deliberately no ``FockState.to_gaussian()``. Once a
computation needs `FockState`, it stays there.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import qutip as qt
from qutip.measurement import measurement_statistics

from catsy.core import (
    TOL_PHYSICALITY,
    TOL_TRACE_WARN,
    _check_non_negative,
    _check_positive_int,
    _check_unit_interval,
    _normalize_phase_vector,
    _qutip_passive_unitary,
)
from catsy.types import Modes

logger = logging.getLogger("catsy")


def _validate_state(
    rho: qt.Qobj,
    N_cutoff: int | None,
    mode_idx: int,
) -> tuple[int, int]:
    if not isinstance(rho, qt.Qobj):
        raise TypeError(f"rho must be a QuTiP Qobj, got {type(rho).__name__}.")
    if not rho.isoper:
        raise ValueError("rho must be a QuTiP operator (density matrix).")

    dims = rho.dims[0]
    if not dims or any(dim != dims[0] for dim in dims):
        raise ValueError(
            "all system modes must have the same Fock dimension; "
            f"got dimensions {dims!r}."
        )
    inferred_cutoff = dims[0]
    if N_cutoff is not None:
        _check_positive_int(N_cutoff, "N_cutoff")
        if N_cutoff != inferred_cutoff:
            raise ValueError(
                "N_cutoff must match every mode dimension of rho; "
                f"rho has dimensions {dims!r}, got N_cutoff={N_cutoff}."
            )

    n_modes = len(dims)
    if not isinstance(mode_idx, int) or not 0 <= mode_idx < n_modes:
        raise ValueError(
            f"mode_idx must be an integer in [0, {n_modes - 1}], got {mode_idx!r}."
        )

    return n_modes, inferred_cutoff


def _expand_operator(
    op_1mode: qt.Qobj,
    dims: list[int],
    mode_idx: int,
) -> qt.Qobj:
    """Expand a single-mode operator onto the selected subsystem."""
    return qt.expand_operator(op_1mode, dims=dims, targets=mode_idx)


def _mode_operator(
    op_1mode: qt.Qobj,
    n_modes: int,
    mode_idx: int,
    N_cutoff: int,
) -> qt.Qobj:
    return _expand_operator(op_1mode, [N_cutoff] * n_modes, mode_idx)


# ========================================================================
# Gaussian-unitary gates (Fock-space implementations)
#
# These are the Fock-space counterparts of the corresponding GaussianState
# methods/gaussian module transforms: squeezing, rotation, displacement,
# beam splitters and vacuum-coupled loss are all Gaussian (quadratic-
# generator) operations, so they have an exact representation in either
# picture. Rotator/BeamSplitter/Noise reuse `_qutip_passive_unitary` (also
# used by `GaussianState.to_fock`) so a given orthogonal-symplectic
# transform produces byte-for-byte the same Fock-space unitary regardless
# of which gate constructed it. Squeezer/Displacer use QuTiP's own
# `squeeze`/`displace` primitives with the parametrizations matched to this
# codebase's phase-space convention (see the README's "Conventions"
# section): `qt.squeeze(N, r * exp(2j * theta))` reproduces
# Var(x) = 0.5 * exp(-2r) at theta=0, exactly like `GaussianState.squeeze`.
# ========================================================================


def squeeze(
    rho: qt.Qobj,
    mode_idx: int = 0,
    r: float = 0.0,
    theta: float = 0.0,
    N_cutoff: int | None = None,
) -> qt.Qobj:
    """Apply single-mode squeezing to ``mode_idx``.

    Matches ``GaussianState.squeeze``: at ``theta=0`` this squeezes the x
    quadrature (``Var(x) = 0.5 * exp(-2r)``); at ``theta=pi/2`` it squeezes p.
    """
    if not np.isfinite(r) or not np.isfinite(theta):
        raise ValueError(f"r and theta must be finite, got r={r!r}, theta={theta!r}.")
    n_modes, cutoff = _validate_state(rho, N_cutoff, mode_idx)
    xi = r * np.exp(2j * theta)
    S = _mode_operator(qt.squeeze(cutoff, xi), n_modes, mode_idx, cutoff)
    return S * rho * S.dag()


def rotate(
    rho: qt.Qobj,
    mode_idx: int = 0,
    phi: float = 0.0,
    N_cutoff: int | None = None,
) -> qt.Qobj:
    """Apply a phase-space rotation by ``phi`` to ``mode_idx``.

    Matches ``GaussianState.rotate``: implemented as the number-conserving
    unitary for the orthogonal symplectic rotation ``R(phi)``, via
    `catsy.core._qutip_passive_unitary` -- the same construction
    `GaussianState.to_fock` uses for the passive part of a general
    symplectic transform.
    """
    if not np.isfinite(phi):
        raise ValueError(f"phi must be finite, got {phi!r}.")
    n_modes, cutoff = _validate_state(rho, N_cutoff, mode_idx)
    R_local = np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])
    a_op = _mode_operator(qt.destroy(cutoff), n_modes, mode_idx, cutoff)
    U = _qutip_passive_unitary(R_local, [a_op])
    return U * rho * U.dag()


def displace(
    rho: qt.Qobj,
    mode_idx: int = 0,
    alpha: complex | None = None,
    *,
    x: float | None = None,
    p: float | None = None,
    N_cutoff: int | None = None,
) -> qt.Qobj:
    """Displace ``mode_idx`` by ``alpha`` (or by ``x``/``p``).

    Uses the same ``alpha = (x + ip) / sqrt(2)`` convention as
    `GaussianState.displace` and `GaussianState.to_fock` (both via
    `catsy.core._normalize_phase_vector`), applied with QuTiP's `displace`.
    """
    alpha_value, _, _ = _normalize_phase_vector(alpha=alpha, x=x, p=p)
    n_modes, cutoff = _validate_state(rho, N_cutoff, mode_idx)
    D = _mode_operator(qt.displace(cutoff, alpha_value), n_modes, mode_idx, cutoff)
    return D * rho * D.dag()


def beam_splitter(
    rho: qt.Qobj,
    mode_a_idx: int,
    mode_b_idx: int,
    eta: float,
    N_cutoff: int | None = None,
) -> qt.Qobj:
    """Apply a lossless beam splitter between ``mode_a_idx`` and ``mode_b_idx``.

    Matches ``GaussianState.beam_splitter``: builds the same orthogonal
    symplectic block (``t = sqrt(eta)`` on the diagonal, ``r = sqrt(1-eta)``
    off-diagonal) and turns it into a Fock-space unitary via
    `catsy.core._qutip_passive_unitary`.
    """
    if mode_a_idx == mode_b_idx:
        raise ValueError("mode_a_idx and mode_b_idx must be different modes.")
    _check_unit_interval(eta, "eta")
    n_modes, cutoff = _validate_state(rho, N_cutoff, mode_a_idx)
    _validate_state(rho, N_cutoff, mode_b_idx)

    t = np.sqrt(eta)
    r_coeff = np.sqrt(1 - eta)
    O = np.block(
        [
            [t * np.eye(2), r_coeff * np.eye(2)],
            [-r_coeff * np.eye(2), t * np.eye(2)],
        ]
    )
    a_a = _mode_operator(qt.destroy(cutoff), n_modes, mode_a_idx, cutoff)
    a_b = _mode_operator(qt.destroy(cutoff), n_modes, mode_b_idx, cutoff)
    U = _qutip_passive_unitary(O, [a_a, a_b])
    return U * rho * U.dag()


def loss(
    rho: qt.Qobj,
    mode_idx: int = 0,
    eta: float = 1.0,
    N_cutoff: int | None = None,
    ancilla_cutoff: int | None = None,
) -> qt.Qobj:
    """Apply vacuum-coupled loss (transmissivity ``eta``) to ``mode_idx``.

    Physically identical to ``GaussianState.loss``: couple ``mode_idx`` to a
    fresh vacuum ancilla mode through the same beam-splitter unitary
    `beam_splitter` builds, then trace the ancilla out. This is the standard
    microscopic derivation of an optical loss channel, so -- unlike
    `photon_subtraction` -- it stays exactly Gaussian whenever the input is.
    ``ancilla_cutoff`` defaults to the system's own cutoff and only needs to
    be raised for very lossy (small ``eta``) channels acting on states with
    substantial photon number.
    """
    _check_unit_interval(eta, "eta")
    n_modes, cutoff = _validate_state(rho, N_cutoff, mode_idx)
    ancilla_cutoff = cutoff if ancilla_cutoff is None else ancilla_cutoff
    _check_positive_int(ancilla_cutoff, "ancilla_cutoff")

    dims = [cutoff] * n_modes + [ancilla_cutoff]
    ancilla_idx = n_modes

    a_sys = _expand_operator(qt.destroy(cutoff), dims=dims, mode_idx=mode_idx)
    a_anc = _expand_operator(qt.destroy(ancilla_cutoff), dims=dims, mode_idx=ancilla_idx)

    t = np.sqrt(eta)
    r_coeff = np.sqrt(1 - eta)
    O = np.block(
        [
            [t * np.eye(2), r_coeff * np.eye(2)],
            [-r_coeff * np.eye(2), t * np.eye(2)],
        ]
    )
    U = _qutip_passive_unitary(O, [a_sys, a_anc])

    rho_extended = qt.tensor(rho, qt.fock_dm(ancilla_cutoff, 0))
    rho_coupled = U * rho_extended * U.dag()
    return rho_coupled.ptrace(list(range(n_modes)))


def thermal_loss(
    rho: qt.Qobj,
    mode_idx: int = 0,
    eta: float = 1.0,
    nbar: float = 0.0,
    N_cutoff: int | None = None,
    ancilla_cutoff: int | None = None,
) -> qt.Qobj:
    """Apply thermal loss with transmissivity ``eta`` and bath occupancy ``nbar``.

    The selected system mode is coupled to a thermal ancilla through the same
    beam-splitter unitary used by :func:`loss`, after which the ancilla is
    traced out.

    ``nbar=0`` reduces exactly to vacuum-coupled loss.  For ``nbar>0`` the
    channel describes attenuation into a thermal environment.

    ``ancilla_cutoff`` controls the Fock-space truncation of the thermal
    environment.  It defaults to the system cutoff; for large ``nbar`` it
    may need to be increased to faithfully represent the thermal state.
    """
    _check_unit_interval(eta, "eta")
    _check_non_negative(nbar, "nbar")

    n_modes, cutoff = _validate_state(rho, N_cutoff, mode_idx)

    ancilla_cutoff = cutoff if ancilla_cutoff is None else ancilla_cutoff
    _check_positive_int(ancilla_cutoff, "ancilla_cutoff")

    dims = [cutoff] * n_modes + [ancilla_cutoff]
    ancilla_idx = n_modes

    a_sys = _expand_operator(
        qt.destroy(cutoff),
        dims=dims,
        mode_idx=mode_idx,
    )
    a_anc = _expand_operator(
        qt.destroy(ancilla_cutoff),
        dims=dims,
        mode_idx=ancilla_idx,
    )

    # Same beamsplitter convention as loss():
    #
    #   a_sys' = sqrt(eta) a_sys + sqrt(1-eta) a_anc
    #
    # For a thermal environment this implements the bosonic thermal-loss
    # channel with transmissivity eta and environment mean occupation nbar.
    t = np.sqrt(eta)
    r_coeff = np.sqrt(1.0 - eta)

    O = np.block(
        [
            [t * np.eye(2), r_coeff * np.eye(2)],
            [-r_coeff * np.eye(2), t * np.eye(2)],
        ]
    )

    U = _qutip_passive_unitary(O, [a_sys, a_anc])

    # Replace the vacuum environment used by loss() with a thermal state.
    ancilla_thermal = qt.thermal_dm(ancilla_cutoff, nbar)
    rho_extended = qt.tensor(rho, ancilla_thermal)

    rho_coupled = U * rho_extended * U.dag()

    return rho_coupled.ptrace(list(range(n_modes)))


def _apply_kraus_operators(
    rho: qt.Qobj,
    kraus_ops: Sequence[qt.Qobj],
    label: str = "apply_kraus_operators",
) -> qt.Qobj:
    """Apply a conditional operation represented by Kraus operators.

    The unnormalized post-selected state is
    ``sum_i K_i rho K_i.dag()``. QuTiP performs the Kraus-to-superoperator
    conversion; Catsy only handles the conditional renormalization and the
    domain-specific zero-success-probability error.
    """
    kraus_ops = list(kraus_ops)
    if not kraus_ops:
        raise ValueError("kraus_ops must contain at least one operator.")
    for kraus_op in kraus_ops:
        if not isinstance(kraus_op, qt.Qobj) or not kraus_op.isoper:
            raise TypeError("every Kraus operator must be a QuTiP operator (Qobj).")
        if kraus_op.dims[0] != rho.dims[0] or kraus_op.dims[1] != rho.dims[1]:
            raise ValueError(
                "all Kraus operators must act on the same Hilbert space as rho; "
                f"got kraus_op.dims={kraus_op.dims!r}, rho.dims={rho.dims!r}."
            )

    channel = qt.kraus_to_super(kraus_ops)
    rho_new = qt.vector_to_operator(channel * qt.operator_to_vector(rho))
    trace_val = rho_new.tr()
    if abs(trace_val) < TOL_PHYSICALITY:
        raise ValueError(f"{label}: heralding success probability is numerically zero.")
    return rho_new / trace_val


def _apply_kraus_operator(
    rho: qt.Qobj,
    kraus_op: qt.Qobj,
    label: str = "apply_kraus_operator",
) -> qt.Qobj:
    """Backward-compatible single-Kraus wrapper."""
    return _apply_kraus_operators(rho, [kraus_op], label)


def photon_subtraction(
    rho: qt.Qobj,
    mode_idx: int = 0,
    N_cutoff: int | None = None,
) -> qt.Qobj:
    """Apply textbook photon subtraction ``rho -> a rho a†``."""
    n_modes, cutoff = _validate_state(rho, N_cutoff, mode_idx)
    a_op = _mode_operator(qt.destroy(cutoff), n_modes, mode_idx, cutoff)
    return _apply_kraus_operators(rho, [a_op], "photon_subtraction")


def photon_addition(
    rho: qt.Qobj,
    mode_idx: int = 0,
    N_cutoff: int | None = None,
) -> qt.Qobj:
    """Apply textbook photon addition ``rho -> a† rho a``."""
    n_modes, cutoff = _validate_state(rho, N_cutoff, mode_idx)
    adag_op = _mode_operator(qt.create(cutoff), n_modes, mode_idx, cutoff)
    return _apply_kraus_operators(rho, [adag_op], "photon_addition")


def _click_heralded_operation(
    rho: qt.Qobj,
    mode_idx: int,
    N_cutoff: int | None,
    ancilla_cutoff: int,
    coupling_strength: float,
    detector_efficiency: float,
    coupling_kind: str,
    label: str,
) -> qt.Qobj:
    """Shared engine for realistic subtraction/addition operations."""
    n_modes, cutoff = _validate_state(rho, N_cutoff, mode_idx)
    _check_positive_int(ancilla_cutoff, "ancilla_cutoff")
    _check_non_negative(coupling_strength, "coupling_strength")
    _check_unit_interval(detector_efficiency, "detector_efficiency")

    dims = [cutoff] * n_modes + [ancilla_cutoff]
    ancilla_idx = n_modes

    a_sys = _expand_operator(qt.destroy(cutoff), dims=dims, mode_idx=mode_idx)
    a_anc = _expand_operator(qt.destroy(ancilla_cutoff), dims=dims, mode_idx=ancilla_idx)

    if coupling_kind == "subtract":
        generator = coupling_strength * (a_sys * a_anc.dag() - a_sys.dag() * a_anc)
        coupling_unitary = generator.expm()
    else:
        # QuTiP's generalized two-mode squeezing operator is
        # exp(1/2 * (z* a_sys a_anc - z a_sys† a_anc†)). For real z=-2g,
        # this is exactly exp(g * (a_sys† a_anc† - a_sys a_anc)).
        coupling_unitary = qt.squeezing(a_sys, a_anc, -2.0 * coupling_strength)

    ancilla_vacuum = qt.fock_dm(ancilla_cutoff, 0)
    rho_extended = qt.tensor(rho, ancilla_vacuum)
    rho_coupled = coupling_unitary * rho_extended * coupling_unitary.dag()

    no_click_diag = (1.0 - detector_efficiency) ** np.arange(ancilla_cutoff)
    click_diag = np.sqrt(np.clip(1.0 - no_click_diag, 0.0, None))
    click_operator_anc = qt.Qobj(np.diag(click_diag))
    click_operator = _expand_operator(
        click_operator_anc,
        dims=dims,
        mode_idx=ancilla_idx,
    )

    rho_heralded = _apply_kraus_operators(rho_coupled, [click_operator], label)
    return rho_heralded.ptrace(list(range(n_modes)))


def realistic_photon_subtraction(
    rho: qt.Qobj,
    mode_idx: int = 0,
    N_cutoff: int | None = None,
    tap_reflectivity: float = 0.05,
    detector_efficiency: float = 0.6,
    ancilla_cutoff: int = 6,
) -> qt.Qobj:
    """Heralded photon subtraction via a beamsplitter tap + click detector."""
    _check_unit_interval(tap_reflectivity, "tap_reflectivity")
    return _click_heralded_operation(
        rho,
        mode_idx,
        N_cutoff,
        ancilla_cutoff,
        coupling_strength=np.arcsin(np.sqrt(tap_reflectivity)),
        detector_efficiency=detector_efficiency,
        coupling_kind="subtract",
        label="realistic_photon_subtraction",
    )


def realistic_photon_addition(
    rho: qt.Qobj,
    mode_idx: int = 0,
    N_cutoff: int | None = None,
    coupling_strength: float = 0.05,
    detector_efficiency: float = 0.6,
    ancilla_cutoff: int = 6,
) -> qt.Qobj:
    """Heralded photon addition via parametric coupling + click detector."""
    return _click_heralded_operation(
        rho,
        mode_idx,
        N_cutoff,
        ancilla_cutoff,
        coupling_strength=coupling_strength,
        detector_efficiency=detector_efficiency,
        coupling_kind="add",
        label="realistic_photon_addition",
    )


def mean_photon_number(
    rho: qt.Qobj,
    mode_idx: int = 0,
    N_cutoff: int | None = None,
) -> float:
    """Return ``<n> = tr(rho * a†a)`` for the selected mode."""
    n_modes, cutoff = _validate_state(rho, N_cutoff, mode_idx)
    n_op = _mode_operator(qt.num(cutoff), n_modes, mode_idx, cutoff)
    return float(np.real(qt.expect(n_op, rho)))


def photon_number_measurement(
    rho: qt.Qobj,
    mode_idx: int = 0,
    N_cutoff: int | None = None,
    outcome: int | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[int, qt.Qobj]:
    """Ideal photon-number-resolving detection on ``mode_idx``."""
    n_modes, cutoff = _validate_state(rho, N_cutoff, mode_idx)
    dims = [cutoff] * n_modes
    projectors = [
        _expand_operator(qt.fock_dm(cutoff, n), dims=dims, mode_idx=mode_idx)
        for n in range(cutoff)
    ]
    collapsed_states, probabilities = measurement_statistics(
        rho, projectors, tol=TOL_PHYSICALITY
    )
    probabilities = np.asarray(probabilities, dtype=float)

    if outcome is not None:
        if not isinstance(outcome, int) or not 0 <= outcome < cutoff:
            raise ValueError(
                f"outcome must be an integer in [0, {cutoff - 1}], got {outcome!r}."
            )
        if probabilities[outcome] < TOL_PHYSICALITY:
            raise ValueError(
                "photon_number_measurement: selected outcome has "
                "numerically zero probability."
            )
    else:
        total = probabilities.sum()
        if total < TOL_PHYSICALITY:
            raise ValueError(
                "photon_number_measurement: outcome probabilities are "
                "numerically zero for every Fock level."
            )
        rng = rng if rng is not None else np.random.default_rng()
        outcome = int(rng.choice(cutoff, p=probabilities / total))

    collapsed = collapsed_states[outcome]
    if n_modes > 1:
        remaining = [i for i in range(n_modes) if i != mode_idx]
        collapsed = collapsed.ptrace(remaining)

    return outcome, collapsed


# ========================================================================
# FockState
# ========================================================================


@dataclass
class FockState:
    """A multi-mode state in an explicit, truncated Fock-space representation.

    The discrete-variable counterpart to
    :class:`~catsy.gaussian.GaussianState`: instead of a displacement vector
    and covariance matrix, the state is a full QuTiP density matrix ``rho``,
    addressed through the same named-mode API (``modes``,
    :meth:`get_mode_index`, :meth:`reorder_modes`) `GaussianState` uses.
    Every mode shares the same Fock-space cutoff ``N_cutoff``.

    Instance methods are thin, mode-name-aware wrappers around this module's
    functional core (e.g. :meth:`squeeze` calls the module-level
    :func:`squeeze` with the resolved ``mode_idx``); the numerics live in
    the functional layer, matching this module's existing style.

    `FockState` is a terminal representation: every `GaussianState` embeds
    into one exactly, up to Fock-space truncation, via
    ``GaussianState.to_fock()``, but not every `FockState` has a Gaussian
    description, so there is no `to_gaussian()`. Once a computation is a
    `FockState`, every gate below keeps returning a `FockState`.
    """

    modes: Modes
    rho: qt.Qobj
    N_cutoff: int

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        n_modes = len(self.modes)
        if len(set(self.modes)) != n_modes:
            raise ValueError(f"Duplicate mode names in {self.modes!r}.")
        if n_modes == 0:
            raise ValueError("FockState must have at least one mode.")
        _check_positive_int(self.N_cutoff, "N_cutoff")

        if not isinstance(self.rho, qt.Qobj) or not self.rho.isoper:
            raise TypeError(
                f"rho must be a QuTiP operator (density matrix), got {type(self.rho).__name__}."
            )

        dims = self.rho.dims[0]
        if len(dims) != n_modes:
            raise ValueError(
                f"rho has {len(dims)} mode(s) but {n_modes} mode name(s) were "
                f"given: {self.modes!r}."
            )
        if any(dim != self.N_cutoff for dim in dims):
            raise ValueError(
                f"rho's per-mode dimensions {dims!r} must all equal "
                f"N_cutoff={self.N_cutoff}."
            )

        trace = complex(self.rho.tr())
        if abs(trace - 1.0) > TOL_TRACE_WARN:
            logger.warning(
                "FockState: rho has trace %s, expected 1 (tol=%.1e).",
                trace,
                TOL_TRACE_WARN,
            )
        hermiticity_gap = (self.rho - self.rho.dag()).norm()
        if hermiticity_gap > TOL_TRACE_WARN:
            raise ValueError(
                f"rho must be Hermitian; ||rho - rho^dagger||_tr = {hermiticity_gap:.3e}."
            )

    def get_mode_index(self, mode_name: str) -> int:
        if mode_name not in self.modes:
            raise ValueError(f"Mode '{mode_name}' is not present in this state.")
        return self.modes.index(mode_name)

    def reorder_modes(self, modes: Sequence[str]) -> FockState:
        """Return an equivalent state with subsystems arranged in ``modes`` order."""
        requested = tuple(modes)
        if len(requested) != len(self.modes) or set(requested) != set(self.modes):
            raise ValueError(
                "Requested mode order must contain exactly the state's modes; "
                f"state={self.modes!r}, requested={requested!r}."
            )
        if requested == self.modes:
            return self.copy()
        order = [self.get_mode_index(mode) for mode in requested]
        return FockState(
            modes=requested, rho=self.rho.permute(order), N_cutoff=self.N_cutoff
        )

    def copy(self) -> FockState:
        return FockState(modes=self.modes, rho=self.rho.copy(), N_cutoff=self.N_cutoff)

    def __repr__(self) -> str:
        purity = float(np.real((self.rho * self.rho).tr()))
        return f"FockState(modes={self.modes}, N_cutoff={self.N_cutoff}, purity~{purity:.3f})"

    # -- Gaussian-unitary gates (exact in either representation) -----------

    def squeeze(self, mode: str, r: float, theta: float = 0.0) -> FockState:
        idx = self.get_mode_index(mode)
        new_rho = squeeze(self.rho, idx, r, theta, N_cutoff=self.N_cutoff)
        return FockState(modes=self.modes, rho=new_rho, N_cutoff=self.N_cutoff)

    def rotate(self, mode: str, phi: float) -> FockState:
        idx = self.get_mode_index(mode)
        new_rho = rotate(self.rho, idx, phi, N_cutoff=self.N_cutoff)
        return FockState(modes=self.modes, rho=new_rho, N_cutoff=self.N_cutoff)

    def displace(
        self,
        mode: str,
        alpha: complex | None = None,
        *,
        x: float | None = None,
        p: float | None = None,
    ) -> FockState:
        idx = self.get_mode_index(mode)
        new_rho = displace(self.rho, idx, alpha, x=x, p=p, N_cutoff=self.N_cutoff)
        return FockState(modes=self.modes, rho=new_rho, N_cutoff=self.N_cutoff)

    def beam_splitter(self, mode_a: str, mode_b: str, eta: float) -> FockState:
        idx_a = self.get_mode_index(mode_a)
        idx_b = self.get_mode_index(mode_b)
        new_rho = beam_splitter(self.rho, idx_a, idx_b, eta, N_cutoff=self.N_cutoff)
        return FockState(modes=self.modes, rho=new_rho, N_cutoff=self.N_cutoff)

    def loss(self, mode: str, eta: float, ancilla_cutoff: int | None = None) -> FockState:
        idx = self.get_mode_index(mode)
        new_rho = loss(
            self.rho, idx, eta, N_cutoff=self.N_cutoff, ancilla_cutoff=ancilla_cutoff
        )
        return FockState(modes=self.modes, rho=new_rho, N_cutoff=self.N_cutoff)

    def thermal_loss(
        self,
        mode: str,
        eta: float,
        nbar: float = 0.0,
        ancilla_cutoff: int | None = None,
    ) -> FockState:
        idx = self.get_mode_index(mode)
        new_rho = thermal_loss(
            self.rho,
            idx,
            eta,
            nbar=nbar,
            N_cutoff=self.N_cutoff,
            ancilla_cutoff=ancilla_cutoff,
        )
        return FockState(
            modes=self.modes,
            rho=new_rho,
            N_cutoff=self.N_cutoff,
        )

    # -- Non-Gaussian operations (Fock-only, by physics) --------------------

    def photon_subtraction(self, mode: str) -> FockState:
        idx = self.get_mode_index(mode)
        new_rho = photon_subtraction(self.rho, idx, N_cutoff=self.N_cutoff)
        return FockState(modes=self.modes, rho=new_rho, N_cutoff=self.N_cutoff)

    def photon_addition(self, mode: str) -> FockState:
        idx = self.get_mode_index(mode)
        new_rho = photon_addition(self.rho, idx, N_cutoff=self.N_cutoff)
        return FockState(modes=self.modes, rho=new_rho, N_cutoff=self.N_cutoff)

    def realistic_photon_subtraction(
        self,
        mode: str,
        tap_reflectivity: float = 0.05,
        detector_efficiency: float = 0.6,
        ancilla_cutoff: int = 6,
    ) -> FockState:
        idx = self.get_mode_index(mode)
        new_rho = realistic_photon_subtraction(
            self.rho,
            idx,
            N_cutoff=self.N_cutoff,
            tap_reflectivity=tap_reflectivity,
            detector_efficiency=detector_efficiency,
            ancilla_cutoff=ancilla_cutoff,
        )
        return FockState(modes=self.modes, rho=new_rho, N_cutoff=self.N_cutoff)

    def realistic_photon_addition(
        self,
        mode: str,
        coupling_strength: float = 0.05,
        detector_efficiency: float = 0.6,
        ancilla_cutoff: int = 6,
    ) -> FockState:
        idx = self.get_mode_index(mode)
        new_rho = realistic_photon_addition(
            self.rho,
            idx,
            N_cutoff=self.N_cutoff,
            coupling_strength=coupling_strength,
            detector_efficiency=detector_efficiency,
            ancilla_cutoff=ancilla_cutoff,
        )
        return FockState(modes=self.modes, rho=new_rho, N_cutoff=self.N_cutoff)

    # -- Observables & measurement -------------------------------------------

    def mean_photon_number(self, mode: str) -> float:
        idx = self.get_mode_index(mode)
        return mean_photon_number(self.rho, idx, N_cutoff=self.N_cutoff)

    def photon_number_measurement(
        self,
        mode: str,
        outcome: int | None = None,
        rng: np.random.Generator | None = None,
    ) -> tuple[int, FockState]:
        """Ideal photon-number-resolving detection on ``mode``.

        Mirrors the module-level :func:`photon_number_measurement`: on a
        multi-mode state the measured mode is traced out of the returned
        `FockState`, exactly like `GaussianMeasurements.homodyne_measurement`
        drops the measured mode from the returned `GaussianState`. On a
        single-mode state there is nothing left to trace out, so the
        (trivial, now-classical) collapsed state of that one mode is kept.
        """
        idx = self.get_mode_index(mode)
        result, collapsed = photon_number_measurement(
            self.rho, idx, N_cutoff=self.N_cutoff, outcome=outcome, rng=rng
        )
        remaining_modes = tuple(m for i, m in enumerate(self.modes) if i != idx)
        if not remaining_modes:
            remaining_modes = self.modes
        return result, FockState(
            modes=remaining_modes, rho=collapsed, N_cutoff=self.N_cutoff
        )


class FockGates:
    """Backward-compatible namespace for the functional Fock API.

    New code should use the module-level functions directly. The class remains
    temporarily so existing callers can migrate without a flag day.
    """

    apply_kraus_operator = staticmethod(_apply_kraus_operator)
    photon_subtraction = staticmethod(photon_subtraction)
    photon_addition = staticmethod(photon_addition)
    realistic_photon_subtraction = staticmethod(realistic_photon_subtraction)
    realistic_photon_addition = staticmethod(realistic_photon_addition)
    mean_photon_number = staticmethod(mean_photon_number)
    photon_number_measurement = staticmethod(photon_number_measurement)
