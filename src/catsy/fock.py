"""Low-level operations on QuTiP Fock-space states.

This is the discrete-variable counterpart to :mod:`catsy.gaussian`: where
the Gaussian layer represents a state by its first and second phase-space
moments ``(d, V)`` and stays exact for as long as every transformation is
linear/Gaussian, this module represents a state by an explicit, truncated
QuTiP density matrix and can express operations -- such as heralded photon
addition/subtraction -- that push a state outside the Gaussian family.

The public API is functional: photon operations act directly on QuTiP
density matrices. ``FockGates`` remains only as a backwards-compatible
namespace for existing callers.

:meth:`photon_subtraction`/:meth:`photon_addition` are the textbook,
unit-efficiency operators ``a``/``a†``. :meth:`realistic_photon_subtraction`/
:meth:`realistic_photon_addition` model how these are actually implemented
on an optical bench: a weak coupling to an ancilla mode (a beamsplitter tap
for subtraction, a weak parametric two-mode-squeezing interaction for
addition) followed by heralding on an imperfect "click" detector on the
ancilla. In the limit of weak coupling and unit detector efficiency, both
converge to the corresponding ideal operator; at finite coupling and
efficiency they reproduce the impurity real heralded sources have.

The Fock layer deliberately operates on QuTiP objects rather than a
bespoke wrapper class. Conversion from a
:class:`~catsy.gaussian.GaussianState` belongs at the phase-space/Fock
boundary via ``GaussianState.to_qutip()``.
"""

from __future__ import annotations

import numpy as np
import qutip as qt

from .core import (
    TOL_PHYSICALITY,
    _check_non_negative,
    _check_positive_int,
    _check_unit_interval,
)


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


def _mode_operator(
    op_1mode: qt.Qobj,
    n_modes: int,
    mode_idx: int,
    N_cutoff: int,
) -> qt.Qobj:
    """Expand a single-mode operator onto the selected subsystem."""
    return qt.expand_operator(op_1mode, [N_cutoff] * n_modes, mode_idx)


def _apply_kraus_operator(
    rho: qt.Qobj,
    kraus_op: qt.Qobj,
    label: str = "apply_kraus_operator",
) -> qt.Qobj:
    """Apply a single Kraus operator and renormalize."""
    if not isinstance(kraus_op, qt.Qobj) or not kraus_op.isoper:
        raise TypeError("kraus_op must be a QuTiP operator (Qobj).")
    if kraus_op.dims[0] != rho.dims[0] or kraus_op.dims[1] != rho.dims[1]:
        raise ValueError(
            "kraus_op must act on the same Hilbert space as rho; "
            f"got kraus_op.dims={kraus_op.dims!r}, rho.dims={rho.dims!r}."
        )

    rho_new = kraus_op * rho * kraus_op.dag()
    trace_val = rho_new.tr()
    if abs(trace_val) < TOL_PHYSICALITY:
        raise ValueError(
            f"{label}: heralding success probability is numerically zero."
        )
    return rho_new / trace_val


def photon_subtraction(
    rho: qt.Qobj,
    mode_idx: int = 0,
    N_cutoff: int | None = None,
) -> qt.Qobj:
    """Apply textbook photon subtraction ``rho -> a rho a†``."""
    n_modes, cutoff = _validate_state(rho, N_cutoff, mode_idx)
    a_op = _mode_operator(qt.destroy(cutoff), n_modes, mode_idx, cutoff)
    return _apply_kraus_operator(rho, a_op, "photon_subtraction")


def photon_addition(
    rho: qt.Qobj,
    mode_idx: int = 0,
    N_cutoff: int | None = None,
) -> qt.Qobj:
    """Apply textbook photon addition ``rho -> a† rho a``."""
    n_modes, cutoff = _validate_state(rho, N_cutoff, mode_idx)
    adag_op = _mode_operator(qt.create(cutoff), n_modes, mode_idx, cutoff)
    return _apply_kraus_operator(rho, adag_op, "photon_addition")


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

    a_sys = _mode_operator(qt.destroy(cutoff), len(dims), mode_idx, cutoff)
    a_anc = _mode_operator(
        qt.destroy(ancilla_cutoff), len(dims), ancilla_idx, ancilla_cutoff
    )

    if coupling_kind == "subtract":
        generator = coupling_strength * (
            a_sys * a_anc.dag() - a_sys.dag() * a_anc
        )
    else:
        generator = coupling_strength * (
            a_sys.dag() * a_anc.dag() - a_sys * a_anc
        )
    coupling_unitary = generator.expm()

    ancilla_vacuum = qt.fock_dm(ancilla_cutoff, 0)
    rho_extended = qt.tensor(rho, ancilla_vacuum)
    rho_coupled = coupling_unitary * rho_extended * coupling_unitary.dag()

    no_click_diag = (1.0 - detector_efficiency) ** np.arange(ancilla_cutoff)
    click_diag = np.sqrt(np.clip(1.0 - no_click_diag, 0.0, None))
    click_operator_anc = qt.Qobj(np.diag(click_diag))
    click_operator = _mode_operator(
        click_operator_anc,
        len(dims),
        ancilla_idx,
        ancilla_cutoff,
    )

    rho_heralded = _apply_kraus_operator(rho_coupled, click_operator, label)
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
        qt.expand_operator(qt.fock_dm(cutoff, n), dims, mode_idx)
        for n in range(cutoff)
    ]
    collapsed_states, probabilities = qt.measurement_statistics_povm(
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
