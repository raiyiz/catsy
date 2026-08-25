"""Low-level operations on QuTiP Fock-space states.

This is the discrete-variable counterpart to :mod:`catsy.gaussian`: where
the Gaussian layer represents a state by its first and second phase-space
moments ``(d, V)`` and stays exact for as long as every transformation is
linear/Gaussian, this module represents a state by an explicit, truncated
QuTiP density matrix and can express operations -- such as heralded photon
addition/subtraction -- that push a state outside the Gaussian family.

``FockGates`` provides one generic primitive, :meth:`FockGates.apply_kraus_operator`
(apply a single-mode Kraus operator and renormalize, i.e. a heralded
quantum operation), plus the specific operators built on top of it:
photon subtraction/addition, ideal photon-number-resolving detection, and
mean-photon-number readout. This mirrors the Gaussian layer's own
generic-channel-plus-presets split (``GaussianChannel`` + ``LossChannels``).

The Fock layer deliberately operates on QuTiP objects rather than a
bespoke wrapper class. Conversion from a
:class:`~catsy.gaussian.GaussianState` belongs at the phase-space/Fock
boundary via ``GaussianState.to_qutip()``; no second convenience layer is
maintained here.
"""

from __future__ import annotations

import numpy as np
import qutip as qt

from .core import TOL_PHYSICALITY, _check_positive_int


class FockGates:
    """Primitive photon operations acting directly on QuTiP states."""

    @staticmethod
    def _validate_state(rho: qt.Qobj, N_cutoff: int, mode_idx: int) -> int:
        if not isinstance(rho, qt.Qobj):
            raise TypeError(f"rho must be a QuTiP Qobj, got {type(rho).__name__}.")
        if not rho.isoper:
            raise ValueError("rho must be a QuTiP operator (density matrix).")

        _check_positive_int(N_cutoff, "N_cutoff")

        dims = rho.dims[0]
        if not dims or any(dim != N_cutoff for dim in dims):
            raise ValueError(
                "N_cutoff must match every mode dimension of rho; "
                f"rho has dimensions {dims!r}, got N_cutoff={N_cutoff}."
            )

        n_modes = len(dims)
        if not isinstance(mode_idx, int) or not 0 <= mode_idx < n_modes:
            raise ValueError(
                f"mode_idx must be an integer in [0, {n_modes - 1}], got {mode_idx!r}."
            )

        return n_modes

    @staticmethod
    def _mode_operator(
        op_1mode: qt.Qobj,
        n_modes: int,
        mode_idx: int,
        N_cutoff: int,
    ) -> qt.Qobj:
        if n_modes == 1:
            return op_1mode
        op_list = [qt.qeye(N_cutoff) for _ in range(n_modes)]
        op_list[mode_idx] = op_1mode
        return qt.tensor(*op_list)

    @staticmethod
    def apply_kraus_operator(
        rho: qt.Qobj,
        kraus_op: qt.Qobj,
        label: str = "apply_kraus_operator",
    ) -> qt.Qobj:
        """Apply a single Kraus operator ``rho -> K rho K†`` and renormalize.

        This is the generic heralded-operation primitive that
        :meth:`photon_subtraction`, :meth:`photon_addition`, and
        :meth:`photon_number_measurement` are all built from. ``kraus_op``
        must act on the same (already mode-embedded) Hilbert space as
        ``rho``; use :meth:`_mode_operator` to embed a single-mode operator
        into a multi-mode space first. ``label`` is used only to identify
        the operation in the error message if the herald fails.
        """
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

    @staticmethod
    def photon_subtraction(
        rho: qt.Qobj,
        mode_idx: int = 0,
        N_cutoff: int = 20,
    ) -> qt.Qobj:
        """Apply photon subtraction ``rho -> a rho a†`` and renormalize.

        ``rho`` must already be represented in the QuTiP Fock basis.  For a
        Gaussian state, call ``state.to_qutip(N_cutoff=...)`` first.
        """
        n_modes = FockGates._validate_state(rho, N_cutoff, mode_idx)
        a_op = FockGates._mode_operator(qt.destroy(N_cutoff), n_modes, mode_idx, N_cutoff)
        return FockGates.apply_kraus_operator(rho, a_op, "photon_subtraction")

    @staticmethod
    def photon_addition(
        rho: qt.Qobj,
        mode_idx: int = 0,
        N_cutoff: int = 20,
    ) -> qt.Qobj:
        """Apply photon addition ``rho -> a† rho a`` and renormalize.

        ``rho`` must already be represented in the QuTiP Fock basis.  For a
        Gaussian state, call ``state.to_qutip(N_cutoff=...)`` first.
        """
        n_modes = FockGates._validate_state(rho, N_cutoff, mode_idx)
        adag_op = FockGates._mode_operator(
            qt.create(N_cutoff), n_modes, mode_idx, N_cutoff
        )
        return FockGates.apply_kraus_operator(rho, adag_op, "photon_addition")

    @staticmethod
    def mean_photon_number(
        rho: qt.Qobj,
        mode_idx: int = 0,
        N_cutoff: int = 20,
    ) -> float:
        """Return ``<n> = tr(rho * a†a)`` for the selected mode.

        The Fock-space analysis counterpart of the Gaussian layer's
        ``compute_*`` diagnostics; unlike those, this is exact at any
        cutoff large enough that ``rho``'s population near ``N_cutoff - 1``
        is negligible.
        """
        n_modes = FockGates._validate_state(rho, N_cutoff, mode_idx)
        n_op = FockGates._mode_operator(qt.num(N_cutoff), n_modes, mode_idx, N_cutoff)
        return float(np.real((n_op * rho).tr()))

    @staticmethod
    def photon_number_measurement(
        rho: qt.Qobj,
        mode_idx: int = 0,
        N_cutoff: int = 20,
        outcome: int | None = None,
        rng: np.random.Generator | None = None,
    ) -> tuple[int, qt.Qobj]:
        """Ideal photon-number-resolving detection on ``mode_idx``.

        Projects onto a definite Fock outcome ``n`` (sampled from the
        measured mode's photon-number distribution, unless ``outcome`` is
        given to force a particular result -- useful for deterministic
        tests), then traces the measured mode out. Returns
        ``(outcome, rho_remaining)``, the Fock-basis analog of
        ``GaussianMeasurements.homodyne_measurement``'s
        ``(outcome, GaussianState)`` contract.

        For a single-mode input, ``rho_remaining`` is the scalar
        1-dimensional Qobj left after the only mode is measured out.
        """
        n_modes = FockGates._validate_state(rho, N_cutoff, mode_idx)

        if outcome is not None:
            if not isinstance(outcome, int) or not 0 <= outcome < N_cutoff:
                raise ValueError(
                    f"outcome must be an integer in [0, {N_cutoff - 1}], got {outcome!r}."
                )
        else:
            reduced = rho.ptrace(mode_idx) if n_modes > 1 else rho
            probs = np.clip(np.real(reduced.diag()), 0.0, None)
            total = probs.sum()
            if total < TOL_PHYSICALITY:
                raise ValueError(
                    "photon_number_measurement: outcome probabilities are "
                    "numerically zero for every Fock level."
                )
            rng = rng if rng is not None else np.random.default_rng()
            outcome = int(rng.choice(N_cutoff, p=probs / total))

        projector = FockGates._mode_operator(
            qt.fock_dm(N_cutoff, outcome), n_modes, mode_idx, N_cutoff
        )
        collapsed = FockGates.apply_kraus_operator(
            rho, projector, "photon_number_measurement"
        )

        if n_modes > 1:
            remaining = [i for i in range(n_modes) if i != mode_idx]
            collapsed = collapsed.ptrace(remaining)

        return outcome, collapsed
