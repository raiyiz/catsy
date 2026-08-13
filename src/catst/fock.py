"""Low-level operations on QuTiP Fock-space states."""

from __future__ import annotations

import qutip as qt

from .core import TOL_PHYSICALITY, _check_positive_int


class FockOperations:
    """Primitive photon operations acting directly on QuTiP states.

    The Fock layer deliberately operates on QuTiP objects.  Conversion from a
    :class:`~catst.gaussian.GaussianState` belongs at the phase-space/Fock
    boundary via ``GaussianState.to_qutip()``; no second convenience layer is
    maintained here.
    """

    @staticmethod
    def _validate_state(rho, N_cutoff: int, mode_idx: int):
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
    def _mode_operator(op_1mode, n_modes: int, mode_idx: int, N_cutoff: int):
        if n_modes == 1:
            return op_1mode
        op_list = [qt.qeye(N_cutoff) for _ in range(n_modes)]
        op_list[mode_idx] = op_1mode
        return qt.tensor(*op_list)

    @staticmethod
    def _apply_and_renormalize(rho, op, label: str):
        rho_new = op * rho * op.dag()
        trace_val = rho_new.tr()
        if abs(trace_val) < TOL_PHYSICALITY:
            raise ValueError(
                f"{label}: heralding success probability is numerically zero."
            )
        return rho_new / trace_val

    @staticmethod
    def photon_subtraction(rho, mode_idx: int = 0, N_cutoff: int = 20):
        """Apply photon subtraction ``rho -> a rho a†`` and renormalize.

        ``rho`` must already be represented in the QuTiP Fock basis.  For a
        Gaussian state, call ``state.to_qutip(N_cutoff=...)`` first.
        """
        n_modes = FockOperations._validate_state(rho, N_cutoff, mode_idx)
        a_op = FockOperations._mode_operator(
            qt.destroy(N_cutoff), n_modes, mode_idx, N_cutoff
        )
        return FockOperations._apply_and_renormalize(rho, a_op, "photon_subtraction")

    @staticmethod
    def photon_addition(rho, mode_idx: int = 0, N_cutoff: int = 20):
        """Apply photon addition ``rho -> a† rho a`` and renormalize.

        ``rho`` must already be represented in the QuTiP Fock basis.  For a
        Gaussian state, call ``state.to_qutip(N_cutoff=...)`` first.
        """
        n_modes = FockOperations._validate_state(rho, N_cutoff, mode_idx)
        adag_op = FockOperations._mode_operator(
            qt.create(N_cutoff), n_modes, mode_idx, N_cutoff
        )
        return FockOperations._apply_and_renormalize(rho, adag_op, "photon_addition")
