"""Fock-space operations and Gaussian-to-Fock non-Gaussian operations."""

from __future__ import annotations

import qutip as qt

from .core import TOL_PHYSICALITY, _check_positive_int
from .gaussian import GaussianState


# ========================================================================
# Fock
# ========================================================================

class FockOperations:
    """Low-level Fock-space photon click operations, shared by the low-level Fock API and Gaussian-to-Fock convenience wrappers so the
    mathematical implementation lives in exactly one place."""

    @staticmethod
    def _mode_operator(op_1mode, n_modes: int, mode_idx: int, N_cutoff: int):
    
        if n_modes == 1:
            return op_1mode
        op_list = [qt.qeye(N_cutoff)] * n_modes
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
        """rho -> a * rho * a^dagger (renormalized). Probabilistic heralding."""
        _check_positive_int(N_cutoff, "N_cutoff")
        n_modes = len(rho.dims[0])
        a_op = FockOperations._mode_operator(
            qt.destroy(N_cutoff), n_modes, mode_idx, N_cutoff
        )
        return FockOperations._apply_and_renormalize(rho, a_op, "photon_subtraction")

    @staticmethod
    def photon_addition(rho, mode_idx: int = 0, N_cutoff: int = 20):
        """rho -> a^dagger * rho * a (renormalized)."""
        _check_positive_int(N_cutoff, "N_cutoff")
        n_modes = len(rho.dims[0])
        adag_op = FockOperations._mode_operator(
            qt.create(N_cutoff), n_modes, mode_idx, N_cutoff
        )
        return FockOperations._apply_and_renormalize(rho, adag_op, "photon_addition")


class NonGaussianOperations:
    """Convenience wrappers that take a GaussianState (converting to Fock
    space internally) instead of an already-converted qutip Qobj."""

    @staticmethod
    def photon_subtraction(state: GaussianState, mode_name: str, N_cutoff: int = 20):
        rho = state.to_qutip(N_cutoff=N_cutoff)
        mode_idx = state.modes.index(mode_name)
        return FockOperations.photon_subtraction(
            rho, mode_idx=mode_idx, N_cutoff=N_cutoff
        )

    @staticmethod
    def photon_addition(state: GaussianState, mode_name: str, N_cutoff: int = 20):
        rho = state.to_qutip(N_cutoff=N_cutoff)
        mode_idx = state.modes.index(mode_name)
        return FockOperations.photon_addition(rho, mode_idx=mode_idx, N_cutoff=N_cutoff)

