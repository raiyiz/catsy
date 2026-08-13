"""Fock-space operations and numerical optical-system simulation."""

from __future__ import annotations

from typing import Any

import numpy as np
import scipy.linalg
import qutip as qt

from .core import (
    TOL_PHYSICALITY,
    TOL_ZERO_ENTRY,
    _check_non_negative,
    _check_positive_int,
    _symplectic_form,
)
from .gaussian import GaussianOperations, GaussianState


# ========================================================================
# Fock
# ========================================================================

class FockOperations:
    """Low-level Fock-space photon click operations, shared by
    NonGaussianOperations (state-in/rho-out) and QBSSimulator (rho-in/rho-out)
    so the math lives in exactly one place."""

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

# ========================================================================
# Simulators
# ========================================================================

class QBSSimulator:
    """Time-dependent master-equation and interferometer simulations."""

    # Backward/forward-compatible entry points onto the single FockOperations
    # implementation (useful when you already hold a converted rho, e.g. mid-
    # simulation, and have no GaussianState to go back to).
    photon_subtraction = staticmethod(FockOperations.photon_subtraction)
    photon_addition = staticmethod(FockOperations.photon_addition)

    @staticmethod
    def run_cavity_with_pulse(
        rho_init,
        tlist: np.ndarray,
        K: float,
        kappa: float,
        amp: float,
        t0: float,
        sigma: float,
        N_cutoff: int,
    ) -> list:
        """Dissipative cavity with a Kerr nonlinearity, driven by a
        time-dependent Gaussian pulse.

        Parameters
        ----------
        rho_init : starting density matrix (e.g. from GaussianState.to_qutip())
        tlist : time grid for the ODE solver
        K : Kerr nonlinearity strength (K * adag^2 * a^2)
        kappa : cavity photon-loss rate
        amp, t0, sigma : Gaussian pulse-shape parameters
        N_cutoff : Hilbert-space dimension
        """
    
        _check_positive_int(N_cutoff, "N_cutoff")
        _check_non_negative(kappa, "kappa")
        if len(tlist) < 2:
            raise ValueError("tlist must contain at least 2 time points.")
        if sigma <= 0:
            raise ValueError(f"sigma must be > 0, got {sigma}.")

        a = qt.destroy(N_cutoff)
        H_kerr = K * a.dag() * a.dag() * a * a

        def pulse_shape(t, amp, t0, sigma):
            return amp * np.exp(-((t - t0) ** 2) / (2 * sigma**2))

        H_total = [H_kerr, [a + a.dag(), pulse_shape]]
        c_ops = [np.sqrt(kappa) * a] if kappa > 0 else []
        args = {"amp": amp, "t0": t0, "sigma": sigma}

        res = qt.mesolve(H_total, rho_init, tlist, c_ops=c_ops, args=args)
        return res.states

    @staticmethod
    def scan_mzi_with_loss(
        psi_cat_single,
        theta_list: np.ndarray,
        kappa: float,
        N_cutoff: int,
        *,
        loss_time: float = 1.0,
    ) -> dict:
        """Scan a cat state through a noisy Mach-Zehnder interferometer.

        ``theta_list`` is now a pure phase scan.  It no longer controls the
        duration for which the arm is exposed to loss.  ``kappa`` is the
        photon-loss rate and ``loss_time`` is the fixed physical exposure
        time of the lossy arm, so its amplitude transmissivity is independent
        of the scanned phase.

        The model is: input -> 50:50 BS -> fixed-time amplitude damping on
        arm 1 -> phase shift on arm 1 -> second 50:50 BS -> measurements.
        Amplitude damping is phase-covariant, so applying the fixed loss before
        the phase is physically equivalent to applying it after the phase.
        """
        _check_positive_int(N_cutoff, "N_cutoff")
        _check_non_negative(kappa, "kappa")
        _check_non_negative(loss_time, "loss_time")
        theta_list = np.asarray(theta_list, dtype=float)
        if theta_list.ndim != 1 or len(theta_list) < 1:
            raise ValueError("theta_list must be a non-empty 1D array.")
        if not np.all(np.isfinite(theta_list)):
            raise ValueError("theta_list must contain only finite values.")

        a1 = qt.tensor(qt.destroy(N_cutoff), qt.qeye(N_cutoff))
        a2 = qt.tensor(qt.qeye(N_cutoff), qt.destroy(N_cutoff))

        n1_op = a1.dag() * a1
        n2_op = a2.dag() * a2
        parity1_op = (1j * np.pi * n1_op).expm()

        U_BS = ((1j * np.pi / 4) * (a1.dag() * a2 + a1 * a2.dag())).expm()

        psi_in = qt.tensor(psi_cat_single, qt.fock(N_cutoff, 0))
        psi_after_BS1 = U_BS * psi_in

        # The loss channel is independent of theta.  For amplitude damping
        # generated by H=0 and collapse operator sqrt(kappa)*a, the fixed
        # exposure time gives eta = exp(-kappa * loss_time).
        c_ops = [np.sqrt(kappa) * a1] if kappa > 0 and loss_time > 0 else []
        if c_ops:
            loss_sim = qt.mesolve(
                0 * n1_op,
                psi_after_BS1,
                [0.0, float(loss_time)],
                c_ops=c_ops,
            )
            rho_after_loss = loss_sim.states[-1]
            if rho_after_loss.isket:
                rho_after_loss = qt.ket2dm(rho_after_loss)
        elif psi_after_BS1.isket:
            rho_after_loss = qt.ket2dm(psi_after_BS1)
        else:
            rho_after_loss = psi_after_BS1

        results = {
            "theta": theta_list,
            "n1": [],
            "n2": [],
            "parity1": [],
        }

        for theta in theta_list:
            U_phase = (1j * float(theta) * n1_op).expm()
            rho_after_phase = U_phase * rho_after_loss * U_phase.dag()
            rho_out = U_BS * rho_after_phase * U_BS.dag()

            results["n1"].append(qt.expect(n1_op, rho_out))
            results["n2"].append(qt.expect(n2_op, rho_out))
            results["parity1"].append(qt.expect(parity1_op, rho_out).real)

        return results
