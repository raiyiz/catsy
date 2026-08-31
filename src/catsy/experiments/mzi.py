"""Mach-Zehnder interferometer simulations in truncated Fock space."""

from __future__ import annotations

from typing import TypedDict

import numpy as np
import qutip as qt

from catsy.fock import make_even_cat

from ..core import _check_non_negative, _check_positive_int
from ..types import FloatArray


class ObservableScanData(TypedDict):
    theta: FloatArray
    n1: list[float]
    n2: list[float]
    parity1: list[float]


class MachZehnderInterferometer:
    """Two-mode Mach-Zehnder interferometer with a lossy phase-sensing arm."""

    DEFAULT_NUM_PHASE_POINTS = 200

    def __init__(
        self, state: qt.Qobj, N_cutoff: int, kappa: float = 0.0, *, loss_time: float = 1.0
    ):
        _check_positive_int(N_cutoff, "N_cutoff")
        _check_non_negative(kappa, "kappa")
        _check_non_negative(loss_time, "loss_time")
        if not isinstance(state, qt.Qobj):
            raise TypeError("state must be a QuTiP Qobj.")
        if not (state.isket or state.isoper):
            raise ValueError("state must be a ket or density matrix.")
        expected_shape = (N_cutoff, 1) if state.isket else (N_cutoff, N_cutoff)
        if state.shape != expected_shape:
            raise ValueError(
                f"state has shape {state.shape}, expected {expected_shape} for N_cutoff={N_cutoff}."
            )
        self.state = state
        self.N_cutoff = N_cutoff
        self.kappa = kappa
        self.loss_time = loss_time
        self.results: ObservableScanData | None = None

    @classmethod
    def even_cat(
        cls,
        *,
        cutoff: int = 22,
        alpha: complex = 4.0 + 2j,
        kappa: float = 0.0,
        loss_time: float = 1.0,
    ) -> MachZehnderInterferometer:
        return cls(
            make_even_cat(cutoff=cutoff, alpha=alpha), cutoff, kappa, loss_time=loss_time
        )

    def scan(self, theta_list: FloatArray | None = None) -> ObservableScanData:
        if theta_list is None:
            theta_list = np.linspace(0.0, 2.0 * np.pi, self.DEFAULT_NUM_PHASE_POINTS)
        else:
            theta_list = np.asarray(theta_list, dtype=float)
        if theta_list.ndim != 1 or len(theta_list) < 1:
            raise ValueError("theta_list must be a non-empty 1D array.")
        if not np.all(np.isfinite(theta_list)):
            raise ValueError("theta_list must contain only finite values.")

        N = self.N_cutoff
        a1 = qt.tensor(qt.destroy(N), qt.qeye(N))
        a2 = qt.tensor(qt.qeye(N), qt.destroy(N))
        n1_op = a1.dag() * a1
        n2_op = a2.dag() * a2
        parity1_op = (1j * np.pi * n1_op).expm()
        U_BS = ((1j * np.pi / 4) * (a1.dag() * a2 + a1 * a2.dag())).expm()

        vacuum = qt.fock(N, 0)
        if self.state.isket:
            psi_in = qt.tensor(self.state, vacuum)
            after_bs = U_BS * psi_in
        else:
            rho_in = qt.tensor(self.state, qt.ket2dm(vacuum))
            after_bs = U_BS * rho_in * U_BS.dag()

        c_ops = (
            [np.sqrt(self.kappa) * a1]
            if self.kappa > 0.0 and self.loss_time > 0.0
            else []
        )
        if c_ops:
            loss_sim = qt.mesolve(0 * n1_op, after_bs, [0.0, self.loss_time], c_ops=c_ops)
            rho_after_loss = loss_sim.states[-1]
            if rho_after_loss.isket:
                rho_after_loss = qt.ket2dm(rho_after_loss)
        elif after_bs.isket:
            rho_after_loss = qt.ket2dm(after_bs)
        else:
            rho_after_loss = after_bs

        results: ObservableScanData = {
            "theta": theta_list,
            "n1": [],
            "n2": [],
            "parity1": [],
        }
        for theta in theta_list:
            U_phase = (1j * theta * n1_op).expm()
            rho_after_phase = U_phase * rho_after_loss * U_phase.dag()
            rho_out = U_BS * rho_after_phase * U_BS.dag()
            results["n1"].append(float(qt.expect(n1_op, rho_out)))
            results["n2"].append(float(qt.expect(n2_op, rho_out)))
            results["parity1"].append(float(qt.expect(parity1_op, rho_out).real))
        self.results = results
        return results
