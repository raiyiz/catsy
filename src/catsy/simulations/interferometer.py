"""Mach-Zehnder interferometer simulations implemented with QuTiP."""

from __future__ import annotations

import numpy as np
import qutip as qt

from ..core import _check_non_negative, _check_positive_int


class MachZehnderInterferometer:
    """Two-mode Mach-Zehnder interferometer with a lossy phase-sensing arm.

    Parameters
    ----------
    kappa:
        Photon-loss rate in the lossy arm.
    N_cutoff:
        Fock-space Hilbert-space dimension for each optical mode.
    loss_time:
        Fixed physical exposure time of the lossy arm. The loss is applied
        before the scanned phase, so its strength is independent of phase.
    """

    def __init__(self, kappa: float, N_cutoff: int, *, loss_time: float = 1.0):
        _check_positive_int(N_cutoff, "N_cutoff")
        _check_non_negative(kappa, "kappa")
        _check_non_negative(loss_time, "loss_time")
        self.kappa = float(kappa)
        self.N_cutoff = N_cutoff
        self.loss_time = float(loss_time)

    def scan(self, psi_cat_single, theta_list: np.ndarray) -> dict:
        """Scan the phase of the lossy arm and return output observables.

        The model is input -> 50:50 beam splitter -> fixed-time amplitude
        damping on arm 1 -> phase shift on arm 1 -> second 50:50 beam splitter.
        The returned dictionary contains ``theta``, ``n1``, ``n2`` and
        ``parity1`` arrays.
        """
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

        psi_in = qt.tensor(psi_cat_single, qt.fock(N, 0))
        psi_after_BS1 = U_BS * psi_in

        c_ops = [np.sqrt(self.kappa) * a1] if self.kappa > 0 and self.loss_time > 0 else []
        if c_ops:
            loss_sim = qt.mesolve(
                0 * n1_op,
                psi_after_BS1,
                [0.0, self.loss_time],
                c_ops=c_ops,
            )
            rho_after_loss = loss_sim.states[-1]
            if rho_after_loss.isket:
                rho_after_loss = qt.ket2dm(rho_after_loss)
        elif psi_after_BS1.isket:
            rho_after_loss = qt.ket2dm(psi_after_BS1)
        else:
            rho_after_loss = psi_after_BS1

        results = {"theta": theta_list, "n1": [], "n2": [], "parity1": []}

        for theta in theta_list:
            U_phase = (1j * float(theta) * n1_op).expm()
            rho_after_phase = U_phase * rho_after_loss * U_phase.dag()
            rho_out = U_BS * rho_after_phase * U_BS.dag()

            results["n1"].append(qt.expect(n1_op, rho_out))
            results["n2"].append(qt.expect(n2_op, rho_out))
            results["parity1"].append(qt.expect(parity1_op, rho_out).real)

        return results
