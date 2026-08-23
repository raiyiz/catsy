"""QuTiP-based physical simulations of specific optical hardware.

These operate directly on QuTiP Fock-space states rather than on
GaussianState/Circuit; they model specific pieces of optical hardware (a
driven cavity, an interferometer) rather than generic phase-space
transformations. Reusable Gaussian gate layouts belong on `Circuit` itself
(see core.py) -- there is no separate optical-bench abstraction here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, TypedDict

import numpy as np
import qutip as qt

from .core import _check_non_negative, _check_positive_int
from .types import FloatArray


@dataclass(slots=True, eq=False)
class Mode:
    """A named runtime optical mode.

    Modes use object identity for equality. A mode may optionally be owned by
    a :class:`~catsy.core.Circuit`; ownership is assigned by the circuit when
    it adopts the mode.
    """

    name: str
    owner: Circuit | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("Mode name must be a non-empty string.")


if TYPE_CHECKING:
    from .core import Circuit


# ---------------------------------------------------------------------------
# Physical-system simulations
# ---------------------------------------------------------------------------


class KerrCavity:
    """Driven, dissipative single-mode cavity with Kerr nonlinearity.

    Parameters
    ----------
    K:
        Kerr nonlinearity strength in ``K * a†² a²``.
    kappa:
        Cavity photon-loss rate.
    N_cutoff:
        Fock-space Hilbert-space dimension.
    """

    def __init__(self, K: float, kappa: float, N_cutoff: int):
        _check_positive_int(N_cutoff, "N_cutoff")
        _check_non_negative(kappa, "kappa")
        if not np.isfinite(K):
            raise ValueError(f"K must be finite, got {K!r}.")
        self.K = K
        self.kappa = kappa
        self.N_cutoff = N_cutoff

    def run(
        self,
        rho_init: qt.Qobj,
        tlist: FloatArray,
        amp: float,
        t0: float,
        sigma: float,
    ) -> list[qt.Qobj]:
        """Evolve ``rho_init`` under the driven Kerr-cavity master equation.

        ``amp``, ``t0`` and ``sigma`` define the Gaussian drive pulse.
        """
        tlist = np.asarray(tlist, dtype=float)
        if tlist.ndim != 1 or len(tlist) < 2:
            raise ValueError("tlist must be a 1D array with at least 2 time points.")
        if not np.all(np.isfinite(tlist)):
            raise ValueError("tlist must contain only finite values.")
        if not np.isfinite(amp):
            raise ValueError(f"amp must be finite, got {amp!r}.")
        if not np.isfinite(t0):
            raise ValueError(f"t0 must be finite, got {t0!r}.")
        if not np.isfinite(sigma) or sigma <= 0:
            raise ValueError(f"sigma must be > 0 and finite, got {sigma!r}.")

        a = qt.destroy(self.N_cutoff)
        H_kerr = self.K * a.dag() * a.dag() * a * a

        def pulse_shape(t: float, amp: float, t0: float, sigma: float) -> float:
            return float(amp * np.exp(-((t - t0) ** 2) / (2 * sigma**2)))

        H_total = [H_kerr, [a + a.dag(), pulse_shape]]
        c_ops = [np.sqrt(self.kappa) * a] if self.kappa > 0 else []
        args = {"amp": amp, "t0": t0, "sigma": sigma}

        result = qt.mesolve(H_total, rho_init, tlist, c_ops=c_ops, args=args)
        states: list[qt.Qobj] = result.states
        return states


class ObservableScanData(TypedDict):
    theta: FloatArray
    n1: list[float]
    n2: list[float]
    parity1: list[float]


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
        self.kappa = kappa
        self.N_cutoff = N_cutoff
        self.loss_time = loss_time

    def scan(self, psi_cat_single: qt.Qobj, theta_list: FloatArray) -> ObservableScanData:
        """Scan the phase of the lossy arm and return output observables.

        ``psi_cat_single`` may be a ket or a density matrix -- the latter
        lets the output of a lossy simulation (e.g. ``KerrCavity.run``) be
        fed in directly without an intermediate purification step.

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

        # psi_cat_single may be a ket (e.g. a hand-built cat state) or a
        # density matrix (e.g. the output of a lossy KerrCavity.run()) -- the
        # vacuum port and the beam-splitter application below must match it,
        # since qt.tensor requires both operands to be the same Qobj type
        # and U*rho (unlike U*psi) needs an explicit U*rho*U.dag() to conjugate.
        if psi_cat_single.isket:
            psi_in = qt.tensor(psi_cat_single, qt.fock(N, 0))
            psi_after_BS1 = U_BS * psi_in
        else:
            psi_in = qt.tensor(psi_cat_single, qt.ket2dm(qt.fock(N, 0)))
            psi_after_BS1 = U_BS * psi_in * U_BS.dag()

        c_ops = (
            [np.sqrt(self.kappa) * a1] if self.kappa > 0 and self.loss_time > 0 else []
        )
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

            results["n1"].append(qt.expect(n1_op, rho_out))
            results["n2"].append(qt.expect(n2_op, rho_out))
            results["parity1"].append(qt.expect(parity1_op, rho_out).real)

        return results
