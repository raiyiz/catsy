"""Driven dissipative Kerr-cavity simulations."""

from __future__ import annotations

import numpy as np
import qutip as qt

from ..core import _check_non_negative, _check_positive_int
from ..types import FloatArray


class KerrCavity:
    """Driven, dissipative single-mode cavity with Kerr nonlinearity."""

    def __init__(self, K: float, kappa: float, N_cutoff: int):
        _check_positive_int(N_cutoff, "N_cutoff")
        _check_non_negative(kappa, "kappa")
        if not np.isfinite(K):
            raise ValueError(f"K must be finite, got {K!r}.")
        self.K = K
        self.kappa = kappa
        self.N_cutoff = N_cutoff

    def run(self, rho_init: qt.Qobj, tlist: FloatArray, amp: float, t0: float, sigma: float) -> list[qt.Qobj]:
        """Evolve ``rho_init`` under the driven Kerr-cavity master equation."""
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
        result = qt.mesolve(H_total, rho_init, tlist, c_ops=c_ops, args={"amp": amp, "t0": t0, "sigma": sigma})
        return list(result.states)
