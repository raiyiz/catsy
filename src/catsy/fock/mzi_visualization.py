"""Visualization helpers for Fock-space Mach–Zehnder scans."""

from __future__ import annotations

from collections.abc import Mapping

import matplotlib.pyplot as plt
import numpy as np

from catsy.visualization import figure_and_axes, finalize_figure


def plot_mach_zehnder_scan(
    scan: Mapping[str, np.ndarray],
    *,
    ax: plt.Axes | None = None,
    show: bool = False,
) -> plt.Figure:
    """Plot mean output photon numbers and parity from an MZI phase scan.

    The scan mapping is expected to contain ``theta``, ``n1``, ``n2`` and
    ``parity1`` arrays as returned by :class:`MachZehnderInterferometer`.
    """
    required = ("theta", "n1", "n2", "parity1")
    missing = [key for key in required if key not in scan]
    if missing:
        raise ValueError(f"scan is missing required fields: {', '.join(missing)}")

    theta = np.asarray(scan["theta"], dtype=float)
    n1 = np.asarray(scan["n1"], dtype=float)
    n2 = np.asarray(scan["n2"], dtype=float)
    parity1 = np.asarray(scan["parity1"], dtype=float)

    if theta.ndim != 1 or theta.size == 0:
        raise ValueError("theta must be a non-empty one-dimensional array.")
    if any(values.shape != theta.shape for values in (n1, n2, parity1)):
        raise ValueError("MZI scan arrays must all have the same shape.")

    fig, ax = figure_and_axes(ax, figsize=(8.0, 5.2))
    ax.plot(theta, n1, label=r"Output 1: $\langle n_1\rangle")
    ax.plot(theta, n2, label=r"Output 2: $\langle n_2\rangle")
    ax.plot(theta, parity1, ls="--", label=r"Output 1 parity")
    ax.set_xlabel(r"Interferometer phase $\theta$ [rad]")
    ax.set_ylabel("Observable")
    ax.set_title("Lossy Mach–Zehnder interferometer scan", pad=12)
    ax.grid(alpha=0.15)
    ax.legend(frameon=False)
    return finalize_figure(fig, show)


__all__ = ["plot_mach_zehnder_scan"]
