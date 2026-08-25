"""Visualizations for truncated QuTiP Fock-space states.

The Fock layer complements the Gaussian visualization API by exposing
quantities that are invisible to first- and second-moment descriptions:
photon-number statistics, Fock-basis coherences, and Wigner negativity.
All helpers return Matplotlib figures and never display them unless
``show=True``.
"""

from __future__ import annotations

from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import qutip as qt


def _mode_state(rho: qt.Qobj, mode_idx: int) -> qt.Qobj:
    """Return a single-mode density matrix, tracing other modes out."""
    if not isinstance(rho, qt.Qobj):
        raise TypeError(f"rho must be a QuTiP Qobj, got {type(rho).__name__}.")
    if not rho.isoper:
        raise ValueError("rho must be a QuTiP operator (density matrix).")

    dims = rho.dims[0]
    if not dims:
        raise ValueError("rho must contain at least one mode.")
    if not isinstance(mode_idx, int) or not 0 <= mode_idx < len(dims):
        raise ValueError(
            f"mode_idx must be an integer in [0, {len(dims) - 1}], got {mode_idx!r}."
        )
    return rho if len(dims) == 1 else rho.ptrace(mode_idx)


def _finalize(fig: plt.Figure, show: bool) -> plt.Figure:
    if show:
        plt.show()
    return fig


def plot_photon_statistics(
    rho: qt.Qobj,
    *,
    mode_idx: int = 0,
    ax: plt.Axes | None = None,
    n_max: int | None = None,
    show: bool = False,
) -> plt.Figure:
    """Plot photon-number probabilities and annotate non-Poissonianity.

    QuTiP provides the standard Fock-distribution rendering; Catsy adds the
    mean photon number and ``g^(2)(0)`` diagnostic on top.
    """
    state = _mode_state(rho, mode_idx)
    cutoff = state.dims[0][0]
    probabilities = np.clip(np.real(state.diag()), 0.0, None)
    probabilities /= probabilities.sum()

    if n_max is None:
        support = np.flatnonzero(probabilities > 1e-8)
        n_max = int(support[-1]) if support.size else 0
        n_max = min(n_max + 2, cutoff - 1)
    if not isinstance(n_max, int) or not 0 <= n_max < cutoff:
        raise ValueError(f"n_max must be an integer in [0, {cutoff - 1}].")

    n = np.arange(n_max + 1)
    p = probabilities[: n_max + 1]
    mean = float(np.dot(n, p))
    factorial_second = float(np.dot(n * (n - 1), p))
    g2 = factorial_second / mean**2 if mean > np.finfo(float).eps else float("nan")

    if ax is None:
        fig, ax = plt.subplots(figsize=(7.0, 4.8), constrained_layout=True)
    else:
        fig = cast(plt.Figure, ax.figure)

    qt.plot_fock_distribution(
        state,
        fig=fig,
        ax=ax,
        n_toconv=n_max,
        show=False,
    )
    ax.axvline(
        mean,
        ls="--",
        lw=1.4,
        alpha=0.65,
        label=fr"$\langle n\rangle={mean:.2f}$",
    )
    ax.set_title(f"Photon-number statistics — mode {mode_idx}", pad=14)
    annotation = (
        fr"$g^{{(2)}}(0) = {g2:.3g}$"
        if np.isfinite(g2)
        else r"$g^{(2)}(0)$ undefined"
    )
    ax.text(
        0.98,
        0.96,
        annotation,
        transform=ax.transAxes,
        ha="right",
        va="top",
        bbox={
            "boxstyle": "round,pad=0.3",
            "facecolor": "white",
            "alpha": 0.85,
            "edgecolor": "none",
        },
    )
    ax.legend(frameon=False)
    return _finalize(fig, show)


def plot_fock_density_matrix(
    rho: qt.Qobj,
    *,
    mode_idx: int = 0,
    show: bool = False,
) -> plt.Figure:
    """Visualize diagonal occupation and off-diagonal Fock coherences.

    The two panels show ``|rho_mn|`` and ``arg(rho_mn)``. This makes coherent
    superpositions, phase structure, and mixed-state decoherence visible in a
    way that a photon-number histogram alone cannot capture.
    """
    state = _mode_state(rho, mode_idx)
    matrix = state.full()
    cutoff = matrix.shape[0]
    magnitude = np.abs(matrix)
    phase = np.angle(matrix)

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.8), constrained_layout=True)
    ax_mag, ax_phase = axes
    image_mag = ax_mag.imshow(magnitude, origin="lower", interpolation="nearest")
    image_phase = ax_phase.imshow(
        phase,
        origin="lower",
        interpolation="nearest",
        cmap="twilight",
        vmin=-np.pi,
        vmax=np.pi,
    )
    ticks = np.arange(cutoff)
    for ax in axes:
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xlabel("Fock index $n$")
        ax.set_ylabel("Fock index $m$")
    ax_mag.set_title(r"Magnitude $|\rho_{mn}|$")
    ax_phase.set_title(r"Phase $\arg(\rho_{mn})$")
    fig.colorbar(image_mag, ax=ax_mag, fraction=0.046, pad=0.04)
    fig.colorbar(image_phase, ax=ax_phase, fraction=0.046, pad=0.04, label="radians")
    fig.suptitle(f"Fock density matrix — mode {mode_idx}", fontweight="medium")
    return _finalize(fig, show)


def plot_wigner(
    rho: qt.Qobj,
    *,
    mode_idx: int = 0,
    xlim: tuple[float, float] = (-5.0, 5.0),
    resolution: int = 180,
    ax: plt.Axes | None = None,
    projection: str = "2d",
    show: bool = False,
) -> plt.Figure:
    """Plot a single-mode Wigner function using QuTiP's renderer.

    ``projection`` can be ``"2d"`` or ``"3d"``. For the 2D view, Catsy adds
    the zero-negativity contour; all Wigner evaluation and rendering are
    delegated to QuTiP.
    """
    if resolution < 32:
        raise ValueError("resolution must be at least 32.")
    if xlim[0] >= xlim[1]:
        raise ValueError("xlim must be an increasing pair of finite values.")
    if not np.all(np.isfinite(xlim)):
        raise ValueError("xlim must contain finite values.")
    if projection not in {"2d", "3d"}:
        raise ValueError("projection must be '2d' or '3d'.")

    state = _mode_state(rho, mode_idx)
    grid = np.linspace(xlim[0], xlim[1], resolution)

    if ax is None:
        if projection == "3d":
            fig = plt.figure(figsize=(6.8, 5.8), constrained_layout=True)
            ax = fig.add_subplot(111, projection="3d")
        else:
            fig, ax = plt.subplots(figsize=(6.6, 5.8), constrained_layout=True)
    else:
        fig = cast(plt.Figure, ax.figure)

    qt.plot_wigner(
        state,
        xvec=grid,
        yvec=grid,
        projection=projection,
        fig=fig,
        ax=ax,
        show=False,
    )
    ax.set_title(f"Wigner function — mode {mode_idx}", pad=14, fontweight="medium")

    if projection == "2d":
        wigner = qt.wigner(state, grid, grid)
        ax.contour(
            grid,
            grid,
            wigner,
            levels=[0.0],
            colors="black",
            linewidths=0.8,
            alpha=0.8,
        )
        ax.axhline(0, lw=0.5, ls="--", alpha=0.25)
        ax.axvline(0, lw=0.5, ls="--", alpha=0.25)
        ax.set_aspect("equal", adjustable="box")

    return _finalize(fig, show)
