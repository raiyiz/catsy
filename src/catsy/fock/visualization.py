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
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d.axes3d import Axes3D

from catsy.visualization import (
    add_colorbar,
    annotate_box,
    figure_and_axes,
    finalize_figure,
    style_phase_axes,
)


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


def _photon_statistics(rho: qt.Qobj) -> tuple[np.ndarray, float, float]:
    """Return probabilities, mean photon number, and ``g^(2)(0)``."""
    probabilities = np.clip(np.real(rho.diag()), 0.0, None)
    total = probabilities.sum()
    if total <= np.finfo(float).eps:
        raise ValueError("rho has no positive diagonal probability mass.")
    probabilities /= total

    n = np.arange(len(probabilities))
    mean = float(np.dot(n, probabilities))
    factorial_second = float(np.dot(n * (n - 1), probabilities))
    g2 = factorial_second / mean**2 if mean > np.finfo(float).eps else float("nan")
    return probabilities, mean, g2


def _state_description(rho: qt.Qobj) -> str:
    """Return a concise physical description suitable for plot titles."""
    probabilities, mean, g2 = _photon_statistics(rho)
    peak = int(np.argmax(probabilities))
    if probabilities[peak] > 1.0 - 1e-10:
        return rf"Fock state $|{peak}\rangle$"
    if np.isfinite(g2) and np.isclose(g2, 1.0, atol=2e-3):
        return rf"Poissonian state ($\langle n\rangle={mean:.2f}$)"
    if np.isclose(mean, 0.0, atol=1e-10):
        return "Vacuum state"
    parity = float(np.sum(probabilities[::2]) - np.sum(probabilities[1::2]))
    if abs(parity) > 0.98:
        parity_name = "even-parity" if parity > 0 else "odd-parity"
        return rf"{parity_name} state ($\langle n\rangle={mean:.2f}$)"
    if np.isfinite(g2):
        return rf"Nonclassical state ($\langle n\rangle={mean:.2f}$, $g^{{(2)}}(0)={g2:.2f}$)"
    return rf"Fock-space state ($\langle n\rangle={mean:.2f}$)"


def plot_photon_statistics(
    rho: qt.Qobj,
    *,
    mode_idx: int = 0,
    ax: plt.Axes | None = None,
    n_max: int | None = None,
    show: bool = False,
) -> plt.Figure:
    """Plot photon-number probabilities and annotate non-Poissonianity."""
    state = _mode_state(rho, mode_idx)
    cutoff = state.dims[0][0]
    probabilities, mean, g2 = _photon_statistics(state)

    if n_max is None:
        support = np.flatnonzero(probabilities > 1e-8)
        n_max = int(support[-1]) if support.size else 0
        n_max = min(n_max + 2, cutoff - 1)
    if not isinstance(n_max, int) or not 0 <= n_max < cutoff:
        raise ValueError(f"n_max must be an integer in [0, {cutoff - 1}].")

    fig, ax = figure_and_axes(ax, figsize=(7.0, 4.8))
    qt.plot_fock_distribution(
        state,
        fig=fig,
        ax=ax,
        unit_y_range=False,
    )
    ax.set_xlim(-0.5, n_max + 0.5)
    ax.axvline(
        mean,
        ls="--",
        lw=1.4,
        alpha=0.65,
        label=rf"$\langle n\rangle={mean:.2f}$",
    )
    ax.set_title(_state_description(state), pad=14)
    annotation = (
        rf"$g^{{(2)}}(0) = {g2:.3g}$" if np.isfinite(g2) else r"$g^{(2)}(0)$ undefined"
    )
    annotate_box(
        ax,
        0.98,
        0.96,
        annotation,
        ha="right",
        va="top",
    )
    ax.legend(frameon=False)
    return finalize_figure(fig, show)


def _draw_density_matrix(
    state: qt.Qobj,
    ax_mag: plt.Axes,
    ax_phase: plt.Axes,
) -> None:
    """Draw magnitude and phase of a single-mode density matrix."""
    matrix = state.full()
    cutoff = matrix.shape[0]
    magnitude = np.abs(matrix)
    phase = np.ma.masked_where(magnitude <= 1e-12, np.angle(matrix))

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
    for ax in (ax_mag, ax_phase):
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xlabel("Fock index $n$")
        ax.set_ylabel("Fock index $m$")
    ax_mag.set_title(r"Magnitude $|\rho_{mn}|$")
    ax_phase.set_title(r"Phase $\arg(\rho_{mn})$")
    fig = cast(plt.Figure, ax_mag.figure)
    add_colorbar(fig, image_mag, ax_mag)
    phase_colorbar = add_colorbar(
        fig, image_phase, ax_phase, label=r"phase $\arg(\rho_{mn})$ [rad]"
    )
    phase_colorbar.set_ticks([-np.pi, -np.pi / 2, 0.0, np.pi / 2, np.pi])
    phase_colorbar.set_ticklabels([r"$-\pi$", r"$-\pi/2$", "0", r"$\pi/2$", r"$\pi$"])


def plot_fock_density_matrix(
    rho: qt.Qobj,
    *,
    mode_idx: int = 0,
    axes: tuple[plt.Axes, plt.Axes] | None = None,
    show: bool = False,
) -> plt.Figure:
    """Visualize diagonal occupation and off-diagonal Fock coherences."""
    state = _mode_state(rho, mode_idx)
    if axes is None:
        fig, created_axes = plt.subplots(
            1, 2, figsize=(10.0, 4.8), constrained_layout=True
        )
        ax_mag, ax_phase = created_axes
    else:
        ax_mag, ax_phase = axes
        fig = cast(plt.Figure, ax_mag.figure)
        if ax_phase.figure is not fig:
            raise ValueError("axes must belong to the same Matplotlib figure.")

    _draw_density_matrix(state, ax_mag, ax_phase)
    fig.suptitle(_state_description(state), fontweight="medium")
    return finalize_figure(fig, show)


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
    fig, ax = figure_and_axes(
        ax,
        figsize=(6.8, 5.8) if projection == "3d" else (6.6, 5.8),
        projection="3d" if projection == "3d" else None,
    )

    if projection == "3d":
        # QuTiP's 3D renderer does not expose the Wigner values' colour
        # normalization. Compute the surface here so the colormap always
        # spans the actual minimum and maximum of this Wigner function.
        wigner = qt.wigner(state, grid, grid)
        X, Y = np.meshgrid(grid, grid)
        norm = Normalize(vmin=float(wigner.min()), vmax=float(wigner.max()))
        ax3d = cast(Axes3D, ax)
        ax3d.plot_surface(
            X,
            Y,
            wigner,
            cmap="viridis",
            norm=norm,
            linewidth=0,
            antialiased=True,
        )
    else:
        qt.plot_wigner(
            state,
            xvec=grid,
            yvec=grid,
            projection=projection,
            fig=fig,
            ax=ax,
            colorbar=True,
        )

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
        style_phase_axes(ax)

    ax.set_title(_state_description(state), pad=14, fontweight="medium")
    return finalize_figure(fig, show)


def plot_fock_dashboard(
    rho: qt.Qobj,
    *,
    mode_idx: int = 0,
    xlim: tuple[float, float] = (-5.0, 5.0),
    resolution: int = 140,
    n_max: int | None = None,
    show: bool = False,
) -> plt.Figure:
    """Render a dense four-panel Fock-state diagnostic dashboard.

    The dashboard deliberately combines complementary views: number
    statistics, Wigner negativity, coherence magnitude, and coherence phase.
    It is intended for exploratory work and preserves the full information
    available in the truncated single-mode density matrix.
    """
    state = _mode_state(rho, mode_idx)
    fig = plt.figure(figsize=(13.5, 9.5), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, width_ratios=(1.0, 1.05), height_ratios=(1.0, 1.0))
    ax_stats = fig.add_subplot(grid[0, 0])
    ax_wigner = fig.add_subplot(grid[0, 1])
    ax_mag = fig.add_subplot(grid[1, 0])
    ax_phase = fig.add_subplot(grid[1, 1])

    plot_photon_statistics(state, ax=ax_stats, n_max=n_max)
    plot_wigner(state, xlim=xlim, resolution=resolution, ax=ax_wigner)
    plot_fock_density_matrix(state, axes=(ax_mag, ax_phase))
    fig.suptitle(_state_description(state), fontsize=16, fontweight="medium")
    return finalize_figure(fig, show)


__all__ = [
    "plot_fock_dashboard",
    "plot_fock_density_matrix",
    "plot_photon_statistics",
    "plot_wigner",
]
