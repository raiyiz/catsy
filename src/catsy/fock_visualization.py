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

    The panel shows ``P(n)`` together with the mean photon number and the
    normalized factorial moment ``g^(2)(0) = <n(n-1)>/<n>^2``. The latter
    distinguishes antibunched/sub-Poissonian states from classical Poissonian
    statistics without relying on a Gaussian approximation.
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
    ax.bar(n, p, width=0.82, alpha=0.82, label=r"$P(n)$")
    ax.axvline(mean, ls="--", lw=1.4, alpha=0.65, label=fr"$\langle n\rangle={mean:.2f}$")
    ax.set_xlabel("photon number $n$")
    ax.set_ylabel("probability")
    ax.set_title(f"Photon-number statistics — mode {mode_idx}", pad=14)
    ax.set_xticks(n)
    ax.grid(axis="y", alpha=0.12, linewidth=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    annotation = fr"$g^{{(2)}}(0) = {g2:.3g}$" if np.isfinite(g2) else r"$g^{(2)}(0)$ undefined"
    ax.text(
        0.98,
        0.96,
        annotation,
        transform=ax.transAxes,
        ha="right",
        va="top",
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.85, "edgecolor": "none"},
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
    show: bool = False,
) -> plt.Figure:
    """Plot a single-mode Wigner function, including its negativity contour.

    For a multimode state, ``mode_idx`` is reduced first. A black zero contour
    separates positive and negative Wigner regions, making non-Gaussian
    structure such as photon-subtraction-induced negativity easy to inspect.
    """
    if resolution < 32:
        raise ValueError("resolution must be at least 32.")
    if xlim[0] >= xlim[1]:
        raise ValueError("xlim must be an increasing pair of finite values.")
    if not np.all(np.isfinite(xlim)):
        raise ValueError("xlim must contain finite values.")

    state = _mode_state(rho, mode_idx)
    grid = np.linspace(xlim[0], xlim[1], resolution)
    wigner = qt.wigner(state, grid, grid)

    if ax is None:
        fig, ax = plt.subplots(figsize=(6.6, 5.8), constrained_layout=True)
    else:
        fig = cast(plt.Figure, ax.figure)
    image = ax.contourf(grid, grid, wigner, levels=80, cmap="RdBu_r")
    ax.contour(grid, grid, wigner, levels=[0.0], colors="black", linewidths=0.8, alpha=0.8)
    ax.axhline(0, lw=0.5, ls="--", alpha=0.25)
    ax.axvline(0, lw=0.5, ls="--", alpha=0.25)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$p$")
    ax.set_title(f"Wigner function — mode {mode_idx}", pad=14, fontweight="medium")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label=r"$W(x,p)$")
    return _finalize(fig, show)
