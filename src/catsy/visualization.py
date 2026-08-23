"""Publication-friendly visualizations for Gaussian states.

The plotting helpers are deliberately side-effect free by default: they return
Matplotlib figures/axes and only call ``show()`` when explicitly requested.
This makes them useful both in notebooks and in automated documentation/tests.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse

from .gaussian import GaussianState


def _finalize(fig: plt.Figure, show: bool) -> plt.Figure:
    """Optionally display and always return a figure."""
    fig.tight_layout()
    if show:
        plt.show()
    return fig


def plot_covariance_matrix(
    state: GaussianState,
    *,
    ax: plt.Axes | None = None,
    annotate: bool = True,
    show: bool = False,
) -> plt.Figure:
    """Plot a labelled covariance matrix as a symmetric heatmap.

    Quadratures are labelled ``x_mode`` and ``p_mode`` in the state's mode
    order. The colour scale is symmetric around zero so correlations are easy
    to distinguish from variances.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(max(5.0, 0.9 * len(state.modes) + 2), 4.8))
    else:
        fig = ax.figure

    labels = [f"{q}_{mode}" for mode in state.modes for q in ("x", "p")]
    covariance = state.covariance
    limit = float(np.max(np.abs(covariance)))
    limit = max(limit, np.finfo(float).eps)

    image = ax.imshow(covariance, cmap="RdBu_r", vmin=-limit, vmax=limit)
    ax.set_xticks(range(len(labels)), labels=labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels=labels)
    ax.set_title("Gaussian covariance matrix")
    ax.set_xlabel("Quadrature")
    ax.set_ylabel("Quadrature")
    fig.colorbar(image, ax=ax, label="Covariance")

    if annotate:
        threshold = 0.45 * limit
        for row in range(covariance.shape[0]):
            for col in range(covariance.shape[1]):
                value = covariance[row, col]
                text_color = "white" if abs(value) > threshold else "black"
                ax.text(col, row, f"{value:.2g}", ha="center", va="center", color=text_color)

    return _finalize(fig, show)


def plot_phase_space(
    state: GaussianState,
    mode_name: str,
    *,
    ax: plt.Axes | None = None,
    n_sigma: float = 2.0,
    show: bool = False,
) -> plt.Figure:
    """Plot one mode's displacement and covariance ellipse in phase space.

    The ellipse encloses the ``n_sigma`` Mahalanobis contour of the Gaussian
    distribution. Principal axes are derived directly from the covariance
    matrix, so squeezing and rotated squeezing are immediately visible.
    """
    if n_sigma <= 0:
        raise ValueError("n_sigma must be positive.")

    idx = state.get_mode_index(mode_name)
    mean = state.displacement[idx : idx + 2]
    covariance = state.covariance[idx : idx + 2, idx : idx + 2]
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    angle = float(np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0])))

    if ax is None:
        fig, ax = plt.subplots(figsize=(5.6, 5.2))
    else:
        fig = ax.figure

    widths = 2.0 * n_sigma * np.sqrt(np.maximum(eigenvalues, 0.0))
    ellipse = Ellipse(
        xy=mean,
        width=float(widths[0]),
        height=float(widths[1]),
        angle=angle,
        fill=False,
        linewidth=2.2,
        label=f"{n_sigma:g}σ contour",
    )
    ax.add_patch(ellipse)
    ax.scatter([mean[0]], [mean[1]], s=55, zorder=3, label="mean")
    ax.axhline(0.0, linewidth=0.8, linestyle="--")
    ax.axvline(0.0, linewidth=0.8, linestyle="--")
    ax.set_xlabel("x")
    ax.set_ylabel("p")
    ax.set_title(f"Phase space — mode '{mode_name}'")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(frameon=False)

    radius = float(np.max(widths) / 2.0)
    radius = max(radius, 0.5)
    ax.set_xlim(float(mean[0] - radius * 1.25), float(mean[0] + radius * 1.25))
    ax.set_ylim(float(mean[1] - radius * 1.25), float(mean[1] + radius * 1.25))
    return _finalize(fig, show)


def plot_wigner(
    state: GaussianState,
    mode_name: str,
    *,
    x_max: float = 4.0,
    num_points: int = 180,
    ax: plt.Axes | None = None,
    show: bool = False,
) -> plt.Figure:
    """Plot the analytically computed single-mode Wigner function."""
    if x_max <= 0:
        raise ValueError("x_max must be positive.")
    if num_points < 2:
        raise ValueError("num_points must be at least 2.")

    idx = state.get_mode_index(mode_name)
    mean = state.displacement[idx : idx + 2]
    covariance = state.covariance[idx : idx + 2, idx : idx + 2]
    x = np.linspace(-x_max, x_max, num_points)
    p = np.linspace(-x_max, x_max, num_points)
    X, P = np.meshgrid(x, p)
    delta = np.stack((X - mean[0], P - mean[1]), axis=-1)
    inverse = np.linalg.inv(covariance)
    exponent = np.einsum("...i,ij,...j->...", delta, inverse, delta)
    wigner = np.exp(-0.5 * exponent) / (2.0 * np.pi * np.sqrt(np.linalg.det(covariance)))

    if ax is None:
        fig, ax = plt.subplots(figsize=(6.0, 5.2))
    else:
        fig = ax.figure

    image = ax.pcolormesh(X, P, wigner, shading="auto", cmap="magma")
    ax.contour(X, P, wigner, levels=6, colors="white", linewidths=0.45, alpha=0.65)
    ax.scatter([mean[0]], [mean[1]], marker="+", s=80, linewidths=1.5, color="white")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("p")
    ax.set_title(f"Wigner function — mode '{mode_name}'")
    fig.colorbar(image, ax=ax, label="W(x, p)")
    return _finalize(fig, show)


def plot_state_dashboard(
    state: GaussianState,
    *,
    mode: str | None = None,
    show: bool = False,
) -> plt.Figure:
    """Create a compact dashboard with covariance and phase-space views."""
    mode_name = state.modes[0] if mode is None else mode
    if mode_name not in state.modes:
        raise ValueError(f"Mode '{mode_name}' is not present in this state.")

    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.6))
    plot_covariance_matrix(state, ax=axes[0], annotate=False)
    plot_phase_space(state, mode_name, ax=axes[1])
    plot_wigner(state, mode_name, ax=axes[2])
    fig.suptitle(f"catsy state dashboard — {', '.join(state.modes)}", fontsize=14)
    return _finalize(fig, show)


__all__ = [
    "plot_covariance_matrix",
    "plot_phase_space",
    "plot_state_dashboard",
    "plot_wigner",
]
