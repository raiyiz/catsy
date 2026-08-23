"""Publication-friendly visualizations for Gaussian states.

The plotting helpers are side-effect free by default: they return Matplotlib
figures or animations and only display them when explicitly requested.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Ellipse

from .gaussian import GaussianState


def _finalize(fig: plt.Figure, show: bool) -> plt.Figure:
    fig.tight_layout()
    if show:
        plt.show()
    return fig


def _states(states: Sequence[GaussianState]) -> tuple[GaussianState, ...]:
    result = tuple(states)
    if not result:
        raise ValueError("states must contain at least one GaussianState.")
    modes = result[0].modes
    if any(state.modes != modes for state in result[1:]):
        raise ValueError("all states must have the same mode ordering.")
    return result


def _mode_geometry(state: GaussianState, mode_name: str) -> tuple[np.ndarray, np.ndarray]:
    idx = state.get_mode_index(mode_name)
    return state.displacement[idx : idx + 2], state.covariance[idx : idx + 2, idx : idx + 2]


def _ellipse_geometry(covariance: np.ndarray, n_sigma: float) -> tuple[float, float, float]:
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0.0)
    vector = eigenvectors[:, order[0]]
    angle = float(np.degrees(np.arctan2(vector[1], vector[0])))
    widths = 2.0 * n_sigma * np.sqrt(eigenvalues)
    return float(widths[0]), float(widths[1]), angle


def _add_ellipse(ax: plt.Axes, mean: np.ndarray, covariance: np.ndarray, n_sigma: float, **kwargs: object) -> Ellipse:
    width, height, angle = _ellipse_geometry(covariance, n_sigma)
    ellipse = Ellipse(
        xy=(float(mean[0]), float(mean[1])),
        width=width,
        height=height,
        angle=angle,
        fill=False,
        linewidth=1.8,
        **kwargs,
    )
    ax.add_patch(ellipse)
    return ellipse


def plot_covariance_matrix(
    state: GaussianState,
    *,
    ax: plt.Axes | None = None,
    annotate: bool = True,
    show: bool = False,
) -> plt.Figure:
    """Plot a labelled covariance matrix as a symmetric heatmap."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(max(5.0, 0.9 * len(state.modes) + 2), 4.8))
    else:
        fig = cast(plt.Figure, ax.figure)
    labels = [f"{q}_{mode}" for mode in state.modes for q in ("x", "p")]
    covariance = state.covariance
    limit = max(float(np.max(np.abs(covariance))), np.finfo(float).eps)
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
                ax.text(col, row, f"{value:.2g}", ha="center", va="center", color="white" if abs(value) > threshold else "black")
    return _finalize(fig, show)


def plot_phase_space(
    state: GaussianState,
    mode_name: str,
    *,
    ax: plt.Axes | None = None,
    n_sigma: float = 2.0,
    show: bool = False,
) -> plt.Figure:
    """Plot displacement, principal covariance axes, and uncertainty ellipse."""
    if n_sigma <= 0:
        raise ValueError("n_sigma must be positive.")
    mean, covariance = _mode_geometry(state, mode_name)
    if ax is None:
        fig, ax = plt.subplots(figsize=(5.6, 5.2))
    else:
        fig = cast(plt.Figure, ax.figure)
    _add_ellipse(ax, mean, covariance, n_sigma, label=f"{n_sigma:g}σ contour")
    ax.scatter([mean[0]], [mean[1]], s=55, zorder=3, label="mean")
    ax.axhline(0.0, linewidth=0.8, linestyle="--")
    ax.axvline(0.0, linewidth=0.8, linestyle="--")
    ax.set_xlabel("x")
    ax.set_ylabel("p")
    ax.set_title(f"Phase space — mode '{mode_name}'")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(frameon=False)
    width, height, _ = _ellipse_geometry(covariance, n_sigma)
    radius = max(width, height, 1.0) / 2.0
    ax.set_xlim(float(mean[0] - radius * 1.25), float(mean[0] + radius * 1.25))
    ax.set_ylim(float(mean[1] - radius * 1.25), float(mean[1] + radius * 1.25))
    return _finalize(fig, show)


def plot_phase_space_trajectory(
    states: Sequence[GaussianState],
    mode_name: str,
    *,
    times: Sequence[float] | None = None,
    ellipse_every: int | None = None,
    n_sigma: float = 2.0,
    ax: plt.Axes | None = None,
    show: bool = False,
) -> plt.Figure:
    """Plot a Gaussian mode's displacement trajectory and evolving ellipses."""
    sequence = _states(states)
    if n_sigma <= 0:
        raise ValueError("n_sigma must be positive.")
    if ellipse_every is not None and ellipse_every < 1:
        raise ValueError("ellipse_every must be positive or None.")
    if times is not None and len(times) != len(sequence):
        raise ValueError("times must have the same length as states.")
    means = np.array([_mode_geometry(state, mode_name)[0] for state in sequence])
    if ax is None:
        fig, ax = plt.subplots(figsize=(6.2, 5.6))
    else:
        fig = cast(plt.Figure, ax.figure)
    ax.plot(means[:, 0], means[:, 1], linewidth=2.0, label="trajectory")
    ax.scatter([means[0, 0]], [means[0, 1]], s=45, label="initial", zorder=3)
    ax.scatter([means[-1, 0]], [means[-1, 1]], s=55, marker="*", label="final", zorder=3)
    indices = range(0, len(sequence), ellipse_every or max(1, len(sequence) // 6))
    if len(sequence) > 1 and (len(sequence) - 1) not in indices:
        indices = list(indices) + [len(sequence) - 1]
    for index in indices:
        mean, covariance = _mode_geometry(sequence[index], mode_name)
        _add_ellipse(ax, mean, covariance, n_sigma, alpha=0.45)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("p")
    ax.set_title(f"Phase-space evolution — mode '{mode_name}'")
    ax.legend(frameon=False)
    if times is not None:
        ax.text(0.02, 0.98, f"t = {times[-1]:g}", transform=ax.transAxes, va="top")
    return _finalize(fig, show)


def animate_phase_space(
    states: Sequence[GaussianState],
    mode_name: str,
    *,
    times: Sequence[float] | None = None,
    n_sigma: float = 2.0,
    interval: int = 80,
    ax: plt.Axes | None = None,
    show: bool = False,
) -> FuncAnimation:
    """Animate a Gaussian mode's phase-space mean and covariance ellipse."""
    sequence = _states(states)
    if n_sigma <= 0:
        raise ValueError("n_sigma must be positive.")
    if interval <= 0:
        raise ValueError("interval must be positive.")
    if times is not None and len(times) != len(sequence):
        raise ValueError("times must have the same length as states.")
    means = np.array([_mode_geometry(state, mode_name)[0] for state in sequence])
    if ax is None:
        fig, ax = plt.subplots(figsize=(6.2, 5.6))
    else:
        fig = cast(plt.Figure, ax.figure)
    ax.plot(means[:, 0], means[:, 1], linestyle="--", linewidth=0.8, alpha=0.35)
    point, = ax.plot([], [], marker="o", linestyle="None", markersize=7)
    trail, = ax.plot([], [], linewidth=2.0)
    ellipse = Ellipse((0.0, 0.0), 0.0, 0.0, fill=False, linewidth=2.0)
    ax.add_patch(ellipse)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("p")
    ax.set_title(f"Phase-space evolution — mode '{mode_name}'")
    span = max(float(np.ptp(means[:, 0])), float(np.ptp(means[:, 1])), 1.0)
    center = means.mean(axis=0)
    ax.set_xlim(float(center[0] - 0.65 * span), float(center[0] + 0.65 * span))
    ax.set_ylim(float(center[1] - 0.65 * span), float(center[1] + 0.65 * span))
    time_text = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top")

    def update(frame: int) -> tuple[object, ...]:
        mean, covariance = _mode_geometry(sequence[frame], mode_name)
        width, height, angle = _ellipse_geometry(covariance, n_sigma)
        point.set_data([mean[0]], [mean[1]])
        trail.set_data(means[: frame + 1, 0], means[: frame + 1, 1])
        ellipse.center = (float(mean[0]), float(mean[1]))
        ellipse.width = width
        ellipse.height = height
        ellipse.angle = angle
        time_text.set_text("" if times is None else f"t = {times[frame]:g}")
        return point, trail, ellipse, time_text

    animation = FuncAnimation(fig, update, frames=len(sequence), interval=interval, blit=False)
    if show:
        plt.show()
    return animation


def plot_covariance_evolution(
    states: Sequence[GaussianState],
    mode_name: str,
    *,
    times: Sequence[float] | None = None,
    ax: plt.Axes | None = None,
    show: bool = False,
) -> plt.Figure:
    """Plot the three independent entries of a single-mode covariance over time."""
    sequence = _states(states)
    if times is not None and len(times) != len(sequence):
        raise ValueError("times must have the same length as states.")
    x = np.arange(len(sequence), dtype=float) if times is None else np.asarray(times, dtype=float)
    values = np.array([_mode_geometry(state, mode_name)[1] for state in sequence])
    if ax is None:
        fig, ax = plt.subplots(figsize=(7.0, 4.6))
    else:
        fig = cast(plt.Figure, ax.figure)
    ax.plot(x, values[:, 0, 0], label=r"$V_{xx}$")
    ax.plot(x, values[:, 1, 1], label=r"$V_{pp}$")
    ax.plot(x, values[:, 0, 1], label=r"$V_{xp}$")
    ax.set_xlabel("time" if times is not None else "step")
    ax.set_ylabel("covariance")
    ax.set_title(f"Covariance evolution — mode '{mode_name}'")
    ax.legend(frameon=False)
    return _finalize(fig, show)


def _symplectic_eigenvalues(covariance: np.ndarray) -> np.ndarray:
    n = covariance.shape[0] // 2
    omega = np.kron(np.eye(n), np.array([[0.0, 1.0], [-1.0, 0.0]]))
    return np.sort(np.abs(np.linalg.eigvals(1j * omega @ covariance).real))[::2]


def plot_diagnostics(
    states: Sequence[GaussianState],
    *,
    times: Sequence[float] | None = None,
    ax: plt.Axes | None = None,
    show: bool = False,
) -> plt.Figure:
    """Plot purity and the minimum symplectic eigenvalue over an evolution."""
    sequence = _states(states)
    if times is not None and len(times) != len(sequence):
        raise ValueError("times must have the same length as states.")
    x = np.arange(len(sequence), dtype=float) if times is None else np.asarray(times, dtype=float)
    purity = np.array([1.0 / (2.0 ** len(s.modes) * np.sqrt(max(np.linalg.det(s.covariance), 0.0))) for s in sequence])
    minimum_nu = np.array([np.min(_symplectic_eigenvalues(s.covariance)) for s in sequence])
    if ax is None:
        fig, ax = plt.subplots(figsize=(7.0, 4.6))
    else:
        fig = cast(plt.Figure, ax.figure)
    ax.plot(x, purity, label="purity")
    ax.plot(x, minimum_nu, label=r"min. symplectic eigenvalue $\nu$")
    ax.set_xlabel("time" if times is not None else "step")
    ax.set_title("Gaussian-state diagnostics")
    ax.legend(frameon=False)
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
    mean, covariance = _mode_geometry(state, mode_name)
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
        fig = cast(plt.Figure, ax.figure)
    image = ax.pcolormesh(X, P, wigner, shading="auto", cmap="magma")
    ax.contour(X, P, wigner, levels=6, colors="white", linewidths=0.45, alpha=0.65)
    ax.scatter([mean[0]], [mean[1]], marker="+", s=80, linewidths=1.5, color="white")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("p")
    ax.set_title(f"Wigner function — mode '{mode_name}'")
    fig.colorbar(image, ax=ax, label="W(x, p)")
    return _finalize(fig, show)


def plot_wigner_evolution(
    states: Sequence[GaussianState],
    mode_name: str,
    *,
    times: Sequence[float] | None = None,
    indices: Sequence[int] | None = None,
    x_max: float = 4.0,
    num_points: int = 120,
    show: bool = False,
) -> plt.Figure:
    """Plot selected Wigner-function snapshots from an evolution."""
    sequence = _states(states)
    if times is not None and len(times) != len(sequence):
        raise ValueError("times must have the same length as states.")
    selected = list(range(len(sequence))) if indices is None else list(indices)
    if not selected:
        raise ValueError("indices must contain at least one frame.")
    if any(index < 0 or index >= len(sequence) for index in selected):
        raise ValueError("indices contain an out-of-range frame.")
    fig, axes = plt.subplots(1, len(selected), figsize=(5.0 * len(selected), 4.6), squeeze=False)
    for position, index in enumerate(selected):
        plot_wigner(sequence[index], mode_name, x_max=x_max, num_points=num_points, ax=axes[0, position])
        axes[0, position].set_title("t = %g" % times[index] if times is not None else f"step {index}")
    return _finalize(fig, show)


def plot_evolution(
    states: Sequence[GaussianState],
    mode_name: str,
    *,
    times: Sequence[float] | None = None,
    wigner_indices: Sequence[int] | None = None,
    n_sigma: float = 2.0,
    show: bool = False,
) -> plt.Figure:
    """Create a four-panel dashboard for Gaussian-state evolution."""
    sequence = _states(states)
    if n_sigma <= 0:
        raise ValueError("n_sigma must be positive.")
    if times is not None and len(times) != len(sequence):
        raise ValueError("times must have the same length as states.")
    selected = list(range(len(sequence))) if wigner_indices is None else list(wigner_indices)
    if not selected:
        raise ValueError("wigner_indices must contain at least one frame.")
    fig, axes = plt.subplots(2, 2, figsize=(13.0, 9.0))
    plot_phase_space_trajectory(sequence, mode_name, times=times, n_sigma=n_sigma, ax=axes[0, 0])
    plot_covariance_evolution(sequence, mode_name, times=times, ax=axes[0, 1])
    plot_diagnostics(sequence, times=times, ax=axes[1, 1])
    snapshot = selected[-1]
    plot_wigner(sequence[snapshot], mode_name, ax=axes[1, 0])
    axes[1, 0].set_title("Wigner function — final snapshot")
    fig.suptitle(f"catsy Gaussian evolution — mode '{mode_name}'", fontsize=14)
    return _finalize(fig, show)


def plot_state_dashboard(
    state: GaussianState,
    *,
    mode: str | None = None,
    show: bool = False,
) -> plt.Figure:
    """Create a compact dashboard with covariance, phase-space, and Wigner views."""
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
    "animate_phase_space",
    "plot_covariance_evolution",
    "plot_covariance_matrix",
    "plot_diagnostics",
    "plot_evolution",
    "plot_phase_space",
    "plot_phase_space_trajectory",
    "plot_state_dashboard",
    "plot_wigner",
    "plot_wigner_evolution",
]
