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


def _ellipse_extents(mean: np.ndarray, covariance: np.ndarray, n_sigma: float) -> tuple[float, float]:
    """Return conservative x/p extents of an uncertainty ellipse."""
    width, height, angle = _ellipse_geometry(covariance, n_sigma)
    theta = np.radians(angle)
    half_x = 0.5 * np.sqrt((width * np.cos(theta)) ** 2 + (height * np.sin(theta)) ** 2)
    half_p = 0.5 * np.sqrt((width * np.sin(theta)) ** 2 + (height * np.cos(theta)) ** 2)
    return float(abs(mean[0]) + half_x), float(abs(mean[1]) + half_p)


def _add_ellipse(
    ax: plt.Axes,
    mean: np.ndarray,
    covariance: np.ndarray,
    n_sigma: float,
    **kwargs: object,
) -> Ellipse:
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


def _set_phase_space_limits(
    ax: plt.Axes,
    means: np.ndarray,
    covariances: Sequence[np.ndarray],
    n_sigma: float,
    *,
    padding: float = 0.18,
) -> None:
    x_extent = max(_ellipse_extents(mean, covariance, n_sigma)[0] for mean, covariance in zip(means, covariances))
    p_extent = max(_ellipse_extents(mean, covariance, n_sigma)[1] for mean, covariance in zip(means, covariances))
    x_extent = max(x_extent, 1.0)
    p_extent = max(p_extent, 1.0)
    extent = max(x_extent, p_extent)
    extent *= 1.0 + padding
    ax.set_xlim(-extent, extent)
    ax.set_ylim(-extent, extent)


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
                ax.text(
                    col,
                    row,
                    f"{value:.2g}",
                    ha="center",
                    va="center",
                    color="white" if abs(value) > threshold else "black",
                )
    return _finalize(fig, show)


def plot_phase_space(
    state: GaussianState,
    mode_name: str,
    *,
    ax: plt.Axes | None = None,
    n_sigma: float = 2.0,
    show: bool = False,
) -> plt.Figure:
    """Plot displacement, covariance ellipse, and principal axes."""
    if n_sigma <= 0:
        raise ValueError("n_sigma must be positive.")
    mean, covariance = _mode_geometry(state, mode_name)
    if ax is None:
        fig, ax = plt.subplots(figsize=(5.6, 5.2))
    else:
        fig = cast(plt.Figure, ax.figure)
    _add_ellipse(ax, mean, covariance, n_sigma, label=f"{n_sigma:g}σ contour")
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    for eigenvalue, vector in zip(eigenvalues[order], eigenvectors[:, order].T):
        length = n_sigma * np.sqrt(max(float(eigenvalue), 0.0))
        ax.plot(
            [mean[0] - vector[0] * length, mean[0] + vector[0] * length],
            [mean[1] - vector[1] * length, mean[1] + vector[1] * length],
            linewidth=1.0,
            alpha=0.55,
        )
    ax.scatter([mean[0]], [mean[1]], s=55, zorder=3, label="mean")
    ax.axhline(0.0, linewidth=0.8, linestyle="--")
    ax.axvline(0.0, linewidth=0.8, linestyle="--")
    ax.set_xlabel("x")
    ax.set_ylabel("p")
    ax.set_title(f"Phase space — mode '{mode_name}'")
    ax.set_aspect("equal", adjustable="box")
    _set_phase_space_limits(ax, np.asarray([mean]), [covariance], n_sigma)
    ax.legend(frameon=False)
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
    """Plot a displacement trajectory with evolving covariance geometry."""
    sequence = _states(states)
    if n_sigma <= 0:
        raise ValueError("n_sigma must be positive.")
    if ellipse_every is not None and ellipse_every < 1:
        raise ValueError("ellipse_every must be positive or None.")
    if times is not None and len(times) != len(sequence):
        raise ValueError("times must have the same length as states.")
    means = np.array([_mode_geometry(state, mode_name)[0] for state in sequence])
    covariances = [_mode_geometry(state, mode_name)[1] for state in sequence]
    if ax is None:
        fig, ax = plt.subplots(figsize=(6.2, 5.6))
    else:
        fig = cast(plt.Figure, ax.figure)
    ax.plot(means[:, 0], means[:, 1], linewidth=2.0, label="trajectory")
    ax.scatter([means[0, 0]], [means[0, 1]], s=45, label="initial", zorder=3)
    ax.scatter([means[-1, 0]], [means[-1, 1]], s=55, marker="*", label="final", zorder=3)
    step = ellipse_every or max(1, len(sequence) // 6)
    indices = list(range(0, len(sequence), step))
    if indices[-1] != len(sequence) - 1:
        indices.append(len(sequence) - 1)
    for index in indices:
        _add_ellipse(ax, means[index], covariances[index], n_sigma, alpha=0.35)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("p")
    ax.set_title(f"Phase-space evolution — mode '{mode_name}'")
    _set_phase_space_limits(ax, means, covariances, n_sigma)
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
    """Animate Gaussian phase-space dynamics with geometry and diagnostics.

    The frame is deliberately fixed for the whole animation.  Each frame shows
    the current mean, accumulated trajectory, uncertainty ellipse, principal
    covariance axes, and instantaneous covariance eigenvalues.  The complete
    trajectory is also drawn faintly in the background so the current state is
    easy to place in the global dynamics.
    """
    sequence = _states(states)
    if n_sigma <= 0:
        raise ValueError("n_sigma must be positive.")
    if interval <= 0:
        raise ValueError("interval must be positive.")
    if times is not None and len(times) != len(sequence):
        raise ValueError("times must have the same length as states.")

    means = np.array([_mode_geometry(state, mode_name)[0] for state in sequence])
    covariances = [_mode_geometry(state, mode_name)[1] for state in sequence]
    if ax is None:
        fig, ax = plt.subplots(figsize=(7.0, 6.2))
    else:
        fig = cast(plt.Figure, ax.figure)

    # Static context: complete path, initial state, and fixed physical frame.
    ax.plot(means[:, 0], means[:, 1], linestyle="--", linewidth=0.9, alpha=0.3, label="full trajectory")
    ax.scatter([means[0, 0]], [means[0, 1]], marker="o", s=45, zorder=4, label="initial")
    _set_phase_space_limits(ax, means, covariances, n_sigma)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("p")
    ax.set_title(f"Gaussian phase-space dynamics — mode '{mode_name}'")

    point, = ax.plot([], [], marker="o", linestyle="None", markersize=8, zorder=6, label="current")
    trail, = ax.plot([], [], linewidth=2.2, zorder=5, label="elapsed trajectory")
    ellipse = Ellipse((0.0, 0.0), 0.0, 0.0, fill=False, linewidth=2.2, zorder=5)
    ax.add_patch(ellipse)
    major_axis, = ax.plot([], [], linewidth=1.3, zorder=5)
    minor_axis, = ax.plot([], [], linewidth=1.0, linestyle="--", zorder=5)
    time_text = ax.text(0.03, 0.97, "", transform=ax.transAxes, va="top", fontsize=11)
    stats_text = ax.text(0.03, 0.03, "", transform=ax.transAxes, va="bottom", fontsize=9)
    ax.legend(frameon=False, loc="upper right")

    def update(frame: int) -> tuple[object, ...]:
        mean, covariance = _mode_geometry(sequence[frame], mode_name)
        width, height, angle = _ellipse_geometry(covariance, n_sigma)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = np.maximum(eigenvalues[order], 0.0)
        vectors = eigenvectors[:, order]

        point.set_data([mean[0]], [mean[1]])
        trail.set_data(means[: frame + 1, 0], means[: frame + 1, 1])
        ellipse.center = (float(mean[0]), float(mean[1]))
        ellipse.width = width
        ellipse.height = height
        ellipse.angle = angle

        axes = []
        for index, line in enumerate((major_axis, minor_axis)):
            vector = vectors[:, index]
            length = n_sigma * np.sqrt(float(eigenvalues[index]))
            line.set_data(
                [mean[0] - vector[0] * length, mean[0] + vector[0] * length],
                [mean[1] - vector[1] * length, mean[1] + vector[1] * length],
            )
            axes.append(line)

        if times is None:
            time_text.set_text(f"step {frame + 1}/{len(sequence)}")
        else:
            time_text.set_text(f"t = {times[frame]:g}")
        stats_text.set_text(
            f"σ₁ = {np.sqrt(eigenvalues[0]):.3g}    "
            f"σ₂ = {np.sqrt(eigenvalues[1]):.3g}    "
            f"det(V) = {np.linalg.det(covariance):.3g}"
        )
        return point, trail, ellipse, major_axis, minor_axis, time_text, stats_text

    animation = FuncAnimation(
        fig,
        update,
        frames=len(sequence),
        interval=interval,
        blit=False,
        repeat=False,
    )
    # Initialize explicitly so callers inspecting the returned artists get a
    # meaningful first frame even before a GUI event loop renders it.
    update(0)
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
    values = np.linalg.eigvals(1j * omega @ covariance).real
    return np.sort(np.abs(values))[::2]


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
    purity = np.array(
        [1.0 / (2.0 ** len(s.modes) * np.sqrt(max(np.linalg.det(s.covariance), np.finfo(float).tiny))) for s in sequence]
    )
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
    if any(index < 0 or index >= len(sequence) for index in selected):
        raise ValueError("wigner_indices contain an out-of-range frame.")
    fig, axes = plt.subplots(2, 2, figsize=(13.0, 9.0))
    plot_phase_space_trajectory(sequence, mode_name, times=times, n_sigma=n_sigma, ax=axes[0, 0])
    plot_covariance_evolution(sequence, mode_name, times=times, ax=axes[0, 1])
    plot_diagnostics(sequence, times=times, ax=axes[1, 1])
    snapshot = selected[-1]
    plot_wigner(sequence[snapshot], mode_name, ax=axes[1, 0])
    axes[1, 0].set_title("Wigner function — selected snapshot")
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
