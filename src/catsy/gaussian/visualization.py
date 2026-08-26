"""Publication-friendly visualizations for Gaussian states.

All helpers return Matplotlib figures or animations and never display them unless
``show=True``. The visual language is shared across static plots, dashboards,
and animations so state and dynamics remain easy to compare.
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.artist import Artist
from matplotlib.cm import ScalarMappable
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.patches import Ellipse

from . import GaussianState, compute_joint_correlation


def _finalize(fig: plt.Figure, show: bool) -> plt.Figure:
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
    return state.displacement[idx : idx + 2], state.covariance[
        idx : idx + 2, idx : idx + 2
    ]


def _state_summary(state: GaussianState, mode_name: str | None = None) -> str:
    if mode_name is None:
        return f"modes: {', '.join(state.modes)}"
    mean, _ = _mode_geometry(state, mode_name)
    return f"mode {mode_name}   ·   d = ({mean[0]:.2f}, {mean[1]:.2f})"


def _state_header(
    ax: plt.Axes, state: GaussianState, mode_name: str | None = None
) -> None:
    ax.text(
        0.02,
        0.98,
        _state_summary(state, mode_name),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        alpha=0.72,
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": "white",
            "alpha": 0.82,
            "edgecolor": "none",
        },
    )


def _ellipse_geometry(
    covariance: np.ndarray, n_sigma: float
) -> tuple[float, float, float]:
    values, vectors = np.linalg.eigh(covariance)
    order = np.argsort(values)[::-1]
    values = np.maximum(values[order], 0.0)
    vector = vectors[:, order[0]]
    angle = float(np.degrees(np.arctan2(vector[1], vector[0])))
    widths = 2.0 * n_sigma * np.sqrt(values)
    return float(widths[0]), float(widths[1]), angle


def _ellipse_extents(
    mean: np.ndarray, covariance: np.ndarray, n_sigma: float
) -> tuple[float, float]:
    width, height, angle = _ellipse_geometry(covariance, n_sigma)
    theta = np.radians(angle)
    hx = 0.5 * np.sqrt((width * np.cos(theta)) ** 2 + (height * np.sin(theta)) ** 2)
    hp = 0.5 * np.sqrt((width * np.sin(theta)) ** 2 + (height * np.cos(theta)) ** 2)
    return float(abs(mean[0]) + hx), float(abs(mean[1]) + hp)


def _add_ellipse(
    ax: plt.Axes,
    mean: np.ndarray,
    covariance: np.ndarray,
    n_sigma: float,
    **kwargs: object,
) -> Ellipse:
    width, height, angle = _ellipse_geometry(covariance, n_sigma)
    ellipse = Ellipse(
        (float(mean[0]), float(mean[1])),
        width,
        height,
        angle=angle,
        fill=False,
        linewidth=1.5,
        **kwargs,
    )
    ax.add_patch(ellipse)
    return ellipse


def _set_phase_limits(
    ax: plt.Axes, means: np.ndarray, covariances: Sequence[np.ndarray], n_sigma: float
) -> None:
    x = max(
        _ellipse_extents(m, c, n_sigma)[0]
        for m, c in zip(means, covariances, strict=True)
    )
    p = max(
        _ellipse_extents(m, c, n_sigma)[1]
        for m, c in zip(means, covariances, strict=True)
    )
    extent = max(x, p, 1.0) * 1.18
    ax.set_xlim(-extent, extent)
    ax.set_ylim(-extent, extent)


def _style_phase_axes(ax: plt.Axes) -> None:
    ax.axhline(0, lw=0.6, ls="--", alpha=0.30, zorder=0)
    ax.axvline(0, lw=0.6, ls="--", alpha=0.30, zorder=0)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.10, linewidth=0.5)
    ax.spines[["top", "right"]].set_visible(False)


def _quadrature_labels(state: GaussianState) -> list[str]:
    return [f"{q}$_{{{mode}}}$" for mode in state.modes for q in ("x", "p")]


def _quadrature_correlation(covariance: np.ndarray) -> np.ndarray:
    variances = np.diag(covariance)
    scale = np.sqrt(np.outer(variances, variances))
    with np.errstate(divide="ignore", invalid="ignore"):
        correlation = np.divide(
            covariance, scale, out=np.zeros_like(covariance), where=scale > 0
        )
    np.fill_diagonal(correlation, 1.0)
    return cast(np.ndarray, correlation)


def plot_covariance_matrix(
    state: GaussianState,
    *,
    ax: plt.Axes | None = None,
    annotate: bool = True,
    show: bool = False,
) -> plt.Figure:
    if ax is None:
        fig, ax = plt.subplots(
            figsize=(max(5.0, 0.9 * len(state.modes) + 2), 5.2), constrained_layout=True
        )
    else:
        fig = cast(plt.Figure, ax.figure)
    labels = _quadrature_labels(state)
    covariance = state.covariance
    limit = max(float(np.max(np.abs(covariance))), np.finfo(float).eps)
    image = ax.imshow(covariance, cmap="RdBu_r", vmin=-limit, vmax=limit)
    ax.set_xticks(range(len(labels)), labels=labels)
    ax.set_yticks(range(len(labels)), labels=labels)
    ax.set_title("Covariance matrix", pad=16, fontweight="medium")
    ax.set_xlabel("quadrature")
    ax.set_ylabel("quadrature")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label=r"$V$")
    _state_header(ax, state)
    if annotate:
        threshold = 0.45 * limit
        for row in range(covariance.shape[0]):
            for col in range(covariance.shape[1]):
                value = covariance[row, col]
                ax.text(
                    col,
                    row,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if abs(value) > threshold else "black",
                )
    return _finalize(fig, show)


def plot_mode_correlation_map(
    state: GaussianState,
    *,
    ax: plt.Axes | None = None,
    annotate: bool = True,
    show: bool = False,
) -> plt.Figure:
    """Plot normalized quadrature correlations with explicit mode boundaries."""
    if ax is None:
        fig, ax = plt.subplots(
            figsize=(max(5.0, 0.9 * len(state.modes) + 2), 5.2), constrained_layout=True
        )
    else:
        fig = cast(plt.Figure, ax.figure)

    labels = _quadrature_labels(state)
    correlation = _quadrature_correlation(state.covariance)
    image = ax.imshow(correlation, cmap="RdBu_r", vmin=-1.0, vmax=1.0)
    ax.set_xticks(range(len(labels)), labels=labels)
    ax.set_yticks(range(len(labels)), labels=labels)
    ax.set_xlabel("quadrature")
    ax.set_ylabel("quadrature")
    ax.set_title("Mode correlation map", pad=16, fontweight="medium")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label=r"$C_{ij}$")
    _state_header(ax, state)

    for boundary in range(1, len(state.modes)):
        position = 2 * boundary - 0.5
        ax.axvline(position, color="black", lw=0.9, alpha=0.45)
        ax.axhline(position, color="black", lw=0.9, alpha=0.45)

    if annotate:
        for row in range(correlation.shape[0]):
            for col in range(correlation.shape[1]):
                value = correlation[row, col]
                ax.text(
                    col,
                    row,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if abs(value) > 0.55 else "black",
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
    if n_sigma <= 0:
        raise ValueError("n_sigma must be positive.")
    mean, covariance = _mode_geometry(state, mode_name)
    if ax is None:
        fig, ax = plt.subplots(figsize=(5.8, 5.4), constrained_layout=True)
    else:
        fig = cast(plt.Figure, ax.figure)
    _add_ellipse(ax, mean, covariance, n_sigma, alpha=0.55, label=rf"{n_sigma:g}$\sigma$")
    values, vectors = np.linalg.eigh(covariance)
    order = np.argsort(values)[::-1]
    for value, vector in zip(values[order], vectors[:, order].T, strict=True):
        length = n_sigma * np.sqrt(max(float(value), 0.0))
        ax.plot(
            [mean[0] - vector[0] * length, mean[0] + vector[0] * length],
            [mean[1] - vector[1] * length, mean[1] + vector[1] * length],
            lw=0.9,
            alpha=0.35,
        )
    ax.scatter([mean[0]], [mean[1]], s=55, zorder=4, label="mean")
    ax.set_xlabel(r"$x$ quadrature")
    ax.set_ylabel(r"$p$ quadrature")
    ax.set_title(f"Phase space — mode {mode_name}", pad=16, fontweight="medium")
    _style_phase_axes(ax)
    _set_phase_limits(ax, np.asarray([mean]), [covariance], n_sigma)
    _state_header(ax, state, mode_name)
    ax.legend(frameon=False, loc="lower right")
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
    sequence = _states(states)
    if n_sigma <= 0:
        raise ValueError("n_sigma must be positive.")
    if ellipse_every is not None and ellipse_every < 1:
        raise ValueError("ellipse_every must be positive or None.")
    if times is not None and len(times) != len(sequence):
        raise ValueError("times must have the same length as states.")
    means = np.array([_mode_geometry(s, mode_name)[0] for s in sequence])
    covariances = [_mode_geometry(s, mode_name)[1] for s in sequence]
    if ax is None:
        fig, ax = plt.subplots(figsize=(6.4, 5.8), constrained_layout=True)
    else:
        fig = cast(plt.Figure, ax.figure)
    ax.plot(means[:, 0], means[:, 1], lw=2.0, label="mean trajectory")
    ax.scatter([means[0, 0]], [means[0, 1]], s=42, label="initial", zorder=4)
    ax.scatter([means[-1, 0]], [means[-1, 1]], s=70, marker="*", label="final", zorder=4)
    step = ellipse_every or max(1, len(sequence) // 6)
    indices = list(range(0, len(sequence), step))
    if indices[-1] != len(sequence) - 1:
        indices.append(len(sequence) - 1)
    for i in indices:
        _add_ellipse(ax, means[i], covariances[i], n_sigma, alpha=0.25)
    _style_phase_axes(ax)
    _set_phase_limits(ax, means, covariances, n_sigma)
    ax.set_xlabel(r"$x$ quadrature")
    ax.set_ylabel(r"$p$ quadrature")
    ax.set_title(f"Phase-space evolution — mode {mode_name}", pad=16, fontweight="medium")
    _state_header(ax, sequence[-1], mode_name)
    ax.legend(frameon=False, loc="lower right")
    return _finalize(fig, show)


def animate_phase_space(
    states: Sequence[GaussianState],
    mode_name: str,
    *,
    times: Sequence[float] | None = None,
    n_sigma: float = 2.0,
    interval: int = 80,
    repeat: bool = False,
    ax: plt.Axes | None = None,
    show: bool = False,
) -> FuncAnimation:
    sequence = _states(states)
    if n_sigma <= 0:
        raise ValueError("n_sigma must be positive.")
    if interval <= 0:
        raise ValueError("interval must be positive.")
    if times is not None and len(times) != len(sequence):
        raise ValueError("times must have the same length as states.")
    means = np.array([_mode_geometry(s, mode_name)[0] for s in sequence])
    covariances = [_mode_geometry(s, mode_name)[1] for s in sequence]
    if ax is None:
        fig, ax = plt.subplots(figsize=(7.0, 6.2), constrained_layout=True)
    else:
        fig = cast(plt.Figure, ax.figure)
    ax.plot(
        means[:, 0], means[:, 1], ls="--", lw=0.8, alpha=0.20, label="full trajectory"
    )
    _set_phase_limits(ax, means, covariances, n_sigma)
    _style_phase_axes(ax)
    ax.set_xlabel(r"$x$ quadrature")
    ax.set_ylabel(r"$p$ quadrature")
    ax.set_title(
        f"Gaussian phase-space dynamics — mode {mode_name}", pad=16, fontweight="medium"
    )
    (point,) = ax.plot([], [], marker="o", ls="None", ms=8, zorder=6, label="current")
    (trail,) = ax.plot([], [], lw=2.2, zorder=5, label="elapsed trajectory")
    ellipse = Ellipse((0, 0), 0, 0, fill=False, lw=2.2, zorder=5)
    ax.add_patch(ellipse)
    (major,) = ax.plot([], [], lw=1.2, zorder=5)
    (minor,) = ax.plot([], [], lw=0.9, ls="--", zorder=5)
    state_text = ax.text(0.03, 0.97, "", transform=ax.transAxes, va="top", fontsize=9)
    time_text = ax.text(0.03, 0.90, "", transform=ax.transAxes, va="top", fontsize=11)
    stats_text = ax.text(0.03, 0.035, "", transform=ax.transAxes, va="bottom", fontsize=9)
    ax.legend(frameon=False, loc="upper right")

    def update(frame: int) -> tuple[Artist, ...]:
        state = sequence[frame]
        mean, covariance = _mode_geometry(state, mode_name)
        width, height, angle = _ellipse_geometry(covariance, n_sigma)
        values, vectors = np.linalg.eigh(covariance)
        order = np.argsort(values)[::-1]
        values = np.maximum(values[order], 0.0)
        vectors = vectors[:, order]
        point.set_data([mean[0]], [mean[1]])
        trail.set_data(means[: frame + 1, 0], means[: frame + 1, 1])
        ellipse.center = (float(mean[0]), float(mean[1]))
        ellipse.width, ellipse.height, ellipse.angle = width, height, angle
        for line, value, vector in zip((major, minor), values, vectors.T, strict=True):
            length = n_sigma * np.sqrt(float(value))
            line.set_data(
                [mean[0] - vector[0] * length, mean[0] + vector[0] * length],
                [mean[1] - vector[1] * length, mean[1] + vector[1] * length],
            )
        state_text.set_text(_state_summary(state, mode_name))
        time_text.set_text(
            f"t = {times[frame]:g}"
            if times is not None
            else f"step {frame + 1} / {len(sequence)}"
        )
        stats_text.set_text(
            f"σ₁ {np.sqrt(values[0]):.3g}   σ₂ {np.sqrt(values[1]):.3g}   det(V) {np.linalg.det(covariance):.3g}"
        )
        return point, trail, ellipse, major, minor, state_text, time_text, stats_text

    animation = FuncAnimation(
        fig,
        update,
        frames=len(sequence),
        interval=interval,
        blit=False,
        repeat=repeat,
    )
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
    sequence = _states(states)
    if times is not None and len(times) != len(sequence):
        raise ValueError("times must have the same length as states.")
    x = (
        np.arange(len(sequence), dtype=float)
        if times is None
        else np.asarray(times, dtype=float)
    )
    values = np.array([_mode_geometry(s, mode_name)[1] for s in sequence])
    if ax is None:
        fig, ax = plt.subplots(figsize=(7.0, 4.6), constrained_layout=True)
    else:
        fig = cast(plt.Figure, ax.figure)
    ax.plot(x, values[:, 0, 0], label=r"$V_{xx}$")
    ax.plot(x, values[:, 1, 1], label=r"$V_{pp}$")
    ax.plot(x, values[:, 0, 1], label=r"$V_{xp}$")
    ax.axhline(0, lw=0.6, ls="--", alpha=0.30)
    ax.set_xlabel("time" if times is not None else "step")
    ax.set_ylabel(r"covariance $V$")
    ax.set_title(f"Covariance evolution — mode {mode_name}", pad=16, fontweight="medium")
    _state_header(ax, sequence[-1], mode_name)
    ax.grid(alpha=0.12, linewidth=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.18))
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
    sequence = _states(states)
    if times is not None and len(times) != len(sequence):
        raise ValueError("times must have the same length as states.")
    x = (
        np.arange(len(sequence), dtype=float)
        if times is None
        else np.asarray(times, dtype=float)
    )
    purity = np.array(
        [
            1.0
            / (
                2.0 ** len(s.modes)
                * np.sqrt(max(np.linalg.det(s.covariance), np.finfo(float).tiny))
            )
            for s in sequence
        ]
    )
    minimum_nu = np.array(
        [np.min(_symplectic_eigenvalues(s.covariance)) for s in sequence]
    )
    if ax is None:
        fig, ax = plt.subplots(figsize=(7.0, 4.6), constrained_layout=True)
    else:
        fig = cast(plt.Figure, ax.figure)
    ax.plot(x, purity, label="purity")
    ax.plot(x, minimum_nu, label=r"min. symplectic eigenvalue $\nu$")
    ax.axhline(1.0, lw=0.6, ls="--", alpha=0.30, label="vacuum threshold")
    ax.set_xlabel("time" if times is not None else "step")
    ax.set_ylabel("value")
    ax.set_title("State diagnostics", pad=16, fontweight="medium")
    _state_header(ax, sequence[-1])
    ax.grid(alpha=0.12, linewidth=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="best")
    return _finalize(fig, show)


def _wigner_grid(
    state: GaussianState,
    mode_name: str,
    *,
    x_max: float,
    num_points: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean, covariance = _mode_geometry(state, mode_name)
    x = np.linspace(-x_max, x_max, num_points)
    p = np.linspace(-x_max, x_max, num_points)
    X, P = np.meshgrid(x, p)
    delta = np.stack((X - mean[0], P - mean[1]), axis=-1)
    inverse = np.linalg.inv(covariance)
    exponent = np.einsum("...i,ij,...j->...", delta, inverse, delta)
    W = np.exp(-0.5 * exponent) / (2.0 * np.pi * np.sqrt(np.linalg.det(covariance)))
    return X, P, W


def plot_wigner(
    state: GaussianState,
    mode_name: str,
    *,
    x_max: float = 4.0,
    num_points: int = 180,
    ax: plt.Axes | None = None,
    show: bool = False,
    colorbar: bool = True,
    vmin: float | None = None,
    vmax: float | None = None,
) -> plt.Figure:
    if x_max <= 0:
        raise ValueError("x_max must be positive.")
    if num_points < 2:
        raise ValueError("num_points must be at least 2.")
    X, P, W = _wigner_grid(state, mode_name, x_max=x_max, num_points=num_points)
    if ax is None:
        fig, ax = plt.subplots(figsize=(6.0, 5.2), constrained_layout=True)
    else:
        fig = cast(plt.Figure, ax.figure)
    scale = float(np.max(np.abs(W))) if vmax is None else vmax
    image = ax.pcolormesh(X, P, W, shading="auto", cmap="magma", vmin=vmin, vmax=scale)
    ax.contour(X, P, W, levels=7, colors="white", linewidths=0.45, alpha=0.60)
    mean, _ = _mode_geometry(state, mode_name)
    ax.scatter(
        [mean[0]], [mean[1]], marker="+", s=85, linewidths=1.5, color="white", zorder=4
    )
    _style_phase_axes(ax)
    ax.set_xlabel(r"$x$ quadrature")
    ax.set_ylabel(r"$p$ quadrature")
    ax.set_title(f"Wigner function — mode {mode_name}", pad=16, fontweight="medium")
    _state_header(ax, state, mode_name)
    if colorbar:
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label=r"$W(x,p)$")
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
    sequence = _states(states)
    if times is not None and len(times) != len(sequence):
        raise ValueError("times must have the same length as states.")
    selected = list(range(len(sequence))) if indices is None else list(indices)
    if not selected:
        raise ValueError("indices must contain at least one frame.")
    if any(i < 0 or i >= len(sequence) for i in selected):
        raise ValueError("indices contain an out-of-range frame.")
    grids = [
        _wigner_grid(sequence[i], mode_name, x_max=x_max, num_points=num_points)
        for i in selected
    ]
    vmax = max(float(np.max(np.abs(grid[2]))) for grid in grids)
    fig, axes = plt.subplots(
        1,
        len(selected),
        figsize=(4.6 * len(selected), 4.5),
        squeeze=False,
        constrained_layout=True,
    )
    axes_flat = axes[0]
    image = None
    for position, (index, (X, P, W)) in enumerate(zip(selected, grids, strict=True)):
        ax = axes_flat[position]
        image = ax.pcolormesh(X, P, W, shading="auto", cmap="magma", vmin=0.0, vmax=vmax)
        ax.contour(X, P, W, levels=7, colors="white", linewidths=0.45, alpha=0.60)
        mean, _ = _mode_geometry(sequence[index], mode_name)
        ax.scatter([mean[0]], [mean[1]], marker="+", s=75, linewidths=1.4, color="white")
        _style_phase_axes(ax)
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$p$")
        label = f"t = {times[index]:g}" if times is not None else f"step {index}"
        ax.set_title(label, pad=12, fontweight="medium")
    assert image is not None
    fig.colorbar(image, ax=list(axes_flat), fraction=0.02, pad=0.03, label=r"$W(x,p)$")
    fig.suptitle(f"Wigner evolution — mode {mode_name}", y=1.02, fontsize=14)
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
    sequence = _states(states)
    if n_sigma <= 0:
        raise ValueError("n_sigma must be positive.")
    if times is not None and len(times) != len(sequence):
        raise ValueError("times must have the same length as states.")
    selected = (
        [0, len(sequence) // 2, len(sequence) - 1]
        if wigner_indices is None
        else list(wigner_indices)
    )
    if any(i < 0 or i >= len(sequence) for i in selected):
        raise ValueError("wigner_indices contain an out-of-range frame.")
    fig = plt.figure(figsize=(13.5, 9.5), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=(1.15, 0.85), hspace=0.12, wspace=0.16)
    ax_phase = fig.add_subplot(grid[0, 0])
    ax_cov = fig.add_subplot(grid[0, 1])
    ax_wig = fig.add_subplot(grid[1, 0])
    ax_diag = fig.add_subplot(grid[1, 1])
    plot_phase_space_trajectory(
        sequence, mode_name, times=times, n_sigma=n_sigma, ax=ax_phase
    )
    plot_covariance_evolution(sequence, mode_name, times=times, ax=ax_cov)
    snapshot = selected[-1]
    plot_wigner(sequence[snapshot], mode_name, ax=ax_wig)
    ax_wig.set_title(
        (
            f"Wigner snapshot — t = {times[snapshot]:g}"
            if times is not None
            else f"Wigner snapshot — step {snapshot}"
        ),
        pad=14,
        fontweight="medium",
    )
    plot_diagnostics(sequence, times=times, ax=ax_diag)
    fig.suptitle(
        f"Gaussian-state evolution · mode {mode_name}", fontsize=16, fontweight="medium"
    )
    return _finalize(fig, show)


def plot_joint_correlation(
    state: GaussianState,
    mode_a: str,
    mode_b: str,
    quadrature: str = "x",
    *,
    ax: plt.Axes | None = None,
    show: bool = False,
) -> plt.Figure:
    P, X_a, X_b, _, _ = compute_joint_correlation(
        state, mode_a, mode_b, quadrature=quadrature
    )
    if ax is None:
        fig, ax = plt.subplots(figsize=(6.0, 5.0), constrained_layout=True)
    else:
        fig = cast(plt.Figure, ax.figure)
    image = ax.contourf(X_a, X_b, P, 100, cmap="viridis")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Probability density")
    ax.set_title(
        f"Correlation: quadrature {quadrature}_{mode_a} vs {quadrature}_{mode_b}"
    )
    ax.set_xlabel(f"{quadrature}_{mode_a}")
    ax.set_ylabel(f"{quadrature}_{mode_b}")
    ax.set_aspect("equal", adjustable="box")
    return _finalize(fig, show)


def plot_state_dashboard(
    state: GaussianState,
    *,
    mode: str | None = None,
    show: bool = False,
) -> plt.Figure:
    mode_name = state.modes[0] if mode is None else mode
    if mode_name not in state.modes:
        raise ValueError(f"Mode '{mode_name}' is not present in this state.")

    multimode = len(state.modes) > 1
    fig = plt.figure(
        figsize=(18.5 if multimode else 14.5, 5.2),
        constrained_layout=True,
    )
    columns = 4 if multimode else 3
    grid = fig.add_gridspec(
        1,
        columns,
        width_ratios=(0.95, 0.95, 1.05, 1.05) if multimode else (0.95, 1.05, 1.05),
    )
    axes = [fig.add_subplot(grid[0, i]) for i in range(columns)]
    plot_covariance_matrix(state, ax=axes[0], annotate=False)
    next_axis = 1
    if multimode:
        plot_mode_correlation_map(state, ax=axes[next_axis], annotate=False)
        next_axis += 1
    plot_phase_space(state, mode_name, ax=axes[next_axis])
    plot_wigner(state, mode_name, ax=axes[next_axis + 1])
    fig.suptitle(
        f"Gaussian state · {', '.join(state.modes)}", fontsize=15, fontweight="medium"
    )
    return _finalize(fig, show)


def plot_phase_space_trajectory_timecoded(
    states: Sequence[GaussianState],
    mode_name: str,
    *,
    times: Sequence[float] | None = None,
    ellipse_every: int | None = None,
    n_sigma: float = 2.0,
    ax: plt.Axes | None = None,
    show: bool = False,
) -> plt.Figure:
    """Plot a phase-space trajectory with continuously time-coded motion.

    The mean trajectory is rendered as a line collection whose segment colors
    encode the supplied time coordinate. Sparse covariance ellipses preserve
    uncertainty geometry without overwhelming the trajectory.
    """
    sequence = _states(states)
    if n_sigma <= 0:
        raise ValueError("n_sigma must be positive.")
    if ellipse_every is not None and ellipse_every < 1:
        raise ValueError("ellipse_every must be positive or None.")
    if times is None:
        time_values = np.arange(len(sequence), dtype=float)
        time_label = "step"
    else:
        if len(times) != len(sequence):
            raise ValueError("times must have the same length as states.")
        time_values = np.asarray(times, dtype=float)
        if not np.all(np.isfinite(time_values)):
            raise ValueError("times must contain only finite values.")
        if np.any(np.diff(time_values) < 0):
            raise ValueError("times must be monotonically increasing.")
        time_label = "time"

    means = np.array([_mode_geometry(state, mode_name)[0] for state in sequence])
    covariances = [_mode_geometry(state, mode_name)[1] for state in sequence]

    if ax is None:
        fig, ax = plt.subplots(figsize=(6.8, 6.1), constrained_layout=True)
    else:
        fig = cast(plt.Figure, ax.figure)

    if len(sequence) > 1:
        segments = [(start, end) for start, end in itertools.pairwise(means)]
        norm = Normalize(vmin=float(time_values[0]), vmax=float(time_values[-1]))
        line_collection = LineCollection(
            segments, cmap="viridis", norm=norm, linewidth=2.4
        )
        line_collection.set_array(time_values[:-1])
        ax.add_collection(line_collection)
        sm = ScalarMappable(norm=norm, cmap=line_collection.cmap)
        sm.set_array(time_values)
        fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04, label=time_label)
    else:
        ax.scatter(means[:, 0], means[:, 1], s=48, zorder=4)

    ax.scatter([means[0, 0]], [means[0, 1]], s=44, label="initial", zorder=5)
    ax.scatter([means[-1, 0]], [means[-1, 1]], s=72, marker="*", label="final", zorder=5)

    step = ellipse_every or max(1, len(sequence) // 6)
    indices = list(range(0, len(sequence), step))
    if indices[-1] != len(sequence) - 1:
        indices.append(len(sequence) - 1)
    for index in indices:
        _add_ellipse(ax, means[index], covariances[index], n_sigma, alpha=0.22)

    _style_phase_axes(ax)
    _set_phase_limits(ax, means, covariances, n_sigma)
    ax.set_xlabel(r"$x$ quadrature")
    ax.set_ylabel(r"$p$ quadrature")
    ax.set_title(
        f"Time-coded phase-space evolution — mode {mode_name}",
        pad=16,
        fontweight="medium",
    )
    _state_header(ax, sequence[-1], mode_name)
    ax.legend(frameon=False, loc="lower right")
    return _finalize(fig, show)


def _cross_mode_correlation(state: GaussianState, mode_a: int, mode_b: int) -> float:
    a = 2 * mode_a
    b = 2 * mode_b
    covariance = state.covariance
    block = covariance[a : a + 2, b : b + 2]
    variances_a = np.diag(covariance[a : a + 2, a : a + 2])
    variances_b = np.diag(covariance[b : b + 2, b : b + 2])
    scale = np.sqrt(np.outer(variances_a, variances_b))
    with np.errstate(divide="ignore", invalid="ignore"):
        normalized = np.divide(block, scale, out=np.zeros_like(block), where=scale > 0)
    return float(np.max(np.abs(normalized)))


def plot_multimode_evolution(
    states: Sequence[GaussianState],
    *,
    times: Sequence[float] | None = None,
    n_sigma: float = 2.0,
    show: bool = False,
) -> plt.Figure:
    """Show per-mode phase-space trajectories and evolving cross-mode correlation."""
    sequence = _states(states)
    if len(sequence[0].modes) < 2:
        raise ValueError("multimode evolution requires at least two modes.")
    if n_sigma <= 0:
        raise ValueError("n_sigma must be positive.")
    if times is not None:
        if len(times) != len(sequence):
            raise ValueError("times must have the same length as states.")
        time_values = np.asarray(times, dtype=float)
        if not np.all(np.isfinite(time_values)):
            raise ValueError("times must contain only finite values.")
        if np.any(np.diff(time_values) < 0):
            raise ValueError("times must be monotonically increasing.")
        x = time_values
        xlabel = "time"
    else:
        x = np.arange(len(sequence), dtype=float)
        xlabel = "step"

    mode_count = len(sequence[0].modes)
    fig = plt.figure(figsize=(5.8 * mode_count, 8.0), constrained_layout=True)
    grid = fig.add_gridspec(2, mode_count, height_ratios=(1.15, 0.85))
    axes = [fig.add_subplot(grid[0, i]) for i in range(mode_count)]
    correlation_ax = fig.add_subplot(grid[1, :])

    for mode_index, (ax, mode_name) in enumerate(
        zip(axes, sequence[0].modes, strict=True)
    ):
        means = np.array([_mode_geometry(state, mode_name)[0] for state in sequence])
        covariances = [_mode_geometry(state, mode_name)[1] for state in sequence]
        ax.plot(means[:, 0], means[:, 1], lw=2.0, label="mean trajectory")
        ax.scatter([means[0, 0]], [means[0, 1]], s=42, label="initial", zorder=4)
        ax.scatter(
            [means[-1, 0]], [means[-1, 1]], s=70, marker="*", label="final", zorder=4
        )
        _style_phase_axes(ax)
        stride = max(1, len(sequence) // 6)
        for mean, covariance in zip(means[::stride], covariances[::stride], strict=True):
            _add_ellipse(ax, mean, covariance, n_sigma, alpha=0.22)
        limits = [ax.get_xlim(), ax.get_ylim()]
        extent = max(
            abs(limits[0][0]),
            abs(limits[0][1]),
            abs(limits[1][0]),
            abs(limits[1][1]),
            1.0,
        )
        ax.set_xlim(-extent, extent)
        ax.set_ylim(-extent, extent)
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$p$")
        ax.set_title(f"Mode {mode_name}", fontweight="medium")
        if mode_index == 0:
            ax.legend(frameon=False, loc="best")

    pair_values = np.array(
        [
            max(
                _cross_mode_correlation(state, i, j)
                for i in range(mode_count)
                for j in range(i + 1, mode_count)
            )
            for state in sequence
        ]
    )
    correlation_ax.plot(x, pair_values, lw=2.2, label="strongest cross-mode correlation")
    correlation_ax.set_ylim(0.0, max(1.0, float(pair_values.max()) * 1.15))
    correlation_ax.set_xlabel(xlabel)
    correlation_ax.set_ylabel("max |quadrature correlation|")
    correlation_ax.set_title("Multimode correlation evolution", fontweight="medium")
    correlation_ax.grid(alpha=0.12, linewidth=0.5)
    correlation_ax.spines[["top", "right"]].set_visible(False)
    correlation_ax.legend(frameon=False, loc="best")
    fig.suptitle(
        f"Multimode Gaussian evolution · {', '.join(sequence[0].modes)}",
        fontsize=16,
        fontweight="medium",
    )
    return _finalize(fig, show)


__all__ = [
    "animate_phase_space",
    "plot_covariance_evolution",
    "plot_covariance_matrix",
    "plot_diagnostics",
    "plot_evolution",
    "plot_joint_correlation",
    "plot_mode_correlation_map",
    "plot_multimode_evolution",
    "plot_phase_space",
    "plot_phase_space_trajectory",
    "plot_phase_space_trajectory_timecoded",
    "plot_state_dashboard",
    "plot_wigner",
    "plot_wigner_evolution",
]
