"""Publication-friendly visualizations for Gaussian states.

All helpers return Matplotlib figures or animations and never display them unless
``show=True``. The visual language is shared across static plots, dashboards,
and animations so state and dynamics remain easy to compare.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.artist import Artist
from matplotlib.patches import Ellipse

from .gaussian import GaussianState, compute_joint_correlation


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
        idx : idx + 2, idx + 2
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
        alpha=0.75,
        bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "none", "alpha": 0.8},
    )


def _ellipse_geometry(
    covariance: np.ndarray, n_sigma: float
) -> tuple[float, float, float]:
    values, vectors = np.linalg.eigh(covariance)
    order = np.argsort(values)[::-1]
    values = np.maximum(values[order], 0.0)
    vectors = vectors[:, order]
    widths = 2.0 * n_sigma * np.sqrt(values)
    angle = float(np.degrees(np.arctan2(vectors[1, 0], vectors[0, 0])))
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
        correlation = cast(
            np.ndarray,
            np.divide(
                covariance,
                scale,
                out=np.zeros_like(covariance),
                where=scale > 0,
            ),
        )
    np.fill_diagonal(correlation, 1.0)
    return correlation


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
    if annotate:
        for row in range(covariance.shape[0]):
            for col in range(covariance.shape[1]):
                ax.text(col, row, f"{covariance[row, col]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="covariance")
    _state_header(ax, state)
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

    correlation = _quadrature_correlation(state.covariance)
    labels = _quadrature_labels(state)
    image = ax.imshow(correlation, cmap="coolwarm", vmin=-1.0, vmax=1.0)
    ax.set_xticks(range(len(labels)), labels=labels)
    ax.set_yticks(range(len(labels)), labels=labels)
    ax.set_title("Quadrature correlation map", pad=16, fontweight="medium")
    if annotate:
        for row in range(correlation.shape[0]):
            for col in range(correlation.shape[1]):
                ax.text(col, row, f"{correlation[row, col]:+.2f}", ha="center", va="center", fontsize=8)
    for mode_index in range(1, len(state.modes)):
        boundary = 2 * mode_index - 0.5
        ax.axhline(boundary, color="white", lw=1.5, alpha=0.9)
        ax.axvline(boundary, color="white", lw=1.5, alpha=0.9)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="normalized correlation")
    _state_header(ax, state)
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
    if ax is None:
        fig, ax = plt.subplots(figsize=(6.5, 6.0), constrained_layout=True)
    else:
        fig = cast(plt.Figure, ax.figure)
    mean, covariance = _mode_geometry(state, mode_name)
    _style_phase_axes(ax)
    _add_ellipse(ax, mean, covariance, n_sigma, fill=False, lw=2.2)
    ax.scatter([mean[0]], [mean[1]], marker="o", s=48, zorder=3)
    _set_phase_limits(ax, np.array([mean]), [covariance], n_sigma)
    ax.set_xlabel(r"$x$ quadrature")
    ax.set_ylabel(r"$p$ quadrature")
    ax.set_title(f"Phase space — mode {mode_name}", pad=16, fontweight="medium")
    _state_header(ax, state, mode_name)
    return _finalize(fig, show)


def animate_phase_space(
    states: Sequence[GaussianState],
    mode_name: str,
    *,
    times: Sequence[float] | None = None,
    interval: int = 120,
    n_sigma: float = 2.0,
    ax: plt.Axes | None = None,
) -> FuncAnimation:
    sequence = _states(states)
    if n_sigma <= 0:
        raise ValueError("n_sigma must be positive.")
    if interval <= 0:
        raise ValueError("interval must be positive.")
    if times is not None:
        if len(times) != len(sequence):
            raise ValueError("times must have the same length as states.")
        time_values = np.asarray(times, dtype=float)
        if not np.all(np.isfinite(time_values)):
            raise ValueError("times must contain only finite values.")
        if np.any(np.diff(time_values) < 0):
            raise ValueError("times must be monotonically increasing.")
    else:
        time_values = None

    means = np.array([_mode_geometry(state, mode_name)[0] for state in sequence])
    covariances = [_mode_geometry(state, mode_name)[1] for state in sequence]
    if ax is None:
        fig, ax = plt.subplots(figsize=(7.0, 6.2), constrained_layout=True)
    else:
        fig = cast(plt.Figure, ax.figure)
    ax.plot(means[:, 0], means[:, 1], ls="--", lw=0.8, alpha=0.20, label="full trajectory")
    _set_phase_limits(ax, means, covariances, n_sigma)
    _style_phase_axes(ax)
    ax.set_xlabel(r"$x$ quadrature")
    ax.set_ylabel(r"$p$ quadrature")
    ax.set_title(f"Gaussian phase-space dynamics — mode {mode_name}", pad=16, fontweight="medium")
    (point,) = ax.plot([], [], marker="o", ls="None", ms=8, zorder=6, label="current")
    (trail,) = ax.plot([], [], lw=2.2, zorder=5, label="elapsed trajectory")
    ellipse = Ellipse((0, 0), 0, 0, fill=False, lw=2.2, zorder=5)
    ax.add_patch(ellipse)
    state_text = ax.text(0.03, 0.96, "", transform=ax.transAxes, va="top")
    time_text = ax.text(0.97, 0.96, "", transform=ax.transAxes, va="top", ha="right")
    stats_text = ax.text(0.03, 0.03, "", transform=ax.transAxes, va="bottom")

    def update(frame: int) -> tuple[Artist, ...]:
        mean = means[frame]
        covariance = covariances[frame]
        values, _ = np.linalg.eigh(covariance)
        width, height, angle = _ellipse_geometry(covariance, n_sigma)
        point.set_data([mean[0]], [mean[1]])
        trail.set_data(means[: frame + 1, 0], means[: frame + 1, 1])
        ellipse.center = (float(mean[0]), float(mean[1]))
        ellipse.width = width
        ellipse.height = height
        ellipse.angle = angle
        state_text.set_text(_state_summary(sequence[frame], mode_name))
        time_text.set_text(
            f"t = {time_values[frame]:g}"
            if time_values is not None
            else f"step {frame + 1} / {len(sequence)}"
        )
        stats_text.set_text(
            f"σ₁ {np.sqrt(values[0]):.3g}   σ₂ {np.sqrt(values[1]):.3g}   det(V) {np.linalg.det(covariance):.3g}"
        )
        return point, trail, ellipse, state_text, time_text, stats_text

    return FuncAnimation(fig, update, frames=len(sequence), interval=interval, repeat=True, blit=False)


def plot_wigner(
    state: GaussianState,
    mode_name: str,
    *,
    x_max: float = 5.0,
    num_points: int = 180,
    ax: plt.Axes | None = None,
    show: bool = False,
) -> plt.Figure:
    if x_max <= 0:
        raise ValueError("x_max must be positive.")
    if num_points < 20:
        raise ValueError("num_points must be at least 20.")
    mean, covariance = _mode_geometry(state, mode_name)
    x = np.linspace(-x_max, x_max, num_points)
    p = np.linspace(-x_max, x_max, num_points)
    X, P = np.meshgrid(x, p)
    centered = np.stack([X - mean[0], P - mean[1]], axis=-1)
    inverse = np.linalg.inv(covariance)
    determinant = np.linalg.det(covariance)
    exponent = np.einsum("...i,ij,...j->...", centered, inverse, centered)
    W = np.exp(-0.5 * exponent) / (2.0 * np.pi * np.sqrt(determinant))
    if ax is None:
        fig, ax = plt.subplots(figsize=(6.8, 6.1), constrained_layout=True)
    else:
        fig = cast(plt.Figure, ax.figure)
    scale = float(np.max(np.abs(W)))
    image = ax.pcolormesh(X, P, W, shading="auto", cmap="magma", vmin=-scale, vmax=scale)
    ax.contour(X, P, W, levels=7, colors="white", linewidths=0.45, alpha=0.60)
    ax.scatter(
        [mean[0]], [mean[1]], marker="+", s=85, linewidths=1.5, color="white", zorder=4
    )
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"$x$ quadrature")
    ax.set_ylabel(r"$p$ quadrature")
    ax.set_title(f"Wigner function — mode {mode_name}", pad=16, fontweight="medium")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Wigner density")
    _state_header(ax, state, mode_name)
    return _finalize(fig, show)
