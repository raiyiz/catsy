"""Multimode evolution visualizations for Gaussian states."""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import matplotlib.pyplot as plt
import numpy as np

from .gaussian import GaussianState
from .visualization import _mode_geometry, _states, _style_phase_axes


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
    """Show per-mode phase-space trajectories and evolving cross-mode correlation.

    The top row contains one phase-space trajectory per mode. The bottom panel
    tracks the strongest normalized quadrature correlation between any pair of
    modes, making correlation build-up or decay visible alongside local motion.
    """
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

    for mode_index, (ax, mode_name) in enumerate(zip(axes, sequence[0].modes, strict=True)):
        means = np.array([_mode_geometry(state, mode_name)[0] for state in sequence])
        covariances = [_mode_geometry(state, mode_name)[1] for state in sequence]
        ax.plot(means[:, 0], means[:, 1], lw=2.0, label="mean trajectory")
        ax.scatter([means[0, 0]], [means[0, 1]], s=42, label="initial", zorder=4)
        ax.scatter([means[-1, 0]], [means[-1, 1]], s=70, marker="*", label="final", zorder=4)
        _style_phase_axes(ax)
        for mean, covariance in zip(means[:: max(1, len(sequence) // 6)], covariances[:: max(1, len(sequence) // 6)], strict=True):
            values, _ = np.linalg.eigh(covariance)
            radius = n_sigma * np.sqrt(np.maximum(values, 0.0))
            ax.add_patch(
                plt.matplotlib.patches.Ellipse(
                    (float(mean[0]), float(mean[1])),
                    2.0 * float(radius.max()),
                    2.0 * float(radius.min()),
                    fill=False,
                    alpha=0.22,
                )
            )
        limits = [ax.get_xlim(), ax.get_ylim()]
        extent = max(abs(limits[0][0]), abs(limits[0][1]), abs(limits[1][0]), abs(limits[1][1]), 1.0)
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
    fig.suptitle(f"Multimode Gaussian evolution · {', '.join(sequence[0].modes)}", fontsize=16, fontweight="medium")
    if show:
        plt.show()
    return cast(plt.Figure, fig)


__all__ = ["plot_multimode_evolution"]
