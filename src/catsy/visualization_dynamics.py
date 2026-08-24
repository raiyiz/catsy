"""Higher-level visualizations for Gaussian-state dynamics."""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.figure import Figure

from .gaussian import GaussianState
from .visualization import (
    _add_ellipse,
    _mode_geometry,
    _set_phase_limits,
    _state_header,
    _style_phase_axes,
    _states,
)


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
        figure, ax = plt.subplots(figsize=(6.8, 6.1), constrained_layout=True)
        fig = cast(Figure, figure)
    else:
        fig = ax.figure

    if len(sequence) > 1:
        segments = [(start, end) for start, end in zip(means[:-1], means[1:], strict=True)]
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

    if show:
        plt.show()
    return fig


__all__ = ["plot_phase_space_trajectory_timecoded"]
