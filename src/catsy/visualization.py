"""Shared plotting primitives used by Catsy's visualization modules.

The Gaussian and Fock visualizers intentionally keep their domain-specific
renderers separate, while common figure lifecycle, color scaling, and
phase-space styling live here. Keeping these details in one place makes mixed
dashboards visually consistent and gives future visualization backends a small
stable surface.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colorbar import Colorbar
from matplotlib.colors import Normalize


def finalize_figure(fig: plt.Figure, show: bool) -> plt.Figure:
    """Optionally display and return a Matplotlib figure."""
    if show:
        plt.show()
    return fig


def figure_and_axes(
    ax: plt.Axes | None,
    *,
    figsize: tuple[float, float],
    projection: str | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Return an existing axes/figure pair or create a new one."""
    if ax is not None:
        return cast(plt.Figure, ax.figure), ax

    if projection is None:
        fig, created_ax = plt.subplots(figsize=figsize, constrained_layout=True)
    else:
        fig = plt.figure(figsize=figsize, constrained_layout=True)
        created_ax = fig.add_subplot(111, projection=projection)
    return fig, created_ax


def annotate_box(
    ax: plt.Axes,
    x: float,
    y: float,
    text: str,
    **kwargs: Any,
) -> None:
    """Add a shared rounded, translucent annotation box to an axes."""
    bbox = {
        "boxstyle": "round,pad=0.32",
        "facecolor": "white",
        "alpha": 0.84,
        "edgecolor": "none",
    }
    custom_bbox = kwargs.pop("bbox", None)
    if isinstance(custom_bbox, dict):
        bbox.update(custom_bbox)
    ax.text(x, y, text, transform=ax.transAxes, bbox=bbox, **kwargs)


def normalize_probabilities(
    probabilities: np.ndarray | Iterable[float],
) -> np.ndarray:
    """Return non-negative probabilities normalized to unit total mass."""
    values = np.asarray(
        list(probabilities)
        if not isinstance(probabilities, np.ndarray)
        else probabilities,
        dtype=float,
    )
    if values.ndim != 1:
        raise ValueError("probabilities must be a one-dimensional sequence.")
    if not np.all(np.isfinite(values)):
        raise ValueError("probabilities must contain only finite values.")
    if np.any(values < 0.0):
        raise ValueError("probabilities must be non-negative.")
    total = float(values.sum())
    if total <= np.finfo(float).eps:
        raise ValueError("probabilities must contain positive total mass.")
    return values / total


def color_norm(
    values: np.ndarray | Iterable[float],
    *,
    norm: Normalize | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    symmetric: bool = False,
) -> Normalize:
    """Build or validate a Matplotlib normalization without rescaling data.

    By default the extrema of ``values`` are used. Passing ``norm`` is useful
    when several panels must share exactly the same color scale. ``symmetric``
    is appropriate for signed quantities such as Wigner functions; it expands
    the range to ``[-max(abs(values)), max(abs(values))]``.

    This helper only controls the display mapping. It never normalizes the
    underlying physical data, so quantities such as thermal-state widths or
    absolute probabilities remain physically meaningful.
    """
    if norm is not None:
        if vmin is not None or vmax is not None:
            raise ValueError("norm cannot be combined with vmin or vmax.")
        return norm

    array = np.asarray(list(values) if not isinstance(values, np.ndarray) else values)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        raise ValueError("values must contain at least one finite value.")

    lower = float(np.min(finite)) if vmin is None else float(vmin)
    upper = float(np.max(finite)) if vmax is None else float(vmax)
    if symmetric:
        limit = max(abs(lower), abs(upper))
        lower, upper = -limit, limit
    if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
        if lower == upper and np.isfinite(lower):
            delta = max(abs(lower) * 1e-12, 1e-12)
            lower -= delta
            upper += delta
        else:
            raise ValueError("color normalization requires finite vmin < vmax.")
    return Normalize(vmin=lower, vmax=upper)


def add_colorbar(
    fig: plt.Figure,
    mappable: ScalarMappable,
    ax: plt.Axes | Iterable[plt.Axes],
    *,
    label: str | None = None,
) -> Colorbar:
    """Add one consistently sized colorbar to one or more axes."""
    axes = list(ax) if not isinstance(ax, plt.Axes) else ax
    colorbar = fig.colorbar(mappable, ax=axes, fraction=0.046, pad=0.04)
    if label is not None:
        colorbar.set_label(label)
    return colorbar


def style_phase_axes(ax: plt.Axes) -> None:
    """Apply the shared visual treatment for x/p phase-space plots."""
    ax.axhline(0, lw=0.6, ls="--", alpha=0.30, zorder=0)
    ax.axvline(0, lw=0.6, ls="--", alpha=0.30, zorder=0)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.10, linewidth=0.5)
    ax.spines[["top", "right"]].set_visible(False)


__all__ = [
    "add_colorbar",
    "annotate_box",
    "color_norm",
    "figure_and_axes",
    "finalize_figure",
    "normalize_probabilities",
    "style_phase_axes",
]
