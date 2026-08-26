"""Shared plotting primitives used by Catsy's visualization modules.

The Gaussian and Fock visualizers intentionally keep their domain-specific
renderers separate, while common figure lifecycle and phase-space styling live
here. Keeping these details in one place makes mixed dashboards visually
consistent and gives future visualization backends a small stable surface.
"""

from __future__ import annotations

from typing import cast

import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable


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
    **kwargs: object,
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


def add_colorbar(
    fig: plt.Figure,
    mappable: ScalarMappable,
    ax: plt.Axes | list[plt.Axes],
    *,
    label: str | None = None,
):
    """Add a consistently sized colorbar to one or more axes."""
    colorbar = fig.colorbar(mappable, ax=ax, fraction=0.046, pad=0.04)
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
    "figure_and_axes",
    "finalize_figure",
    "style_phase_axes",
]
