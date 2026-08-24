"""Compact scientific summaries for Gaussian-state visualization."""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np

from .gaussian import GaussianState


def _symplectic_eigenvalues(covariance: np.ndarray) -> np.ndarray:
    n = covariance.shape[0] // 2
    omega = np.kron(np.eye(n), np.array([[0.0, 1.0], [-1.0, 0.0]]))
    values = np.linalg.eigvals(1j * omega @ covariance).real
    return np.sort(np.abs(values))[::2]


def plot_state_fingerprint(
    state: GaussianState,
    *,
    ax: plt.Axes | None = None,
    show: bool = False,
) -> plt.Figure:
    """Render a compact, human-readable fingerprint of a Gaussian state."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(5.2, 4.8), constrained_layout=True)
    else:
        fig = ax.figure

    covariance = state.covariance
    symplectic = _symplectic_eigenvalues(covariance)
    purity = 1.0 / (
        2.0**len(state.modes)
        * np.sqrt(max(np.linalg.det(covariance), np.finfo(float).tiny))
    )

    rows: list[tuple[str, str]] = [
        ("modes", ", ".join(state.modes)),
        ("dimension", str(covariance.shape[0])),
        ("purity", f"{purity:.4f}"),
        ("det(V)", f"{np.linalg.det(covariance):.4g}"),
        (
            "symplectic spectrum",
            "(" + ", ".join(f"{value:.4f}" for value in symplectic) + ")",
        ),
    ]

    for index, mode in enumerate(state.modes):
        start = 2 * index
        x, p = state.displacement[start : start + 2]
        rows.append((f"d[{mode}]", f"({x:.3f}, {p:.3f})"))

    ax.axis("off")
    table = ax.table(
        cellText=[[label, value] for label, value in rows],
        colLabels=["quantity", "value"],
        cellLoc="left",
        colLoc="left",
        loc="center",
        colWidths=[0.42, 0.58],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.45)
    ax.set_title("State fingerprint", pad=18, fontweight="medium")

    if show:
        plt.show()
    return fig


__all__: Sequence[str] = ("plot_state_fingerprint",)
