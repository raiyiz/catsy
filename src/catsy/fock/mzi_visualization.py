"""Visualization for catsy.optics.MachZehnderInterferometer.

The interferometer itself (state, physical parameters, and the phase scan)
lives in catsy.optics -- this module holds the one thing that belongs in a
"_visualization" module: turning an already-scanned MachZehnderInterferometer
into a figure.
"""

from __future__ import annotations

from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import qutip as qt

from catsy.fock.visualization import plot_wigner
from catsy.optics import MachZehnderInterferometer


def plot_mzi_scan(
    mzi: MachZehnderInterferometer,
    *,
    theta_list: np.ndarray | None = None,
    show_state_panel: bool = True,
    state_title: str = "MZI input state",
    state_xlim: tuple[float, float] = (-6.0, 6.0),
    resolution: int = 120,
    phase: float | None = None,
    axes: tuple[plt.Axes, plt.Axes] | None = None,
    figsize: tuple[float, float] = (13.5, 5.5),
    show: bool = False,
) -> plt.Figure:
    """Render an interferometer's fringes beside its input-state panel.

    Reads ``mzi.results`` (from ``mzi.scan()``) and ``mzi.state`` directly, so
    the two are always shown consistently together; runs ``mzi.scan(theta_list)``
    first if it hasn't been called yet, or if ``theta_list`` is given explicitly.
    Set ``show_state_panel=False`` to get a text scan-summary panel instead of
    the Wigner inset (e.g. for a state whose Wigner function isn't the point).
    """
    if mzi.results is None or theta_list is not None:
        mzi.scan(theta_list)
    results = cast("dict[str, object]", mzi.results)

    theta = np.asarray(results["theta"], dtype=float)
    if resolution <= 0:
        raise ValueError("resolution must be positive.")

    if axes is None:
        _, (scan_ax, state_ax) = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)
    else:
        scan_ax, state_ax = axes
    fig = cast(plt.Figure, scan_ax.figure)

    x_phase = theta / np.pi
    scan_ax.plot(x_phase, results["n1"], label="Output port 1", lw=2)
    scan_ax.plot(x_phase, results["n2"], label="Output port 2", lw=2, ls="--")
    scan_ax.plot(x_phase, results["parity1"], label="Parity, port 1", lw=2.2, alpha=0.85)
    scan_ax.axhline(0.0, lw=0.7, ls=":")
    scan_ax.set_xlabel(r"Phase shift $\theta$ ($\times \pi$)")
    scan_ax.set_ylabel("Observable")
    scan_ax.set_title("Mach–Zehnder interference fringes")
    scan_ax.grid(True, ls="--", alpha=0.25)
    scan_ax.legend(frameon=False)

    if not show_state_panel:
        state_ax.axis("off")
        state_ax.text(
            0.5,
            0.5,
            f"{len(theta)} phase points\n\n"
            f"parity range: {np.ptp(np.asarray(results['parity1'])):.3f}",
            ha="center",
            va="center",
            transform=state_ax.transAxes,
        )
        state_ax.set_title("Scan summary")
    else:
        # Delegate to the one canonical Wigner renderer instead of a second,
        # hand-rolled copy of "compute wigner, symmetric norm, colorbar" --
        # plot_wigner already does exactly that (see its color_norm(...,
        # symmetric=True) call), including the zero-negativity contour.
        # plot_wigner requires a density matrix; MachZehnderInterferometer
        # (correctly) also accepts a ket, so purify it here rather than
        # pushing that conversion onto every caller.
        state_rho = mzi.state if mzi.state.isoper else qt.ket2dm(mzi.state)
        plot_wigner(
            state_rho,
            xlim=state_xlim,
            resolution=resolution,
            ax=state_ax,
        )
        title = state_title
        if phase is not None:
            title += rf" · $\theta={phase / np.pi:.2f}\pi$"
        # plot_wigner's own title reports the physically meaningful bits
        # (mean photon number, parity/g^(2) classification); keep that and
        # prefix it with what this panel actually is.
        state_ax.set_title(f"{title}\n{state_ax.get_title()}")

    fig.suptitle("Mach–Zehnder interference scan", fontsize=15, fontweight="medium")
    if show:
        plt.show()
    return fig


__all__ = ["plot_mzi_scan"]
