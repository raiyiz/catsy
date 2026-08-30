"""Reusable Mach-Zehnder cat-state scan helpers and visualizations."""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import qutip as qt

from catsy.optics import MachZehnderInterferometer, ObservableScanData
from catsy.visualization import add_colorbar


def make_even_cat(*, cutoff: int, alpha: complex) -> qt.Qobj:
    """Return a normalized even cat ket in a truncated Fock basis."""
    if cutoff <= 0:
        raise ValueError("cutoff must be positive.")
    return (qt.coherent(cutoff, alpha) + qt.coherent(cutoff, -alpha)).unit()


def run_mzi_phase_scan(
    state: qt.Qobj,
    *,
    cutoff: int,
    theta_list: Sequence[float] | None = None,
    kappa: float = 0.0,
    loss_time: float = 1.0,
) -> ObservableScanData:
    """Run a reusable Mach-Zehnder phase scan for an arbitrary Fock state."""
    phases = (
        np.linspace(0.0, 2.0 * np.pi, 200)
        if theta_list is None
        else np.asarray(theta_list, dtype=float)
    )
    return MachZehnderInterferometer(
        kappa=kappa,
        N_cutoff=cutoff,
        loss_time=loss_time,
    ).scan(state, phases)


def run_cat_mzi_phase_scan(
    *,
    cutoff: int = 22,
    alpha: complex = 4.0 + 2j,
    theta_list: Sequence[float] | None = None,
    kappa: float = 0.0,
    loss_time: float = 1.0,
) -> tuple[qt.Qobj, ObservableScanData]:
    """Prepare an even cat and run it through a Mach-Zehnder phase scan."""
    cat = make_even_cat(cutoff=cutoff, alpha=alpha)
    return cat, run_mzi_phase_scan(
        cat,
        cutoff=cutoff,
        theta_list=theta_list,
        kappa=kappa,
        loss_time=loss_time,
    )


def plot_mzi_scan(
    results: ObservableScanData,
    *,
    state: qt.Qobj | None = None,
    state_title: str = "MZI input state",
    state_xlim: tuple[float, float] = (-6.0, 6.0),
    resolution: int = 120,
    phase: float | None = None,
    axes: tuple[plt.Axes, plt.Axes] | None = None,
    figsize: tuple[float, float] = (13.5, 5.5),
    show: bool = False,
) -> plt.Figure:
    """Render MZI interference fringes beside an optional Fock-state panel."""
    theta = np.asarray(results["theta"], dtype=float)
    if theta.ndim != 1 or len(theta) == 0:
        raise ValueError("results['theta'] must be a non-empty 1D array.")
    for key in ("n1", "n2", "parity1"):
        if len(results[key]) != len(theta):
            raise ValueError(f"results['{key}'] must have the same length as theta.")
    if resolution <= 0:
        raise ValueError("resolution must be positive.")

    if axes is None:
        _, (scan_ax, state_ax) = plt.subplots(
            1, 2, figsize=figsize, constrained_layout=True
        )
    else:
        scan_ax, state_ax = axes
    fig = cast(plt.Figure, scan_ax.figure)

    x_phase = theta / np.pi
    scan_ax.plot(x_phase, results["n1"], label="Output port 1", lw=2)
    scan_ax.plot(x_phase, results["n2"], label="Output port 2", lw=2, ls="--")
    scan_ax.plot(
        x_phase,
        results["parity1"],
        label="Parity, port 1",
        lw=2.2,
        alpha=0.85,
    )
    scan_ax.axhline(0.0, lw=0.7, ls=":")
    scan_ax.set_xlabel(r"Phase shift $\theta$ ($\times \pi$)")
    scan_ax.set_ylabel("Observable")
    scan_ax.set_title("Mach–Zehnder interference fringes")
    scan_ax.grid(True, ls="--", alpha=0.25)
    scan_ax.legend(frameon=False)

    if state is None:
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
        xvec = np.linspace(state_xlim[0], state_xlim[1], resolution)
        wigner = qt.wigner(state, xvec, xvec)
        # Symmetric-about-zero normalization, matching
        # catsy.fock.visualization.plot_wigner: with RdBu_r, an
        # unnormalized (data-driven) range can leave zero off-center,
        # which visually understates Wigner negativity -- the reason this
        # inset is here in the first place.
        wlim = float(np.max(np.abs(wigner)))
        image = state_ax.contourf(
            xvec, xvec, wigner, 100, cmap="RdBu_r", vmin=-wlim, vmax=wlim
        )
        add_colorbar(fig, image, ax=state_ax, label=r"$W(x,p)$")
        state_ax.set_xlabel("x")
        state_ax.set_ylabel("p")
        state_ax.set_aspect("equal", adjustable="box")
        title = state_title
        if phase is not None:
            title += rf" · $\theta={phase / np.pi:.2f}\pi$"
        state_ax.set_title(title)

    fig.suptitle("Mach–Zehnder interference scan", fontsize=15, fontweight="medium")
    if show:
        plt.show()
    return fig


__all__ = [
    "make_even_cat",
    "plot_mzi_scan",
    "run_cat_mzi_phase_scan",
    "run_mzi_phase_scan",
]
