"""Curated visualization gallery.

The contract suites test individual plotting APIs and their semantic behavior.
This module instead collects richer, physically meaningful examples that are
useful as visual smoke tests and as a record of the states catsy is intended
to make easy to explore.

Gallery tests deliberately keep assertions to figure-level rendering checks,
except for genuinely gallery-specific physical properties. API-specific
assertions belong in the contract suites.

Curation bar: each entry earns its place by showing something the others
don't -- a distinct plotting function, a distinct part of Hilbert space, or a
genuinely different physical story. A state or plot combination the contract
suites already exercise belongs there, not here; duplicating it here doesn't
make it a better showcase, just a longer file. The current arc, roughly in
increasing order of "how far from a textbook Gaussian state":

    single entangled Gaussian state -> quantum vs. classical correlation
    -> Kerr-driven cat dynamics -> catsy's own heralded non-Gaussian gates
    -> a four-component compass state's full Fock diagnostics
    -> that state's interferometric (Mach-Zehnder) readout
"""

import matplotlib.pyplot as plt
import numpy as np
import pytest
import qutip as qt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from catsy import GaussianState
from catsy.fock import realistic_photon_addition, realistic_photon_subtraction
from catsy.fock.mzi_visualization import plot_mzi_scan, run_mzi_phase_scan
from catsy.fock.visualization import plot_fock_dashboard
from catsy.fock.visualization import plot_wigner as plot_fock_wigner
from catsy.gaussian import LossChannels
from catsy.gaussian.visualization import (
    plot_covariance_matrix,
    plot_joint_correlation,
    plot_mode_correlation_map,
    plot_phase_space,
    plot_wigner,
)


@pytest.mark.visualize
def test_showcase_gaussian_state_gallery(assert_no_empty_axes, assert_layout_can_render):
    """Show one rich two-mode Gaussian state as a coherent visual gallery."""
    state = (
        GaussianState.tmsv("a", "b", r=0.9)
        .squeeze("a", r=0.65, theta=0.3)
        .squeeze("b", r=0.35, theta=-0.8)
        .rotate("a", 0.7)
        .rotate("b", -0.45)
        .displace("a", 1.25 + 0.55j)
        .displace("b", -0.8 + 0.35j)
    )

    figure = plt.figure(figsize=(13.5, 10.5), constrained_layout=True)
    grid = figure.add_gridspec(2, 2, hspace=0.16, wspace=0.14)
    axes = [
        figure.add_subplot(grid[0, 0]),
        figure.add_subplot(grid[0, 1]),
        figure.add_subplot(grid[1, 0]),
        figure.add_subplot(grid[1, 1]),
    ]

    plot_phase_space(state, "a", ax=axes[0])
    plot_wigner(state, "a", num_points=72, ax=axes[1])
    plot_covariance_matrix(state, ax=axes[2], annotate=False)
    plot_mode_correlation_map(state, ax=axes[3], annotate=False)

    figure.suptitle(
        "Two-mode Gaussian state · phase space, Wigner function, and correlations",
        fontsize=16,
        fontweight="medium",
    )
    assert_no_empty_axes(figure)
    assert_layout_can_render(figure)


@pytest.mark.visualize
def test_showcase_gaussian_entanglement_gallery(
    assert_no_empty_axes, assert_layout_can_render
):
    """Contrast genuine two-mode squeezing with correlated classical noise."""
    tmsv = GaussianState.tmsv("a", "b", r=1.0)
    classical = LossChannels.correlated_thermal_noise(
        "a", "b", eta=0.3, n_thermal=1.5, c_correlation=1.4
    ).apply(GaussianState.vacuum(("a", "b")))

    figure, axes = plt.subplots(2, 2, figsize=(11, 10), constrained_layout=True)
    panels = [
        ("Two-mode squeezed vacuum", tmsv, "x"),
        ("Two-mode squeezed vacuum", tmsv, "p"),
        ("Correlated classical noise", classical, "x"),
        ("Correlated classical noise", classical, "p"),
    ]
    for (label, state, quadrature), ax in zip(panels, axes.flat, strict=True):
        plot_joint_correlation(state, "a", "b", quadrature=quadrature, ax=ax)
        # plot_joint_correlation's own title only names the quadrature, which
        # is identical across both states here -- prefix which state this
        # panel is, or the quantum/classical contrast is invisible at a
        # glance.
        ax.set_title(f"{label}\n{ax.get_title()}")

    figure.suptitle("Quantum entanglement versus classical correlation")
    assert_no_empty_axes(figure)
    assert_layout_can_render(figure)


def _even_cat(cutoff: int = 20, alpha: complex = 2.2) -> qt.Qobj:
    """Create a normalized even cat density matrix for gallery rendering."""
    state = (qt.coherent(cutoff, alpha) + qt.coherent(cutoff, -alpha)).unit()
    return qt.ket2dm(state)


@pytest.mark.visualize
def test_showcase_cat_state_evolution_gallery(
    assert_no_empty_axes, assert_layout_can_render
):
    """Show a cat state changing under Kerr evolution and photon loss."""
    cutoff = 20
    cat = _even_cat(cutoff=cutoff, alpha=2.2)
    a = qt.destroy(cutoff)
    number = a.dag() * a
    kerr_strength = 0.08
    loss_rate = 0.025
    hamiltonian = kerr_strength * number * number
    times = np.linspace(0.0, 10.0, 9)

    result = qt.mesolve(
        hamiltonian,
        cat,
        times,
        c_ops=[np.sqrt(loss_rate) * a],
    )

    indices = [0, 2, 4, 6]
    grid_values = np.linspace(-6.5, 6.5, 96)
    wigners = [qt.wigner(result.states[index], grid_values, grid_values) for index in indices]
    vmax = max(float(np.max(np.abs(wigner))) for wigner in wigners)
    norm = Normalize(vmin=-vmax, vmax=vmax)

    figure, axes = plt.subplots(2, 2, figsize=(11, 10), constrained_layout=True)
    for ax, index, wigner in zip(axes.flat, indices, wigners, strict=True):
        image = ax.imshow(
            wigner,
            origin="lower",
            extent=(grid_values[0], grid_values[-1], grid_values[0], grid_values[-1]),
            cmap="RdBu_r",
            norm=norm,
            interpolation="nearest",
            aspect="equal",
        )
        ax.contour(
            grid_values,
            grid_values,
            wigner,
            levels=[0.0],
            colors="black",
            linewidths=0.8,
            alpha=0.8,
        )
        ax.axhline(0, lw=0.6, ls="--", alpha=0.30, zorder=0)
        ax.axvline(0, lw=0.6, ls="--", alpha=0.30, zorder=0)
        ax.set_xlabel("x")
        ax.set_ylabel("p")
        ax.set_title(f"t = {times[index]:.1f}")

    figure.colorbar(
        ScalarMappable(norm=norm, cmap=image.cmap),
        ax=axes,
        fraction=0.046,
        pad=0.04,
        label="Wigner function",
    )
    figure.suptitle(
        "Even cat evolution · Kerr nonlinearity and photon loss",
        fontsize=16,
        fontweight="medium",
    )

    initial_wigner = qt.wigner(result.states[0], grid_values, grid_values)
    final_wigner = qt.wigner(result.states[-1], grid_values, grid_values)
    assert np.min(initial_wigner) < -0.01
    assert np.min(final_wigner) < 0.0
    assert_no_empty_axes(figure)
    assert_layout_can_render(figure)


@pytest.mark.visualize
def test_showcase_heralded_cat_processing_gallery(
    assert_no_empty_axes, assert_layout_can_render
):
    """Show an even cat through realistic photon subtraction and addition."""
    cat = _even_cat(cutoff=18, alpha=1.8 + 0.2j)
    subtracted = realistic_photon_subtraction(
        cat,
        tap_reflectivity=0.08,
        detector_efficiency=0.75,
        ancilla_cutoff=6,
    )
    added = realistic_photon_addition(
        subtracted,
        coupling_strength=0.045,
        detector_efficiency=0.75,
        ancilla_cutoff=6,
    )

    states = [
        ("Initial even cat", cat),
        ("After photon subtraction", subtracted),
        ("After photon addition", added),
    ]

    figure = plt.figure(figsize=(16, 5.5), constrained_layout=True)
    grid = figure.add_gridspec(1, 3, wspace=0.08)

    for column, (title, state) in enumerate(states):
        ax = figure.add_subplot(grid[0, column])
        plot_fock_wigner(state, xlim=(-6, 6), resolution=64, ax=ax)
        ax.set_title(title)

    figure.suptitle(
        "Heralded non-Gaussian processing · cat → subtraction → addition",
        fontsize=16,
        fontweight="medium",
    )

    assert abs(float(cat.tr()) - 1.0) < 1e-10
    assert float(subtracted.tr()) > 0.0
    assert float(added.tr()) > 0.0
    assert_no_empty_axes(figure)
    assert_layout_can_render(figure)


@pytest.mark.visualize
def test_showcase_compass_state_gallery(assert_no_empty_axes, assert_layout_can_render):
    """Show a four-component compass state's full Fock-space diagnostics."""
    cutoff = 20
    alpha = 2.4
    state = (
        qt.coherent(cutoff, alpha)
        + qt.coherent(cutoff, 1j * alpha)
        + qt.coherent(cutoff, -alpha)
        + qt.coherent(cutoff, -1j * alpha)
    ).unit()
    rho = qt.ket2dm(state)

    # plot_fock_dashboard already composes photon-number statistics, the
    # Wigner function, and the density-matrix magnitude/phase into one call
    # -- reuse it directly rather than hand-assembling a subset of the same
    # panels, and it's a plotting entry point that otherwise had no showcase
    # representation at all.
    figure = plot_fock_dashboard(rho, xlim=(-7, 7), resolution=96)
    figure.suptitle(
        "Four-component compass state · full Fock diagnostics",
        fontsize=16,
        fontweight="medium",
    )

    grid_values = np.linspace(-7, 7, 96)
    wigner = qt.wigner(rho, grid_values, grid_values)
    assert np.min(wigner) < -0.01
    assert_no_empty_axes(figure)
    assert_layout_can_render(figure)


@pytest.mark.visualize
def test_showcase_mzi_interference_gallery(
    assert_no_empty_axes, assert_layout_can_render
):
    """Send the compass state through a lossy, Kerr-coupled Mach-Zehnder scan.

    The rest of the gallery is phase-space/Wigner-family plots; this is the
    one entry built entirely differently, on swept-phase line data rather
    than a 2D quadrature grid, showing the interferometric readout side of
    the project instead of state tomography.
    """
    cutoff = 20
    alpha = 2.4
    state = (
        qt.coherent(cutoff, alpha)
        + qt.coherent(cutoff, 1j * alpha)
        + qt.coherent(cutoff, -alpha)
        + qt.coherent(cutoff, -1j * alpha)
    ).unit()

    theta = np.linspace(0.0, 2.0 * np.pi, 120)
    results = run_mzi_phase_scan(
        state,
        cutoff=cutoff,
        theta_list=theta,
        kappa=0.05,
        loss_time=0.85,
    )

    figure = plot_mzi_scan(
        results,
        state=state,
        state_title="Compass state entering the MZI",
        phase=float(theta[len(theta) // 3]),
    )
    figure.suptitle(
        "Compass state · lossy, Kerr-coupled Mach–Zehnder interference",
        fontsize=15,
        fontweight="medium",
    )

    parity1 = np.asarray(results["parity1"])
    assert np.ptp(parity1) > 0.05, "phase scan should show a genuine oscillation"
    assert_no_empty_axes(figure)
    assert_layout_can_render(figure)
