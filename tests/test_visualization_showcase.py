"""Curated visualization gallery.

The contract suites test individual plotting APIs and their semantic behavior.
This module instead collects richer, physically meaningful examples that are
useful as visual smoke tests and as a record of the states catsy is intended
to make easy to explore.

Gallery tests deliberately keep assertions to figure-level rendering checks,
except for genuinely gallery-specific physical properties. API-specific
assertions belong in the contract suites.

Curation bar: each entry earns its place by showing something the others don't
-- a distinct plotting function, a distinct part of Hilbert space, or a
genuinely different physical story. A state or plot combination the contract
suites already exercise belongs there, not here; duplicating it here doesn't
make it a better showcase, just a longer file. The current arc, roughly in
increasing order of "how far from a textbook Gaussian state":

    single entangled Gaussian state -> quantum vs. classical correlation
    -> that entanglement building up in time under its own Hamiltonian
    -> Kerr-driven cat dynamics -> catsy's own heralded non-Gaussian gates
    -> a four-component compass state's full Fock diagnostics
    -> that state's interferometric (Mach-Zehnder) readout
"""

import matplotlib.pyplot as plt
import numpy as np
import pytest
import qutip as qt
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
from catsy.visualization import color_norm

_TMSV_COUPLING = 0.7  # kappa in H = i*kappa*(a^dagger b^dagger - a b)


def _tmsv_hamiltonian_evolution() -> tuple[list[GaussianState], np.ndarray]:
    """Exact two-mode-squeezing evolution from vacuum under H = iκ(a†b† - ab).

    Kept in sync with the identically-named fixture in
    test_gaussian_visualization.py, which uses it for
    plot_multimode_evolution's own contract test -- this fixture is the
    physical scenario, the two files just look at it differently (contract
    check vs. hand-built narrative gallery).
    """
    times = np.linspace(0.0, 1.5, 17)
    states = [GaussianState.tmsv("a", "b", r=_TMSV_COUPLING * time) for time in times]
    return states, times


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
        ax.set_title(f"{label}\n{ax.get_title()}")

    figure.suptitle("Quantum entanglement versus classical correlation")
    assert_no_empty_axes(figure)
    assert_layout_can_render(figure)


@pytest.mark.visualize
def test_showcase_tmsv_hamiltonian_evolution_gallery(
    assert_no_empty_axes, assert_layout_can_render
):
    """Watch two-mode squeezing -- and the entanglement from the gallery
    entry above -- actually build up under its generating Hamiltonian,
    H = iκ(a†b† - ab), from vacuum.
    """
    states, times = _tmsv_hamiltonian_evolution()
    dt = times[1] - times[0]

    figure = plt.figure(figsize=(11.5, 9.0), constrained_layout=True)
    grid = figure.add_gridspec(2, 2, height_ratios=(1.1, 0.9), hspace=0.18, wspace=0.16)
    mode_axes = [figure.add_subplot(grid[0, 0]), figure.add_subplot(grid[0, 1])]
    correlation_ax = figure.add_subplot(grid[1, :])

    n_sigma = 2.0
    for mode_index, (ax, mode_name) in enumerate(
        zip(mode_axes, states[0].modes, strict=True)
    ):
        idx = 2 * mode_index
        variances = np.array(
            [state.covariance[idx : idx + 2, idx : idx + 2] for state in states]
        )
        radii = n_sigma * np.sqrt(np.maximum(variances[:, 0, 0], 0.0))
        for frame, radius in enumerate(radii):
            circle = plt.Circle(
                (0.0, 0.0),
                float(radius),
                fill=False,
                linewidth=2.0 if frame in (0, len(states) - 1) else 0.9,
                alpha=0.65 if frame in (0, len(states) - 1) else 0.1,
                edgecolor=(
                    "tab:blue"
                    if frame == 0
                    else "tab:red"
                    if frame == len(states) - 1
                    else "black"
                ),
                zorder=3 if frame in (0, len(states) - 1) else 2,
            )
            ax.add_patch(circle)
        ax.scatter(0.0, 0.0, s=38, zorder=4)
        ax.set_xlim(-1.0, float(radii[-1]) * 1.2)
        ax.set_ylim(-1.0, float(radii[-1]) * 1.2)
        ax.set_aspect("equal")
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$p$")
        ax.set_title(
            f"Mode {mode_name} · local uncertainty\n"
            f"start: t={times[0]:g} · end: t={times[-1]:g} · Δt={dt:g}",
            fontweight="medium",
        )

    covariance = np.array([state.covariance for state in states])
    correlation_x = covariance[:, 0, 2] / np.sqrt(
        covariance[:, 0, 0] * covariance[:, 2, 2]
    )
    correlation_p = covariance[:, 1, 3] / np.sqrt(
        covariance[:, 1, 1] * covariance[:, 3, 3]
    )
    correlation_ax.plot(times, correlation_x, lw=2.2, label=r"$C_{x_ax_b}$")
    correlation_ax.plot(times, correlation_p, lw=2.2, label=r"$C_{p_ap_b}$")
    correlation_ax.axhspan(-0.005, 0.005, alpha=0.12)
    correlation_ax.set_ylim(-1.05, 1.05)
    correlation_ax.set_xlabel("time")
    correlation_ax.set_ylabel("normalized quadrature correlation")
    correlation_ax.set_title(
        f"TMSV correlation build-up · t={times[0]:g} → {times[-1]:g} · Δt={dt:g}",
        fontweight="medium",
    )
    correlation_ax.grid(alpha=0.12, linewidth=0.5)
    correlation_ax.spines[["top", "right"]].set_visible(False)
    correlation_ax.legend(frameon=False, loc="best")

    figure.suptitle(
        rf"Two-mode squeezing evolution · $H=i\kappa(a^\dagger b^\dagger-ab)$, "
        rf"$\kappa={_TMSV_COUPLING:g}$, $t\in[{times[0]:g}, {times[-1]:g}]$",
        fontsize=16,
        fontweight="medium",
    )

    assert len(figure.axes) == 3
    assert_no_empty_axes(figure)
    assert_layout_can_render(figure)
    assert len(correlation_ax.lines) == 2
    np.testing.assert_allclose(correlation_ax.get_lines()[0].get_xdata(), times)
    np.testing.assert_allclose(correlation_ax.get_lines()[1].get_xdata(), times)
    assert correlation_ax.get_lines()[0].get_ydata()[-1] > 0.0
    assert correlation_ax.get_lines()[1].get_ydata()[-1] < 0.0


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
    wigners = [
        qt.wigner(result.states[index], grid_values, grid_values) for index in indices
    ]
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
        plt.cm.ScalarMappable(norm=norm, cmap=image.cmap),
        ax=axes,
        fraction=0.046,
        pad=0.04,
        label="Wigner function",
    )
    figure.suptitle(
        rf"Even cat evolution · Kerr $\chi={kerr_strength:g}$, "
        rf"loss $\gamma={loss_rate:g}$, $t\in[{times[0]:g}, {times[-1]:g}]$",
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
    cat = _even_cat(cutoff=14, alpha=1.8 + 0.2j)
    subtracted = realistic_photon_subtraction(
        cat,
        tap_reflectivity=0.08,
        detector_efficiency=0.85,
        ancilla_cutoff=6,
    )
    added = realistic_photon_addition(
        subtracted,
        coupling_strength=0.045,
        detector_efficiency=0.85,
        ancilla_cutoff=6,
    )

    states = [
        ("Initial even cat", cat),
        ("After photon subtraction", subtracted),
        ("After photon addition", added),
    ]

    figure = plt.figure(figsize=(16, 5.5), constrained_layout=True)
    grid = figure.add_gridspec(1, 3, wspace=0.08)
    axes = [figure.add_subplot(grid[0, column]) for column in range(len(states))]

    wigners = [
        qt.wigner(state, np.linspace(-6, 6, 64), np.linspace(-6, 6, 64))
        for _, state in states
    ]
    norm = color_norm(
        np.concatenate([wigner.ravel() for wigner in wigners]), symmetric=True
    )
    plot_param = dict(
        xlim=(-5, 5),
        resolution=256,
        norm=norm,
        contour=False,
    )

    for ax, (title, state) in zip(axes, states, strict=True):
        plot_fock_wigner(
            state,
            ax=ax,
            colorbar=False,
            **plot_param,
        )
        ax.set_title(f"{title}\n{ax.get_title()}")

    plot_fock_wigner(
        states[-1][1],
        ax=axes[-1],
        colorbar=True,
        **plot_param,
    )

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

    figure = plot_fock_dashboard(rho, xlim=(-7, 7), resolution=96)
    figure.suptitle(
        "Four-component compass state · full Fock diagnostics",
        fontsize=16,
        fontweight="medium",
    )
    assert_no_empty_axes(figure)
    assert_layout_can_render(figure)


@pytest.mark.visualize
def test_showcase_compass_mzi_gallery(assert_no_empty_axes, assert_layout_can_render):
    """Show the compass state's Mach-Zehnder phase readout."""
    cutoff = 20
    alpha = 2.4
    state = (
        qt.coherent(cutoff, alpha)
        + qt.coherent(cutoff, 1j * alpha)
        + qt.coherent(cutoff, -alpha)
        + qt.coherent(cutoff, -1j * alpha)
    ).unit()
    rho = qt.ket2dm(state)

    results = run_mzi_phase_scan(rho)
    # TODO: stupid hack, need fix
    figure, (scan_ax, state_ax) = plt.subplots(
        2,
        figsize=(13.5, 5.5),
        constrained_layout=True,
    )
    plot_mzi_scan(results, axes=(scan_ax, state_ax))
    state_ax.set_title("Compass-state interferometric readout")
    assert_no_empty_axes(figure)
    assert_layout_can_render(figure)
