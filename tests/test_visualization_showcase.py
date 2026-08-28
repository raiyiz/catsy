"""Curated visualization gallery.

The contract suites test individual plotting APIs and their semantic behavior.
This module instead collects richer, physically meaningful examples that are
useful as visual smoke tests and as a record of the states catsy is intended
to make easy to explore.

Gallery tests deliberately keep assertions to figure-level rendering checks,
except for genuinely gallery-specific physical properties. API-specific
assertions belong in the contract suites.
"""

import matplotlib.pyplot as plt
import numpy as np
import pytest
import qutip as qt

from catsy import GaussianState
from catsy.fock import realistic_photon_addition, realistic_photon_subtraction
from catsy.fock.visualization import plot_fock_density_matrix
from catsy.fock.visualization import plot_wigner as plot_fock_wigner
from catsy.gaussian import LossChannels
from catsy.gaussian.visualization import (
    plot_covariance_matrix,
    plot_evolution,
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
    for (state, quadrature), ax in zip(
        [(tmsv, "x"), (tmsv, "p"), (classical, "x"), (classical, "p")],
        axes.flat,
        strict=True,
    ):
        plot_joint_correlation(state, "a", "b", quadrature=quadrature, ax=ax)

    figure.suptitle("Quantum entanglement versus classical correlation")
    assert_no_empty_axes(figure)
    assert_layout_can_render(figure)


@pytest.mark.visualize
def test_showcase_gaussian_evolution_gallery(
    assert_no_empty_axes, assert_layout_can_render
):
    """Show a nontrivial single-mode evolution at several representative times."""
    state = (
        GaussianState.vacuum(("a",))
        .squeeze("a", r=1.0, theta=0.2)
        .displace("a", 1.5 + 0.4j)
    )
    states = []
    for step in range(17):
        fraction = step / 16
        states.append(
            state.rotate("a", 3.4 * fraction)
            .loss("a", eta=1.0 - 0.6 * fraction)
            .displace("a", 0.5 * np.exp(1j * 2.4 * fraction))
            .squeeze("a", r=0.25 * np.sin(np.pi * fraction), theta=0.6)
        )
    times = np.linspace(0.0, 4.0, len(states))

    figure = plot_evolution(states, "a", times=times, wigner_indices=[0, 8, 16])
    assert_no_empty_axes(figure)
    assert_layout_can_render(figure)


def _even_cat(cutoff: int = 32, alpha: complex = 2.2) -> qt.Qobj:
    """Create a normalized even cat density matrix for gallery rendering."""
    state = (qt.coherent(cutoff, alpha) + qt.coherent(cutoff, -alpha)).unit()
    return qt.ket2dm(state)


@pytest.mark.visualize
def test_showcase_cat_state_evolution_gallery(
    assert_no_empty_axes, assert_layout_can_render
):
    """Show a cat state changing under Kerr evolution and photon loss."""
    cutoff = 32
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

    indices = [0, 2, 4, 6, 8]
    figure = plt.figure(figsize=(17, 7), constrained_layout=True)
    grid = figure.add_gridspec(1, len(indices), wspace=0.08)

    for column, index in enumerate(indices):
        ax = figure.add_subplot(grid[0, column])
        plot_fock_wigner(
            result.states[index],
            xlim=(-6.5, 6.5),
            resolution=96,
            ax=ax,
        )
        ax.set_title(f"t = {times[index]:.1f}")

    figure.suptitle(
        "Even cat evolution · Kerr nonlinearity and photon loss",
        fontsize=16,
        fontweight="medium",
    )

    grid = np.linspace(-6.5, 6.5, 96)
    initial_wigner = qt.wigner(result.states[0], grid, grid)
    final_wigner = qt.wigner(result.states[-1], grid, grid)
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
def test_showcase_compass_state_gallery(
    assert_no_empty_axes, assert_layout_can_render
):
    """Show a four-component compass state and its nonclassical interference."""
    cutoff = 32
    alpha = 2.4
    state = (
        qt.coherent(cutoff, alpha)
        + qt.coherent(cutoff, 1j * alpha)
        + qt.coherent(cutoff, -alpha)
        + qt.coherent(cutoff, -1j * alpha)
    ).unit()
    rho = qt.ket2dm(state)

    figure = plt.figure(figsize=(13, 6), constrained_layout=True)
    grid = figure.add_gridspec(1, 3, width_ratios=(1.0, 1.0, 0.9), wspace=0.1)
    wigner_ax = figure.add_subplot(grid[0, 0])
    magnitude_ax = figure.add_subplot(grid[0, 1])
    phase_ax = figure.add_subplot(grid[0, 2])

    plot_fock_wigner(rho, xlim=(-7, 7), resolution=96, ax=wigner_ax)
    plot_fock_density_matrix(rho, axes=(magnitude_ax, phase_ax))

    figure.suptitle(
        "Four-component compass state · phase-space interference and Fock coherence",
        fontsize=16,
        fontweight="medium",
    )

    grid_values = np.linspace(-7, 7, 96)
    wigner = qt.wigner(rho, grid_values, grid_values)
    assert np.min(wigner) < -0.01
    assert_no_empty_axes(figure)
    assert_layout_can_render(figure)
