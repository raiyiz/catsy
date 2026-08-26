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
from catsy.fock_visualization import (
    plot_photon_statistics,
)
from catsy.fock_visualization import (
    plot_wigner as plot_fock_wigner,
)
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


@pytest.mark.visualize
def test_showcase_fock_state_diagnostics_gallery(
    assert_no_empty_axes, assert_layout_can_render
):
    """Compare photon-number statistics with Wigner structure for two states."""
    cutoff = 32
    alpha = 2.4
    states = [
        (
            "Even cat",
            (qt.coherent(cutoff, alpha) + qt.coherent(cutoff, -alpha)).unit(),
        ),
        (
            "Four-component compass",
            (
                qt.coherent(cutoff, alpha)
                + qt.coherent(cutoff, 1j * alpha)
                + qt.coherent(cutoff, -alpha)
                + qt.coherent(cutoff, -1j * alpha)
            ).unit(),
        ),
    ]

    figure = plt.figure(figsize=(13, 9), constrained_layout=True)
    grid = figure.add_gridspec(2, 2, hspace=0.18, wspace=0.12)

    for row, (name, state) in enumerate(states):
        photon_ax = figure.add_subplot(grid[row, 0])
        wigner_ax = figure.add_subplot(grid[row, 1])
        rho = qt.ket2dm(state)
        plot_photon_statistics(rho, ax=photon_ax)
        plot_fock_wigner(rho, xlim=(-7, 7), resolution=72, ax=wigner_ax)
        photon_ax.set_title(f"{name} · photon-number statistics")
        wigner_ax.set_title(f"{name} · Wigner function")

    figure.suptitle(
        "Non-Gaussian states · photon-number structure and phase-space interference",
        fontsize=16,
        fontweight="medium",
    )

    compass_wigner = qt.wigner(
        states[1][1], np.linspace(-7, 7, 72), np.linspace(-7, 7, 72)
    )
    assert np.min(compass_wigner) < -0.01
    assert_no_empty_axes(figure)
    assert_layout_can_render(figure)
