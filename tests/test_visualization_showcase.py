"""Curated visualization gallery.

The contract suites test individual plotting APIs. This module instead collects
richer, physically meaningful examples that are useful as a visual smoke test
and as a record of the states catsy is intended to make easy to explore.
"""

import matplotlib.pyplot as plt
import numpy as np
import pytest
import qutip as qt

from catsy import GaussianState
from catsy.core import DUAN_SEPARABILITY_BOUND
from catsy.fock_visualization import plot_fock_dashboard, plot_wigner as plot_fock_wigner
from catsy.gaussian import LossChannels, compute_duan_inseparability
from catsy.gaussian.visualization import (
    plot_evolution,
    plot_joint_correlation,
    plot_phase_space,
    plot_state_dashboard,
    plot_wigner,
)


@pytest.mark.visualize
def test_showcase_gaussian_state_gallery(assert_no_empty_axes, assert_layout_can_render):
    """Show a displaced, squeezed two-mode Gaussian state from several views."""
    state = (
        GaussianState.tmsv("a", "b", r=0.9)
        .squeeze("a", r=0.65, theta=0.3)
        .squeeze("b", r=0.35, theta=-0.8)
        .rotate("a", 0.7)
        .rotate("b", -0.45)
        .displace("a", 1.25 + 0.55j)
        .displace("b", -0.8 + 0.35j)
    )

    figure = plot_state_dashboard(state, mode="b")
    phase_space = plot_phase_space(state, "a")
    wigner = plot_wigner(state, "a", num_points=60)

    figure.suptitle("Two-mode squeezed, rotated, and displaced Gaussian state")
    assert len(figure.axes) == 7
    assert len(phase_space.axes) == 1
    assert len(wigner.axes) == 2
    assert_no_empty_axes(figure)
    assert_no_empty_axes(phase_space)
    assert_no_empty_axes(wigner)
    assert_layout_can_render(figure)
    assert_layout_can_render(phase_space)
    assert_layout_can_render(wigner)


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
    duan_tmsv = compute_duan_inseparability(tmsv, "a", "b")
    duan_classical = compute_duan_inseparability(classical, "a", "b")
    assert duan_tmsv < DUAN_SEPARABILITY_BOUND < duan_classical
    assert_no_empty_axes(figure)
    assert_layout_can_render(figure)


@pytest.mark.visualize
def test_showcase_gaussian_evolution_gallery(assert_no_empty_axes, assert_layout_can_render):
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
    assert len(figure.axes) >= 4
    assert figure._suptitle is not None
    assert "mode a" in figure._suptitle.get_text()
    assert_no_empty_axes(figure)
    assert_layout_can_render(figure)


@pytest.mark.visualize
def test_showcase_fock_dashboard_gallery(assert_no_empty_axes, assert_layout_can_render):
    """Show the complementary views of a non-Gaussian cat state."""
    cutoff = 16
    cat = (qt.coherent(cutoff, 1.8) + qt.coherent(cutoff, -1.8)).unit()
    rho = qt.ket2dm(cat)

    figure = plot_fock_dashboard(rho, xlim=(-5, 5), resolution=48)
    figure.suptitle("Even cat state: Fock-space and phase-space views")
    assert len(figure.axes) >= 7
    assert_no_empty_axes(figure)
    assert_layout_can_render(figure)


@pytest.mark.visualize
def test_showcase_fock_number_state_gallery(assert_no_empty_axes, assert_layout_can_render):
    """Compare the radial Wigner structure of several photon-number states."""
    n_values = (0, 1, 2, 3)
    cutoff = max(n_values) + 4

    figure = plt.figure(figsize=(12, 9), constrained_layout=True)
    for index, n in enumerate(n_values, start=1):
        ax = figure.add_subplot(2, 2, index, projection="3d")
        plot_fock_wigner(
            qt.ket2dm(qt.fock(cutoff, n)),
            xlim=(-4, 4),
            resolution=48,
            ax=ax,
            projection="3d",
        )
        ax.set_title(fr"$|{n}\\rangle$")

    figure.suptitle("Wigner functions of n-photon states", fontweight="medium")
    figure.canvas.draw()
    assert len(figure.axes) == len(n_values)
    assert all(ax.collections for ax in figure.axes)
    assert_no_empty_axes(figure)
    assert_layout_can_render(figure)


@pytest.mark.visualize
def test_showcase_fock_compass_state_has_interference(assert_no_empty_axes, assert_layout_can_render):
    """Show the higher-order interference of a four-component compass state."""
    cutoff = 32
    alpha = 2.4
    compass = (
        qt.coherent(cutoff, alpha)
        + qt.coherent(cutoff, 1j * alpha)
        + qt.coherent(cutoff, -alpha)
        + qt.coherent(cutoff, -1j * alpha)
    ).unit()
    rho = qt.ket2dm(compass)

    figure = plot_fock_wigner(rho, xlim=(-7, 7), resolution=72)
    grid = np.linspace(-7, 7, 72)
    wigner = qt.wigner(compass, grid, grid)
    figure.suptitle("Four-component compass state")
    assert np.min(wigner) < -0.01
    assert_no_empty_axes(figure)
    assert_layout_can_render(figure)
