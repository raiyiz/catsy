import numpy as np
import pytest
import qutip as qt

from catsy import GaussianState
from catsy.fock_visualization import plot_fock_dashboard, plot_wigner
from catsy.gaussian.visualization import (
    plot_multimode_evolution,
    plot_state_dashboard,
)


@pytest.mark.visualize
def test_gaussian_dashboard_showcases_squeezed_displaced_state(
    assert_no_empty_axes, assert_layout_can_render
):
    """Keep a nontrivial single-mode Gaussian state as a canonical visual case."""
    state = (
        GaussianState.vacuum(("signal",))
        .squeeze("signal", r=0.85, theta=0.28)
        .displace("signal", alpha=1.15 + 0.55j)
    )
    figure = plot_state_dashboard(state)
    assert len(figure.axes) >= 3
    assert_no_empty_axes(figure)
    assert_layout_can_render(figure)


@pytest.mark.visualize
def test_gaussian_dashboard_showcases_two_mode_entanglement(
    assert_no_empty_axes, assert_layout_can_render
):
    """The two-mode squeezed vacuum exercises cross-mode structure in the dashboard."""
    state = GaussianState.tmsv("signal", "idler", r=0.72)
    figure = plot_state_dashboard(state, mode="signal")
    assert len(figure.axes) >= 4
    assert_no_empty_axes(figure)
    assert_layout_can_render(figure)


@pytest.mark.visualize
def test_gaussian_multimode_evolution_showcases_correlation_dynamics(
    assert_no_empty_axes, assert_layout_can_render
):
    """Keep a short entangling evolution as the canonical multimode animation frame set."""
    initial = GaussianState.vacuum(("a", "b"))
    states = [
        initial,
        initial.squeeze("a", r=0.45).beam_splitter("a", "b", eta=0.5),
        initial.squeeze("a", r=0.75).beam_splitter("a", "b", eta=0.5),
        initial.squeeze("a", r=1.0).beam_splitter("a", "b", eta=0.5),
    ]
    figure = plot_multimode_evolution(states, times=np.linspace(0.0, 1.0, len(states)))
    assert len(figure.axes) >= 3
    assert_no_empty_axes(figure)
    assert_layout_can_render(figure)


@pytest.mark.visualize
def test_fock_dashboard_showcases_odd_cat_state(
    assert_no_empty_axes, assert_layout_can_render
):
    """An odd cat keeps parity interference and Wigner negativity visible."""
    cutoff = 20
    cat = (qt.coherent(cutoff, 2.2) - qt.coherent(cutoff, -2.2)).unit()
    rho = qt.ket2dm(cat)
    figure = plot_fock_dashboard(rho, xlim=(-6, 6), resolution=64)
    assert len(figure.axes) >= 7
    assert_no_empty_axes(figure)
    assert_layout_can_render(figure)


@pytest.mark.visualize
def test_fock_wigner_showcases_squeezed_cat_interference(
    assert_no_empty_axes, assert_layout_can_render
):
    """A squeezed cat exposes both non-Gaussian interference and anisotropy."""
    cutoff = 24
    cat = (qt.coherent(cutoff, 1.9) + qt.coherent(cutoff, -1.9)).unit()
    squeezed_cat = (qt.squeeze(cutoff, 0.65) * cat).unit()
    rho = qt.ket2dm(squeezed_cat)
    figure = plot_wigner(rho, xlim=(-6, 6), resolution=64)
    assert len(figure.axes[0].collections) >= 2
    assert_no_empty_axes(figure)
    assert_layout_can_render(figure)


@pytest.mark.visualize
def test_fock_dashboard_showcases_high_order_fock_superposition(
    assert_no_empty_axes, assert_layout_can_render
):
    """Widely separated Fock components expose genuinely higher-order coherence."""
    cutoff = 24
    state = (qt.fock(cutoff, 2) + 1j * qt.fock(cutoff, 11)).unit()
    rho = qt.ket2dm(state)
    figure = plot_fock_dashboard(rho, xlim=(-6, 6), resolution=64, n_max=13)
    magnitude = figure.axes[2].images[0].get_array()
    assert magnitude[2, 11] == pytest.approx(0.5, abs=1e-12)
    assert magnitude[11, 2] == pytest.approx(0.5, abs=1e-12)
    assert_no_empty_axes(figure)
    assert_layout_can_render(figure)


@pytest.mark.visualize
def test_fock_wigner_showcases_higher_fock_state(
    assert_no_empty_axes, assert_layout_can_render
):
    """A higher Fock state makes radial Wigner oscillations visible."""
    cutoff = 22
    state = qt.fock(cutoff, 8)
    rho = qt.ket2dm(state)
    grid = np.linspace(-7, 7, 72)
    figure = plot_wigner(rho, xlim=(-7, 7), resolution=72)
    wigner = qt.wigner(state, grid, grid)
    assert np.any(wigner > 0.05)
    assert np.any(wigner < -0.02)
    assert_no_empty_axes(figure)
    assert_layout_can_render(figure)


@pytest.mark.visualize
def test_fock_wigner_showcases_four_component_compass_state(
    assert_no_empty_axes, assert_layout_can_render
):
    """A four-component cat produces higher-order phase-space interference."""
    cutoff = 32
    alpha = 2.4
    compass = (
        qt.coherent(cutoff, alpha)
        + qt.coherent(cutoff, 1j * alpha)
        + qt.coherent(cutoff, -alpha)
        + qt.coherent(cutoff, -1j * alpha)
    ).unit()
    rho = qt.ket2dm(compass)
    grid = np.linspace(-7, 7, 72)
    figure = plot_wigner(rho, xlim=(-7, 7), resolution=72)
    wigner = qt.wigner(compass, grid, grid)
    assert np.min(wigner) < -0.01
    assert_no_empty_axes(figure)
    assert_layout_can_render(figure)


@pytest.mark.visualize
def test_fock_dashboard_showcases_fock_coherence_superposition(
    assert_no_empty_axes, assert_layout_can_render
):
    """A finite Fock superposition makes off-diagonal coherence directly visible."""
    cutoff = 18
    state = (qt.fock(cutoff, 1) + 1j * qt.fock(cutoff, 7)).unit()
    rho = qt.ket2dm(state)
    figure = plot_fock_dashboard(rho, xlim=(-5, 5), resolution=56)
    magnitude = figure.axes[2].images[0].get_array()
    assert magnitude[1, 7] == pytest.approx(0.5, abs=1e-12)
    assert magnitude[7, 1] == pytest.approx(0.5, abs=1e-12)
    assert_no_empty_axes(figure)
    assert_layout_can_render(figure)
