import numpy as np
import pytest
import qutip as qt

from catsy import GaussianState
from catsy.fock_visualization import plot_wigner
from catsy.gaussian.visualization import plot_state_dashboard


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
