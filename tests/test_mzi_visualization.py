import numpy as np
import pytest
from matplotlib import pyplot as plt

from catsy.fock import make_even_cat
from catsy.fock.mzi_visualization import plot_mzi_scan
from catsy.optics import MachZehnderInterferometer


@pytest.mark.visualize
def test_cat_mzi_scan_produces_a_two_panel_figure(
    assert_no_empty_axes, assert_layout_can_render
):
    theta = np.linspace(0.0, 2.0 * np.pi, 24)
    mzi = MachZehnderInterferometer.even_cat(cutoff=12, alpha=1.8 + 0.2j)
    results = mzi.scan(theta)

    assert mzi.state.isket
    assert len(results["theta"]) == len(theta)
    assert np.isfinite(results["parity1"]).all()

    figure = plot_mzi_scan(mzi, state_title="Even cat entering MZI", resolution=64)

    # scan_ax + state_ax, plus the state panel's own Wigner colorbar axes.
    assert len(figure.axes) == 3
    assert_no_empty_axes(figure)
    assert_layout_can_render(figure)
    plt.show(figure)


def test_generic_mzi_scan_reuses_an_existing_fock_state():
    theta = np.array([0.0, 0.7, 1.9])
    cat = make_even_cat(cutoff=10, alpha=1.2)
    results = MachZehnderInterferometer(cat, N_cutoff=10).scan(theta)

    np.testing.assert_allclose(results["theta"], theta)
    assert len(results["n1"]) == len(theta)
    assert len(results["n2"]) == len(theta)
    assert len(results["parity1"]) == len(theta)


def test_plot_mzi_scan_can_show_a_summary_panel_instead_of_the_wigner_inset(
    assert_no_empty_axes,
):
    mzi = MachZehnderInterferometer.even_cat(cutoff=8, alpha=1.0)
    mzi.scan(np.linspace(0.0, 2.0 * np.pi, 10))

    figure = plot_mzi_scan(mzi, show_state_panel=False)

    assert len(figure.axes) == 2  # scan_ax + text summary panel, no Wigner/colorbar
    assert_no_empty_axes(figure)
