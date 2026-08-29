import numpy as np
import pytest
from matplotlib import pyplot as plt

from catsy.fock.mzi_visualization import (
    make_even_cat,
    plot_mzi_scan,
    run_cat_mzi_phase_scan,
    run_mzi_phase_scan,
)


@pytest.mark.visualize
def test_cat_mzi_phase_scan_helpers_produce_a_two_panel_figure(
    assert_no_empty_axes, assert_layout_can_render
):
    theta = np.linspace(0.0, 2.0 * np.pi, 24)
    cat, results = run_cat_mzi_phase_scan(
        cutoff=12,
        alpha=1.8 + 0.2j,
        theta_list=theta,
    )

    assert cat.isket
    assert len(results["theta"]) == len(theta)
    assert np.isfinite(results["parity1"]).all()

    figure = plot_mzi_scan(
        results,
        state=cat,
        state_title="Even cat entering MZI",
        resolution=64,
    )

    assert len(figure.axes) == 2
    assert_no_empty_axes(figure)
    assert_layout_can_render(figure)
    plt.close(figure)


def test_generic_mzi_phase_scan_reuses_an_existing_fock_state():
    theta = np.array([0.0, 0.7, 1.9])
    cat = make_even_cat(cutoff=10, alpha=1.2)
    results = run_mzi_phase_scan(cat, cutoff=10, theta_list=theta)

    np.testing.assert_allclose(results["theta"], theta)
    assert len(results["n1"]) == len(theta)
    assert len(results["n2"]) == len(theta)
    assert len(results["parity1"]) == len(theta)
