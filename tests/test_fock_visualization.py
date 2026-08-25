import numpy as np
import pytest
import qutip as qt
from matplotlib import pyplot as plt

from catsy.fock_visualization import (
    plot_fock_dashboard,
    plot_fock_density_matrix,
    plot_photon_statistics,
    plot_wigner,
)


@pytest.mark.visualize
def test_fock_visualizations_render_representative_state(
    assert_no_empty_axes, assert_layout_can_render
):
    rho = qt.ket2dm((qt.squeeze(10, 0.5) * qt.fock(10, 0)).unit())
    figures = [
        plot_photon_statistics(rho),
        plot_fock_density_matrix(rho),
        plot_wigner(rho, resolution=48),
    ]

    assert [len(figure.axes) for figure in figures] == [1, 4, 2]
    for figure in figures:
        assert_no_empty_axes(figure)
        assert_layout_can_render(figure)


def test_photon_statistics_for_fock_state_has_single_peak():
    rho = qt.ket2dm(qt.fock(10, 3))
    figure = plot_photon_statistics(rho)
    heights = np.array([bar.get_height() for bar in figure.axes[0].patches])

    assert heights[3] == pytest.approx(1.0)
    assert np.count_nonzero(heights > 1e-12) == 1


def test_photon_statistics_reports_sub_poissonian_fock_state():
    rho = qt.ket2dm(qt.fock(10, 3))
    figure = plot_photon_statistics(rho)
    annotations = [text.get_text() for text in figure.axes[0].texts]

    assert any("g^{(2)}" in text and "0.667" in text for text in annotations)


def test_density_matrix_visualization_shows_coherences_for_cat_like_state():
    cutoff = 12
    state = (qt.fock(cutoff, 0) + qt.fock(cutoff, 4)).unit()
    rho = qt.ket2dm(state)
    figure = plot_fock_density_matrix(rho)
    magnitude_image = figure.axes[0].images[0].get_array()

    assert magnitude_image[0, 4] == pytest.approx(0.5, abs=1e-12)
    assert magnitude_image[4, 0] == pytest.approx(0.5, abs=1e-12)


@pytest.mark.visualize
def test_wigner_visualization_contains_zero_contour_for_non_gaussian_state(
    assert_no_empty_axes, assert_layout_can_render
):
    cutoff = 14
    state = (qt.fock(cutoff, 0) + qt.fock(cutoff, 2)).unit()
    figure = plot_wigner(qt.ket2dm(state), xlim=(-4, 4), resolution=48)

    assert len(figure.axes[0].collections) >= 2
    assert_no_empty_axes(figure)
    assert_layout_can_render(figure)


@pytest.mark.visualize
def test_fock_dashboard_keeps_all_complementary_views(
    assert_no_empty_axes, assert_layout_can_render
):
    cutoff = 16
    cat = (qt.coherent(cutoff, 1.8) + qt.coherent(cutoff, -1.8)).unit()
    rho = qt.ket2dm(cat)
    figure = plot_fock_dashboard(rho, xlim=(-5, 5), resolution=48)

    assert len(figure.axes) >= 7
    assert_no_empty_axes(figure)
    assert_layout_can_render(figure)


@pytest.mark.visualize
def test_multimode_visualizations_reduce_to_selected_mode(
    assert_no_empty_axes, assert_layout_can_render
):
    cutoff = 8
    rho = qt.tensor(
        qt.ket2dm(qt.coherent(cutoff, 0.4)),
        qt.ket2dm(qt.fock(cutoff, 2)),
    )
    figures = [
        plot_photon_statistics(rho, mode_idx=1),
        plot_wigner(rho, mode_idx=0, resolution=48),
    ]
    heights = np.array([bar.get_height() for bar in figures[0].axes[0].patches])

    assert heights[2] == pytest.approx(1.0)
    for figure in figures:
        assert_no_empty_axes(figure)
        assert_layout_can_render(figure)


@pytest.mark.visualize
def test_wigner_surfaces_for_n_photon_states(
    assert_no_empty_axes, assert_layout_can_render
):
    """Compare the radial Wigner oscillations of several number states."""
    n_values = (0, 1, 2, 3)
    cutoff = max(n_values) + 4

    fig = plt.figure(figsize=(12, 9), constrained_layout=True)
    for index, n in enumerate(n_values, start=1):
        ax = fig.add_subplot(2, 2, index, projection="3d")
        plot_wigner(
            qt.ket2dm(qt.fock(cutoff, n)),
            xlim=(-4, 4),
            resolution=48,
            ax=ax,
            projection="3d",
        )
        ax.set_title(fr"$|{n}\rangle$")

    fig.suptitle("Wigner functions of n-photon states", fontweight="medium")
    fig.canvas.draw()

    assert len(fig.axes) == len(n_values)
    for ax in fig.axes:
        assert ax.collections
    assert_no_empty_axes(fig)
    assert_layout_can_render(fig)


@pytest.mark.parametrize(
    "call, match",
    [
        pytest.param(
            lambda rho: plot_photon_statistics(rho, n_max=-1),
            "n_max",
            id="invalid-photon-number-range",
        ),
        pytest.param(
            lambda rho: plot_wigner(rho, resolution=16),
            "resolution",
            id="invalid-wigner-resolution",
        ),
        pytest.param(
            lambda rho: plot_wigner(rho, xlim=(1.0, -1.0)),
            "xlim",
            id="invalid-wigner-limits",
        ),
        pytest.param(
            lambda rho: plot_wigner(rho, projection="invalid"),
            "projection",
            id="invalid-wigner-projection",
        ),
    ],
)
def test_fock_visualization_validation(call, match):
    rho = qt.ket2dm(qt.fock(8, 1))
    with pytest.raises(ValueError, match=match):
        call(rho)
