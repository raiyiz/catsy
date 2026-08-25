import numpy as np
import pytest
import qutip as qt

from catsy.fock_visualization import (
    plot_fock_density_matrix,
    plot_photon_statistics,
    plot_wigner,
)


STATIC_VISUALIZATIONS = [
    pytest.param(plot_photon_statistics, 1, id="photon-statistics"),
    pytest.param(plot_fock_density_matrix, 4, id="fock-density-matrix"),
    pytest.param(plot_wigner, 2, id="wigner"),
]


@pytest.mark.visualize
@pytest.mark.parametrize("plotter, expected_axes", STATIC_VISUALIZATIONS)
def test_fock_visualization_renders_representative_state(
    plotter, expected_axes, assert_no_empty_axes, assert_layout_can_render
):
    rho = qt.ket2dm((qt.squeeze(10, 0.5) * qt.fock(10, 0)).unit())
    kwargs = {"resolution": 48} if plotter is plot_wigner else {}
    figure = plotter(rho, **kwargs)

    assert len(figure.axes) == expected_axes
    assert_no_empty_axes(figure)
    assert_layout_can_render(figure)


@pytest.mark.visualize
@pytest.mark.parametrize("n", [0, 1, 3, 6], ids=lambda n: f"n={n}")
def test_photon_statistics_for_fock_state_has_single_peak(
    n, assert_no_empty_axes, assert_layout_can_render
):
    rho = qt.ket2dm(qt.fock(10, n))
    figure = plot_photon_statistics(rho)
    bars = figure.axes[0].patches
    heights = np.array([bar.get_height() for bar in bars])

    assert heights[n] == pytest.approx(1.0)
    assert np.count_nonzero(heights > 1e-12) == 1
    assert_no_empty_axes(figure)
    assert_layout_can_render(figure)


@pytest.mark.visualize
def test_density_matrix_visualization_shows_coherences_for_cat_like_state(
    assert_no_empty_axes, assert_layout_can_render
):
    cutoff = 12
    state = (qt.fock(cutoff, 0) + qt.fock(cutoff, 4)).unit()
    rho = qt.ket2dm(state)
    figure = plot_fock_density_matrix(rho)
    magnitude_image = figure.axes[0].images[0].get_array()

    assert magnitude_image[0, 4] == pytest.approx(0.5, abs=1e-12)
    assert magnitude_image[4, 0] == pytest.approx(0.5, abs=1e-12)
    assert_no_empty_axes(figure)
    assert_layout_can_render(figure)


@pytest.mark.visualize
def test_wigner_visualization_contains_zero_contour_for_non_gaussian_state(
    assert_no_empty_axes, assert_layout_can_render
):
    cutoff = 14
    state = (qt.fock(cutoff, 0) + qt.fock(cutoff, 2)).unit()
    rho = qt.ket2dm(state)
    figure = plot_wigner(rho, xlim=(-4, 4), resolution=48)

    assert len(figure.axes[0].collections) >= 2
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
    figure = plot_photon_statistics(rho, mode_idx=1)
    heights = np.array([bar.get_height() for bar in figure.axes[0].patches])

    assert heights[2] == pytest.approx(1.0)
    assert_no_empty_axes(figure)
    assert_layout_can_render(figure)


class TestFockVisualizationValidation:
    """Argument validation for Fock-space visualization helpers."""

    @pytest.mark.parametrize(
        "call, match",
        [
            pytest.param(
                lambda rho: plot_photon_statistics(rho, n_max=0),
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
        ],
    )
    def test_validation(self, call, match):
        rho = qt.ket2dm(qt.fock(8, 1))
        with pytest.raises(ValueError, match=match):
            call(rho)
