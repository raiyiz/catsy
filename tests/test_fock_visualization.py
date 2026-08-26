import numpy as np
import pytest
import qutip as qt
from matplotlib import pyplot as plt

from catsy.fock.visualization import (
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
    figure, axes = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True)
    plot_photon_statistics(rho, ax=axes[0])
    plot_wigner(rho, resolution=48, ax=axes[1])
    figure.suptitle("Squeezed-vacuum Fock representation", fontweight="medium")

    assert len(figure.axes) == 3  # Wigner adds a colorbar axis.
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


def test_photon_statistics_respects_explicit_n_max():
    rho = qt.ket2dm(qt.fock(10, 3))
    figure = plot_photon_statistics(rho, n_max=5)
    heights = np.array([bar.get_height() for bar in figure.axes[0].patches])

    assert len(heights) == 6
    assert heights[3] == pytest.approx(1.0)
    assert np.count_nonzero(heights > 1e-12) == 1


def test_photon_statistics_infers_cutoff_beyond_occupied_support():
    rho = qt.ket2dm(qt.fock(10, 3))
    figure = plot_photon_statistics(rho)
    heights = np.array([bar.get_height() for bar in figure.axes[0].patches])

    assert len(heights) == 6
    assert heights[3] == pytest.approx(1.0)
    assert np.allclose(heights[4:], 0.0)


def test_density_matrix_visualization_shows_coherences_for_cat_like_state():
    cutoff = 12
    phase = 0.7
    state = (qt.fock(cutoff, 0) + np.exp(1j * phase) * qt.fock(cutoff, 4)).unit()
    rho = qt.ket2dm(state)
    figure = plot_fock_density_matrix(rho)
    magnitude_image = figure.axes[0].images[0].get_array()
    phase_image = figure.axes[1].images[0].get_array()

    assert magnitude_image[0, 4] == pytest.approx(0.5, abs=1e-12)
    assert magnitude_image[4, 0] == pytest.approx(0.5, abs=1e-12)
    assert phase_image[0, 4] == pytest.approx(-phase, abs=1e-12)
    assert phase_image[4, 0] == pytest.approx(phase, abs=1e-12)
    assert phase_image.mask[0, 1]


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


def test_fock_dashboard_selects_requested_mode():
    cutoff = 8
    rho = qt.tensor(
        qt.ket2dm(qt.coherent(cutoff, 0.4)),
        qt.ket2dm(qt.fock(cutoff, 2)),
    )

    figure = plot_fock_dashboard(rho, mode_idx=1, resolution=48)

    heights = np.array([bar.get_height() for bar in figure.axes[0].patches])
    assert heights[2] == pytest.approx(1.0)
    assert np.count_nonzero(heights > 1e-12) == 1
    assert figure.axes[0].get_title().startswith("Fock state")


@pytest.mark.visualize
def test_multimode_visualizations_show_selected_mode_pair(
    assert_no_empty_axes, assert_layout_can_render
):
    cutoff = 8
    rho = qt.tensor(
        qt.ket2dm(qt.coherent(cutoff, 0.4)),
        qt.ket2dm(qt.fock(cutoff, 2)),
    )
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.5), constrained_layout=True)
    plot_photon_statistics(rho, mode_idx=1, ax=axes[0])
    plot_wigner(rho, mode_idx=1, resolution=48, ax=axes[1])
    figure.suptitle("Photon-number and Wigner views of mode 1", fontweight="medium")

    heights = np.array([bar.get_height() for bar in axes[0].patches])
    assert heights[2] == pytest.approx(1.0)
    assert_no_empty_axes(figure)
    assert_layout_can_render(figure)


@pytest.mark.visualize
def test_wigner_surfaces_for_n_photon_states(
    assert_no_empty_axes, assert_layout_can_render
):
    """Compare the radial Wigner oscillations of several number states."""
    n_values = (0, 1, 2, 3)
    cutoff = max(n_values) + 4
    resolution = 48
    grid = np.linspace(-4, 4, resolution)

    fig = plt.figure(figsize=(12, 9), constrained_layout=True)
    for index, n in enumerate(n_values, start=1):
        ax = fig.add_subplot(2, 2, index, projection="3d")
        state = qt.ket2dm(qt.fock(cutoff, n))
        plot_wigner(
            state,
            xlim=(-4, 4),
            resolution=resolution,
            ax=ax,
            projection="3d",
        )
        expected_wigner = qt.wigner(state, grid, grid)
        surface = ax.collections[0]

        assert surface.norm.vmin == pytest.approx(float(expected_wigner.min()))
        assert surface.norm.vmax == pytest.approx(float(expected_wigner.max()))
        ax.set_title(rf"$|{n}\rangle$")

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
            lambda rho: plot_wigner(rho, xlim=(0.0, np.inf)),
            "xlim",
            id="nonfinite-wigner-limits",
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
