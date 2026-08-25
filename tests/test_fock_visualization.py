import matplotlib.pyplot as plt
import numpy as np
import pytest
import qutip as qt

from catsy.fock_visualization import (
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


def test_multimode_visualization_reduces_to_selected_mode():
    cutoff = 8
    rho = qt.tensor(
        qt.ket2dm(qt.coherent(cutoff, 0.4)),
        qt.ket2dm(qt.fock(cutoff, 2)),
    )
    figure = plot_photon_statistics(rho, mode_idx=1)
    heights = np.array([bar.get_height() for bar in figure.axes[0].patches])

    assert heights[2] == pytest.approx(1.0)


@pytest.mark.visualize
def test_wigner_surfaces_for_n_photon_states(
    assert_no_empty_axes, assert_layout_can_render
):
    """Compare the radial Wigner oscillations of several number states."""
    n_values = (0, 1, 2, 3)
    resolution = 60
    grid = np.linspace(-4.0, 4.0, resolution)
    x, p = np.meshgrid(grid, grid)

    fig = plt.figure(figsize=(12, 9), constrained_layout=True)
    for index, n in enumerate(n_values, start=1):
        ax = fig.add_subplot(2, 2, index, projection="3d")
        wigner = qt.wigner(qt.fock_dm(n + 1, n), grid, grid)
        ax.plot_surface(
            x,
            p,
            wigner,
            rcount=resolution,
            ccount=resolution,
            linewidth=0,
            antialiased=True,
            cmap="RdBu_r",
        )
        ax.contour(
            x,
            p,
            wigner,
            levels=[0.0],
            offset=float(np.min(wigner)),
            colors="black",
            linewidths=0.7,
        )
        ax.set_title(fr"$|{n}\rangle$")
        ax.set_xlabel("$x$")
        ax.set_ylabel("$p$")
        ax.set_zlabel("$W(x,p)$")

    fig.suptitle("Wigner functions of n-photon states", fontweight="medium")
    assert len(fig.axes) == len(n_values)
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
    ],
)
def test_fock_visualization_validation(call, match):
    rho = qt.ket2dm(qt.fock(8, 1))
    with pytest.raises(ValueError, match=match):
        call(rho)
