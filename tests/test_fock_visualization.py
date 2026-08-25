import numpy as np
import pytest
import qutip as qt

from catsy.fock_visualization import (
    plot_fock_density_matrix,
    plot_photon_statistics,
    plot_wigner,
)


@pytest.mark.parametrize(
    "rho",
    [
        qt.ket2dm(qt.fock(10, 0)),
        qt.ket2dm(qt.coherent(10, 1.0)),
        qt.ket2dm((qt.squeeze(10, 0.5) * qt.fock(10, 0)).unit()),
    ],
    ids=["vacuum", "coherent", "squeezed"],
)
def test_fock_visualizations_render_representative_states(rho):
    figures = [
        plot_photon_statistics(rho),
        plot_fock_density_matrix(rho),
        plot_wigner(rho, resolution=48),
    ]
    assert all(len(figure.axes) > 0 for figure in figures)


@pytest.mark.parametrize("n", [0, 1, 3, 6])
def test_photon_statistics_for_fock_state_has_single_peak(n):
    rho = qt.ket2dm(qt.fock(10, n))
    figure = plot_photon_statistics(rho)
    bars = figure.axes[0].patches
    heights = np.array([bar.get_height() for bar in bars])
    assert heights[n] == pytest.approx(1.0)
    assert np.count_nonzero(heights > 1e-12) == 1


def test_density_matrix_visualization_shows_coherences_for_cat_like_state():
    cutoff = 12
    state = (qt.fock(cutoff, 0) + qt.fock(cutoff, 4)).unit()
    rho = qt.ket2dm(state)
    figure = plot_fock_density_matrix(rho)
    magnitude_image = figure.axes[0].images[0].get_array()
    assert magnitude_image[0, 4] == pytest.approx(0.5, abs=1e-12)
    assert magnitude_image[4, 0] == pytest.approx(0.5, abs=1e-12)


def test_wigner_visualization_contains_zero_contour_for_non_gaussian_state():
    cutoff = 14
    state = (qt.fock(cutoff, 0) + qt.fock(cutoff, 2)).unit()
    rho = qt.ket2dm(state)
    figure = plot_wigner(rho, xlim=(-4, 4), resolution=48)
    assert len(figure.axes[0].collections) >= 2


def test_multimode_visualizations_reduce_to_selected_mode():
    cutoff = 8
    rho = qt.tensor(
        qt.ket2dm(qt.coherent(cutoff, 0.4)),
        qt.ket2dm(qt.fock(cutoff, 2)),
    )
    figure = plot_photon_statistics(rho, mode_idx=1)
    heights = np.array([bar.get_height() for bar in figure.axes[0].patches])
    assert heights[2] == pytest.approx(1.0)
