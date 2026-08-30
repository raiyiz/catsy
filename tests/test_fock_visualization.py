import numpy as np
import pytest
import qutip as qt
from matplotlib import pyplot as plt

from catsy.fock.visualization import (
    _state_description,
    plot_fock_dashboard,
    plot_fock_density_matrix,
    plot_photon_statistics,
    plot_wigner,
)


@pytest.mark.parametrize(
    "state, expected",
    [
        pytest.param(qt.fock(8, 3), r"Fock state $|3\rangle$", id="fock"),
        pytest.param(qt.fock(8, 0), r"Fock state $|0\rangle$", id="vacuum-fock"),
        pytest.param(qt.coherent(20, 1.2), "Poissonian state", id="poissonian"),
        pytest.param(
            (qt.fock(8, 0) + qt.fock(8, 2)).unit(),
            "Poissonian state",
            id="even-parity",
        ),
        pytest.param(
            (qt.fock(8, 1) + qt.fock(8, 3)).unit(),
            "odd-parity state",
            id="odd-parity",
        ),
    ],
)
def test_state_description_classifies_representative_fock_states(state, expected):
    description = _state_description(qt.ket2dm(state))
    assert description.startswith(expected)


def test_state_description_classifies_nonclassical_state():
    state = (qt.fock(10, 0) + qt.fock(10, 1)).unit()
    description = _state_description(qt.ket2dm(state))

    assert description.startswith("Nonclassical state")
    assert "g^{(2)}" in description


def test_state_description_falls_back_when_g2_is_undefined():
    rho = qt.ket2dm(qt.fock(8, 0))
    rho = rho + 0.0 * qt.ket2dm(qt.fock(8, 1))
    description = _state_description(rho)

    assert description.startswith("Fock state")


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

    # Also covers cutoff inference: with no explicit n_max, the plotted
    # range still extends to the full N_cutoff=10 Hilbert space.
    assert len(heights) == 10
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

    assert len(heights) == 10
    assert heights[3] == pytest.approx(1.0)
    assert np.count_nonzero(heights > 1e-12) == 1


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
    wigner_axes = []
    for index, n in enumerate(n_values, start=1):
        ax = fig.add_subplot(2, 2, index, projection="3d")
        wigner_axes.append(ax)
        state = qt.ket2dm(qt.fock(cutoff, n))
        plot_wigner(
            state,
            xlim=(-4, 4),
            resolution=resolution,
            ax=ax,
            projection="3d",
        )
        expected_wigner = qt.wigner(state, grid, grid)
        expected_limit = float(np.max(np.abs(expected_wigner)))
        surface = ax.collections[0]

        # Symmetric about zero (not just data min/max) so the colormap's
        # neutral color always sits at W=0 -- see plot_wigner's 3D branch.
        assert surface.norm.vmin == pytest.approx(-expected_limit)
        assert surface.norm.vmax == pytest.approx(expected_limit)
        ax.set_title(rf"$|{n}\rangle$")

    fig.suptitle("Wigner functions of n-photon states", fontweight="medium")
    fig.canvas.draw()

    # Each 3D panel now also gets its own colorbar axes (see plot_wigner's
    # 3D branch), so len(fig.axes) is no longer len(n_values) -- check the
    # data axes specifically instead of the whole figure.
    assert len(wigner_axes) == len(n_values)
    for ax in wigner_axes:
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
