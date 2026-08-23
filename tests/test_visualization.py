import matplotlib
import numpy as np

matplotlib.use("Agg")

from catsy import GaussianState
from catsy.visualization import (
    plot_covariance_matrix,
    plot_phase_space,
    plot_state_dashboard,
    plot_wigner,
)


def test_visualizations_return_figures_without_showing() -> None:
    state = GaussianState.tmsv("a", "b", r=0.4)

    covariance = plot_covariance_matrix(state)
    phase_space = plot_phase_space(state, "a")
    wigner = plot_wigner(state, "a", num_points=40)
    dashboard = plot_state_dashboard(state, mode="b")

    assert len(covariance.axes) == 2
    assert len(phase_space.axes) == 1
    assert len(wigner.axes) == 2
    assert len(dashboard.axes) == 5


def test_phase_space_uses_displacement() -> None:
    state = GaussianState.coherent(("a",), 0.8 + 0.3j)
    figure = plot_phase_space(state, "a")
    scatter = figure.axes[0].collections[0]

    offsets = np.asarray(scatter.get_offsets())
    np.testing.assert_allclose(offsets[0], state.displacement, atol=1e-12)


def test_visualization_arguments_are_validated() -> None:
    state = GaussianState.vacuum(("a",))

    try:
        plot_phase_space(state, "a", n_sigma=0)
    except ValueError as exc:
        assert "n_sigma" in str(exc)
    else:
        raise AssertionError("expected n_sigma validation")

    try:
        plot_wigner(state, "a", num_points=1)
    except ValueError as exc:
        assert "num_points" in str(exc)
    else:
        raise AssertionError("expected num_points validation")
