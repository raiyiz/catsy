import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from catsy import GaussianState
from catsy.visualization import (
    animate_phase_space,
    plot_covariance_evolution,
    plot_covariance_matrix,
    plot_diagnostics,
    plot_evolution,
    plot_phase_space,
    plot_phase_space_trajectory,
    plot_state_dashboard,
    plot_wigner,
    plot_wigner_evolution,
)


@pytest.mark.visual
def test_visualizations_return_figures_without_showing() -> None:
    state = (
        GaussianState.tmsv("a", "b", r=0.85)
        .squeeze("a", r=0.55, theta=0.35)
        .displace("a", 1.1 + 0.7j)
        .displace("b", -0.55 + 0.35j)
        .rotate("b", 0.7)
    )

    covariance = plot_covariance_matrix(state)
    phase_space = plot_phase_space(state, "a")
    wigner = plot_wigner(state, "a", num_points=40)
    dashboard = plot_state_dashboard(state, mode="b")

    assert len(covariance.axes) == 2
    assert len(phase_space.axes) == 1
    assert len(wigner.axes) == 2
    assert len(dashboard.axes) == 5


@pytest.mark.visual
def test_phase_space_uses_displacement() -> None:
    state = (
        GaussianState.vacuum(("a",))
        .squeeze("a", r=0.9, theta=np.pi / 5)
        .displace("a", 1.2 + 0.65j)
        .rotate("a", np.pi / 7)
    )
    figure = plot_phase_space(state, "a")
    scatter = figure.axes[0].collections[0]

    offsets = np.asarray(scatter.get_offsets())
    np.testing.assert_allclose(offsets[0], state.displacement, atol=1e-12)


@pytest.mark.visual
def test_evolution_visualizations() -> None:
    # A deliberately nontrivial Gaussian evolution: a strongly squeezed,
    # displaced state rotates while progressively coupling to vacuum loss.
    state = (
        GaussianState.vacuum(("a",))
        .squeeze("a", r=1.0, theta=0.2)
        .displace("a", 1.5 + 0.4j)
    )
    states = []
    for step in range(13):
        fraction = step / 12
        states.append(
            state
            .rotate("a", 2.8 * fraction)
            .loss("a", eta=1.0 - 0.55 * fraction)
            .displace("a", 0.35 * np.exp(1j * 2.0 * fraction))
        )
    times = np.linspace(0.0, 3.0, len(states))

    trajectory = plot_phase_space_trajectory(
        states, "a", times=times, ellipse_every=2, n_sigma=2.0
    )
    covariance = plot_covariance_evolution(states, "a", times=times)
    diagnostics = plot_diagnostics(states, times=times)
    wigner = plot_wigner_evolution(
        states, "a", times=times, indices=[0, 6, 12], num_points=30
    )
    dashboard = plot_evolution(
        states, "a", times=times, wigner_indices=[0, 6, 12]
    )
    animation = animate_phase_space(states, "a", times=times, interval=10)

    assert len(trajectory.axes) == 1
    assert len(covariance.axes) == 1
    assert len(diagnostics.axes) == 1
    assert len(wigner.axes) == 6
    assert len(dashboard.axes) == 5


@pytest.mark.visual
def test_visualization_arguments_are_validated() -> None:
    state = GaussianState.vacuum(("a",))

    with pytest.raises(ValueError, match="n_sigma"):
        plot_phase_space(state, "a", n_sigma=0)
    with pytest.raises(ValueError, match="num_points"):
        plot_wigner(state, "a", num_points=1)
    with pytest.raises(ValueError, match="same length"):
        plot_phase_space_trajectory([state], "a", times=[0.0, 1.0])
    with pytest.raises(ValueError, match="positive"):
        animate_phase_space([state], "a", interval=0)
