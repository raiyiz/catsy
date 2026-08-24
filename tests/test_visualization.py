import numpy as np
import pytest
from test_gaussian import wigner_test_state

from catsy import GaussianState
from catsy.visualization import (
    animate_phase_space,
    plot_covariance_evolution,
    plot_covariance_matrix,
    plot_diagnostics,
    plot_evolution,
    plot_mode_correlation_map,
    plot_phase_space,
    plot_phase_space_trajectory,
    plot_state_dashboard,
    plot_wigner,
    plot_wigner_evolution,
)
from catsy.visualization_dynamics import plot_phase_space_trajectory_timecoded


def _complex_state() -> GaussianState:
    """A representative multimode Gaussian state for showcase plots."""
    return (
        GaussianState.tmsv("a", "b", r=0.9)
        .squeeze("a", r=0.65, theta=0.3)
        .squeeze("b", r=0.35, theta=-0.8)
        .rotate("a", 0.7)
        .rotate("b", -0.45)
        .displace("a", 1.25 + 0.55j)
        .displace("b", -0.8 + 0.35j)
    )


def _evolution() -> tuple[list[GaussianState], np.ndarray]:
    """A nontrivial single-mode evolution for trajectory and Wigner plots."""
    state = (
        GaussianState.vacuum(("a",))
        .squeeze("a", r=1.0, theta=0.2)
        .displace("a", 1.5 + 0.4j)
    )
    states = []
    for step in range(17):
        fraction = step / 16
        states.append(
            state.rotate("a", 3.4 * fraction)
            .loss("a", eta=1.0 - 0.6 * fraction)
            .displace("a", 0.5 * np.exp(1j * 2.4 * fraction))
            .squeeze("a", r=0.25 * np.sin(np.pi * fraction), theta=0.6)
        )
    return states, np.linspace(0.0, 4.0, len(states))


class TestStateVisualizations:
    """Static views of representative Gaussian states."""

    @pytest.mark.visualize
    def test_static_visualizations(
        self, assert_no_empty_axes, assert_layout_can_render
    ) -> None:
        state = _complex_state()
        figures = [
            plot_covariance_matrix(state),
            plot_phase_space(state, "a"),
            plot_wigner(state, "a", num_points=50),
            plot_mode_correlation_map(state),
            plot_state_dashboard(state, mode="b"),
        ]

        assert len(figures[0].axes) == 2
        assert len(figures[1].axes) == 1
        assert len(figures[2].axes) == 2
        assert len(figures[3].axes) == 2
        assert len(figures[4].axes) == 7
        for figure in figures:
            assert_no_empty_axes(figure)
            assert_layout_can_render(figure)

    @pytest.mark.visualize
    def test_wigner_analytical_visualization_demo(self) -> None:
        state = wigner_test_state()
        (plot_wigner(state, "a", num_points=50),)

    @pytest.mark.visualize
    def test_phase_space_geometry_is_consistent(self) -> None:
        state = _complex_state()
        figure = plot_phase_space(state, "a")
        ax = figure.axes[0]

        index = state.get_mode_index("a")
        mean = state.displacement[index : index + 2]
        covariance = state.covariance[index : index + 2, index : index + 2]
        values, vectors = np.linalg.eigh(covariance)
        order = np.argsort(values)[::-1]
        values = values[order]
        vector = vectors[:, order[0]]
        expected_angle = np.degrees(np.arctan2(vector[1], vector[0]))
        ellipse = next(patch for patch in ax.patches if hasattr(patch, "angle"))
        np.testing.assert_allclose(ellipse.width, 4.0 * np.sqrt(values[0]))
        np.testing.assert_allclose(ellipse.height, 4.0 * np.sqrt(values[1]))
        np.testing.assert_allclose(ellipse.angle, expected_angle)
        np.testing.assert_allclose(ax.collections[0].get_offsets()[0], mean)
        assert np.isclose(ax.get_aspect(), 1.0)

    @pytest.mark.visualize
    def test_mode_correlation_map_matches_normalized_covariance(
        self, assert_layout_can_render
    ) -> None:
        state = GaussianState.tmsv("a", "b", r=0.9)
        figure = plot_mode_correlation_map(state)
        ax = figure.axes[0]
        actual = np.asarray(ax.images[0].get_array())
        covariance = state.covariance
        expected = covariance / np.sqrt(
            np.outer(np.diag(covariance), np.diag(covariance))
        )

        np.testing.assert_allclose(actual, expected, atol=1e-12)
        np.testing.assert_allclose(np.diag(actual), 1.0, atol=1e-12)
        assert actual[0, 2] > 0.0
        assert actual[1, 3] < 0.0
        assert len(ax.lines) == 2
        assert ax.get_title() == "Mode correlation map"
        assert ax.get_xticklabels()[0].get_text().startswith("x")
        assert_layout_can_render(figure)

    @pytest.mark.visualize
    def test_wigner_and_covariance_are_structurally_well_formed(
        self, assert_layout_can_render
    ) -> None:
        state = _complex_state()
        covariance = plot_covariance_matrix(state)
        wigner = plot_wigner(state, "a", num_points=60)

        assert covariance.axes[0].images[0].get_array().shape == state.covariance.shape
        assert wigner.axes[0].collections
        assert wigner.axes[0].get_xlabel() == "$x$ quadrature"
        assert wigner.axes[0].get_ylabel() == "$p$ quadrature"
        assert_layout_can_render(covariance)
        assert_layout_can_render(wigner)


class TestEvolutionVisualizations:
    """Time-dependent views of nontrivial Gaussian dynamics."""

    @pytest.mark.visualize
    def test_phase_space_evolution_has_shared_geometry(
        self, assert_no_empty_axes, assert_layout_can_render
    ) -> None:
        states, times = _evolution()
        figure = plot_phase_space_trajectory(
            states, "a", times=times, ellipse_every=2, n_sigma=2.0
        )
        ax = figure.axes[0]
        assert len(ax.lines) >= 1
        assert len(ax.patches) >= 8
        assert ax.get_xlabel() == "$x$ quadrature"
        assert ax.get_ylabel() == "$p$ quadrature"
        assert "Phase-space evolution" in ax.get_title()
        np.testing.assert_allclose(ax.get_xlim(), ax.get_ylim())
        assert_no_empty_axes(figure)
        assert_layout_can_render(figure)

    @pytest.mark.visualize
    def test_wigner_covariance_and_diagnostics_evolution(
        self, assert_no_empty_axes, assert_layout_can_render
    ) -> None:
        states, times = _evolution()
        covariance = plot_covariance_evolution(states, "a", times=times)
        wigner = plot_wigner_evolution(
            states, "a", times=times, indices=[0, 8, 16], num_points=40
        )
        diagnostics = plot_diagnostics(states, times=times)

        assert covariance.axes[0].get_xlabel() == "time"
        assert "Covariance evolution" in covariance.axes[0].get_title()
        wigner_axes = [ax for ax in wigner.axes if ax.get_title().startswith("t =")]
        assert len(wigner_axes) == 3
        assert len(wigner.axes) == 4
        assert all("t =" in ax.get_title() for ax in wigner_axes)
        assert diagnostics.axes[0].get_title() == "State diagnostics"

        for left, right in zip(wigner_axes, wigner_axes[1:]):
            np.testing.assert_allclose(left.get_xlim(), right.get_xlim())
            np.testing.assert_allclose(left.get_ylim(), right.get_ylim())
        for figure in (covariance, wigner, diagnostics):
            assert_no_empty_axes(figure)
            assert_layout_can_render(figure)

    @pytest.mark.visualize
    def test_evolution_animation_is_loopable_and_renderable(self, tmp_path) -> None:
        states, times = _evolution()
        animation = animate_phase_space(
            states, "a", times=times, interval=30, repeat=True
        )
        assert animation._repeat is True
        animation._draw_next_frame(0, blit=False)
        animation._draw_next_frame(len(states) - 1, blit=False)
        output = tmp_path / "phase_space.gif"
        animation.save(output, writer="pillow", fps=30)
        assert output.exists()
        assert output.stat().st_size > 0

    @pytest.mark.visualize
    def test_evolution_dashboard_has_no_empty_panels(
        self, assert_no_empty_axes, assert_layout_can_render
    ) -> None:
        states, times = _evolution()
        dashboard = plot_evolution(states, "a", times=times, wigner_indices=[0, 8, 16])

        assert len(dashboard.axes) >= 4
        assert dashboard._suptitle is not None
        assert "mode a" in dashboard._suptitle.get_text()
        assert_no_empty_axes(dashboard)
        assert_layout_can_render(dashboard)

    @pytest.mark.visualize
    def test_multimode_state_dashboard_has_expected_structure(
        self, assert_layout_can_render
    ) -> None:
        dashboard = plot_state_dashboard(_complex_state(), mode="b")
        assert dashboard._suptitle is not None
        assert "a, b" in dashboard._suptitle.get_text()
        assert len(dashboard.axes) == 7
        assert_layout_can_render(dashboard)

    @pytest.mark.visualize
    def test_timecoded_phase_space_has_continuous_time_structure(
        self, assert_layout_can_render
    ) -> None:
        states = [
            GaussianState.coherent(("a",), 0.2),
            GaussianState.coherent(("a",), 0.8 + 0.4j),
            GaussianState.coherent(("a",), 1.2 + 0.8j),
            GaussianState.coherent(("a",), 0.3 + 1.1j),
        ]
        times = [0.0, 0.5, 1.5, 3.0]
        figure = plot_phase_space_trajectory_timecoded(
            states, "a", times=times, ellipse_every=2
        )

        assert figure._suptitle is None
        axes = [axis for axis in figure.axes if axis.get_xlabel() == "$x$ quadrature"]
        assert len(axes) == 1
        ax = axes[0]
        assert len(ax.collections) >= 3
        assert len(ax.patches) >= 2
        assert "Time-coded phase-space evolution" in ax.get_title()
        np.testing.assert_allclose(ax.get_xlim(), ax.get_ylim())
        assert_layout_can_render(figure)

    @pytest.mark.visualize
    def test_timecoded_phase_space_accepts_implicit_steps(
        self, assert_layout_can_render
    ) -> None:
        states = [
            GaussianState.vacuum(("a",)),
            GaussianState.vacuum(("a",)).displace("a", alpha=0.7),
        ]
        figure = plot_phase_space_trajectory_timecoded(states, "a")
        ax = figure.axes[0]
        assert "mode a" in ax.texts[-1].get_text()
        assert figure.axes[1].get_ylabel() == "step"
        assert_layout_can_render(figure)

    def test_timecoded_phase_space_rejects_invalid_times(self) -> None:
        state = GaussianState.vacuum(("a",))
        with pytest.raises(ValueError, match="same length"):
            plot_phase_space_trajectory_timecoded([state, state], "a", times=[0.0])
        with pytest.raises(ValueError, match="finite"):
            plot_phase_space_trajectory_timecoded(
                [state, state], "a", times=[0.0, np.nan]
            )
        with pytest.raises(ValueError, match="monotonically"):
            plot_phase_space_trajectory_timecoded([state, state], "a", times=[1.0, 0.0])


class TestVisualizationValidation:
    """Argument and input validation for visualization helpers."""

    def test_visualization_arguments_are_validated(self) -> None:
        state = GaussianState.vacuum(("a",))

        with pytest.raises(ValueError, match="n_sigma"):
            plot_phase_space(state, "a", n_sigma=0)
        with pytest.raises(ValueError, match="num_points"):
            plot_wigner(state, "a", num_points=1)
        with pytest.raises(ValueError, match="same length"):
            plot_phase_space_trajectory([state], "a", times=[0.0, 1.0])
        with pytest.raises(ValueError, match="positive"):
            animate_phase_space([state], "a", interval=0)
        with pytest.raises(ValueError, match="at least one"):
            plot_phase_space_trajectory([], "a")
        with pytest.raises(ValueError, match="same mode ordering"):
            plot_phase_space_trajectory([state, GaussianState.vacuum(("a", "b"))], "a")
