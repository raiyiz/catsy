import matplotlib.pyplot as plt
import numpy as np
import pytest

from catsy import GaussianState
from catsy.core import DUAN_SEPARABILITY_BOUND
from catsy.gaussian import LossChannels, compute_duan_inseparability
from catsy.gaussian.visualization import (
    animate_phase_space,
    plot_evolution,
    plot_joint_correlation,
    plot_multimode_evolution,
    plot_phase_space,
    plot_phase_space_trajectory_timecoded,
    plot_state_dashboard,
    plot_wigner,
)


def _complex_state() -> GaussianState:
    """Representative multimode Gaussian state for showcase plots."""
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
    """Nontrivial single-mode evolution for trajectory and dashboard plots."""
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


def _phase_space_trajectory_states() -> list[GaussianState]:
    return [
        GaussianState.coherent(("a",), 0.2),
        GaussianState.coherent(("a",), 0.8 + 0.4j),
        GaussianState.coherent(("a",), 1.2 + 0.8j),
        GaussianState.coherent(("a",), 0.3 + 1.1j),
    ]


class TestStaticVisualizations:
    """A small gallery of the richest static Gaussian views."""

    @pytest.mark.visualize
    def test_multimode_dashboard_showcase(
        self, assert_no_empty_axes, assert_layout_can_render
    ) -> None:
        figure = plot_state_dashboard(_complex_state(), mode="b")
        assert len(figure.axes) == 7
        assert_no_empty_axes(figure)
        assert_layout_can_render(figure)

    @pytest.mark.visualize
    def test_phase_space_and_wigner_showcase(
        self, assert_no_empty_axes, assert_layout_can_render
    ) -> None:
        state = _complex_state()
        phase_space = plot_phase_space(state, "a")
        wigner = plot_wigner(state, "a", num_points=60)

        assert len(phase_space.axes) == 1
        assert len(wigner.axes) == 2
        assert wigner.axes[0].collections
        assert wigner.axes[0].get_xlabel() == "$x$ quadrature"
        assert wigner.axes[0].get_ylabel() == "$p$ quadrature"
        assert_no_empty_axes(phase_space)
        assert_no_empty_axes(wigner)
        assert_layout_can_render(phase_space)
        assert_layout_can_render(wigner)

    @pytest.mark.visualize
    def test_joint_correlation_distinguishes_entanglement_from_classical_noise(
        self,
        assert_no_empty_axes,
        assert_layout_can_render,
    ) -> None:
        tmsv = GaussianState.tmsv("a", "b", r=1.0)
        classical = LossChannels.correlated_thermal_noise(
            "a", "b", eta=0.3, n_thermal=1.5, c_correlation=1.4
        ).apply(GaussianState.vacuum(("a", "b")))

        fig, axes = plt.subplots(2, 2, figsize=(11, 10), constrained_layout=True)
        for (state, quad), ax in zip(
            [(tmsv, "x"), (tmsv, "p"), (classical, "x"), (classical, "p")],
            axes.flat,
            strict=True,
        ):
            plot_joint_correlation(state, "a", "b", quadrature=quad, ax=ax)

        fig.suptitle(
            f"Genuine entanglement vs classical correlation "
            f"(separability bound = {DUAN_SEPARABILITY_BOUND})"
        )
        assert_no_empty_axes(fig)
        assert_layout_can_render(fig)

        duan_tmsv = compute_duan_inseparability(tmsv, "a", "b")
        duan_classical = compute_duan_inseparability(classical, "a", "b")
        assert duan_tmsv < DUAN_SEPARABILITY_BOUND < duan_classical


class TestEvolutionVisualizations:
    """A few high-information views of nontrivial Gaussian dynamics."""

    @pytest.mark.visualize
    def test_evolution_dashboard_showcase(
        self, assert_no_empty_axes, assert_layout_can_render
    ) -> None:
        states, times = _evolution()
        figure = plot_evolution(states, "a", times=times, wigner_indices=[0, 8, 16])
        assert len(figure.axes) >= 4
        assert figure._suptitle is not None
        assert "mode a" in figure._suptitle.get_text()
        assert_no_empty_axes(figure)
        assert_layout_can_render(figure)

    @pytest.mark.visualize
    def test_multimode_evolution_showcase(
        self, assert_no_empty_axes, assert_layout_can_render
    ) -> None:
        states = [
            (
                GaussianState.vacuum(("a", "b"))
                .squeeze("a", r=float(r))
                .squeeze("b", r=float(r), theta=np.pi / 2)
                .beam_splitter("a", "b", eta=0.5)
            )
            for r in np.linspace(0.0, 0.8, 5)
        ]
        times = np.linspace(0.0, 1.0, len(states))
        figure = plot_multimode_evolution(states, times=times)

        assert len(figure.axes) == 3
        assert_no_empty_axes(figure)
        assert_layout_can_render(figure)
        correlation_axis = figure.axes[-1]
        assert len(correlation_axis.lines) == 1
        np.testing.assert_allclose(correlation_axis.get_lines()[0].get_xdata(), times)
        assert np.max(correlation_axis.get_lines()[0].get_ydata()) > 0.0

    @pytest.mark.visualize
    def test_phase_space_animation_is_renderable(self, tmp_path) -> None:
        states, times = _evolution()
        animation = animate_phase_space(
            states, "a", times=times, interval=30, repeat=False
        )
        assert animation._repeat is False
        animation._draw_next_frame(0, blit=False)
        animation._draw_next_frame(len(states) - 1, blit=False)
        output = tmp_path / "phase_space.gif"
        animation.save(output, writer="pillow", fps=30)
        assert output.exists()
        assert output.stat().st_size > 0

    @pytest.mark.visualize
    def test_timecoded_phase_space_showcase(self, assert_layout_can_render) -> None:
        states = _phase_space_trajectory_states()
        figure = plot_phase_space_trajectory_timecoded(
            states,
            "a",
            times=[0.0, 0.5, 1.5, 3.0],
            ellipse_every=2,
        )
        ax = figure.axes[0]
        assert "Time-coded phase-space evolution" in ax.get_title()
        assert len(ax.collections) >= 3
        assert len(ax.patches) >= 2
        assert figure.axes[1].get_ylabel() == "time"
        assert np.isclose(ax.get_aspect(), 1.0)
        assert_layout_can_render(figure)


class TestVisualizationValidation:
    """Argument and input validation for visualization helpers."""

    @pytest.mark.parametrize(
        "call, match",
        [
            pytest.param(
                lambda state: plot_phase_space(state, "a", n_sigma=0),
                "n_sigma",
                id="nonpositive-sigma",
            ),
            pytest.param(
                lambda state: plot_wigner(state, "a", num_points=1),
                "num_points",
                id="invalid-wigner-grid",
            ),
            pytest.param(
                lambda state: plot_phase_space_trajectory_timecoded(
                    [state], "a", times=[0.0, 1.0]
                ),
                "same length",
                id="timecoded-length",
            ),
            pytest.param(
                lambda state: animate_phase_space([state], "a", interval=0),
                "positive",
                id="animation-interval",
            ),
            pytest.param(
                lambda state: plot_phase_space_trajectory_timecoded([], "a"),
                "at least one",
                id="empty-sequence",
            ),
            pytest.param(
                lambda state: plot_phase_space_trajectory_timecoded(
                    [state, GaussianState.vacuum(("a", "b"))], "a"
                ),
                "same mode ordering",
                id="mixed-mode-sequence",
            ),
        ],
    )
    def test_visualization_arguments_are_validated(self, call, match) -> None:
        with pytest.raises(ValueError, match=match):
            call(GaussianState.vacuum(("a",)))

    @pytest.mark.parametrize(
        "times, match",
        [
            pytest.param([0.0], "same length", id="wrong-length"),
            pytest.param([0.0, np.nan], "finite", id="nonfinite"),
            pytest.param([1.0, 0.0], "monotonically", id="decreasing"),
        ],
    )
    def test_timecoded_phase_space_rejects_invalid_times(self, times, match) -> None:
        state = GaussianState.vacuum(("a",))
        with pytest.raises(ValueError, match=match):
            plot_phase_space_trajectory_timecoded([state, state], "a", times=times)

    def test_multimode_evolution_rejects_single_mode(self) -> None:
        state = GaussianState.vacuum(("a",))
        with pytest.raises(ValueError, match="at least two modes"):
            plot_multimode_evolution([state])
