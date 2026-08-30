import matplotlib.pyplot as plt
import numpy as np
import pytest

from catsy import GaussianState
from catsy.gaussian.visualization import (
    animate_phase_space,
    plot_evolution,
    plot_multimode_evolution,
    plot_phase_space,
    plot_phase_space_trajectory_timecoded,
    plot_wigner,
)


def _complex_state() -> GaussianState:
    """Representative multimode Gaussian state for visualization tests."""
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


def _tmsv_hamiltonian_evolution() -> tuple[list[GaussianState], np.ndarray]:
    """Exact two-mode-squeezing evolution from vacuum under H = iκ(a†b† - ab)."""
    times = np.linspace(0.0, 1.5, 17)
    coupling = 0.7
    states = [GaussianState.tmsv("a", "b", r=coupling * time) for time in times]
    return states, times


class TestEvolutionVisualizations:
    """Contract tests for Gaussian dynamics visualizations."""

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
        states, times = _tmsv_hamiltonian_evolution()
        figure = plt.figure(figsize=(11.5, 9.0), constrained_layout=True)
        grid = figure.add_gridspec(2, 2, height_ratios=(1.1, 0.9), hspace=0.18, wspace=0.16)
        mode_axes = [figure.add_subplot(grid[0, 0]), figure.add_subplot(grid[0, 1])]
        correlation_ax = figure.add_subplot(grid[1, :])

        n_sigma = 2.0
        for mode_index, (ax, mode_name) in enumerate(
            zip(mode_axes, states[0].modes, strict=True)
        ):
            idx = 2 * mode_index
            variances = np.array([state.covariance[idx : idx + 2, idx : idx + 2] for state in states])
            radii = n_sigma * np.sqrt(np.maximum(variances[:, 0, 0], 0.0))
            for frame, radius in enumerate(radii):
                circle = plt.Circle(
                    (0.0, 0.0),
                    float(radius),
                    fill=False,
                    linewidth=2.0 if frame in (0, len(states) - 1) else 0.9,
                    alpha=0.65 if frame in (0, len(states) - 1) else 0.13,
                    edgecolor=("tab:blue" if frame == 0 else "tab:red" if frame == len(states) - 1 else "black"),
                    zorder=3 if frame in (0, len(states) - 1) else 2,
                )
                ax.add_patch(circle)
            ax.scatter(0.0, 0.0, s=38, zorder=4)
            ax.set_xlim(-1.0, float(radii[-1]) * 1.2)
            ax.set_ylim(-1.0, float(radii[-1]) * 1.2)
            ax.set_aspect("equal")
            ax.set_xlabel(r"$x$")
            ax.set_ylabel(r"$p$")
            ax.set_title(f"Mode {mode_name} · local uncertainty", fontweight="medium")

        covariance = np.array([state.covariance for state in states])
        correlation_x = covariance[:, 0, 2] / np.sqrt(covariance[:, 0, 0] * covariance[:, 2, 2])
        correlation_p = covariance[:, 1, 3] / np.sqrt(covariance[:, 1, 1] * covariance[:, 3, 3])
        correlation_ax.plot(times, correlation_x, lw=2.2, label=r"$C_{x_ax_b}$")
        correlation_ax.plot(times, correlation_p, lw=2.2, label=r"$C_{p_ap_b}$")
        correlation_ax.axhline(0.0, lw=0.7, ls="--", alpha=0.35)
        correlation_ax.set_ylim(-1.05, 1.05)
        correlation_ax.set_xlabel("time")
        correlation_ax.set_ylabel("normalized quadrature correlation")
        correlation_ax.set_title("TMSV correlation build-up", fontweight="medium")
        correlation_ax.grid(alpha=0.12, linewidth=0.5)
        correlation_ax.spines[["top", "right"]].set_visible(False)
        correlation_ax.legend(frameon=False, loc="best")

        figure.suptitle(
            r"Two-mode squeezing evolution · $H=i\kappa(a^\dagger b^\dagger-ab)$",
            fontsize=16,
            fontweight="medium",
        )

        assert len(figure.axes) == 3
        assert_no_empty_axes(figure)
        assert_layout_can_render(figure)
        assert len(correlation_ax.lines) == 2
        np.testing.assert_allclose(correlation_ax.get_lines()[0].get_xdata(), times)
        np.testing.assert_allclose(correlation_ax.get_lines()[1].get_xdata(), times)
        assert correlation_ax.get_lines()[0].get_ydata()[-1] > 0.0
        assert correlation_ax.get_lines()[1].get_ydata()[-1] < 0.0

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
        states, times = _evolution()
        figure = plot_phase_space_trajectory_timecoded(
            states,
            "a",
            times=times,
            ellipse_every=1,
        )
        ax = figure.axes[0]
        ax.set_title(
            "Driven, lossy squeezed-state evolution — mode a",
            pad=16,
            fontweight="medium",
        )
        assert len(ax.collections) >= 3
        assert len(ax.patches) == len(states)
        assert figure.axes[1].get_ylabel() == "time"
        assert np.isclose(ax.get_aspect(), 1.0)

        for ellipse in ax.patches:
            ellipse.set_alpha(0.10)
            ellipse.set_linewidth(0.8)

        initial_ellipse = ax.patches[0]
        final_ellipse = ax.patches[-1]
        initial_ellipse.set_alpha(0.55)
        initial_ellipse.set_linewidth(2.0)
        initial_ellipse.set_edgecolor("tab:blue")
        final_ellipse.set_alpha(0.55)
        final_ellipse.set_linewidth(2.0)
        final_ellipse.set_edgecolor("tab:red")

        initial_marker = ax.collections[2]
        final_marker = ax.collections[3]
        initial_marker.set_facecolor("tab:blue")
        initial_marker.set_edgecolor("white")
        initial_marker.set_sizes([90])
        final_marker.set_facecolor("tab:red")
        final_marker.set_edgecolor("white")
        final_marker.set_sizes([120])

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
