import numpy as np
import pytest

from catsy import GaussianState
from catsy.visualization_dynamics import plot_phase_space_trajectory_timecoded


@pytest.mark.visualize
def test_timecoded_phase_space_has_continuous_time_structure():
    states = [
        GaussianState.coherent(("a",), 0.2),
        GaussianState.coherent(("a",), 0.8 + 0.4j),
        GaussianState.coherent(("a",), 1.2 + 0.8j),
        GaussianState.coherent(("a",), 0.3 + 1.1j),
    ]
    times = [0.0, 0.5, 1.5, 3.0]

    figure = plot_phase_space_trajectory_timecoded(
        states,
        "a",
        times=times,
        ellipse_every=2,
    )

    assert figure._suptitle is None
    axes = [axis for axis in figure.axes if axis.get_xlabel() == "$x$ quadrature"]
    assert len(axes) == 1
    ax = axes[0]
    assert len(ax.collections) >= 3
    assert len(ax.patches) >= 2
    assert "Time-coded phase-space evolution" in ax.get_title()
    np.testing.assert_allclose(ax.get_xlim(), ax.get_ylim())
    figure.canvas.draw()


@pytest.mark.visualize
def test_timecoded_phase_space_accepts_implicit_steps():
    states = [
        GaussianState.vacuum(("a",)),
        GaussianState.vacuum(("a",)).displace("a", alpha=0.7),
    ]
    figure = plot_phase_space_trajectory_timecoded(states, "a")
    ax = figure.axes[0]
    assert "mode a" in ax.texts[-1].get_text()
    assert figure.axes[1].get_ylabel() == "step"
    figure.canvas.draw()


def test_timecoded_phase_space_rejects_invalid_times():
    state = GaussianState.vacuum(("a",))
    with pytest.raises(ValueError, match="same length"):
        plot_phase_space_trajectory_timecoded([state, state], "a", times=[0.0])
    with pytest.raises(ValueError, match="finite"):
        plot_phase_space_trajectory_timecoded([state, state], "a", times=[0.0, np.nan])
    with pytest.raises(ValueError, match="monotonically"):
        plot_phase_space_trajectory_timecoded([state, state], "a", times=[1.0, 0.0])
