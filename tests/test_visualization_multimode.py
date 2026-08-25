import numpy as np
import pytest

from catsy.gaussian import GaussianState
from catsy.visualization_multimode import plot_multimode_evolution


@pytest.mark.visualize
def test_multimode_evolution_dashboard_has_mode_panels_and_correlation(
    assert_no_empty_axes,
    assert_layout_can_render,
):
    states = []
    for r in np.linspace(0.0, 0.8, 6):
        state = (
            GaussianState.vacuum(("a", "b"))
            .squeeze("a", r=float(r))
            .squeeze("b", r=float(r), theta=np.pi / 2)
            .beam_splitter("a", "b", eta=0.5)
        )
        states.append(state)

    figure = plot_multimode_evolution(states, times=np.linspace(0.0, 1.0, len(states)))

    assert len(figure.axes) == 3
    assert_no_empty_axes(figure)
    assert_layout_can_render(figure)

    correlation_axis = figure.axes[-1]
    assert len(correlation_axis.lines) == 1
    np.testing.assert_allclose(correlation_axis.get_lines()[0].get_xdata(), np.linspace(0.0, 1.0, 6))
    assert np.max(correlation_axis.get_lines()[0].get_ydata()) > 0.0


def test_multimode_evolution_rejects_single_mode():
    state = GaussianState.vacuum(("a",))
    with pytest.raises(ValueError, match="at least two modes"):
        plot_multimode_evolution([state])
