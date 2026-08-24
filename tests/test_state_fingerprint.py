import numpy as np
import pytest

from catsy import GaussianState
from catsy.state_fingerprint import plot_state_fingerprint


def _symplectic_eigenvalues(covariance: np.ndarray) -> np.ndarray:
    n = covariance.shape[0] // 2
    omega = np.kron(np.eye(n), np.array([[0.0, 1.0], [-1.0, 0.0]]))
    values = np.linalg.eigvals(1j * omega @ covariance).real
    return np.sort(np.abs(values))[::2]


@pytest.mark.visualize
def test_state_fingerprint_matches_state_invariants() -> None:
    state = (
        GaussianState.tmsv("a", "b", r=0.85)
        .squeeze("a", r=0.35, theta=0.2)
        .rotate("b", -0.4)
        .displace("a", 0.8 + 0.3j)
        .displace("b", -0.4 + 0.6j)
        .loss("b", eta=0.72)
    )
    figure = plot_state_fingerprint(state)
    figure.canvas.draw()

    axes = figure.axes[0]
    assert axes.get_title() == "State fingerprint"
    table = next(
        artist for artist in axes.get_children() if artist.__class__.__name__ == "Table"
    )
    cells = table.get_celld()
    displayed = {
        cells[(row, 0)].get_text().get_text(): cells[(row, 1)].get_text().get_text()
        for row in range(1, 6 + len(state.modes))
        if (row, 0) in cells and (row, 1) in cells
    }

    purity = 1.0 / (
        2.0**len(state.modes) * np.sqrt(np.linalg.det(state.covariance))
    )
    symplectic = _symplectic_eigenvalues(state.covariance)

    assert displayed["modes"] == "a, b"
    assert displayed["dimension"] == "4"
    assert displayed["purity"] == f"{purity:.4f}"
    assert displayed["det(V)"] == f"{np.linalg.det(state.covariance):.4g}"
    assert displayed["symplectic spectrum"] == "(" + ", ".join(f"{value:.4f}" for value in symplectic) + ")"
    for index, mode in enumerate(state.modes):
        start = 2 * index
        expected = f"({state.displacement[start]:.3f}, {state.displacement[start + 1]:.3f})"
        assert displayed[f"d[{mode}]"] == expected


@pytest.mark.visualize
def test_state_fingerprint_uses_single_structural_panel() -> None:
    figure = plot_state_fingerprint(
        GaussianState.coherent(("a", "b"), [0.4 + 0.1j, -0.2 + 0.3j])
    )
    figure.canvas.draw()

    assert len(figure.axes) == 1
    assert figure.axes[0].get_visible()
