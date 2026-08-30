"""Contract tests for shared visualization scaling primitives."""

import matplotlib.pyplot as plt
import numpy as np
import qutip as qt
from matplotlib.colors import Normalize

from catsy.fock.visualization import plot_wigner
from catsy.visualization import color_norm, normalize_probabilities


def test_normalize_probabilities_preserves_relative_weights():
    probabilities = normalize_probabilities([1.0, 2.0, 3.0])

    np.testing.assert_allclose(probabilities, [1 / 6, 2 / 6, 3 / 6])
    np.testing.assert_allclose(probabilities.sum(), 1.0)


def test_color_norm_can_be_shared_without_rescaling_data():
    first = np.array([-1.0, 0.0, 2.0])
    second = np.array([-3.0, 0.0, 4.0])

    norm = color_norm(np.concatenate([first, second]), symmetric=True)

    assert isinstance(norm, Normalize)
    assert norm.vmin == -4.0
    assert norm.vmax == 4.0
    np.testing.assert_array_equal(first, [-1.0, 0.0, 2.0])


def test_color_norm_accepts_explicit_limits():
    norm = color_norm([0.0, 1.0], vmin=-2.0, vmax=3.0)

    assert norm.vmin == -2.0
    assert norm.vmax == 3.0


def test_wigner_uses_supplied_shared_norm_and_optional_colorbar():
    state = qt.fock_dm(12, 2)
    norm = Normalize(vmin=-0.25, vmax=0.25)

    figure = plt.figure(figsize=(10.0, 4.0), constrained_layout=True)
    axes = [figure.add_subplot(1, 2, index) for index in (1, 2)]
    plot_wigner(state, xlim=(-4, 4), resolution=48, ax=axes[0], norm=norm, colorbar=False)
    plot_wigner(state, xlim=(-4, 4), resolution=48, ax=axes[1], norm=norm, colorbar=True)

    assert len(figure.axes) == 3
    assert axes[0].images[-1].norm is norm
    assert axes[1].images[-1].norm is norm

    plt.close(figure)
