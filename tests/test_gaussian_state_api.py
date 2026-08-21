import numpy as np
import pytest

from catsy.core import Circuit
from catsy.gaussian import GaussianState, beam_splitter, loss, squeeze


def test_gaussian_state_constructors_and_fluent_transformations():
    vacuum = GaussianState.vacuum(("a",))
    coherent = GaussianState.coherent(("a",), 0.7 + 0.2j)

    squeezed = (
        vacuum.squeeze("a", r=0.5).rotate("a", phi=0.25).displace("a", alpha=0.3 + 0.1j)
    )

    assert vacuum is not squeezed
    np.testing.assert_allclose(coherent.covariance, vacuum.covariance)
    assert not np.array_equal(squeezed.displacement, vacuum.displacement)
    assert squeezed.modes == ("a",)


def test_tmsv_is_the_two_mode_squeezed_vacuum_constructor():
    state = GaussianState.tmsv("a", "b", r=0.7)

    assert state.modes == ("a", "b")
    assert np.isclose(np.linalg.det(state.covariance), 1 / 16, rtol=1e-10)
    assert state.covariance[0, 2] > 0
    assert state.covariance[1, 3] < 0


def test_circuit_matches_fluent_state_chain():
    direct = (
        GaussianState.vacuum(("a", "b"))
        .squeeze("a", r=0.4)
        .squeeze("b", r=0.4, theta=np.pi / 2)
        .beam_splitter("a", "b", eta=0.5)
        .loss("a", eta=0.8)
    )

    circuit = Circuit().add_mode("a").add_mode("b")
    circuit.add_operation(squeeze, ("a",), r=0.4, theta=0.0).add_operation(
        squeeze, ("b",), r=0.4, theta=np.pi / 2
    ).add_operation(beam_splitter, ("a", "b"), eta=0.5).add_operation(
        loss, ("a",), eta=0.8
    )
    compiled = circuit.run(GaussianState.vacuum(("a", "b")))

    np.testing.assert_allclose(compiled.displacement, direct.displacement)
    np.testing.assert_allclose(compiled.covariance, direct.covariance)


def test_repr_reports_modes_and_purity():
    # A pure two-mode state (squeeze + lossless beam splitter) must report
    # purity~1.000; this also exercises purity() on a non-trivial covariance
    # rather than only ever being read as a side effect of debugging.
    pure = (
        GaussianState.vacuum(("a", "b"))
        .squeeze("a", r=0.6)
        .beam_splitter("a", "b", eta=0.5)
    )
    text = repr(pure)
    assert "GaussianState" in text
    assert "modes=('a', 'b')" in text
    assert "purity~1.000" in text

    # Loss strictly reduces purity below 1, so the printed value must move
    # off the pure-state baseline too, not just be present.
    lossy = pure.loss("a", eta=0.5)
    assert "purity~1.000" not in repr(lossy)


@pytest.mark.visual
def test_plot_covariance_renders_without_error():
    state = (
        GaussianState.vacuum(("a", "b"))
        .squeeze("a", r=0.7)
        .beam_splitter("a", "b", eta=0.5)
    )
    state.plot_covariance()
