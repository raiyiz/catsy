import numpy as np

from catsy.gaussian import GaussianCircuit, GaussianState


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

    circuit = (
        GaussianCircuit()
        .add_mode("a")
        .add_mode("b")
        .squeeze("a", r=0.4)
        .squeeze("b", r=0.4, theta=np.pi / 2)
        .beam_splitter("a", "b", eta=0.5)
        .loss("a", eta=0.8)
    )
    compiled = circuit.compile_and_run()

    np.testing.assert_allclose(compiled.displacement, direct.displacement)
    np.testing.assert_allclose(compiled.covariance, direct.covariance)
