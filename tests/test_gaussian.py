import numpy as np
import pytest
import qutip as qt
from matplotlib import pyplot as plt

from catsy.core import DUAN_SEPARABILITY_BOUND, _williamson_decomposition
from catsy.gaussian import (
    GaussianChannel,
    GaussianCircuit,
    GaussianMeasurements,
    GaussianState,
    LossChannels,
    compute_duan_inseparability,
    compute_joint_correlation,
    compute_wigner_analytically,
    plot_joint_correlation,
    plot_wigner,
)

# Analytic primitive checks


@pytest.mark.parametrize("r", [0.0, 0.25, 0.8, 1.2])
def test_squeezing_has_expected_principal_variances(r):
    state = GaussianState.vacuum(("a",))
    squeezed = state.squeeze("a", r=r, theta=0.0)

    expected = 0.5 * np.diag([np.exp(-2 * r), np.exp(2 * r)])
    np.testing.assert_allclose(squeezed.covariance, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize(
    ("phi", "alpha"),
    [
        (0.0, 0.7 + 0.2j),
        (np.pi / 2, 0.7 + 0.2j),
        (np.pi, 0.7 + 0.2j),
        (-0.3, -0.4 + 0.9j),
    ],
)
def test_phase_rotation_rotates_displacement(phi, alpha):
    state = GaussianState.coherent(("a",), alpha)
    rotated = state.rotate("a", phi=phi)
    x, p = state.displacement
    expected = np.array(
        [
            np.cos(phi) * x - np.sin(phi) * p,
            np.sin(phi) * x + np.cos(phi) * p,
        ]
    )
    np.testing.assert_allclose(rotated.displacement, expected, atol=1e-12)
    np.testing.assert_allclose(rotated.covariance, state.covariance, atol=1e-12)


@pytest.mark.parametrize("eta", [0.0, 0.2, 0.5, 0.9, 1.0])
def test_loss_matches_vacuum_environment_formula(eta):
    state = GaussianState.vacuum(("a",)).squeeze("a", r=0.6)
    lossy = state.loss("a", eta=eta)
    expected = eta * state.covariance + (1.0 - eta) * 0.5 * np.eye(2)
    np.testing.assert_allclose(lossy.covariance, expected, atol=1e-12)
    np.testing.assert_allclose(lossy.displacement, np.sqrt(eta) * state.displacement)


@pytest.mark.parametrize(
    ("eta", "n_thermal"),
    [(0.0, 0.0), (0.25, 0.5), (0.8, 1.2), (1.0, 2.0)],
)
def test_thermal_loss_matches_environment_formula(eta, n_thermal):
    state = GaussianState.coherent(("a",), 0.4 - 0.2j).squeeze("a", r=0.5, theta=0.2)
    lossy = LossChannels.thermal_loss(mode="a", eta=eta, n_thermal=n_thermal).apply(state)
    expected = eta * state.covariance + (1.0 - eta) * (n_thermal + 0.5) * np.eye(2)
    np.testing.assert_allclose(lossy.covariance, expected, atol=1e-12)
    np.testing.assert_allclose(lossy.displacement, np.sqrt(eta) * state.displacement)


@pytest.mark.parametrize("eta", [0.0, 0.25, 0.5, 0.75, 1.0])
def test_beam_splitter_has_expected_coherent_amplitude_map(eta):
    alpha_a, alpha_b = 0.8 + 0.1j, -0.3 + 0.6j
    state = GaussianState.coherent(("a", "b"), [alpha_a, alpha_b])
    mixed = state.beam_splitter("a", "b", eta=eta)
    t, r = np.sqrt(eta), np.sqrt(1.0 - eta)
    expected = np.array(
        [
            np.sqrt(2.0) * np.real(t * alpha_a + r * alpha_b),
            np.sqrt(2.0) * np.imag(t * alpha_a + r * alpha_b),
            np.sqrt(2.0) * np.real(-r * alpha_a + t * alpha_b),
            np.sqrt(2.0) * np.imag(-r * alpha_a + t * alpha_b),
        ]
    )
    np.testing.assert_allclose(mixed.displacement, expected, atol=1e-12)
    np.testing.assert_allclose(mixed.covariance, 0.5 * np.eye(4), atol=1e-12)


# States and operations


def test_vacuum_is_shot_noise_limited(two_mode_vacuum):
    state = two_mode_vacuum
    assert state.displacement.shape == (4,)
    np.testing.assert_allclose(state.covariance, 0.5 * np.eye(4))


def test_squeezing_preserves_purity_determinant(single_mode_vacuum):
    state = single_mode_vacuum
    squeezed = state.squeeze("a", r=0.8, theta=0.3)
    assert np.linalg.det(squeezed.covariance) == pytest.approx(0.25, rel=1e-9)

    unrotated = state.squeeze("a", r=0.8, theta=0.0)
    assert unrotated.covariance[0, 0] < 0.5
    assert unrotated.covariance[1, 1] > 0.5


def test_beam_splitter_entangles_independent_modes(two_mode_vacuum):
    state = two_mode_vacuum.squeeze("a", r=0.6, theta=0.0).squeeze(
        "b", r=0.6, theta=np.pi / 2
    )

    np.testing.assert_allclose(state.covariance[0:2, 2:4], np.zeros((2, 2)), atol=1e-12)
    mixed = state.beam_splitter("a", "b", eta=0.5)
    assert np.abs(mixed.covariance[0:2, 2:4]).max() > 1e-6


@pytest.mark.parametrize("eta", [-0.1, 1.1])
def test_beam_splitter_rejects_invalid_eta(eta):
    state = GaussianState.vacuum(("a", "b"))
    with pytest.raises(ValueError):
        state.beam_splitter("a", "b", eta=eta)


def test_beam_splitter_rejects_duplicate_modes():
    state = GaussianState.vacuum(("a", "b"))
    with pytest.raises(ValueError):
        state.beam_splitter("a", "a", eta=0.5)


def test_loss_reduces_toward_vacuum(single_mode_vacuum):
    state = single_mode_vacuum.squeeze("a", r=1.0, theta=0.0)
    lossy = state.loss("a", eta=0.0)
    np.testing.assert_allclose(lossy.covariance, 0.5 * np.eye(2), atol=1e-9)


def test_displacement_shifts_mean_leaves_covariance_untouched(single_mode_vacuum):
    squeezed = single_mode_vacuum.squeeze("a", r=0.4, theta=0.2)
    displaced = squeezed.displace("a", alpha=1.0)
    np.testing.assert_allclose(displaced.covariance, squeezed.covariance)
    np.testing.assert_allclose(
        displaced.displacement - squeezed.displacement, [np.sqrt(2.0), 0.0]
    )


def test_displacement_alpha_and_xp_are_equivalent(single_mode_vacuum):
    state = single_mode_vacuum
    alpha = 0.6 - 0.9j
    via_alpha = state.displace("a", alpha=alpha)
    via_xp = state.displace("a", x=np.sqrt(2.0) * alpha.real, p=np.sqrt(2.0) * alpha.imag)
    np.testing.assert_allclose(via_alpha.displacement, via_xp.displacement)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"alpha": 1.0, "x": 1.0}, "not both"),
        ({}, "alpha"),
        ({"x": 1.0}, "p"),
    ],
)
def test_displacement_rejects_invalid_argument_combinations(kwargs, match):
    state = GaussianState.vacuum(("a",))
    with pytest.raises(ValueError, match=match):
        state.displace("a", **kwargs)


def test_coherent_matches_displaced_vacuum():
    alpha = 1.2 + 0.4j
    coherent = GaussianState.coherent(("a",), alpha)
    manual = GaussianState.vacuum(("a",)).displace("a", alpha=alpha)
    np.testing.assert_allclose(coherent.displacement, manual.displacement)
    np.testing.assert_allclose(coherent.covariance, 0.5 * np.eye(2))


def test_coherent_broadcasts_scalar_alpha_across_modes():
    state = GaussianState.coherent(("a", "b"), 1.0j)
    np.testing.assert_allclose(state.displacement[0:2], state.displacement[2:4])


def test_coherent_rejects_mismatched_alpha_count():
    with pytest.raises(ValueError):
        GaussianState.coherent(("a", "b"), [1.0])


def test_coherent_state_mean_photon_number_matches_alpha_squared():
    alpha = 1.5 + 0.7j
    state = GaussianState.coherent(("a",), alpha)
    rho = state.to_qutip(N_cutoff=25)
    mean_n = qt.expect(qt.num(25), rho)
    assert mean_n == pytest.approx(np.abs(alpha) ** 2, rel=1e-3)


def test_circuit_displace_matches_manual_displacement():
    manual = (
        GaussianState.vacuum(("a",)).squeeze("a", r=0.3).displace("a", alpha=0.5 - 0.2j)
    )

    circuit = GaussianCircuit().add_mode("a")
    circuit.squeeze(mode="a", r=0.3).displace(mode="a", alpha=0.5 - 0.2j)
    compiled = circuit.compile_and_run()

    np.testing.assert_allclose(compiled.displacement, manual.displacement, atol=1e-10)
    np.testing.assert_allclose(compiled.covariance, manual.covariance, atol=1e-10)


def test_circuit_add_mode_with_alpha_seeds_coherent_starting_state():
    circuit = GaussianCircuit().add_mode("c", alpha=1.0 + 1.0j)
    compiled = circuit.compile_and_run()
    expected = GaussianState.coherent(("c",), 1.0 + 1.0j)
    np.testing.assert_allclose(compiled.displacement, expected.displacement)

    overridden = circuit.compile_and_run(initial_state=GaussianState.vacuum(("c",)))
    np.testing.assert_allclose(overridden.displacement, np.zeros(2))


def test_get_mode_index_unknown_mode_raises(two_mode_vacuum):
    state = two_mode_vacuum
    with pytest.raises(ValueError):
        state.get_mode_index("z")


# Mode ordering and circuits


def test_duplicate_mode_names_rejected():
    with pytest.raises(ValueError):
        GaussianState(
            modes=("a", "a"),
            displacement=np.zeros(4),
            covariance=0.5 * np.eye(4),
        )


def test_state_reorder_modes_preserves_physical_state(two_mode_vacuum):
    state = two_mode_vacuum.displace("a", alpha=0.7 + 0.2j).squeeze("b", r=0.4, theta=0.3)

    reordered = state.reorder_modes(("b", "a"))
    roundtrip = reordered.reorder_modes(("a", "b"))
    assert reordered.modes == ("b", "a")
    np.testing.assert_allclose(roundtrip.displacement, state.displacement)
    np.testing.assert_allclose(roundtrip.covariance, state.covariance)


def test_state_reorder_modes_rejects_wrong_mode_set(two_mode_vacuum):
    with pytest.raises(ValueError, match="exactly the state's modes"):
        two_mode_vacuum.reorder_modes(("a", "c"))


def test_circuit_canonicalizes_initial_state_mode_order():
    initial = GaussianState.coherent(("b", "a"), alphas=[0.0 + 1.0j, 1.0 + 0.0j])

    circuit = GaussianCircuit().add_mode("a").add_mode("b")
    result = circuit.compile_and_run(initial_state=initial)
    assert result.modes == ("a", "b")

    expected = GaussianState.coherent(("a", "b"), alphas=[1.0 + 0.0j, 0.0 + 1.0j])
    np.testing.assert_allclose(result.displacement, expected.displacement)
    np.testing.assert_allclose(result.covariance, expected.covariance)


def test_classical_phase_jitter_channel_applies():
    state = GaussianState.vacuum(modes=("a",))
    channel = LossChannels.classical_phase_jitter(mode="a", sigma_phi=0.3)
    jittered = channel.apply(state)
    assert jittered.covariance[0, 0] == pytest.approx(0.5)
    assert jittered.covariance[1, 1] > 0.5


def test_gaussian_state_rejects_unphysical_covariance():
    with pytest.raises(ValueError, match="uncertainty relation"):
        GaussianState(
            modes=("a",),
            displacement=np.zeros(2),
            covariance=0.1 * np.eye(2),
        )


# Channels


def test_gaussian_state_rejects_nonsymmetric_covariance():
    covariance = np.array([[0.5, 0.1], [0.0, 0.5]])
    with pytest.raises(ValueError, match="symmetric"):
        GaussianState(
            modes=("a",),
            displacement=np.zeros(2),
            covariance=covariance,
        )


def test_gaussian_state_rejects_nonfinite_values():
    covariance = np.diag([np.inf, np.inf])
    with pytest.raises(ValueError, match="finite"):
        GaussianState(
            modes=("a",),
            displacement=np.zeros(2),
            covariance=covariance,
        )


def test_gaussian_channel_rejects_non_cp_channel():
    with pytest.raises(ValueError, match="complete positivity"):
        GaussianChannel(
            target_modes=("a",),
            X=2.0 * np.eye(2),
            Y=np.zeros((2, 2)),
            d0=np.zeros(2),
        )


def test_gaussian_channel_rejects_nonsymmetric_noise():
    with pytest.raises(ValueError, match="symmetric"):
        GaussianChannel(
            target_modes=("a",),
            X=np.eye(2),
            Y=np.array([[0.1, 0.2], [0.0, 0.1]]),
            d0=np.zeros(2),
        )


def test_thermal_loss_is_a_valid_gaussian_channel():
    channel = LossChannels.thermal_loss(
        mode="a",
        eta=0.7,
        n_thermal=0.2,
    )
    assert channel.X.shape == (2, 2)


@pytest.mark.parametrize(
    ("n_thermal", "c_correlation", "valid"),
    [
        (0.5, 0.5, True),
        (0.5, -0.5, True),
        (0.5, 0.500001, False),
        (0.0, 1e-6, False),
    ],
)
def test_correlated_thermal_noise_validates_correlation(n_thermal, c_correlation, valid):
    if valid:
        channel = LossChannels.correlated_thermal_noise(
            "a",
            "b",
            eta=0.5,
            n_thermal=n_thermal,
            c_correlation=c_correlation,
        )
        assert channel.Y.shape == (4, 4)
    else:
        with pytest.raises(ValueError, match="c_correlation"):
            LossChannels.correlated_thermal_noise(
                "a",
                "b",
                eta=0.5,
                n_thermal=n_thermal,
                c_correlation=c_correlation,
            )


# Circuit validation and serialization


def test_circuit_matches_manual_operation_chain():
    manual = (
        GaussianState.vacuum(("a", "b"))
        .squeeze("a", r=0.6, theta=0.0)
        .squeeze("b", r=0.6, theta=np.pi / 2)
        .beam_splitter("a", "b", eta=0.5)
    )
    manual = LossChannels.thermal_loss(mode="b", eta=0.7, n_thermal=0.3).apply(manual)

    circuit = GaussianCircuit()
    circuit.add_mode("a").add_mode("b")
    circuit.squeeze(mode="a", r=0.6, theta=0.0).squeeze(
        mode="b", r=0.6, theta=np.pi / 2
    ).beam_splitter(mode_a="a", mode_b="b", eta=0.5).thermal_loss(
        mode="b", eta=0.7, n_thermal=0.3
    )
    compiled = circuit.compile_and_run()

    np.testing.assert_allclose(compiled.displacement, manual.displacement, atol=1e-10)
    np.testing.assert_allclose(compiled.covariance, manual.covariance, atol=1e-10)


def test_circuit_executes_the_callable_directly():
    calls = []

    def my_operation(state, modes, **kwargs):
        calls.append((modes, kwargs))
        return state.rotate(modes[0], phi=kwargs["phi"])

    circuit = GaussianCircuit().add_mode("a")
    circuit.add_operation(my_operation, ("a",), phi=0.25)
    result = circuit.compile_and_run()

    assert calls == [(('a',), {"phi": 0.25})]
    expected = GaussianState.vacuum(("a",)).rotate("a", phi=0.25)
    np.testing.assert_allclose(result.displacement, expected.displacement)
    np.testing.assert_allclose(result.covariance, expected.covariance)


def test_circuit_rejects_unregistered_mode():
    circuit = GaussianCircuit()
    circuit.add_mode("a")
    circuit.squeeze(mode="z", r=0.5)
    with pytest.raises(ValueError):
        circuit.compile_and_run()


def test_circuit_rejects_empty_mode_set():
    with pytest.raises(ValueError):
        GaussianCircuit().compile_and_run()


def test_circuit_serializes_operation_function_name():
    circuit = GaussianCircuit().add_mode("a")
    circuit.squeeze(mode="a", r=0.4)

    assert circuit.to_dict()["operations"] == [
        {"name": "squeeze", "modes": ["a"], "kwargs": {"r": 0.4, "theta": 0.0}}
    ]


def test_circuit_register_is_only_for_custom_operation_deserialization():
    def my_operation(state, modes, **kwargs):
        return state

    GaussianCircuit.register("my_operation", my_operation)
    try:
        circuit = GaussianCircuit().add_mode("a")
        circuit.add_operation(my_operation, ("a",), foo=1)
        restored = GaussianCircuit.from_dict(circuit.to_dict())
        assert restored.to_dict() == circuit.to_dict()
    finally:
        from catsy.gaussian import _OPERATION_DESERIALIZERS

        del _OPERATION_DESERIALIZERS["my_operation"]


def test_gaussian_state_roundtrips_through_dict():
    state = GaussianState.vacuum(modes=("a", "b")).squeeze("a", r=0.4)
    restored = GaussianState.from_dict(state.to_dict())
    assert restored.modes == state.modes
    np.testing.assert_allclose(restored.displacement, state.displacement)
    np.testing.assert_allclose(restored.covariance, state.covariance)


def test_gaussian_state_roundtrips_through_file(tmp_path):
    state = GaussianState.vacuum(modes=("a",)).squeeze("a", r=0.9, theta=0.2)
    path = tmp_path / "state.json"
    state.save(path)
    restored = GaussianState.load(path)
    np.testing.assert_allclose(restored.covariance, state.covariance)


# Measurements


def test_circuit_roundtrips_through_file(tmp_path):
    circuit = GaussianCircuit()
    circuit.add_mode("a").add_mode("b")
    circuit.squeeze(mode="a", r=0.6).squeeze(
        mode="b", r=0.6, theta=np.pi / 2
    ).beam_splitter(mode_a="a", mode_b="b", eta=0.5)
    path = tmp_path / "circuit.json"
    circuit.save(path)
    restored = GaussianCircuit.load(path)
    original_result = circuit.compile_and_run()
    restored_result = restored.compile_and_run()
    np.testing.assert_allclose(restored_result.covariance, original_result.covariance)


def test_circuit_roundtrips_seeded_coherent_alpha_through_file(tmp_path):
    circuit = GaussianCircuit()
    circuit.add_mode("a", alpha=0.5 + 1.3j).add_mode("b")
    circuit.displace(mode="b", x=0.2, p=-0.4)
    path = tmp_path / "circuit.json"
    circuit.save(path)
    restored = GaussianCircuit.load(path)
    original_result = circuit.compile_and_run()
    restored_result = restored.compile_and_run()
    np.testing.assert_allclose(restored_result.displacement, original_result.displacement)


def test_homodyne_measurement_collapses_tmsv_correlation():
    circuit = GaussianCircuit()
    circuit.add_mode("a").add_mode("b")
    circuit.squeeze(mode="a", r=1.0).squeeze(
        mode="b", r=1.0, theta=np.pi / 2
    ).beam_splitter(mode_a="a", mode_b="b", eta=0.5)
    state = circuit.compile_and_run()
    val, collapsed = GaussianMeasurements.homodyne_measurement(
        state, measured_mode="a", phi=0.0, outcome=2.5
    )
    assert val == 2.5
    assert collapsed.modes == ("b",)

    _val_neg, collapsed_neg = GaussianMeasurements.homodyne_measurement(
        state, measured_mode="a", phi=0.0, outcome=-2.5
    )
    assert np.sign(collapsed.displacement[0]) != np.sign(collapsed_neg.displacement[0])


def test_homodyne_measurement_is_reproducible_with_seeded_rng():
    state = GaussianState.vacuum(modes=("a",))
    v1, _ = GaussianMeasurements.homodyne_measurement(
        state, measured_mode="a", phi=0.0, rng=np.random.default_rng(42)
    )
    v2, _ = GaussianMeasurements.homodyne_measurement(
        state, measured_mode="a", phi=0.0, rng=np.random.default_rng(42)
    )
    assert v1 == v2


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"phi": np.nan}, "phi must be finite"),
        ({"phi": 0.0, "outcome": np.inf}, "finite scalar"),
    ],
)
def test_homodyne_rejects_nonfinite_inputs(kwargs, match):
    state = GaussianState.vacuum(("a",))
    with pytest.raises(ValueError, match=match):
        GaussianMeasurements.homodyne_measurement(state, measured_mode="a", **kwargs)


def test_homodyne_single_mode_returns_valid_empty_state():
    state = GaussianState.coherent(modes=("a",), alphas=0.7 + 0.2j)
    outcome, collapsed = GaussianMeasurements.homodyne_measurement(
        state, measured_mode="a", phi=0.0, outcome=1.25
    )

    assert outcome == pytest.approx(1.25)
    assert collapsed.modes == ()
    assert collapsed.displacement.shape == (0,)
    assert collapsed.covariance.shape == (0, 0)


# Phase-space analysis


@pytest.mark.parametrize(
    ("outcome", "match"),
    [
        (np.array([1.0]), r"shape \(2,\)"),
        (np.array([1.0, np.nan]), "finite values"),
    ],
)
def test_heterodyne_rejects_invalid_outcomes(outcome, match):
    state = GaussianState.vacuum(("a",))
    with pytest.raises(ValueError, match=match):
        GaussianMeasurements.heterodyne_measurement(
            state, measured_mode="a", outcome=outcome
        )


def test_heterodyne_single_mode_returns_valid_empty_state():
    state = GaussianState.vacuum(modes=("a",))
    outcome, collapsed = GaussianMeasurements.heterodyne_measurement(
        state, measured_mode="a", outcome=np.array([0.2, -0.3])
    )

    np.testing.assert_allclose(outcome, [0.2, -0.3])
    assert collapsed.modes == ()
    assert collapsed.displacement.shape == (0,)
    assert collapsed.covariance.shape == (0, 0)


def test_wigner_analytical_matches_gaussian_normalization(plot_enabled):
    circuit = GaussianCircuit()
    circuit.add_mode("a")
    circuit.squeeze(mode="a", r=1.1, theta=30.0)
    test_state = circuit.compile_and_run()
    test_state.displacement[0] = 2.0
    test_state.displacement[1] = 3.0

    W, X, P, M = compute_wigner_analytically(
        test_state, mode_name="a", x_max=8.0, num_points=200
    )
    dx = (2 * 8.0) / 199
    integral = W.sum() * dx * dx
    assert integral == pytest.approx(1.0, rel=1e-2)
    if plot_enabled:
        plot_wigner(W, X, P, M)


def test_joint_correlation_computes_valid_grid(plot_enabled):
    circuit = GaussianCircuit()
    circuit.add_mode("a").add_mode("b")
    circuit.squeeze(mode="a", r=0.6).squeeze(mode="b", r=0.6, theta=np.pi).beam_splitter(
        mode_a="a", mode_b="b", eta=0.5
    )
    cv_state = circuit.compile_and_run()
    P, X_a, X_b, mode_a, mode_b = compute_joint_correlation(cv_state, "a", "b")
    assert P.shape == (150, 150)
    assert np.all(P >= 0)
    if plot_enabled:
        plot_joint_correlation(P, X_a, X_b, mode_a, mode_b)


def test_joint_correlation_rejects_invalid_quadrature():
    state = GaussianState.vacuum(modes=("a", "b"))
    with pytest.raises(ValueError):
        compute_joint_correlation(state, "a", "b", quadrature="z")


def test_joint_correlation_x_correlated_p_anticorrelated_for_tmsv():
    state = GaussianState.tmsv("a", "b", r=1.0)
    P_x, Xa_x, Xb_x, _, _ = compute_joint_correlation(
        state, "a", "b", x_max=6.0, quadrature="x"
    )
    P_p, Xa_p, Xb_p, _, _ = compute_joint_correlation(
        state, "a", "b", x_max=6.0, quadrature="p"
    )
    dx = Xa_x[0, 1] - Xa_x[0, 0]

    empirical_cov_x = np.sum(Xa_x * Xb_x * P_x) * dx * dx
    empirical_cov_p = np.sum(Xa_p * Xb_p * P_p) * dx * dx
    assert empirical_cov_x == pytest.approx(state.covariance[0, 2], rel=5e-3)
    assert empirical_cov_p == pytest.approx(state.covariance[1, 3], rel=5e-3)
    assert empirical_cov_x > 0
    assert empirical_cov_p < 0


def test_tmsv_matches_manual_squeeze_squeeze_bs():
    manual = (
        GaussianState.vacuum(("a", "b"))
        .squeeze("a", r=0.8, theta=0.0)
        .squeeze("b", r=0.8, theta=np.pi / 2)
        .beam_splitter("a", "b", eta=0.5)
    )
    tmsv = GaussianState.tmsv("a", "b", r=0.8)
    np.testing.assert_allclose(tmsv.displacement, manual.displacement)
    np.testing.assert_allclose(tmsv.covariance, manual.covariance)


def test_duan_witness_independent_vacua_saturate_separability_bound():
    state = GaussianState.vacuum(modes=("a", "b"))
    witness = compute_duan_inseparability(state, "a", "b")
    assert witness == pytest.approx(DUAN_SEPARABILITY_BOUND)


def test_duan_witness_confirms_genuine_entanglement_for_tmsv():
    r = 1.0
    tmsv = GaussianState.tmsv("a", "b", r=r)
    witness = compute_duan_inseparability(tmsv, "a", "b")
    assert witness == pytest.approx(2.0 * np.exp(-2.0 * r), rel=1e-6)
    assert witness < DUAN_SEPARABILITY_BOUND


def test_duan_witness_strengthens_with_more_squeezing():
    weak = compute_duan_inseparability(GaussianState.tmsv("a", "b", r=0.3), "a", "b")
    strong = compute_duan_inseparability(GaussianState.tmsv("a", "b", r=1.2), "a", "b")
    assert DUAN_SEPARABILITY_BOUND > weak > strong > 0.0


def test_classical_correlation_does_not_violate_duan_bound():
    vacuum = GaussianState.vacuum(modes=("a", "b"))
    correlated = LossChannels.correlated_thermal_noise(
        "a", "b", eta=0.5, n_thermal=0.5, c_correlation=0.3
    ).apply(vacuum)
    assert abs(correlated.covariance[0, 2]) > 1e-6
    witness = compute_duan_inseparability(correlated, "a", "b")
    assert witness >= DUAN_SEPARABILITY_BOUND - 1e-9


def test_tmsv_entanglement_survives_but_weakens_under_loss():
    tmsv = GaussianState.tmsv("a", "b", r=1.0)
    witness_clean = compute_duan_inseparability(tmsv, "a", "b")
    lossy = tmsv.loss("a", eta=0.9)
    witness_light_loss = compute_duan_inseparability(lossy, "a", "b")

    very_lossy = tmsv.loss("a", eta=0.1)
    witness_heavy_loss = compute_duan_inseparability(very_lossy, "a", "b")

    assert witness_clean < witness_light_loss < witness_heavy_loss
    assert witness_light_loss < DUAN_SEPARABILITY_BOUND
    assert witness_heavy_loss > DUAN_SEPARABILITY_BOUND


@pytest.mark.visual
def test_tmsv_entanglement_visualization_demo():
    r = 1.0
    tmsv = GaussianState.tmsv("a", "b", r=r)
    vacuum = GaussianState.vacuum(("a", "b"))
    classical = LossChannels.correlated_thermal_noise(
        "a", "b", eta=0.3, n_thermal=1.5, c_correlation=1.4
    ).apply(vacuum)

    duan_tmsv = compute_duan_inseparability(tmsv, "a", "b")
    duan_classical = compute_duan_inseparability(classical, "a", "b")
    fig, axes = plt.subplots(2, 2, figsize=(11, 10))
    panels = [
        (
            tmsv,
            "x",
            axes[0][0],
            f"TMSV: x_a vs x_b\n(Duan sum = {duan_tmsv:.2f})",
        ),
        (
            tmsv,
            "p",
            axes[0][1],
            f"TMSV: p_a vs p_b\n(Duan sum = {duan_tmsv:.2f})",
        ),
        (
            classical,
            "x",
            axes[1][0],
            f"Classical noise: x_a vs x_b\n(Duan sum = {duan_classical:.2f})",
        ),
        (
            classical,
            "p",
            axes[1][1],
            f"Classical noise: p_a vs p_b\n(Duan sum = {duan_classical:.2f})",
        ),
    ]
    for state, quad, ax, title in panels:
        P, X_a, X_b, _, _ = compute_joint_correlation(
            state, "a", "b", x_max=6.0, quadrature=quad
        )
        ax.contourf(X_a, X_b, P, 100, cmap="viridis")
        ax.set_title(title)
        ax.set_xlabel(f"{quad}_a")
        ax.set_ylabel(f"{quad}_b")
        ax.axis("equal")
    fig.suptitle(
        f"Genuine entanglement vs classical correlation "
        f"(separability bound = {DUAN_SEPARABILITY_BOUND})"
    )
    plt.tight_layout()
    plt.show()


# Gaussian-to-Fock boundary


def test_phase_rotation_preserves_photon_number_and_purity():
    state = GaussianState.vacuum(modes=("a",))
    squeezed = state.squeeze("a", r=0.7, theta=0.0)
    rotated = squeezed.rotate("a", phi=1.1)
    assert np.linalg.det(rotated.covariance) == pytest.approx(
        np.linalg.det(squeezed.covariance)
    )
    full_turn = squeezed.rotate("a", phi=2 * np.pi)
    np.testing.assert_allclose(full_turn.covariance, squeezed.covariance, atol=1e-9)


def test_circuit_rotate_matches_manual_rotation():
    manual = GaussianState.vacuum(("a",)).squeeze("a", r=0.5).rotate("a", phi=0.4)

    circuit = GaussianCircuit().add_mode("a")
    circuit.squeeze(mode="a", r=0.5).rotate(mode="a", phi=0.4)
    compiled = circuit.compile_and_run()

    np.testing.assert_allclose(compiled.covariance, manual.covariance, atol=1e-10)


def test_williamson_decomposition_is_genuinely_symplectic():
    rng = np.random.default_rng(1234)
    A = rng.normal(size=(6, 6))
    covariance = A @ A.T + 0.5 * np.eye(6)

    nus, S, D = _williamson_decomposition(covariance)
    Omega = np.kron(np.eye(3), np.array([[0.0, 1.0], [-1.0, 0.0]]))

    assert np.all(nus >= 0.5 - 1e-10)
    np.testing.assert_allclose(S @ Omega @ S.T, Omega, atol=1e-8)
    np.testing.assert_allclose(S @ D @ S.T, covariance, atol=1e-8)


def test_to_qutip_reconstructs_gaussian_covariance():
    state = (
        GaussianState.vacuum(("a", "b"))
        .squeeze("a", r=0.45, theta=0.2)
        .squeeze("b", r=0.35, theta=-0.4)
        .beam_splitter("a", "b", eta=0.37)
        .displace("a", alpha=0.4 + 0.2j)
    )
    N_cutoff = 30
    rho = state.to_qutip(N_cutoff=N_cutoff)

    a_ops = []
    for i in range(2):
        op_list = [qt.qeye(N_cutoff) for _ in range(2)]
        op_list[i] = qt.destroy(N_cutoff)
        a_ops.append(qt.tensor(*op_list))

    r_ops = []
    for a in a_ops:
        r_ops.extend(
            [
                (a + a.dag()) / np.sqrt(2.0),
                (a - a.dag()) / (1j * np.sqrt(2.0)),
            ]
        )
    covariance = qt.covariance_matrix(r_ops, rho, symmetrized=True).real
    displacement = np.array([qt.expect(op, rho).real for op in r_ops])

    np.testing.assert_allclose(displacement, state.displacement, atol=2e-5)
    np.testing.assert_allclose(covariance, state.covariance, atol=2e-5)


def test_to_qutip_handles_plain_vacuum_and_pure_displacement():
    vacuum = GaussianState.vacuum(modes=("a", "b"))
    rho = vacuum.to_qutip(N_cutoff=10)
    assert rho.tr() == pytest.approx(1.0, abs=1e-9)
    displaced_only = vacuum.copy()
    displaced_only.displacement[0] = 1.5
    rho2 = displaced_only.to_qutip(N_cutoff=15)
    assert rho2.tr() == pytest.approx(1.0, abs=1e-9)


def test_to_qutip_trace_always_exactly_one_even_with_ill_conditioned_v():
    state = (
        GaussianState.vacuum(("a", "b"))
        .squeeze("a", r=0.5)
        .squeeze("b", r=0.5, theta=np.pi / 2)
        .beam_splitter("a", "b", eta=0.5)
    )
    noisy_state = LossChannels.thermal_loss(mode="a", eta=0.9, n_thermal=0.2).apply(state)
    rho = noisy_state.to_qutip(N_cutoff=18)
    assert rho.tr() == pytest.approx(1.0, abs=1e-9)


def test_to_qutip_rejects_invalid_n_cutoff():
    state = GaussianState.vacuum(modes=("a",))
    with pytest.raises(ValueError):
        state.to_qutip(N_cutoff=0)
    with pytest.raises(ValueError):
        state.to_qutip(N_cutoff=-5)


def test_gaussian_channel_roundtrips_through_file(tmp_path):
    channel = LossChannels.thermal_loss(mode="a", eta=0.8, n_thermal=0.1)
    path = tmp_path / "channel.json"
    channel.save(path)
    restored = GaussianChannel.load(path)
    state = GaussianState.vacuum(modes=("a",))
    original_out = channel.apply(state)
    restored_out = restored.apply(state)
    np.testing.assert_allclose(restored_out.covariance, original_out.covariance)


# Heterodyne checks


def test_heterodyne_measurement_adds_vacuum_noise_and_collapses_to_coherent():
    state = (
        GaussianState.vacuum(("a", "b"))
        .squeeze("a", r=1.0)
        .squeeze("b", r=1.0, theta=np.pi / 2)
        .beam_splitter("a", "b", eta=0.5)
    )
    outcome, collapsed = GaussianMeasurements.heterodyne_measurement(
        state, measured_mode="a", outcome=np.array([1.0, 0.5])
    )
    np.testing.assert_allclose(outcome, [1.0, 0.5])
    assert collapsed.modes == ("b",)
    eigvals = np.linalg.eigvalsh(collapsed.covariance)
    assert (eigvals >= 0.5 - 1e-9).all()


def test_heterodyne_measurement_is_reproducible_with_seeded_rng():
    state = GaussianState.vacuum(modes=("a",))
    v1, _ = GaussianMeasurements.heterodyne_measurement(
        state, measured_mode="a", rng=np.random.default_rng(7)
    )
    v2, _ = GaussianMeasurements.heterodyne_measurement(
        state, measured_mode="a", rng=np.random.default_rng(7)
    )
    np.testing.assert_allclose(v1, v2)
