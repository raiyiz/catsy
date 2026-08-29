import numpy as np
import pytest
import qutip as qt

from catsy.core import DUAN_SEPARABILITY_BOUND, _williamson_decomposition
from catsy.gaussian import (
    GaussianChannel,
    GaussianMeasurements,
    GaussianState,
    LossChannels,
    beam_splitter,
    compute_duan_inseparability,
    compute_joint_correlation,
    compute_wigner_analytically,
    displace,
    loss,
    rotate,
    squeeze,
    thermal_loss,
)
from catsy.gaussian.visualization import plot_joint_correlation
from catsy.optics import Circuit, Gate

# Analytic primitive checks


@pytest.mark.parametrize("r", [0.0, 0.25, 1.2])
def test_squeezing_has_expected_principal_variances(r):
    squeezed = GaussianState.vacuum(("a",)).squeeze("a", r=r, theta=0.0)
    expected = 0.5 * np.diag([np.exp(-2 * r), np.exp(2 * r)])
    np.testing.assert_allclose(squeezed.covariance, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("phi", [0.0, np.pi / 2, -0.3])
def test_phase_rotation_rotates_displacement(phi):
    state = GaussianState.coherent(("a",), 0.7 + 0.2j)
    rotated = state.rotate("a", phi=phi)
    x, p = state.displacement
    expected = [np.cos(phi) * x - np.sin(phi) * p, np.sin(phi) * x + np.cos(phi) * p]
    np.testing.assert_allclose(rotated.displacement, expected, atol=1e-12)
    np.testing.assert_allclose(rotated.covariance, state.covariance, atol=1e-12)


@pytest.mark.parametrize("eta", [0.0, 0.5, 1.0])
def test_loss_matches_vacuum_environment_formula(eta):
    state = GaussianState.vacuum(("a",)).squeeze("a", r=0.6)
    lossy = state.loss("a", eta=eta)
    expected = eta * state.covariance + (1.0 - eta) * 0.5 * np.eye(2)
    np.testing.assert_allclose(lossy.covariance, expected, atol=1e-12)
    np.testing.assert_allclose(lossy.displacement, np.sqrt(eta) * state.displacement)


@pytest.mark.parametrize(("eta", "n_thermal"), [(0.0, 0.0), (0.25, 0.5), (1.0, 2.0)])
def test_thermal_loss_matches_environment_formula(eta, n_thermal):
    state = GaussianState.coherent(("a",), 0.4 - 0.2j).squeeze("a", r=0.5, theta=0.2)
    lossy = LossChannels.thermal_loss(mode="a", eta=eta, n_thermal=n_thermal).apply(state)
    expected = eta * state.covariance + (1.0 - eta) * (n_thermal + 0.5) * np.eye(2)
    np.testing.assert_allclose(lossy.covariance, expected, atol=1e-12)
    np.testing.assert_allclose(lossy.displacement, np.sqrt(eta) * state.displacement)


@pytest.mark.parametrize("eta", [0.0, 0.5, 1.0])
def test_beam_splitter_has_expected_coherent_amplitude_map(eta):
    alpha_a, alpha_b = 0.8 + 0.1j, -0.3 + 0.6j
    state = GaussianState.coherent(("a", "b"), [alpha_a, alpha_b])
    mixed = state.beam_splitter("a", "b", eta=eta)
    t, r = np.sqrt(eta), np.sqrt(1.0 - eta)
    expected = [
        np.sqrt(2.0) * np.real(t * alpha_a + r * alpha_b),
        np.sqrt(2.0) * np.imag(t * alpha_a + r * alpha_b),
        np.sqrt(2.0) * np.real(-r * alpha_a + t * alpha_b),
        np.sqrt(2.0) * np.imag(-r * alpha_a + t * alpha_b),
    ]
    np.testing.assert_allclose(mixed.displacement, expected, atol=1e-12)
    np.testing.assert_allclose(mixed.covariance, 0.5 * np.eye(4), atol=1e-12)


# States and operations


def test_gate_instances_use_the_explicit_gate_contract():
    gates = (
        Gate("Squeezer", squeeze, ("a",), {"r": 0.5}),
        Gate("Rotator", rotate, ("a",), {"phi": 0.2}),
        Gate("Displacer", displace, ("a",), {"alpha": 0.3 + 0.1j}),
        Gate("BeamSplitter", beam_splitter, ("a", "b"), {"eta": 0.5}),
        Gate("Noise", loss, ("a",), {"eta": 0.9}),
        Gate("ThermalLoss", thermal_loss, ("a",), {"eta": 0.9, "n_thermal": 0.2}),
    )
    assert [g.name for g in gates] == [
        "Squeezer",
        "Rotator",
        "Displacer",
        "BeamSplitter",
        "Noise",
        "ThermalLoss",
    ]
    assert all(callable(g.transform) and g.modes for g in gates)


def test_vacuum_is_shot_noise_limited(two_mode_vacuum):
    assert two_mode_vacuum.displacement.shape == (4,)
    np.testing.assert_allclose(two_mode_vacuum.covariance, 0.5 * np.eye(4))


def test_squeezing_preserves_purity_determinant(single_mode_vacuum):
    squeezed = single_mode_vacuum.squeeze("a", r=0.8, theta=0.3)
    assert np.linalg.det(squeezed.covariance) == pytest.approx(0.25, rel=1e-9)
    unrotated = single_mode_vacuum.squeeze("a", r=0.8, theta=0.0)
    assert unrotated.covariance[0, 0] < 0.5 < unrotated.covariance[1, 1]


def test_beam_splitter_entangles_independent_modes(two_mode_vacuum):
    state = two_mode_vacuum.squeeze("a", r=0.6, theta=0.0).squeeze(
        "b", r=0.6, theta=np.pi / 2
    )
    np.testing.assert_allclose(state.covariance[0:2, 2:4], np.zeros((2, 2)), atol=1e-12)
    mixed = state.beam_splitter("a", "b", eta=0.5)
    assert np.abs(mixed.covariance[0:2, 2:4]).max() > 1e-6


@pytest.mark.parametrize("eta", [-0.1, 1.1])
def test_beam_splitter_rejects_invalid_eta(eta):
    with pytest.raises(ValueError):
        GaussianState.vacuum(("a", "b")).beam_splitter("a", "b", eta=eta)


def test_beam_splitter_rejects_duplicate_modes():
    with pytest.raises(ValueError):
        GaussianState.vacuum(("a", "b")).beam_splitter("a", "a", eta=0.5)


def test_loss_reduces_toward_vacuum(single_mode_vacuum):
    lossy = single_mode_vacuum.squeeze("a", r=1.0, theta=0.0).loss("a", eta=0.0)
    np.testing.assert_allclose(lossy.covariance, 0.5 * np.eye(2), atol=1e-9)


def test_displacement_shifts_mean_leaves_covariance_untouched(single_mode_vacuum):
    squeezed = single_mode_vacuum.squeeze("a", r=0.4, theta=0.2)
    displaced = squeezed.displace("a", alpha=1.0)
    np.testing.assert_allclose(displaced.covariance, squeezed.covariance)
    np.testing.assert_allclose(
        displaced.displacement - squeezed.displacement, [np.sqrt(2.0), 0.0]
    )


def test_displacement_alpha_and_xp_are_equivalent(single_mode_vacuum):
    alpha = 0.6 - 0.9j
    via_alpha = single_mode_vacuum.displace("a", alpha=alpha)
    via_xp = single_mode_vacuum.displace(
        "a", x=np.sqrt(2.0) * alpha.real, p=np.sqrt(2.0) * alpha.imag
    )
    np.testing.assert_allclose(via_alpha.displacement, via_xp.displacement)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"alpha": 1.0, "x": 1.0}, "either `alpha` or"),
        ({}, "need some input"),
        ({"x": 1.0}, "both `x` and `p`"),
    ],
)
def test_displacement_rejects_invalid_argument_combinations(kwargs, match):
    with pytest.raises((ValueError, TypeError), match=match):
        GaussianState.vacuum(("a",)).displace("a", **kwargs)


def test_coherent_matches_displaced_vacuum():
    alpha = 1.2 + 0.4j
    coherent = GaussianState.coherent(("a",), alpha)
    manual = GaussianState.vacuum(("a",)).displace("a", alpha=alpha)
    np.testing.assert_allclose(coherent.displacement, manual.displacement)
    np.testing.assert_allclose(coherent.covariance, 0.5 * np.eye(2))


def test_coherent_broadcasts_and_validates_alpha_count():
    state = GaussianState.coherent(("a", "b"), 1.0j)
    np.testing.assert_allclose(state.displacement[0:2], state.displacement[2:4])
    with pytest.raises(ValueError):
        GaussianState.coherent(("a", "b"), [1.0])


def test_coherent_state_mean_photon_number_matches_alpha_squared():
    alpha = 1.5 + 0.7j
    rho = GaussianState.coherent(("a",), alpha).to_qutip(N_cutoff=25)
    mean_n = qt.expect(qt.num(25), rho)
    assert mean_n == pytest.approx(np.abs(alpha) ** 2, rel=1e-3)


def test_circuit_runs_against_explicit_initial_state():
    circuit = Circuit().add_mode("c")
    compiled = circuit.run(GaussianState.coherent(("c",), 1.0 + 1.0j))
    expected = GaussianState.coherent(("c",), 1.0 + 1.0j)
    np.testing.assert_allclose(compiled.displacement, expected.displacement)
    overridden = circuit.run(GaussianState.vacuum(("c",)))
    np.testing.assert_allclose(overridden.displacement, np.zeros(2))


def test_get_mode_index_unknown_mode_raises(two_mode_vacuum):
    with pytest.raises(ValueError):
        two_mode_vacuum.get_mode_index("z")


# Mode ordering and construction validation


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
    circuit = Circuit().add_mode("a").add_mode("b")
    result = circuit.run(initial)
    assert result.modes == ("a", "b")
    expected = GaussianState.coherent(("a", "b"), alphas=[1.0 + 0.0j, 0.0 + 1.0j])
    np.testing.assert_allclose(result.displacement, expected.displacement)
    np.testing.assert_allclose(result.covariance, expected.covariance)


def test_classical_phase_jitter_channel_applies():
    channel = LossChannels.classical_phase_jitter(mode="a", sigma_phi=0.3)
    jittered = channel.apply(GaussianState.vacuum(modes=("a",)))
    assert jittered.covariance[0, 0] == pytest.approx(0.5)
    assert jittered.covariance[1, 1] > 0.5


# GaussianState construction validation, in the order it runs: shape checks,
# then duplicate modes, then symmetry/finiteness/physicality.
_BAD_CONSTRUCTIONS = [
    (("a", "a"), np.zeros(4), 0.5 * np.eye(4), "Duplicate mode"),
    (("a",), np.zeros(3), 0.5 * np.eye(2), "displacement must have shape"),
    (("a",), np.zeros(2), 0.5 * np.eye(3), "covariance must have shape"),
    (("a",), np.zeros(2), np.array([[0.5, 0.1], [0.0, 0.5]]), "symmetric"),
    (("a",), np.zeros(2), np.diag([np.inf, np.inf]), "finite"),
    (("a",), np.zeros(2), 0.1 * np.eye(2), "uncertainty relation"),
]


@pytest.mark.parametrize(
    ("modes", "displacement", "covariance", "match"), _BAD_CONSTRUCTIONS
)
def test_gaussian_state_rejects_invalid_construction(
    modes, displacement, covariance, match
):
    with pytest.raises(ValueError, match=match):
        GaussianState(modes=modes, displacement=displacement, covariance=covariance)


# Channels

_BAD_CHANNELS = [
    (("a",), 2.0 * np.eye(2), np.zeros((2, 2)), np.zeros(2), "complete positivity"),
    (("a",), np.eye(2), np.array([[0.1, 0.2], [0.0, 0.1]]), np.zeros(2), "symmetric"),
    (("a", "a"), np.eye(4), np.zeros((4, 4)), np.zeros(4), "Duplicate target mode"),
    (("a",), np.eye(3), np.eye(2), np.zeros(2), "X must have shape"),
    (("a",), np.eye(2), np.eye(3), np.zeros(2), "Y must have shape"),
    (("a",), np.eye(2), np.eye(2), np.zeros(3), "d0 must have shape"),
]


@pytest.mark.parametrize(("target_modes", "X", "Y", "d0", "match"), _BAD_CHANNELS)
def test_gaussian_channel_rejects_invalid_construction(target_modes, X, Y, d0, match):
    with pytest.raises(ValueError, match=match):
        GaussianChannel(target_modes=target_modes, X=X, Y=Y, d0=d0)


@pytest.mark.parametrize(
    ("n_thermal", "c_correlation", "valid"),
    [(0.5, 0.5, True), (0.5, -0.5, True), (0.5, 0.500001, False)],
)
def test_correlated_thermal_noise_validates_correlation(n_thermal, c_correlation, valid):
    kwargs = dict(eta=0.5, n_thermal=n_thermal, c_correlation=c_correlation)
    if valid:
        assert LossChannels.correlated_thermal_noise("a", "b", **kwargs).Y.shape == (4, 4)
    else:
        with pytest.raises(ValueError, match="c_correlation"):
            LossChannels.correlated_thermal_noise("a", "b", **kwargs)


# Circuit validation and serialization


def test_circuit_matches_manual_gate_chain():
    manual = (
        GaussianState.vacuum(("a", "b"))
        .squeeze("a", r=0.6, theta=0.0)
        .squeeze("b", r=0.6, theta=np.pi / 2)
        .beam_splitter("a", "b", eta=0.5)
        .displace("a", alpha=0.5 - 0.2j)
        .rotate("b", phi=0.4)
    )
    manual = LossChannels.thermal_loss(mode="b", eta=0.7, n_thermal=0.3).apply(manual)

    circuit = Circuit().add_mode("a").add_mode("b")
    circuit.add_gate(
        Gate(
            name="Squeezer",
            transform=squeeze,
            modes=("a",),
            kwargs={"r": 0.6, "theta": 0.0},
        )
    ).add_gate(
        Gate(
            name="Squeezer",
            transform=squeeze,
            modes=("b",),
            kwargs={"r": 0.6, "theta": np.pi / 2},
        )
    ).add_gate(
        Gate(
            name="BeamSplitter",
            transform=beam_splitter,
            modes=("a", "b"),
            kwargs={"eta": 0.5},
        )
    ).add_gate(
        Gate(
            name="Displacer",
            transform=displace,
            modes=("a",),
            kwargs={"x": np.sqrt(2.0) * 0.5, "p": np.sqrt(2.0) * -0.2},
        )
    ).add_gate(
        Gate(name="Rotator", transform=rotate, modes=("b",), kwargs={"phi": 0.4})
    ).add_gate(
        Gate(
            name="ThermalLoss",
            transform=thermal_loss,
            modes=("b",),
            kwargs={"eta": 0.7, "n_thermal": 0.3},
        )
    )
    compiled = circuit.run(GaussianState.vacuum(("a", "b")))
    np.testing.assert_allclose(compiled.displacement, manual.displacement, atol=1e-10)
    np.testing.assert_allclose(compiled.covariance, manual.covariance, atol=1e-10)


def test_circuit_executes_the_callable_directly():
    calls = []

    def my_gate(state, modes, **kwargs):
        calls.append((modes, kwargs))
        return state.rotate(modes[0], phi=kwargs["phi"])

    circuit = Circuit().add_mode("a")
    circuit.add_gate(
        Gate(name="MyGate", transform=my_gate, modes=("a",), kwargs={"phi": 0.25})
    )
    result = circuit.run(GaussianState.vacuum(("a",)))

    assert calls == [(("a",), {"phi": 0.25})]
    expected = GaussianState.vacuum(("a",)).rotate("a", phi=0.25)
    np.testing.assert_allclose(result.displacement, expected.displacement)
    np.testing.assert_allclose(result.covariance, expected.covariance)


@pytest.mark.parametrize(
    ("modes", "match"),
    [
        ((), "at least one mode"),
        (("",), "is not registered on this circuit"),
        (("a", "a"), "cannot target the same mode more than once"),
    ],
)
def test_add_gate_rejects_invalid_mode_tuples(modes, match):
    # Gate owns its target modes, so Circuit validates the bound Gate here.
    def my_gate(state, modes, **kwargs):
        return state

    circuit = Circuit().add_mode("a")
    with pytest.raises(ValueError, match=match):
        circuit.add_gate(Gate(name="MyGate", transform=my_gate, modes=modes, kwargs={}))


def test_circuit_rejects_unregistered_mode():
    # add_gate() is the sole enforcement point -- rejected immediately, not deferred to run().
    circuit = Circuit().add_mode("a")
    with pytest.raises(ValueError, match="not registered on this circuit"):
        circuit.add_gate(
            Gate(
                name="Squeezer",
                transform=squeeze,
                modes=("z",),
                kwargs={"r": 0.5, "theta": 0.0},
            )
        )


def test_circuit_rejects_empty_mode_set():
    with pytest.raises(ValueError):
        Circuit().run(GaussianState.vacuum(()))


@pytest.mark.skip("uniqueness needs to be implemented yet")
def test_circuit_rejects_duplicate_mode_registration():
    circuit = Circuit().add_mode("a")
    with pytest.raises(ValueError, match="already registered"):
        circuit.add_mode("a")


def test_gate_applies_its_bound_transform():
    gate = Gate("Squeezer", squeeze, ("a",), {"r": 0.5, "theta": 0.0})
    state = GaussianState.vacuum(("a",))
    result = gate.apply(state)
    expected = state.squeeze("a", r=0.5, theta=0.0)
    np.testing.assert_allclose(result.displacement, expected.displacement)
    np.testing.assert_allclose(result.covariance, expected.covariance)


def test_registered_gates_are_available_as_circuit_methods():
    circuit = Circuit().add_mode("a").squeeze("a", r=0.4).displace("a", alpha=0.2 - 0.4j)
    gates = circuit.to_dict()["gates"]
    assert gates[0] == {"gate": "Squeezer", "modes": ["a"], "kwargs": {"r": 0.4}}
    # Displacer kwargs are canonicalized to real-valued (x, p) at gate construction
    # time, not stored as the raw complex `alpha`, which isn't JSON serializable.
    assert gates[1]["gate"] == "Displacer"
    assert gates[1]["kwargs"] == pytest.approx(
        {"x": np.sqrt(2.0) * 0.2, "p": np.sqrt(2.0) * -0.4}
    )


@pytest.mark.parametrize(
    ("kind", "modes", "kwargs", "expected"),
    [
        ("vacuum", ("a",), {}, lambda: GaussianState.vacuum(("a",))),
        ("tmsv", ("a", "b"), {"r": 0.6}, lambda: GaussianState.tmsv("a", "b", 0.6)),
    ],
)
def test_initial_state_gate_constructs_state(kind, modes, kwargs, expected):
    circuit = Circuit()
    for mode in modes:
        circuit.add_mode(mode)
    circuit.initial_state(*modes, kind=kind, **kwargs)
    result = circuit.run()
    expected_state = expected()
    np.testing.assert_allclose(result.displacement, expected_state.displacement)
    np.testing.assert_allclose(result.covariance, expected_state.covariance)


def test_initial_state_gate_can_be_overridden_by_explicit_state():
    circuit = Circuit().add_mode("a").initial_state("a", kind="coherent", alpha=2.0)
    result = circuit.run(GaussianState.vacuum(("a",)))
    np.testing.assert_allclose(result.displacement, np.zeros(2))


def test_initial_state_tmsv_requires_exactly_two_modes():
    circuit = Circuit().add_mode("a").initial_state("a", kind="tmsv", r=0.5)
    with pytest.raises(ValueError, match="exactly two modes"):
        circuit.run()


def test_initial_state_rejects_unknown_kind():
    circuit = Circuit().add_mode("a").initial_state("a", kind="squeezed")
    with pytest.raises(ValueError, match="Unknown Gaussian initial state kind"):
        circuit.run()


def test_circuit_register_is_only_for_custom_gate_deserialization():
    def my_gate(state, modes, **kwargs):
        return state

    Circuit.register("MyGate", my_gate)
    circuit = Circuit().add_mode("a")
    circuit.add_gate(
        Gate(name="MyGate", transform=my_gate, modes=("a",), kwargs={"foo": 1})
    )
    restored = Circuit.from_dict(circuit.to_dict())
    assert restored.to_dict() == circuit.to_dict()


def test_circuit_register_requires_matching_gate_name():
    with pytest.raises(ValueError, match="non-empty"):
        Circuit.register("", lambda state, modes, **kwargs: state)


def test_gaussian_state_roundtrips_through_dict():
    state = GaussianState.vacuum(modes=("a", "b")).squeeze("a", r=0.4)
    restored = GaussianState.from_dict(state.to_dict())
    assert restored.modes == state.modes
    np.testing.assert_allclose(restored.displacement, state.displacement)
    np.testing.assert_allclose(restored.covariance, state.covariance)


# Measurements

# Circuit file roundtripping (save/load) is Circuit's own concern and is
# covered in test_optics.py (test_circuit_roundtrips_through_file,
# test_circuit_gate_roundtrip_agrees); no need to duplicate it here.


def test_circuit_from_dict_rejects_unknown_gate():
    # from_dict executes whatever callable is attached, with no name-based
    # registry lookup, so a stale/corrupted gate name fails here with KeyError.
    data = {
        "name": "Bad",
        "modes": ["a"],
        "gates": [{"gate": "NotARealOp", "modes": ["a"], "kwargs": {}}],
    }
    with pytest.raises(KeyError, match="NotARealOp"):
        Circuit.from_dict(data)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [({"alpha": 1.0, "x": 1.0}, "not both"), ({}, "need some input"), ({"x": 1.0}, "p")],
)
def test_circuit_displace_rejects_invalid_argument_combinations(kwargs, match):
    # Circuit's Displacer gate has its own copy of the alpha/(x, p) validation.
    circuit = Circuit().add_mode("a")
    circuit.add_gate(
        Gate(name="Displacer", transform=displace, modes=("a",), kwargs={**kwargs})
    )
    with pytest.raises((ValueError, TypeError), match=match):
        circuit.run(GaussianState.vacuum(("a",)))


def test_homodyne_measurement_collapses_tmsv_correlation():
    # tmsv() is proven equivalent to a squeeze/squeeze/beam-splitter chain by
    # test_tmsv_matches_manual_squeeze_squeeze_bs below.
    state = GaussianState.tmsv("a", "b", r=1.0)
    val, collapsed = GaussianMeasurements.homodyne_measurement(
        state, measured_mode="a", phi=0.0, outcome=2.5
    )
    assert val == 2.5
    assert collapsed.modes == ("b",)
    _val_neg, collapsed_neg = GaussianMeasurements.homodyne_measurement(
        state, measured_mode="a", phi=0.0, outcome=-2.5
    )
    assert np.sign(collapsed.displacement[0]) != np.sign(collapsed_neg.displacement[0])


@pytest.mark.parametrize(
    ("measure", "kwargs", "seed"),
    [
        (GaussianMeasurements.homodyne_measurement, {"phi": 0.0}, 42),
        (GaussianMeasurements.heterodyne_measurement, {}, 7),
    ],
)
def test_measurement_is_reproducible_with_seeded_rng(measure, kwargs, seed):
    state = GaussianState.vacuum(modes=("a",))
    v1, _ = measure(state, measured_mode="a", rng=np.random.default_rng(seed), **kwargs)
    v2, _ = measure(state, measured_mode="a", rng=np.random.default_rng(seed), **kwargs)
    np.testing.assert_allclose(v1, v2)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"phi": np.nan}, "phi must be finite"),
        ({"phi": 0.0, "outcome": np.inf}, "finite scalar"),
    ],
)
def test_homodyne_rejects_nonfinite_inputs(kwargs, match):
    with pytest.raises(ValueError, match=match):
        GaussianMeasurements.homodyne_measurement(
            GaussianState.vacuum(("a",)), measured_mode="a", **kwargs
        )


@pytest.mark.parametrize(
    ("measure", "kwargs", "expected"),
    [
        (GaussianMeasurements.homodyne_measurement, {"phi": 0.0, "outcome": 1.25}, 1.25),
        (
            GaussianMeasurements.heterodyne_measurement,
            {"outcome": np.array([0.2, -0.3])},
            np.array([0.2, -0.3]),
        ),
    ],
)
def test_single_mode_measurement_returns_valid_empty_state(measure, kwargs, expected):
    state = GaussianState.coherent(modes=("a",), alphas=0.7 + 0.2j)
    outcome, collapsed = measure(state, measured_mode="a", **kwargs)
    np.testing.assert_allclose(outcome, expected)
    assert collapsed.modes == ()
    assert collapsed.displacement.shape == (0,)
    assert collapsed.covariance.shape == (0, 0)


def test_homodyne_rejects_numerically_singular_measured_variance():
    # r=25 squeezes Var(x) to ~1e-22, well below TOL_PHYSICALITY: homodyning
    # along exactly the squeezed quadrature must be rejected, not silently
    # divide by a near-zero variance.
    state = GaussianState.vacuum(("a",)).squeeze("a", r=25.0, theta=0.0)
    with pytest.raises(ValueError, match="measurement variance"):
        GaussianMeasurements.homodyne_measurement(state, measured_mode="a", phi=0.0)


# Phase-space analysis


@pytest.mark.parametrize(
    ("outcome", "match"),
    [(np.array([1.0]), r"shape \(2,\)"), (np.array([1.0, np.nan]), "finite values")],
)
def test_heterodyne_rejects_invalid_outcomes(outcome, match):
    with pytest.raises(ValueError, match=match):
        GaussianMeasurements.heterodyne_measurement(
            GaussianState.vacuum(("a",)), measured_mode="a", outcome=outcome
        )


def test_heterodyne_rejects_non_positive_definite_effective_covariance(monkeypatch):
    # No physical state naturally drives V_eff below positive-definiteness
    # (the vacuum floor keeps it well-conditioned), so force Cholesky to fail.
    def _always_fails(_matrix):
        raise np.linalg.LinAlgError("forced failure for test")

    monkeypatch.setattr(np.linalg, "cholesky", _always_fails)
    with pytest.raises(ValueError, match="positive definite"):
        GaussianMeasurements.heterodyne_measurement(
            GaussianState.vacuum(("a",)), measured_mode="a"
        )


def test_heterodyne_rejects_nonfinite_effective_covariance():
    # Construction-time validation rejects a non-finite covariance outright,
    # so mutate it in place post-hoc (the dataclass isn't frozen) to reach this.
    state = GaussianState.vacuum(("a",))
    state.covariance[0, 0] = np.inf
    with pytest.raises(ValueError, match="effective covariance must be finite"):
        GaussianMeasurements.heterodyne_measurement(state, measured_mode="a")


def test_wigner_analytical_matches_gaussian_normalization():
    state = GaussianState.vacuum(("a",)).squeeze("a", r=1.1, theta=30.0)
    state.displacement[:] = [2.0, 3.0]
    W, _X, _P, _M = compute_wigner_analytically(
        state, mode_name="a", x_max=8.0, num_points=200
    )
    dx = (2 * 8.0) / 199
    assert W.sum() * dx * dx == pytest.approx(1.0, rel=1e-2)


def _correlated_two_mode_state() -> GaussianState:
    circuit = (
        Circuit()
        .add_mode("a")
        .add_mode("b")
        .squeeze("a", r=0.6, theta=0.0)
        .squeeze("b", r=0.6, theta=np.pi)
        .beam_splitter("a", "b", eta=0.5)
    )
    return circuit.run(GaussianState.vacuum(("a", "b")))


def test_joint_correlation_computes_valid_grid():
    P, _X_a, _X_b, _mode_a, _mode_b = compute_joint_correlation(
        _correlated_two_mode_state(), "a", "b"
    )
    assert P.shape == (150, 150)
    assert np.all(P >= 0)


@pytest.mark.visualize
def test_joint_correlation_visualization_demo():
    # Visual counterpart to test_joint_correlation_computes_valid_grid above;
    # grid validity is checked there, this is only the plot, gated behind --plot.
    plot_joint_correlation(_correlated_two_mode_state(), "a", "b")


def test_joint_correlation_rejects_invalid_quadrature():
    with pytest.raises(ValueError):
        compute_joint_correlation(
            GaussianState.vacuum(modes=("a", "b")), "a", "b", quadrature="z"
        )


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
    assert empirical_cov_x > 0 > empirical_cov_p


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


def test_duan_witness_behaves_correctly_for_vacuum_and_tmsv():
    vacuum_witness = compute_duan_inseparability(
        GaussianState.vacuum(modes=("a", "b")), "a", "b"
    )
    assert vacuum_witness == pytest.approx(DUAN_SEPARABILITY_BOUND)

    r = 1.0
    tmsv_witness = compute_duan_inseparability(
        GaussianState.tmsv("a", "b", r=r), "a", "b"
    )
    assert tmsv_witness == pytest.approx(2.0 * np.exp(-2.0 * r), rel=1e-6)
    assert tmsv_witness < DUAN_SEPARABILITY_BOUND

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
    witness_light_loss = compute_duan_inseparability(tmsv.loss("a", eta=0.9), "a", "b")
    witness_heavy_loss = compute_duan_inseparability(tmsv.loss("a", eta=0.1), "a", "b")
    assert witness_clean < witness_light_loss < witness_heavy_loss
    assert witness_light_loss < DUAN_SEPARABILITY_BOUND < witness_heavy_loss


# Gaussian-to-Fock boundary


def test_phase_rotation_preserves_photon_number_and_purity():
    squeezed = GaussianState.vacuum(modes=("a",)).squeeze("a", r=0.7, theta=0.0)
    rotated = squeezed.rotate("a", phi=1.1)
    assert np.linalg.det(rotated.covariance) == pytest.approx(
        np.linalg.det(squeezed.covariance)
    )
    full_turn = squeezed.rotate("a", phi=2 * np.pi)
    np.testing.assert_allclose(full_turn.covariance, squeezed.covariance, atol=1e-9)


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
    N_cutoff = 20
    rho = state.to_qutip(N_cutoff=N_cutoff)

    a_ops = []
    for i in range(2):
        op_list = [qt.qeye(N_cutoff) for _ in range(2)]
        op_list[i] = qt.destroy(N_cutoff)
        a_ops.append(qt.tensor(*op_list))
    r_ops = []
    for a in a_ops:
        r_ops.extend([(a + a.dag()) / np.sqrt(2.0), (a - a.dag()) / (1j * np.sqrt(2.0))])
    covariance = qt.covariance_matrix(r_ops, rho, symmetrized=True).real
    displacement = np.array([qt.expect(op, rho).real for op in r_ops])

    np.testing.assert_allclose(displacement, state.displacement, atol=2e-5)
    np.testing.assert_allclose(covariance, state.covariance, atol=2e-5)


def test_to_qutip_trace_is_always_exactly_one():
    # Covers a plain vacuum, a pure displacement, and a thermal-noise state
    # with an ill-conditioned V -- all must still normalize exactly.
    vacuum = GaussianState.vacuum(modes=("a", "b"))
    assert vacuum.to_qutip(N_cutoff=10).tr() == pytest.approx(1.0, abs=1e-9)

    displaced_only = vacuum.copy()
    displaced_only.displacement[0] = 1.5
    assert displaced_only.to_qutip(N_cutoff=15).tr() == pytest.approx(1.0, abs=1e-9)

    state = (
        vacuum.squeeze("a", r=0.5)
        .squeeze("b", r=0.5, theta=np.pi / 2)
        .beam_splitter("a", "b", eta=0.5)
    )
    noisy_state = LossChannels.thermal_loss(mode="a", eta=0.9, n_thermal=0.2).apply(state)
    assert noisy_state.to_qutip(N_cutoff=15).tr() == pytest.approx(1.0, abs=1e-9)


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
