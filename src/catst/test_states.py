from time import perf_counter

import numpy as np
import pytest
import qutip as qt
from matplotlib import pyplot as plt

from .states import (
    OPERATION_REGISTRY,
    FockOperations,
    GaussianChannel,
    GaussianCircuit,
    GaussianMeasurements,
    GaussianOperations,
    GaussianState,
    NonGaussianOperations,
    QBSChannels,
    QBSSimulator,
    compute_joint_correlation,
    compute_wigner_analytically,
    plot_joint_correlation,
    plot_wigner,
)

PLOT = 0  # flip on locally to pop up matplotlib windows while developing

RNG = np.random.default_rng(1234)  # shared, seeded RNG -> reproducible test outcomes


# ---------------------------------------------------------------------------
# Phase-space layer: gates, channels, circuit compiler
# ---------------------------------------------------------------------------


def test_vacuum_is_shot_noise_limited():
    state = GaussianOperations.create_vacuum(modes=("a", "b"))
    assert state.displacement.shape == (4,)
    np.testing.assert_allclose(state.covariance, 0.5 * np.eye(4))


def test_squeezing_preserves_purity_determinant():
    # A pure Gaussian state has det(V) == 0.25 in this (hbar=1, vacuum=0.5*I)
    # convention; squeezing is a symplectic (purity-preserving) operation.
    state = GaussianOperations.create_vacuum(modes=("a",))
    squeezed = GaussianOperations.apply_squeezing(state, mode="a", r=0.8, theta=0.3)
    assert np.linalg.det(squeezed.covariance) == pytest.approx(0.25, rel=1e-9)
    # And squeezing should actually squeeze: at theta=0 the p-variance grows,
    # the q-variance shrinks below the vacuum value.
    unrotated = GaussianOperations.apply_squeezing(state, mode="a", r=0.8, theta=0.0)
    assert unrotated.covariance[0, 0] < 0.5
    assert unrotated.covariance[1, 1] > 0.5


def test_beam_splitter_entangles_independent_modes():
    state = GaussianOperations.create_vacuum(modes=("a", "b"))
    state = GaussianOperations.apply_squeezing(state, mode="a", r=0.6, theta=0.0)
    state = GaussianOperations.apply_squeezing(state, mode="b", r=0.6, theta=np.pi / 2)

    # Independent modes -> block-diagonal covariance beforehand.
    np.testing.assert_allclose(state.covariance[0:2, 2:4], np.zeros((2, 2)), atol=1e-12)

    mixed = GaussianOperations.apply_beam_splitter(
        state, mode_a="a", mode_b="b", eta=0.5
    )
    # A 50:50 BS on two independently squeezed modes must generate nonzero
    # cross-correlations (entanglement) between them.
    assert np.abs(mixed.covariance[0:2, 2:4]).max() > 1e-6


def test_beam_splitter_rejects_invalid_eta():
    state = GaussianOperations.create_vacuum(modes=("a", "b"))
    with pytest.raises(ValueError):
        GaussianOperations.apply_beam_splitter(state, mode_a="a", mode_b="b", eta=1.5)
    with pytest.raises(ValueError):
        GaussianOperations.apply_beam_splitter(state, mode_a="a", mode_b="a", eta=0.5)


def test_loss_reduces_toward_vacuum():
    state = GaussianOperations.create_vacuum(modes=("a",))
    state = GaussianOperations.apply_squeezing(state, mode="a", r=1.0, theta=0.0)
    lossy = GaussianOperations.apply_loss(state, mode="a", eta=0.0)
    # eta=0 is total loss -> mode is replaced by vacuum regardless of input.
    np.testing.assert_allclose(lossy.covariance, 0.5 * np.eye(2), atol=1e-9)


def test_get_mode_index_unknown_mode_raises():
    state = GaussianOperations.create_vacuum(modes=("a", "b"))
    with pytest.raises(ValueError):
        state.get_mode_index("z")


def test_duplicate_mode_names_rejected():
    with pytest.raises(ValueError):
        GaussianState(
            modes=("a", "a"),
            displacement=np.zeros(4),
            covariance=0.5 * np.eye(4),
        )


def test_classical_phase_jitter_channel_applies():
    # Regression test: this channel used to build a (1,2)-shaped Y matrix,
    # which fails GaussianChannel's (2,2) validation. It must now apply cleanly
    # and only add noise to the p-quadrature.
    state = GaussianOperations.create_vacuum(modes=("a",))
    channel = QBSChannels.classical_phase_jitter(mode="a", sigma_phi=0.3)
    jittered = channel.apply(state)
    assert jittered.covariance[0, 0] == pytest.approx(0.5)  # q untouched
    assert jittered.covariance[1, 1] > 0.5  # p gained noise


def test_channel_dimension_validation():
    with pytest.raises(ValueError):
        GaussianChannel(target_modes=("a",), X=np.eye(2), Y=np.eye(3), d0=np.zeros(2))


def test_circuit_matches_manual_operation_chain():
    manual = GaussianOperations.create_vacuum(modes=("a", "b"))
    manual = GaussianOperations.apply_squeezing(manual, mode="a", r=0.6, theta=0.0)
    manual = GaussianOperations.apply_squeezing(
        manual, mode="b", r=0.6, theta=np.pi / 2
    )
    manual = GaussianOperations.apply_beam_splitter(
        manual, mode_a="a", mode_b="b", eta=0.5
    )
    manual = QBSChannels.thermal_loss(mode="b", eta=0.7, n_thermal=0.3).apply(manual)

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


def test_circuit_rejects_unregistered_mode():
    circuit = GaussianCircuit()
    circuit.add_mode("a")
    circuit.squeeze(mode="z", r=0.5)  # 'z' was never added
    with pytest.raises(ValueError):
        circuit.compile_and_run()


def test_circuit_rejects_empty_mode_set():
    with pytest.raises(ValueError):
        GaussianCircuit().compile_and_run()


def test_circuit_extensible_via_registry():
    # New gates plug into the same dispatch the built-ins use — no need to
    # touch compile_and_run.
    calls = []

    def _my_op(state, modes, **kwargs):
        calls.append((modes, kwargs))
        return state

    GaussianCircuit.register("MyCustomOp", _my_op)
    try:
        circuit = GaussianCircuit()
        circuit.add_mode("a")
        circuit._add_op("MyCustomOp", ("a",), foo=1)
        circuit.compile_and_run()
        assert calls == [(("a",), {"foo": 1})]
    finally:
        del OPERATION_REGISTRY["MyCustomOp"]


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


def test_gaussian_state_roundtrips_through_dict():
    state = GaussianOperations.create_vacuum(modes=("a", "b"))
    state = GaussianOperations.apply_squeezing(state, mode="a", r=0.4)
    restored = GaussianState.from_dict(state.to_dict())
    assert restored.modes == state.modes
    np.testing.assert_allclose(restored.displacement, state.displacement)
    np.testing.assert_allclose(restored.covariance, state.covariance)


def test_gaussian_state_roundtrips_through_file(tmp_path):
    state = GaussianOperations.create_vacuum(modes=("a",))
    state = GaussianOperations.apply_squeezing(state, mode="a", r=0.9, theta=0.2)
    path = tmp_path / "state.json"
    state.save(path)
    restored = GaussianState.load(path)
    np.testing.assert_allclose(restored.covariance, state.covariance)


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


# ---------------------------------------------------------------------------
# Phase-space <-> Fock-space bridge (Williamson conversion)
# ---------------------------------------------------------------------------


def test_cv_channel_to_fock_purity_drops_with_loss():
    state = GaussianOperations.create_vacuum(modes=("a", "b"))
    state = GaussianOperations.apply_squeezing(state, mode="a", r=0.5)
    state = GaussianOperations.apply_squeezing(state, mode="b", r=0.5, theta=np.pi / 2)
    state = GaussianOperations.apply_beam_splitter(
        state, mode_a="a", mode_b="b", eta=0.5
    )

    clean_rho = state.to_qutip(N_cutoff=18)
    noisy_state = QBSChannels.thermal_loss(mode="a", eta=0.9, n_thermal=0.2).apply(
        state
    )
    noisy_rho = noisy_state.to_qutip(N_cutoff=18)

    assert clean_rho.tr() == pytest.approx(1.0, abs=1e-6)
    # NOTE: this particular covariance matrix hits a known, cutoff-independent
    # precision limit of the sqrtm/logm-based symplectic decomposition in
    # to_qutip (~0.7% trace error even at N_cutoff=35) — see the docstring on
    # GaussianState.to_qutip. 1e-2 tolerance here reflects that limitation,
    # not truncation error.
    assert noisy_rho.tr() == pytest.approx(1.0, abs=1e-2)

    # The two-mode state started pure (r=0.5 squeezing + a lossless BS is
    # unitary). A non-unitary thermal-loss channel on top of it must make the
    # *global* state strictly mixed. (Reduced-subsystem entropy on just mode
    # A isn't a safe proxy here: loss on A both adds local noise, which raises
    # its entropy, and weakens A-B entanglement, which lowers it — the two
    # effects can go either way. Global purity has no such ambiguity.)
    purity_clean = (clean_rho * clean_rho).tr().real
    purity_noisy = (noisy_rho * noisy_rho).tr().real
    assert purity_clean == pytest.approx(1.0, abs=1e-2)
    assert purity_noisy < purity_clean - 1e-3


def test_qo_epr_purity_drops_below_one_after_loss():
    circuit = GaussianCircuit()
    circuit.add_mode("a").add_mode("b")
    circuit.squeeze(mode="a", r=0.6, theta=0.0).squeeze(
        mode="b", r=0.6, theta=np.pi / 2
    ).beam_splitter(mode_a="a", mode_b="b", eta=0.5).thermal_loss(
        mode="b", eta=0.7, n_thermal=0.3
    )
    final_cv_state = circuit.compile_and_run()
    rho_qutip = final_cv_state.to_qutip(N_cutoff=15)

    purity = (rho_qutip * rho_qutip).tr().real
    assert 0.0 < purity < 1.0 - 1e-6


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------


def test_homodyne_measurement_collapses_epr_correlation():
    circuit = GaussianCircuit()
    circuit.add_mode("a").add_mode("b")
    circuit.squeeze(mode="a", r=1.0).squeeze(
        mode="b", r=1.0, theta=np.pi / 2
    ).beam_splitter(mode_a="a", mode_b="b", eta=0.5)
    epr_state = circuit.compile_and_run()

    val, collapsed = GaussianMeasurements.homodyne_measurement(
        epr_state, measured_mode="a", phi=0.0, outcome=2.5
    )
    assert val == 2.5
    assert collapsed.modes == ("b",)
    # EPR correlations mean a measurement on 'a' shifts b's mean the opposite way.
    val_neg, collapsed_neg = GaussianMeasurements.homodyne_measurement(
        epr_state, measured_mode="a", phi=0.0, outcome=-2.5
    )
    assert np.sign(collapsed.displacement[0]) != np.sign(collapsed_neg.displacement[0])


def test_homodyne_measurement_is_reproducible_with_seeded_rng():
    state = GaussianOperations.create_vacuum(modes=("a",))
    v1, _ = GaussianMeasurements.homodyne_measurement(
        state, measured_mode="a", phi=0.0, rng=np.random.default_rng(42)
    )
    v2, _ = GaussianMeasurements.homodyne_measurement(
        state, measured_mode="a", phi=0.0, rng=np.random.default_rng(42)
    )
    assert v1 == v2


# ---------------------------------------------------------------------------
# Analytic phase-space plotting
# ---------------------------------------------------------------------------


def test_wigner_analytical_matches_gaussian_normalization():
    circuit = GaussianCircuit()
    circuit.add_mode("a")
    circuit.squeeze(mode="a", r=1.0, theta=0.0)
    test_state = circuit.compile_and_run()
    test_state.displacement[0] = 2.0
    test_state.displacement[1] = 1.0

    W, X, P = compute_wigner_analytically(
        test_state, mode_name="a", x_max=8.0, num_points=200
    )
    # A properly normalized Wigner function integrates to ~1 over phase space.
    dx = (2 * 8.0) / 199
    integral = W.sum() * dx * dx
    assert integral == pytest.approx(1.0, rel=1e-2)
    if PLOT:
        plot_wigner(W, X, P, mode_name="a")


def test_joint_correlation_plot_runs():
    circuit = GaussianCircuit()
    circuit.add_mode("a").add_mode("b")
    circuit.squeeze(mode="a", r=0.6).squeeze(
        mode="b", r=0.6, theta=np.pi
    ).beam_splitter(mode_a="a", mode_b="b", eta=0.5)
    cv_state = circuit.compile_and_run()
    P, X_a, X_b = compute_joint_correlation(cv_state, "a", "b")
    assert P.shape == (150, 150)
    assert np.all(P >= 0)
    if PLOT:
        plot_joint_correlation(P, X_a, X_b, "a", "b")


# ---------------------------------------------------------------------------
# Non-Gaussian operations (single shared FockOperations implementation)
# ---------------------------------------------------------------------------


def test_photon_subtraction_state_and_rho_entry_points_agree():
    circuit = GaussianCircuit()
    circuit.add_mode("a")
    circuit.squeeze(mode="a", r=0.55)
    gaussian_squeezed = circuit.compile_and_run()

    via_state = NonGaussianOperations.photon_subtraction(
        gaussian_squeezed, mode_name="a", N_cutoff=25
    )
    rho = gaussian_squeezed.to_qutip(N_cutoff=25)
    via_rho = FockOperations.photon_subtraction(rho, mode_idx=0, N_cutoff=25)

    assert via_state == via_rho  # both entry points dispatch to the same code
    # A photon-subtracted squeezed vacuum should show Wigner negativity —
    # i.e. it's genuinely non-Gaussian. Purity must still be < 1 (mixed by loss? No,
    # it's pure) so instead check it's a valid, normalized state:
    assert via_state.tr() == pytest.approx(1.0, abs=1e-6)


def test_photon_subtraction_zero_probability_raises():
    N_cutoff = 5
    vacuum = qt.ket2dm(qt.fock(N_cutoff, 0))
    with pytest.raises(ValueError):
        FockOperations.photon_subtraction(vacuum, mode_idx=0, N_cutoff=N_cutoff)


def test_qbs_simulator_photon_ops_delegate_to_fock_operations():
    # QBSSimulator.photon_subtraction/addition must be the same function
    # object as FockOperations' — not a second, drifting implementation.
    assert QBSSimulator.photon_subtraction is FockOperations.photon_subtraction
    assert QBSSimulator.photon_addition is FockOperations.photon_addition


def test_phase_rotation_preserves_photon_number_and_purity():
    state = GaussianOperations.create_vacuum(modes=("a",))
    squeezed = GaussianOperations.apply_squeezing(state, mode="a", r=0.7, theta=0.0)
    rotated = GaussianOperations.apply_phase_rotation(squeezed, mode="a", phi=1.1)
    # Rotation is passive/energy-preserving: purity (det V) is unchanged...
    assert np.linalg.det(rotated.covariance) == pytest.approx(
        np.linalg.det(squeezed.covariance)
    )
    # ...but a full 2*pi rotation must return exactly to the start.
    full_turn = GaussianOperations.apply_phase_rotation(
        squeezed, mode="a", phi=2 * np.pi
    )
    np.testing.assert_allclose(full_turn.covariance, squeezed.covariance, atol=1e-9)


def test_circuit_rotate_matches_manual_rotation():
    manual = GaussianOperations.create_vacuum(modes=("a",))
    manual = GaussianOperations.apply_squeezing(manual, mode="a", r=0.5)
    manual = GaussianOperations.apply_phase_rotation(manual, mode="a", phi=0.4)

    circuit = GaussianCircuit().add_mode("a")
    circuit.squeeze(mode="a", r=0.5).rotate(mode="a", phi=0.4)
    compiled = circuit.compile_and_run()

    np.testing.assert_allclose(compiled.covariance, manual.covariance, atol=1e-10)


def test_to_qutip_handles_plain_vacuum_and_pure_displacement():
    # Regression test: to_qutip used to crash on any state with no squeezing
    # at all (H_cv stayed a plain int 0, which has no .expm()).
    vacuum = GaussianOperations.create_vacuum(modes=("a", "b"))
    rho = vacuum.to_qutip(N_cutoff=10)
    assert rho.tr() == pytest.approx(1.0, abs=1e-9)

    displaced_only = vacuum.copy()
    displaced_only.displacement[0] = 1.5
    rho2 = displaced_only.to_qutip(N_cutoff=15)
    assert rho2.tr() == pytest.approx(1.0, abs=1e-9)


def test_to_qutip_trace_always_exactly_one_even_with_ill_conditioned_v():
    # Regression test: this covariance matrix (squeeze + BS + thermal loss)
    # used to leave tr(rho) ~0.7% below 1 due to a mildly non-symplectic S
    # from sqrtm(). Symmetrizing the generator now guarantees tr==1 exactly,
    # regardless of the underlying decomposition's precision.
    state = GaussianOperations.create_vacuum(modes=("a", "b"))
    state = GaussianOperations.apply_squeezing(state, mode="a", r=0.5)
    state = GaussianOperations.apply_squeezing(state, mode="b", r=0.5, theta=np.pi / 2)
    state = GaussianOperations.apply_beam_splitter(
        state, mode_a="a", mode_b="b", eta=0.5
    )
    noisy_state = QBSChannels.thermal_loss(mode="a", eta=0.9, n_thermal=0.2).apply(
        state
    )
    rho = noisy_state.to_qutip(N_cutoff=18)
    assert rho.tr() == pytest.approx(1.0, abs=1e-9)


def test_to_qutip_rejects_invalid_n_cutoff():
    state = GaussianOperations.create_vacuum(modes=("a",))
    with pytest.raises(ValueError):
        state.to_qutip(N_cutoff=0)
    with pytest.raises(ValueError):
        state.to_qutip(N_cutoff=-5)


def test_gaussian_channel_roundtrips_through_file(tmp_path):
    channel = QBSChannels.thermal_loss(mode="a", eta=0.8, n_thermal=0.1)
    path = tmp_path / "channel.json"
    channel.save(path)
    restored = GaussianChannel.load(path)

    state = GaussianOperations.create_vacuum(modes=("a",))
    original_out = channel.apply(state)
    restored_out = restored.apply(state)
    np.testing.assert_allclose(restored_out.covariance, original_out.covariance)


def test_heterodyne_measurement_adds_vacuum_noise_and_collapses_to_coherent():
    state = GaussianOperations.create_vacuum(modes=("a", "b"))
    state = GaussianOperations.apply_squeezing(state, mode="a", r=1.0)
    state = GaussianOperations.apply_squeezing(state, mode="b", r=1.0, theta=np.pi / 2)
    state = GaussianOperations.apply_beam_splitter(
        state, mode_a="a", mode_b="b", eta=0.5
    )

    outcome, collapsed = GaussianMeasurements.heterodyne_measurement(
        state, measured_mode="a", outcome=np.array([1.0, 0.5])
    )
    np.testing.assert_allclose(outcome, [1.0, 0.5])
    assert collapsed.modes == ("b",)
    # Heterodyne always adds 0.5*I of extra measurement noise relative to
    # homodyne, so the remaining mode's conditional state can't be squeezed
    # below the vacuum level in the measured quadrature's conjugate way that
    # homodyne allows -- concretely, its covariance eigenvalues stay >= 0.5.
    eigvals = np.linalg.eigvalsh(collapsed.covariance)
    assert (eigvals >= 0.5 - 1e-9).all()


def test_heterodyne_measurement_is_reproducible_with_seeded_rng():
    state = GaussianOperations.create_vacuum(modes=("a",))
    v1, _ = GaussianMeasurements.heterodyne_measurement(
        state, measured_mode="a", rng=np.random.default_rng(7)
    )
    v2, _ = GaussianMeasurements.heterodyne_measurement(
        state, measured_mode="a", rng=np.random.default_rng(7)
    )
    np.testing.assert_allclose(v1, v2)


# ---------------------------------------------------------------------------
# Visual-only demos (no assertions beyond "it runs") -- opt in with PLOT=True
# ---------------------------------------------------------------------------


def test_native_qutip_wigner_plot_demo():
    if not PLOT:
        pytest.skip("visual-only demo; set PLOT=True to view")
    import matplotlib.pyplot as plt

    state = GaussianCircuit().add_mode("a")
    state.squeeze(mode="a", r=0.6, theta=0.0)
    cv_state = state.compile_and_run()
    rho = cv_state.to_qutip(N_cutoff=15)

    xvec = np.linspace(-5, 5, 150)
    W = qt.wigner(rho, xvec, xvec)
    plt.figure(figsize=(5, 4))
    plt.contourf(xvec, xvec, W, 100, cmap="RdBu_r")
    plt.title("Native QuTiP Wigner function (squeezed vacuum)")
    qt.matrix_histogram(rho.full().real[:10, :10])
    plt.show()


def test_laser_pulse_cavity_plot_demo():
    if not PLOT:
        pytest.skip("visual-only demo; set PLOT=True to view")
    import matplotlib.pyplot as plt

    N_cutoff = 15
    rho_vacuum = qt.ket2dm(qt.fock(N_cutoff, 0))
    tlist = np.linspace(0, 5, 100)
    states = QBSSimulator.run_cavity_with_pulse(
        rho_init=rho_vacuum,
        tlist=tlist,
        K=0.3,
        kappa=0.05,
        amp=3.0,
        t0=1.5,
        sigma=0.5,
        N_cutoff=N_cutoff,
    )
    n_op = qt.num(N_cutoff)
    photon_numbers = [qt.expect(n_op, s) for s in states]

    plt.figure(figsize=(6, 4))
    plt.plot(tlist, photon_numbers)
    plt.xlabel("time")
    plt.ylabel("<n>")
    plt.title("Driven Kerr cavity: photon number vs time")
    plt.show()


def test_full_cavity_multipanel_plot_demo():
    if not PLOT:
        pytest.skip("visual-only demo; set PLOT=True to view")
    import matplotlib.pyplot as plt

    N_cutoff = 12
    alpha = 1.5
    psi_cat = (qt.coherent(N_cutoff, alpha) + qt.coherent(N_cutoff, -alpha)).unit()
    theta_list = np.linspace(0, 2 * np.pi, 60)
    results = QBSSimulator.scan_mzi_with_loss(
        psi_cat, theta_list, kappa=0.2, N_cutoff=N_cutoff
    )

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].plot(theta_list, results["n1"])
    axes[0].set_title("Mean photon number, arm 1")
    axes[1].plot(theta_list, results["n2"])
    axes[1].set_title("Mean photon number, arm 2")
    axes[2].plot(theta_list, results["parity1"])
    axes[2].set_title("Parity, arm 1")
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Time-dependent simulation (slower, integration-style tests)
# ---------------------------------------------------------------------------


def test_triggered_cavity_end_to_end():
    cv_circuit = GaussianCircuit().add_mode("c")
    cv_circuit.squeeze(mode="c", r=0.1, theta=0.0)
    initial_state = cv_circuit.compile_and_run()

    N_fock = 15
    rho_vacuum = initial_state.to_qutip(N_cutoff=N_fock)

    tlist = np.linspace(0, 5, 60)
    states = QBSSimulator.run_cavity_with_pulse(
        rho_init=rho_vacuum,
        tlist=tlist,
        K=0.4,
        kappa=0.05,
        amp=3.5,
        t0=1.5,
        sigma=0.6,
        N_cutoff=N_fock,
    )
    assert len(states) == len(tlist)
    rho_kerr_cat = states[-1]
    assert rho_kerr_cat.tr() == pytest.approx(1.0, abs=1e-6)

    rho_final_non_gaussian = QBSSimulator.photon_subtraction(
        rho_kerr_cat, N_cutoff=N_fock
    )
    purity = (rho_final_non_gaussian * rho_final_non_gaussian).tr().real
    assert 0.0 < purity <= 1.0 + 1e-9


def test_decoherence_mzi_parity_visibility_drops_with_loss():
    start_time = perf_counter()
    N_cutoff = 10
    alpha = 1.5
    psi_cat = (qt.coherent(N_cutoff, alpha) + qt.coherent(N_cutoff, -alpha)).unit()

    # theta doubles as the arm's loss "exposure time" in this model (see
    # scan_mzi_with_loss), so decoherence only becomes visible after enough
    # theta has accumulated. Compare visibility over the back half of the
    # scan rather than the raw peak-to-peak of the whole trace, which can be
    # dominated by the still-clean early-theta region even at nonzero kappa.
    theta_list = np.linspace(0, 2 * np.pi, 80)
    results_clean = QBSSimulator.scan_mzi_with_loss(
        psi_cat, theta_list, kappa=0.0, N_cutoff=N_cutoff
    )
    results_noisy = QBSSimulator.scan_mzi_with_loss(
        psi_cat, theta_list, kappa=0.4, N_cutoff=N_cutoff
    )

    tail = slice(len(theta_list) // 2, None)
    visibility_clean = np.ptp(np.array(results_clean["parity1"])[tail])
    visibility_noisy = np.ptp(np.array(results_noisy["parity1"])[tail])
    # Loss must wash out (not enhance) the super-resolved parity fringes.
    assert visibility_noisy < visibility_clean
    print(f"MZI decoherence scan runtime: {perf_counter() - start_time:.2f}s")


def test_kerr_state():
    N_cutoff = (
        35  # Höherer Cutoff wichtig, da Kerr-Zustände breite Fock-Verteilungen haben!
    )
    rho_0 = qt.ket2dm(qt.fock(N_cutoff, 0))  # Start im reinen Vakuum

    a = qt.destroy(N_cutoff)

    K = 0.5  # Stärke der Kerr-Nichtlinearität
    pulse_amplitude = 5.0  # Laser-Stärke

    # Laser-Pulsform: Ein schneller, starker Puls zu Beginn, um die Kavität zu laden
    def pulse_shape(t, **kwargs):
        return kwargs["amp"] * np.exp(
            -((t - kwargs["t0"]) ** 2) / (2 * kwargs["sigma"] ** 2)
        )

    # H = H_Kerr + Omega(t) * (a + a^dagger)
    H_kerr = K * a.dag() * a.dag() * a * a  # Statischer Kerr-Term
    H_drive = a + a.dag()  # Zeitabhängiger Laser-Treiber

    H_total = [H_kerr, [H_drive, pulse_shape]]
    pulse_args = {"amp": pulse_amplitude, "t0": 2.0, "sigma": 0.8}

    # Minimale Kavitäten-Dämpfung (Katzen-Zustände sind extrem empfindlich gegen Verlust!)
    kappa = 0.01
    c_ops = [np.sqrt(kappa) * a]

    tlist = np.linspace(0, 6, 200)
    result = qt.mesolve(H_total, rho_0, tlist, c_ops=c_ops, args=pulse_args)
    assert len(result.states) == 200

    if PLOT:
        rho_t0 = result.states[10]  # t ~ 0.0 (gleich am Anfang)
        rho_t1 = result.states[40]  # t ~ 1.8 (kurz nach dem Puls)
        rho_t2 = result.states[110]  # t ~ 3.3 (Kerr fängt an zu verzerren)
        rho_t3 = result.states[-1]  # t = 6.0 (Kollision -> Katze!)

        fig, axes = plt.subplots(2, 2, figsize=(10, 10))
        xvec = np.linspace(-5, 5, 200)

        # Zeitschritt 0
        W0 = qt.wigner(rho_t0, xvec, xvec)
        axes[0][0].contourf(xvec, xvec, W0, 100, cmap="RdBu_r", vmin=-0.25, vmax=0.25)
        axes[0][0].set_title(r"t = 0.3: Kohärenter Laser-Puls")
        axes[0][0].set_xlabel("x")
        axes[0][0].set_ylabel("p")
        axes[0][0].axis("equal")

        # Zeitschritt 1: Der Laser hat ein verschobenes "Gauß-Blob" (Coherent State) erzeugt
        W1 = qt.wigner(rho_t1, xvec, xvec)
        axes[0][1].contourf(xvec, xvec, W1, 100, cmap="RdBu_r", vmin=-0.25, vmax=0.25)
        axes[0][1].set_title(r"t = 1.2: Defomrierter Laser-Puls")
        axes[0][1].set_xlabel("x")
        axes[0][1].set_ylabel("p")
        axes[0][1].axis("equal")

        # Zeitschritt 2: Die Kerr-Nichtlinearität verbiegt den Zustand zu einer Banane
        W2 = qt.wigner(rho_t2, xvec, xvec)
        axes[1][0].contourf(xvec, xvec, W2, 100, cmap="RdBu_r", vmin=-0.25, vmax=0.25)
        axes[1][0].set_title(r"t = 3.3: Kerr-Squeezing & Verbiegung")
        axes[1][0].set_xlabel("x")
        axes[1][0].set_ylabel("p")
        axes[1][0].axis("equal")

        W3 = qt.wigner(rho_t3, xvec, xvec)
        cont = axes[1][1].contourf(
            xvec, xvec, W3, 100, cmap="RdBu_r", vmin=-0.25, vmax=0.25
        )
        axes[1][1].set_title(r"t = 6.0: Kerr-Cat State")
        axes[1][1].set_xlabel("x")
        axes[1][1].set_ylabel("p")
        axes[1][1].axis("equal")

        fig.colorbar(cont, ax=axes[:, :], label="Wigner-Dichte")

        plt.show()


def test_cat_in_mzi():
    N_cutoff = 22
    a = qt.destroy(N_cutoff)

    alpha = 2
    psi_cat = (qt.coherent(N_cutoff, alpha) + qt.coherent(N_cutoff, -alpha)).unit()

    a1 = qt.tensor(qt.destroy(N_cutoff), qt.qeye(N_cutoff))
    a2 = qt.tensor(qt.qeye(N_cutoff), qt.destroy(N_cutoff))

    # Der 50:50 Strahlteiler-Operator zwischen den beiden Armen
    # U_BS = exp(i * pi/4 * (a1^dagger * a2 + a1 * a2^dagger))
    H_BS = (1j * np.pi / 4) * (a1.dag() * a2 + a1 * a2.dag())
    U_BS = H_BS.expm()

    # Eingangs-Zustand des MZI: Katze auf Port 1, Vakuum auf Port 2
    psi_in = qt.tensor(psi_cat, qt.fock(N_cutoff, 0))

    # --- SCHRITT A: Erster Strahlteiler (BS1) ---
    # Erzeugt Verschränkung zwischen beiden Pfaden!
    psi_after_BS1 = U_BS * psi_in

    # --- SCHRITT B: Phasenverschiebung theta im oberen Arm (Arm 1) ---
    # Wir wählen eine Verschiebung von theta = pi/2 (90 Grad) für maximale Interferenzänderung
    theta = np.pi / 4
    U_phase = (1j * theta * a1.dag() * a1).expm()
    psi_after_phase = U_phase * psi_after_BS1

    # --- SCHRITT C: Zweiter Strahlteiler (BS2) ---
    # Rekombination der Pfade
    psi_out = U_BS * psi_after_phase

    rho_out_port1 = qt.ptrace(psi_out, 0)
    rho_out_port2 = qt.ptrace(psi_out, 1)

    if PLOT:
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        xvec = np.linspace(-4, 4, 200)

        # Port 1 Wigner-Funktion
        W_port1 = qt.wigner(rho_out_port1, xvec, xvec)
        axes[0].contourf(xvec, xvec, W_port1, 100, cmap="RdBu_r", vmin=-0.3, vmax=0.3)
        axes[0].set_title(
            "MZI Ausgangsport 1\n(Verschobene Katze bei $\\theta=\\pi/2$)"
        )
        axes[0].set_xlabel("x")
        axes[0].set_ylabel("p")
        axes[0].axis("equal")

        # Port 2 Wigner-Funktion
        W_port2 = qt.wigner(rho_out_port2, xvec, xvec)
        axes[1].contourf(xvec, xvec, W_port2, 100, cmap="RdBu_r", vmin=-0.3, vmax=0.3)
        axes[1].set_title("MZI Ausgangsport 2\n(Gegenphasige Interferenz)")
        axes[1].set_xlabel("x")
        axes[1].set_ylabel("p")
        axes[1].axis("equal")

        plt.tight_layout()
        plt.show()


def test_time_cat_mzi():
    N_cutoff = 22  # Hilbertraum-Dimension pro Mode
    a = qt.destroy(N_cutoff)

    # Eine reine Katze vorbereiten: |cat> = N * (|alpha> + |-alpha>)
    alpha = 4.0 + 2j
    psi_cat = (qt.coherent(N_cutoff, alpha) + qt.coherent(N_cutoff, -alpha)).unit()

    # Zwei-Moden-Operatoren im Gesamtraum aufbauen
    a1 = qt.tensor(qt.destroy(N_cutoff), qt.qeye(N_cutoff))
    a2 = qt.tensor(qt.qeye(N_cutoff), qt.destroy(N_cutoff))

    # Teilchenzahl-Operatoren für die Ausgänge
    n1_op = a1.dag() * a1
    n2_op = a2.dag() * a2

    # Paritäts-Operator für Port 1: P = exp(i * pi * a1^dagger * a1)
    # Photonenzahl gerade (+1) oder ungerade (-1)
    parity1_op = (1j * np.pi * n1_op).expm()

    # 50:50 (BS)
    H_BS = (1j * np.pi / 4) * (a1.dag() * a2 + a1 * a2.dag())
    U_BS = H_BS.expm()

    # Eingangszustand: Katze auf Port 1, Vakuum auf Port 2
    psi_in = qt.tensor(psi_cat, qt.fock(N_cutoff, 0))
    theta_list = np.linspace(0, 2 * np.pi, 200)

    # Listen für die Messergebnisse
    mean_n1 = []
    mean_n2 = []
    parity_port1 = []

    # Erster Strahlteiler ist fix für alle Phasen (Verschränkung im MZI)
    psi_after_BS1 = U_BS * psi_in

    print("🔄 Scanne Phase theta von 0 bis 2pi...")
    for theta in theta_list:
        # 1. Phasenverschiebung im oberen Arm anwenden
        U_phase = (1j * theta * a1.dag() * a1).expm()
        psi_after_phase = U_phase * psi_after_BS1

        # 2. Zweiter Strahlteiler (Rekombination)
        psi_out = U_BS * psi_after_phase

        # 3. Erwartungswerte für Phasenwert
        mean_n1.append(qt.expect(n1_op, psi_out))
        mean_n2.append(qt.expect(n2_op, psi_out))
        parity_port1.append(qt.expect(parity1_op, psi_out).real)

    if PLOT:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

        # Plot A: Klassische Intensitäten (Photonenzahlen) an den Ausgängen
        ax1.plot(
            theta_list / np.pi, mean_n1, label="Ausgangs-Port 1", color="darkblue", lw=2
        )
        ax1.plot(
            theta_list / np.pi,
            mean_n2,
            label="Ausgangs-Port 2",
            color="crimson",
            lw=2,
            ls="--",
        )
        ax1.set_ylabel(r"Mittlere Photonenzahl $\langle n \rangle$")
        ax1.set_title("Mach-Zehnder Interferenz-Fransen (Intensität)")
        ax1.grid(True, ls="--")
        ax1.legend()

        # Plot B: Quanten-Parität am Ausgang 1
        # zeigt die Verschiebung der mikroskopischen Interferenzstreifen
        ax2.plot(
            theta_list / np.pi,
            parity_port1,
            label="Parität Port 1",
            color="purple",
            lw=2.5,
        )
        ax2.axhline(0, color="black", lw=0.5, ls="-")
        ax2.set_xlabel(r"Phasenverschiebung $\theta$ ($\times \pi$)")
        ax2.set_ylabel("Erwartungswert der Parität")
        ax2.set_title("Quanten-Paritäts-Oszillation (Super-Auflösung)")
        ax2.grid(True, ls="--")
        ax2.legend()

        plt.tight_layout()
        plt.show()
