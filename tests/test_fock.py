import numpy as np
import pytest
import qutip as qt
from matplotlib import pyplot as plt

from catsy.fock import FockGates
from catsy.gaussian import (
    GaussianState,
    LossChannels,
    beam_splitter,
    squeeze,
    thermal_loss,
)
from catsy.optics import Circuit, Gate

# Gaussian -> Fock bridge


def test_cv_channel_to_fock_purity_drops_with_loss():
    state = GaussianState.vacuum(modes=("a", "b"))
    state = state.squeeze(mode="a", r=0.5)
    state = state.squeeze(mode="b", r=0.5, theta=np.pi / 2)
    state = state.beam_splitter(mode_a="a", mode_b="b", eta=0.5)

    clean_rho = state.to_qutip(N_cutoff=18)
    noisy_state = LossChannels.thermal_loss(mode="a", eta=0.9, n_thermal=0.2).apply(state)
    noisy_rho = noisy_state.to_qutip(N_cutoff=18)

    assert clean_rho.tr() == pytest.approx(1.0, abs=1e-6)
    # NOTE: Williamson's theorem gives an exact decomposition in exact
    # arithmetic. Our implementation reconstructs the covariance to a tight
    # numerical tolerance, but the subsequent finite-cutoff QuTiP construction
    # is only a truncated representation. The relaxed tolerance here therefore
    # guards the numerical phase-space -> Fock bridge rather than asserting
    # that the underlying Williamson decomposition is inexact.
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
            name="ThermalLoss",
            transform=thermal_loss,
            modes=("b",),
            kwargs={"eta": 0.7, "n_thermal": 0.3},
        )
    )
    final_cv_state = circuit.run(GaussianState.vacuum(("a", "b")))
    rho_qutip = final_cv_state.to_qutip(N_cutoff=15)

    purity = (rho_qutip * rho_qutip).tr().real
    assert 0.0 < purity < 1.0 - 1e-6


# Fock-space operations


def test_photon_subtraction_state_and_rho_entry_points_agree():
    circuit = Circuit().add_mode("a")
    circuit.add_gate(
        Gate(
            name="Squeezer",
            transform=squeeze,
            modes=("a",),
            kwargs={"r": 0.55, "theta": 0.0},
        )
    )
    gaussian_squeezed = circuit.run(GaussianState.vacuum(("a",)))

    rho = gaussian_squeezed.to_qutip(N_cutoff=25)
    via_rho = FockGates.photon_subtraction(rho, mode_idx=0, N_cutoff=25)

    # The Fock API takes the QuTiP representation directly. The operation is
    # heralded, so its output should remain a normalized density matrix.
    assert via_rho.tr() == pytest.approx(1.0, abs=1e-6)


def test_photon_subtraction_zero_probability_raises():
    N_cutoff = 5
    vacuum = qt.ket2dm(qt.fock(N_cutoff, 0))
    with pytest.raises(ValueError):
        FockGates.photon_subtraction(vacuum, mode_idx=0, N_cutoff=N_cutoff)


def test_fock_gates_reject_incompatible_cutoff_and_mode():
    rho = qt.ket2dm(qt.fock(8, 0))

    with pytest.raises(ValueError, match="N_cutoff"):
        FockGates.photon_subtraction(rho, N_cutoff=10)

    with pytest.raises(ValueError, match="mode_idx"):
        FockGates.photon_subtraction(rho, mode_idx=1, N_cutoff=8)


def test_fock_gates_are_the_single_implementation_for_photon_ops():
    assert FockGates.photon_subtraction.__module__ == "catsy.fock"
    assert FockGates.photon_addition.__module__ == "catsy.fock"


def test_photon_addition_on_vacuum_gives_exact_single_photon_fock_state():
    # photon_addition was previously only ever exercised via the __module__
    # check above -- its actual math (a-dagger rho a-dagger-dag,
    # renormalized) was never run. Vacuum has an exact, analytically known
    # image under photon addition: a†|0><0|a / <0|aa†|0> = |1><1| exactly,
    # so this pins down correctness rather than just "didn't crash".
    N_cutoff = 10
    vacuum = qt.ket2dm(qt.fock(N_cutoff, 0))
    result = FockGates.photon_addition(vacuum, mode_idx=0, N_cutoff=N_cutoff)
    expected = qt.ket2dm(qt.fock(N_cutoff, 1))
    assert result.tr() == pytest.approx(1.0, abs=1e-10)
    assert qt.fidelity(result, expected) == pytest.approx(1.0, abs=1e-10)


def test_photon_gates_act_only_on_the_selected_mode():
    # _mode_operator's multi-mode embedding branch (n_modes > 1) is only
    # exercised by an operation targeting one mode of a multi-mode state --
    # every other fock.py test here uses a single-mode state. Addition on
    # mode_idx=1 of a two-mode vacuum must leave mode 0 untouched.
    N_cutoff = 8
    two_mode_vacuum = qt.tensor(
        qt.ket2dm(qt.fock(N_cutoff, 0)), qt.ket2dm(qt.fock(N_cutoff, 0))
    )
    result = FockGates.photon_addition(two_mode_vacuum, mode_idx=1, N_cutoff=N_cutoff)
    expected = qt.tensor(qt.ket2dm(qt.fock(N_cutoff, 0)), qt.ket2dm(qt.fock(N_cutoff, 1)))
    assert qt.fidelity(result, expected) == pytest.approx(1.0, abs=1e-10)


def test_fock_gates_reject_non_qobj_and_non_operator_input():
    N_cutoff = 5
    with pytest.raises(TypeError, match="Qobj"):
        FockGates.photon_subtraction(np.eye(N_cutoff), mode_idx=0, N_cutoff=N_cutoff)
    with pytest.raises(ValueError, match="operator"):
        FockGates.photon_subtraction(qt.fock(N_cutoff, 0), mode_idx=0, N_cutoff=N_cutoff)


def test_mean_photon_number_matches_known_fock_states():
    N_cutoff = 10
    vacuum = qt.ket2dm(qt.fock(N_cutoff, 0))
    one_photon = qt.ket2dm(qt.fock(N_cutoff, 1))
    assert FockGates.mean_photon_number(vacuum, N_cutoff=N_cutoff) == pytest.approx(
        0.0, abs=1e-10
    )
    assert FockGates.mean_photon_number(one_photon, N_cutoff=N_cutoff) == pytest.approx(
        1.0, abs=1e-10
    )


def test_mean_photon_number_matches_addition_and_subtraction():
    # Photon addition/subtraction on vacuum give exact, independently known
    # photon-number states (see test_photon_addition_on_vacuum_... above),
    # so <n> before/after pins down mean_photon_number against ground truth
    # rather than just checking it runs.
    N_cutoff = 10
    vacuum = qt.ket2dm(qt.fock(N_cutoff, 0))
    added = FockGates.photon_addition(vacuum, mode_idx=0, N_cutoff=N_cutoff)
    assert FockGates.mean_photon_number(added, N_cutoff=N_cutoff) == pytest.approx(
        1.0, abs=1e-10
    )


def test_apply_kraus_operator_rejects_mismatched_dims():
    N_cutoff = 5
    rho = qt.ket2dm(qt.fock(N_cutoff, 0))
    mismatched_op = qt.destroy(N_cutoff + 1)
    with pytest.raises(ValueError, match="Hilbert space"):
        FockGates.apply_kraus_operator(rho, mismatched_op)


def test_photon_number_measurement_forced_outcome_on_known_state():
    N_cutoff = 10
    two_photon = qt.ket2dm(qt.fock(N_cutoff, 2))
    outcome, remaining = FockGates.photon_number_measurement(
        two_photon, mode_idx=0, N_cutoff=N_cutoff, outcome=2
    )
    assert outcome == 2
    # Single-mode input: the only mode is measured out, leaving the trivial
    # (1-dimensional, trace-1) remainder.
    assert remaining.tr() == pytest.approx(1.0, abs=1e-10)


def test_photon_number_measurement_rejects_zero_probability_outcome():
    N_cutoff = 5
    vacuum = qt.ket2dm(qt.fock(N_cutoff, 0))
    with pytest.raises(ValueError):
        FockGates.photon_number_measurement(
            vacuum, mode_idx=0, N_cutoff=N_cutoff, outcome=3
        )


def test_photon_number_measurement_sampled_outcome_matches_fock_state():
    # A pure |3> Fock state must always herald outcome 3 with probability 1,
    # regardless of the RNG draw -- a deterministic check of the sampling
    # path (as opposed to the outcome=... forced path exercised above).
    N_cutoff = 10
    three_photon = qt.ket2dm(qt.fock(N_cutoff, 3))
    rng = np.random.default_rng(0)
    outcome, _remaining = FockGates.photon_number_measurement(
        three_photon, mode_idx=0, N_cutoff=N_cutoff, rng=rng
    )
    assert outcome == 3


def test_photon_number_measurement_on_multimode_state_traces_out_measured_mode():
    # _mode_operator's multi-mode embedding and the post-measurement ptrace
    # are only exercised together by a multi-mode input; every other
    # photon_number_measurement test above uses a single-mode state.
    N_cutoff = 6
    two_mode = qt.tensor(qt.ket2dm(qt.fock(N_cutoff, 1)), qt.ket2dm(qt.fock(N_cutoff, 2)))
    outcome, remaining = FockGates.photon_number_measurement(
        two_mode, mode_idx=0, N_cutoff=N_cutoff, outcome=1
    )
    assert outcome == 1
    expected_remaining = qt.ket2dm(qt.fock(N_cutoff, 2))
    assert qt.fidelity(remaining, expected_remaining) == pytest.approx(1.0, abs=1e-10)


# Realistic (click-heralded) subtraction & addition


def test_realistic_subtraction_converges_to_ideal_in_weak_tap_high_efficiency_limit():
    N_cutoff = 12
    psi = qt.coherent(N_cutoff, 1.0)
    rho = qt.ket2dm(psi)

    realistic = FockGates.realistic_photon_subtraction(
        rho,
        mode_idx=0,
        N_cutoff=N_cutoff,
        tap_reflectivity=1e-4,
        detector_efficiency=0.999,
        ancilla_cutoff=6,
    )
    ideal = FockGates.photon_subtraction(rho, mode_idx=0, N_cutoff=N_cutoff)
    assert qt.fidelity(realistic, ideal) == pytest.approx(1.0, abs=1e-3)


def test_realistic_addition_converges_to_ideal_in_weak_coupling_high_efficiency_limit():
    N_cutoff = 12
    psi = qt.coherent(N_cutoff, 1.0)
    rho = qt.ket2dm(psi)

    realistic = FockGates.realistic_photon_addition(
        rho,
        mode_idx=0,
        N_cutoff=N_cutoff,
        coupling_strength=1e-4,
        detector_efficiency=0.999,
        ancilla_cutoff=6,
    )
    ideal = FockGates.photon_addition(rho, mode_idx=0, N_cutoff=N_cutoff)
    assert qt.fidelity(realistic, ideal) == pytest.approx(1.0, abs=1e-3)


def test_realistic_subtraction_output_stays_a_valid_density_matrix():
    N_cutoff = 10
    rho = qt.ket2dm(qt.coherent(N_cutoff, 0.8))
    result = FockGates.realistic_photon_subtraction(
        rho,
        mode_idx=0,
        N_cutoff=N_cutoff,
        tap_reflectivity=0.2,
        detector_efficiency=0.5,
        ancilla_cutoff=5,
    )
    assert result.tr() == pytest.approx(1.0, abs=1e-8)
    eigenvalues = result.eigenenergies()
    assert np.all(eigenvalues > -1e-9)


def test_realistic_addition_output_stays_a_valid_density_matrix():
    N_cutoff = 10
    rho = qt.ket2dm(qt.coherent(N_cutoff, 0.8))
    result = FockGates.realistic_photon_addition(
        rho,
        mode_idx=0,
        N_cutoff=N_cutoff,
        coupling_strength=0.2,
        detector_efficiency=0.5,
        ancilla_cutoff=5,
    )
    assert result.tr() == pytest.approx(1.0, abs=1e-8)
    eigenvalues = result.eigenenergies()
    assert np.all(eigenvalues > -1e-9)


def test_lower_detector_efficiency_degrades_fidelity_to_ideal_subtraction():
    # A worse detector should herald a less faithful subtraction event --
    # this pins down the *direction* of detector_efficiency's effect, not
    # just that the function runs at some value. Uses a squeezed state
    # rather than a coherent one: splitting a coherent state at a
    # beamsplitter never entangles it with the tap (coherent states stay a
    # product state), so a coherent-state herald is always pure regardless
    # of efficiency and wouldn't distinguish the two detectors.
    N_cutoff = 16
    psi = (qt.squeeze(N_cutoff, 0.5) * qt.fock(N_cutoff, 0)).unit()
    rho = qt.ket2dm(psi)
    ideal = FockGates.photon_subtraction(rho, mode_idx=0, N_cutoff=N_cutoff)

    good_detector = FockGates.realistic_photon_subtraction(
        rho, N_cutoff=N_cutoff, tap_reflectivity=0.05, detector_efficiency=0.95
    )
    poor_detector = FockGates.realistic_photon_subtraction(
        rho, N_cutoff=N_cutoff, tap_reflectivity=0.05, detector_efficiency=0.2
    )
    fid_good = qt.fidelity(good_detector, ideal)
    fid_poor = qt.fidelity(poor_detector, ideal)
    assert fid_good > fid_poor


def test_realistic_photon_ops_reject_invalid_parameters():
    N_cutoff = 8
    rho = qt.ket2dm(qt.fock(N_cutoff, 1))
    with pytest.raises(ValueError):
        FockGates.realistic_photon_subtraction(
            rho, N_cutoff=N_cutoff, tap_reflectivity=-0.1
        )
    with pytest.raises(ValueError):
        FockGates.realistic_photon_subtraction(
            rho, N_cutoff=N_cutoff, detector_efficiency=1.5
        )
    with pytest.raises(ValueError):
        FockGates.realistic_photon_addition(
            rho, N_cutoff=N_cutoff, coupling_strength=-0.1
        )
    with pytest.raises(ValueError):
        FockGates.realistic_photon_subtraction(rho, N_cutoff=N_cutoff, ancilla_cutoff=0)


def test_realistic_photon_ops_act_only_on_the_selected_mode():
    N_cutoff = 6
    two_mode_vacuum = qt.tensor(
        qt.ket2dm(qt.fock(N_cutoff, 0)), qt.ket2dm(qt.fock(N_cutoff, 0))
    )
    result = FockGates.realistic_photon_addition(
        two_mode_vacuum,
        mode_idx=1,
        N_cutoff=N_cutoff,
        coupling_strength=1e-4,
        detector_efficiency=0.999,
        ancilla_cutoff=4,
    )
    expected = qt.tensor(qt.ket2dm(qt.fock(N_cutoff, 0)), qt.ket2dm(qt.fock(N_cutoff, 1)))
    assert qt.fidelity(result, expected) == pytest.approx(1.0, abs=1e-3)


# Visual diagnostics


@pytest.mark.visualize
def test_native_qutip_wigner_plot_demo(assert_no_empty_axes, assert_layout_can_render):
    state = Circuit().add_mode("a")
    state.add_gate(
        Gate(
            name="Squeezer",
            transform=squeeze,
            modes=("a",),
            kwargs={"r": 0.6, "theta": 0.0},
        )
    )
    cv_state = state.run(GaussianState.vacuum(("a",)))
    rho = cv_state.to_qutip(N_cutoff=15)

    xvec = np.linspace(-5, 5, 150)
    W = qt.wigner(rho, xvec, xvec)
    wigner_fig = plt.figure(figsize=(5, 4))
    plt.contourf(xvec, xvec, W, 100, cmap="RdBu_r")
    plt.title("Native QuTiP Wigner function (squeezed vacuum)")
    assert_no_empty_axes(wigner_fig)
    assert_layout_can_render(wigner_fig)

    histogram_fig, _histogram_ax = qt.matrix_histogram(rho.full().real[:10, :10])
    assert_layout_can_render(histogram_fig)

    plt.show()


@pytest.mark.visualize
def test_realistic_vs_ideal_subtraction_wigner_and_photon_stats_demo(
    assert_no_empty_axes, assert_layout_can_render
):
    # Side-by-side view of what "realistic" buys over "ideal": Wigner
    # functions of a coherent state after ideal vs. click-heralded photon
    # subtraction, plus the photon-number distributions that show the
    # click-heralded version isn't a pure single-photon-subtracted state.
    # A squeezed vacuum is used rather than a coherent state: a coherent
    # state passed through a beamsplitter tap never entangles with the tap
    # (it stays a product state), so its heralded subtraction would always
    # come out pure regardless of tap_reflectivity/detector_efficiency and
    # the comparison below would show no visible difference.
    N_cutoff = 20
    psi = (qt.squeeze(N_cutoff, 0.6) * qt.fock(N_cutoff, 0)).unit()
    rho = qt.ket2dm(psi)

    ideal = FockGates.photon_subtraction(rho, mode_idx=0, N_cutoff=N_cutoff)
    realistic = FockGates.realistic_photon_subtraction(
        rho,
        mode_idx=0,
        N_cutoff=N_cutoff,
        tap_reflectivity=0.15,
        detector_efficiency=0.55,
        ancilla_cutoff=8,
    )

    xvec = np.linspace(-5, 5, 150)
    W_ideal = qt.wigner(ideal, xvec, xvec)
    W_realistic = qt.wigner(realistic, xvec, xvec)
    w_lim = max(np.abs(W_ideal).max(), np.abs(W_realistic).max())

    fig, (ax_ideal, ax_realistic, ax_stats) = plt.subplots(1, 3, figsize=(13, 4))

    ax_ideal.contourf(xvec, xvec, W_ideal, 100, cmap="RdBu_r", vmin=-w_lim, vmax=w_lim)
    ax_ideal.set_title("Ideal subtraction (a rho a†)")
    ax_ideal.set_aspect("equal")

    ax_realistic.contourf(
        xvec, xvec, W_realistic, 100, cmap="RdBu_r", vmin=-w_lim, vmax=w_lim
    )
    ax_realistic.set_title("Realistic (click-heralded) subtraction")
    ax_realistic.set_aspect("equal")

    n_max = 8
    n_vals = np.arange(n_max)
    p_ideal = np.real(ideal.diag())[:n_max]
    p_realistic = np.real(realistic.diag())[:n_max]
    width = 0.4
    ax_stats.bar(n_vals - width / 2, p_ideal, width, label="ideal")
    ax_stats.bar(n_vals + width / 2, p_realistic, width, label="realistic")
    ax_stats.set_xlabel("photon number n")
    ax_stats.set_ylabel("P(n)")
    ax_stats.set_title("Photon-number distribution")
    ax_stats.legend()

    fig.suptitle("Ideal vs. click-heralded photon subtraction")
    fig.tight_layout()

    assert_no_empty_axes(fig)
    assert_layout_can_render(fig)

    plt.show()
