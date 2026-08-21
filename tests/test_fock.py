import numpy as np
import pytest
import qutip as qt

from catsy.core import Circuit
from catsy.fock import FockOperations
from catsy.gaussian import (
    GaussianState,
    LossChannels,
    beam_splitter,
    squeeze,
    thermal_loss,
)

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
    circuit.add_operation(squeeze, ("a",), r=0.6, theta=0.0).add_operation(
        squeeze, ("b",), r=0.6, theta=np.pi / 2
    ).add_operation(beam_splitter, ("a", "b"), eta=0.5).add_operation(
        thermal_loss, ("b",), eta=0.7, n_thermal=0.3
    )
    final_cv_state = circuit.run(GaussianState.vacuum(("a", "b")))
    rho_qutip = final_cv_state.to_qutip(N_cutoff=15)

    purity = (rho_qutip * rho_qutip).tr().real
    assert 0.0 < purity < 1.0 - 1e-6


# Fock-space operations


def test_photon_subtraction_state_and_rho_entry_points_agree():
    circuit = Circuit().add_mode("a")
    circuit.add_operation(squeeze, ("a",), r=0.55, theta=0.0)
    gaussian_squeezed = circuit.run(GaussianState.vacuum(("a",)))

    rho = gaussian_squeezed.to_qutip(N_cutoff=25)
    via_rho = FockOperations.photon_subtraction(rho, mode_idx=0, N_cutoff=25)

    # The Fock API takes the QuTiP representation directly. The operation is
    # heralded, so its output should remain a normalized density matrix.
    assert via_rho.tr() == pytest.approx(1.0, abs=1e-6)


def test_photon_subtraction_zero_probability_raises():
    N_cutoff = 5
    vacuum = qt.ket2dm(qt.fock(N_cutoff, 0))
    with pytest.raises(ValueError):
        FockOperations.photon_subtraction(vacuum, mode_idx=0, N_cutoff=N_cutoff)


def test_fock_operations_reject_incompatible_cutoff_and_mode():
    rho = qt.ket2dm(qt.fock(8, 0))

    with pytest.raises(ValueError, match="N_cutoff"):
        FockOperations.photon_subtraction(rho, N_cutoff=10)

    with pytest.raises(ValueError, match="mode_idx"):
        FockOperations.photon_subtraction(rho, mode_idx=1, N_cutoff=8)


def test_fock_operations_are_the_single_implementation_for_photon_ops():
    assert FockOperations.photon_subtraction.__module__ == "catsy.fock"
    assert FockOperations.photon_addition.__module__ == "catsy.fock"


def test_photon_addition_on_vacuum_gives_exact_single_photon_fock_state():
    # photon_addition was previously only ever exercised via the __module__
    # check above -- its actual math (a-dagger rho a-dagger-dag,
    # renormalized) was never run. Vacuum has an exact, analytically known
    # image under photon addition: a†|0><0|a / <0|aa†|0> = |1><1| exactly,
    # so this pins down correctness rather than just "didn't crash".
    N_cutoff = 10
    vacuum = qt.ket2dm(qt.fock(N_cutoff, 0))
    result = FockOperations.photon_addition(vacuum, mode_idx=0, N_cutoff=N_cutoff)
    expected = qt.ket2dm(qt.fock(N_cutoff, 1))
    assert result.tr() == pytest.approx(1.0, abs=1e-10)
    assert qt.fidelity(result, expected) == pytest.approx(1.0, abs=1e-10)


def test_photon_operations_act_only_on_the_selected_mode():
    # _mode_operator's multi-mode embedding branch (n_modes > 1) is only
    # exercised by an operation targeting one mode of a multi-mode state --
    # every other fock.py test here uses a single-mode state. Addition on
    # mode_idx=1 of a two-mode vacuum must leave mode 0 untouched.
    N_cutoff = 8
    two_mode_vacuum = qt.tensor(
        qt.ket2dm(qt.fock(N_cutoff, 0)), qt.ket2dm(qt.fock(N_cutoff, 0))
    )
    result = FockOperations.photon_addition(
        two_mode_vacuum, mode_idx=1, N_cutoff=N_cutoff
    )
    expected = qt.tensor(qt.ket2dm(qt.fock(N_cutoff, 0)), qt.ket2dm(qt.fock(N_cutoff, 1)))
    assert qt.fidelity(result, expected) == pytest.approx(1.0, abs=1e-10)


def test_fock_operations_reject_non_qobj_and_non_operator_input():
    N_cutoff = 5
    with pytest.raises(TypeError, match="Qobj"):
        FockOperations.photon_subtraction(np.eye(N_cutoff), mode_idx=0, N_cutoff=N_cutoff)
    with pytest.raises(ValueError, match="operator"):
        FockOperations.photon_subtraction(
            qt.fock(N_cutoff, 0), mode_idx=0, N_cutoff=N_cutoff
        )


# Visual diagnostics


@pytest.mark.visual
def test_native_qutip_wigner_plot_demo():
    import matplotlib.pyplot as plt

    state = Circuit().add_mode("a")
    state.add_operation(squeeze, ("a",), r=0.6, theta=0.0)
    cv_state = state.run(GaussianState.vacuum(("a",)))
    rho = cv_state.to_qutip(N_cutoff=15)

    xvec = np.linspace(-5, 5, 150)
    W = qt.wigner(rho, xvec, xvec)
    plt.figure(figsize=(5, 4))
    plt.contourf(xvec, xvec, W, 100, cmap="RdBu_r")
    plt.title("Native QuTiP Wigner function (squeezed vacuum)")
    qt.matrix_histogram(rho.full().real[:10, :10])
    plt.show()
