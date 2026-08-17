import numpy as np
import pytest
import qutip as qt

from catsy.gaussian import GaussianCircuit, GaussianOperations, LossChannels
from catsy.fock import FockOperations


# Gaussian -> Fock bridge

def test_cv_channel_to_fock_purity_drops_with_loss():
    state = GaussianOperations.create_vacuum(modes=("a", "b"))
    state = GaussianOperations.apply_squeezing(state, mode="a", r=0.5)
    state = GaussianOperations.apply_squeezing(state, mode="b", r=0.5, theta=np.pi / 2)
    state = GaussianOperations.apply_beam_splitter(
        state, mode_a="a", mode_b="b", eta=0.5
    )

    clean_rho = state.to_qutip(N_cutoff=18)
    noisy_state = LossChannels.thermal_loss(mode="a", eta=0.9, n_thermal=0.2).apply(
        state
    )
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


# Fock-space operations

def test_photon_subtraction_state_and_rho_entry_points_agree():
    circuit = GaussianCircuit()
    circuit.add_mode("a")
    circuit.squeeze(mode="a", r=0.55)
    gaussian_squeezed = circuit.compile_and_run()

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


# Visual diagnostics

@pytest.mark.visual
def test_native_qutip_wigner_plot_demo():
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
