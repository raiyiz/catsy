import numpy as np
import pytest
import qutip as qt

from catsy.fock import FockState, beam_splitter, displace, loss, rotate, squeeze
from catsy.gaussian import GaussianState


CUTOFF = 18


def _assert_equivalent(left: qt.Qobj, right: qt.Qobj) -> None:
    assert left.tr() == pytest.approx(1.0, abs=1e-8)
    assert right.tr() == pytest.approx(1.0, abs=1e-8)
    assert qt.fidelity(left, right) == pytest.approx(1.0, abs=1e-7)


def _initial_state() -> GaussianState:
    return GaussianState.coherent(("a",), 0.45 + 0.2j).squeeze(
        "a", r=0.35, theta=0.17
    )


@pytest.mark.parametrize(
    ("gaussian_gate", "fock_gate"),
    [
        (
            lambda s: s.squeeze("a", r=0.25, theta=0.3),
            lambda s: s.squeeze("a", r=0.25, theta=0.3),
        ),
        (
            lambda s: s.rotate("a", phi=-0.4),
            lambda s: s.rotate("a", phi=-0.4),
        ),
        (
            lambda s: s.displace("a", alpha=-0.2 + 0.35j),
            lambda s: s.displace("a", alpha=-0.2 + 0.35j),
        ),
        (
            lambda s: s.loss("a", eta=0.72),
            lambda s: s.loss("a", eta=0.72),
        ),
    ],
)
def test_gaussian_and_fock_gate_paths_are_equivalent(gaussian_gate, fock_gate):
    gaussian = _initial_state()
    fock = FockState(gaussian.modes, gaussian.to_qutip(CUTOFF), CUTOFF)

    gaussian_result = gaussian_gate(gaussian).to_qutip(CUTOFF)
    fock_result = fock_gate(fock).rho

    _assert_equivalent(gaussian_result, fock_result)


@pytest.mark.parametrize("eta", [0.2, 0.5, 0.8])
def test_beam_splitter_gaussian_and_fock_paths_are_equivalent(eta):
    gaussian = (
        GaussianState.coherent(("a", "b"), [0.45 + 0.2j, -0.15 + 0.3j])
        .squeeze("a", r=0.3, theta=0.1)
        .squeeze("b", r=0.2, theta=-0.25)
    )
    fock = FockState(gaussian.modes, gaussian.to_qutip(CUTOFF), CUTOFF)

    gaussian_result = gaussian.beam_splitter("a", "b", eta=eta).to_qutip(CUTOFF)
    fock_result = fock.beam_splitter("a", "b", eta=eta).rho

    _assert_equivalent(gaussian_result, fock_result)


def test_fock_gate_chain_stays_in_fock_representation():
    gaussian = _initial_state()
    state = FockState(gaussian.modes, gaussian.to_qutip(CUTOFF), CUTOFF)
    result = (
        state.squeeze("a", r=0.2)
        .rotate("a", phi=0.15)
        .displace("a", alpha=0.1 - 0.2j)
        .loss("a", eta=0.9)
    )

    assert isinstance(result, FockState)
    assert result.modes == ("a",)
    assert result.N_cutoff == CUTOFF
    assert result.rho.isoper
    assert result.rho.tr() == pytest.approx(1.0, abs=1e-8)


def test_fock_gate_preserves_named_mode_order():
    gaussian = GaussianState.coherent(("b", "a"), [0.2 + 0.1j, 0.5 - 0.2j])
    state = FockState(gaussian.modes, gaussian.to_qutip(CUTOFF), CUTOFF)

    result = state.beam_splitter("b", "a", eta=0.6)

    assert result.modes == ("b", "a")
    assert result.get_mode_index("b") == 0
    assert result.get_mode_index("a") == 1


def test_functional_and_state_gate_paths_are_equivalent():
    gaussian = _initial_state()
    state = FockState(gaussian.modes, gaussian.to_qutip(CUTOFF), CUTOFF)

    assert qt.fidelity(
        squeeze(state.rho, 0, r=0.2, theta=0.1, N_cutoff=CUTOFF),
        state.squeeze("a", r=0.2, theta=0.1).rho,
    ) == pytest.approx(1.0, abs=1e-8)
    assert qt.fidelity(
        rotate(state.rho, 0, phi=0.2, N_cutoff=CUTOFF),
        state.rotate("a", phi=0.2).rho,
    ) == pytest.approx(1.0, abs=1e-8)
    assert qt.fidelity(
        displace(state.rho, 0, alpha=0.2 - 0.1j, N_cutoff=CUTOFF),
        state.displace("a", alpha=0.2 - 0.1j).rho,
    ) == pytest.approx(1.0, abs=1e-8)
    assert qt.fidelity(
        loss(state.rho, 0, eta=0.8, N_cutoff=CUTOFF),
        state.loss("a", eta=0.8).rho,
    ) == pytest.approx(1.0, abs=1e-8)


def test_beam_splitter_functional_and_state_paths_are_equivalent():
    gaussian = GaussianState.coherent(("a", "b"), [0.4 + 0.1j, -0.2 + 0.3j])
    state = FockState(gaussian.modes, gaussian.to_qutip(CUTOFF), CUTOFF)

    functional = beam_splitter(state.rho, 0, 1, eta=0.65, N_cutoff=CUTOFF)
    method = state.beam_splitter("a", "b", eta=0.65).rho

    _assert_equivalent(functional, method)
