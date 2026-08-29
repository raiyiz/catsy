import pytest
import qutip as qt

from catsy.fock import FockState, beam_splitter
from catsy.gaussian import GaussianState

CUTOFF = 16
TOLERANCE = 1e-5


def _assert_rho_equivalent(
    left: qt.Qobj,
    right: qt.Qobj,
    atol: float = TOLERANCE,
) -> None:
    assert left.tr() == pytest.approx(1.0, abs=atol)
    assert right.tr() == pytest.approx(1.0, abs=atol)
    assert qt.fidelity(left, right) == pytest.approx(1.0, abs=atol)


def _initial_state() -> GaussianState:
    return GaussianState.coherent(("a",), 0.45 + 0.2j).squeeze("a", r=0.35, theta=0.17)


@pytest.mark.parametrize(
    ("apply_gate"),
    [
        lambda s: s.squeeze("a", r=0.25, theta=0.3),
        lambda s: s.rotate("a", phi=-0.4),
        lambda s: s.displace("a", alpha=-0.2 + 0.35j),
        lambda s: s.loss("a", eta=0.72),
    ],
)
def test_gaussian_to_fock_gate_equivalence(apply_gate):
    gaussian = apply_gate(_initial_state())

    initial_fock = FockState(
        _initial_state().modes,
        _initial_state().to_qutip(CUTOFF),
        CUTOFF,
    )
    fock = apply_gate(initial_fock)

    assert fock.modes == gaussian.modes
    assert fock.N_cutoff == CUTOFF
    _assert_rho_equivalent(gaussian.to_qutip(CUTOFF), fock.rho)


@pytest.mark.parametrize("eta", [0.2, 0.5, 0.8])
def test_gaussian_to_fock_beam_splitter_equivalence(eta):
    gaussian = (
        GaussianState.coherent(
            ("a", "b"),
            [0.45 + 0.2j, -0.15 + 0.3j],
        )
        .squeeze("a", r=0.3, theta=0.1)
        .squeeze("b", r=0.2, theta=-0.25)
    )

    gaussian_result = gaussian.beam_splitter("a", "b", eta=eta).to_qutip(CUTOFF)

    fock = FockState(
        gaussian.modes,
        gaussian.to_qutip(CUTOFF),
        CUTOFF,
    )
    fock_result = fock.beam_splitter("a", "b", eta=eta)

    assert fock_result.modes == gaussian.modes
    _assert_rho_equivalent(gaussian_result, fock_result.rho)


def test_fock_gate_chain_stays_in_fock_representation():
    initial = _initial_state()
    state = FockState(
        initial.modes,
        initial.to_qutip(CUTOFF),
        CUTOFF,
    )

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


def test_two_mode_fock_gate_preserves_named_mode_order():
    gaussian = GaussianState.coherent(
        ("b", "a"),
        [0.2 + 0.1j, 0.5 - 0.2j],
    )
    state = FockState(
        gaussian.modes,
        gaussian.to_qutip(CUTOFF),
        CUTOFF,
    )

    result = state.beam_splitter("b", "a", eta=0.6)

    assert result.modes == ("b", "a")
    assert result.get_mode_index("b") == 0
    assert result.get_mode_index("a") == 1


def test_beam_splitter_functional_api_matches_state_method():
    gaussian = GaussianState.coherent(
        ("a", "b"),
        [0.4 + 0.1j, -0.2 + 0.3j],
    )
    state = FockState(
        gaussian.modes,
        gaussian.to_qutip(CUTOFF),
        CUTOFF,
    )

    functional = beam_splitter(
        state.rho,
        0,
        1,
        eta=0.65,
        N_cutoff=CUTOFF,
    )
    method = state.beam_splitter(
        "a",
        "b",
        eta=0.65,
    ).rho

    _assert_rho_equivalent(functional, method)
