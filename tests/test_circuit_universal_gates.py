import pytest
import qutip as qt

from catsy.fock import FockState
from catsy.gaussian import GaussianState
from catsy.optics import Circuit

CUTOFF = 16
TOLERANCE = 1e-5


def _assert_rho_equivalent(left: qt.Qobj, right: qt.Qobj, atol: float = TOLERANCE) -> None:
    assert left.tr() == pytest.approx(1.0, abs=atol)
    assert right.tr() == pytest.approx(1.0, abs=atol)
    assert qt.fidelity(left, right) == pytest.approx(1.0, abs=atol)


# -- to_fock ------------------------------------------------------------


def test_to_fock_returns_fock_state_with_matching_modes_and_cutoff():
    state = GaussianState.coherent(("a", "b"), [0.3 + 0.1j, -0.2 + 0.4j])
    fock = state.to_fock(N_cutoff=CUTOFF)

    assert isinstance(fock, FockState)
    assert fock.modes == state.modes
    assert fock.N_cutoff == CUTOFF


def test_to_qutip_is_still_a_thin_alias_for_to_fock():
    state = GaussianState.coherent(("a",), 0.4 - 0.15j).squeeze("a", r=0.3, theta=0.2)
    _assert_rho_equivalent(state.to_qutip(CUTOFF), state.to_fock(CUTOFF).rho)


# -- pure-Gaussian circuits: no accidental promotion ---------------------


def test_circuit_with_only_universal_gates_stays_gaussian():
    circuit = Circuit(name="gaussian only", modes=("a", "b"))
    circuit.initial_state("a", "b", kind="vacuum")
    circuit.squeeze("a", r=0.4)
    circuit.beam_splitter("a", "b", eta=0.5)
    circuit.loss("b", eta=0.85)
    circuit.rotate("a", phi=0.3)

    result = circuit.run()
    assert isinstance(result, GaussianState)


# -- auto-promotion on first non-Gaussian gate ---------------------------


def test_circuit_promotes_to_fock_on_non_gaussian_gate_and_stays_there():
    circuit = Circuit(name="promote", modes=("a",))
    circuit.initial_state("a", kind="coherent", alpha=0.4 + 0.2j)
    circuit.squeeze("a", r=0.3, theta=0.1)
    circuit.photon_subtraction("a", N_cutoff=CUTOFF)
    circuit.rotate("a", phi=0.25)  # a universal gate, now dispatched to Fock

    result = circuit.run()
    assert isinstance(result, FockState)
    assert result.N_cutoff == CUTOFF
    assert result.rho.tr() == pytest.approx(1.0, abs=1e-6)


def test_promotion_matches_manual_to_fock_plus_fock_gate():
    def build_gaussian_prefix() -> GaussianState:
        return GaussianState.coherent(("a",), 0.4 + 0.2j).squeeze("a", r=0.3, theta=0.1)

    circuit = Circuit(name="promote-equivalence", modes=("a",))
    circuit.initial_state("a", kind="coherent", alpha=0.4 + 0.2j)
    circuit.squeeze("a", r=0.3, theta=0.1)
    circuit.photon_subtraction("a", N_cutoff=CUTOFF)
    via_circuit = circuit.run()

    manual = build_gaussian_prefix().to_fock(CUTOFF).photon_subtraction("a")

    _assert_rho_equivalent(via_circuit.rho, manual.rho)


def test_non_gaussian_gate_without_n_cutoff_raises_clear_error():
    circuit = Circuit(name="missing cutoff", modes=("a",))
    circuit.initial_state("a", kind="coherent", alpha=0.3)
    circuit.photon_subtraction("a")

    with pytest.raises(ValueError, match="N_cutoff"):
        circuit.run()


def test_redundant_n_cutoff_after_promotion_is_ignored_not_an_error():
    circuit = Circuit(name="already fock", modes=("a",))
    circuit.initial_state("a", kind="coherent", alpha=0.3)
    circuit.photon_subtraction("a", N_cutoff=CUTOFF)
    circuit.photon_addition("a", N_cutoff=CUTOFF)  # already a FockState by now

    result = circuit.run()
    assert isinstance(result, FockState)
    assert result.N_cutoff == CUTOFF


# -- schematic reflects the Gaussian -> Fock transition -------------------


def test_schematic_shows_non_gaussian_gate_label():
    circuit = Circuit(name="schematic", modes=("a",))
    circuit.initial_state("a", kind="vacuum")
    circuit.photon_subtraction("a", N_cutoff=CUTOFF)

    assert "PSUB" in circuit.render_schematic()
