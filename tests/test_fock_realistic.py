import numpy as np
import pytest
import qutip as qt

from catsy.fock import FockGates


N_CUTOFF = 18


def _squeezed_vacuum(r: float = 0.6) -> qt.Qobj:
    psi = (qt.squeeze(N_CUTOFF, r) * qt.fock(N_CUTOFF, 0)).unit()
    return qt.ket2dm(psi)


@pytest.mark.parametrize(
    ("ideal", "realistic", "realistic_kwargs"),
    [
        pytest.param(
            FockGates.photon_subtraction,
            FockGates.realistic_photon_subtraction,
            {"tap_reflectivity": 1e-4},
            id="subtraction",
        ),
        pytest.param(
            FockGates.photon_addition,
            FockGates.realistic_photon_addition,
            {"coupling_strength": 1e-4},
            id="addition",
        ),
    ],
)
def test_realistic_photon_operation_approaches_ideal_in_weak_coupling(
    ideal, realistic, realistic_kwargs
):
    rho = _squeezed_vacuum()
    ideal_state = ideal(rho, mode_idx=0, N_cutoff=N_CUTOFF)
    realistic_state = realistic(
        rho,
        mode_idx=0,
        N_cutoff=N_CUTOFF,
        detector_efficiency=1.0,
        ancilla_cutoff=8,
        **realistic_kwargs,
    )

    assert qt.fidelity(realistic_state, ideal_state) == pytest.approx(1.0, abs=1e-3)


@pytest.mark.parametrize(
    ("ideal", "realistic", "coupling_values", "coupling_parameter"),
    [
        pytest.param(
            FockGates.photon_subtraction,
            FockGates.realistic_photon_subtraction,
            (0.2, 0.05, 0.01),
            "tap_reflectivity",
            id="subtraction",
        ),
        pytest.param(
            FockGates.photon_addition,
            FockGates.realistic_photon_addition,
            (0.2, 0.05, 0.01),
            "coupling_strength",
            id="addition",
        ),
    ],
)
def test_realistic_photon_operation_gets_closer_to_ideal_as_coupling_weakens(
    ideal, realistic, coupling_values, coupling_parameter
):
    rho = _squeezed_vacuum()
    ideal_state = ideal(rho, mode_idx=0, N_cutoff=N_CUTOFF)

    fidelities = []
    for coupling in coupling_values:
        realistic_state = realistic(
            rho,
            mode_idx=0,
            N_cutoff=N_CUTOFF,
            detector_efficiency=1.0,
            ancilla_cutoff=8,
            **{coupling_parameter: coupling},
        )
        fidelities.append(qt.fidelity(realistic_state, ideal_state))

    assert fidelities[0] < fidelities[1] < fidelities[2]


@pytest.mark.parametrize(
    ("realistic", "operation_kwargs"),
    [
        pytest.param(
            FockGates.realistic_photon_subtraction,
            {"tap_reflectivity": 0.05},
            id="subtraction",
        ),
        pytest.param(
            FockGates.realistic_photon_addition,
            {"coupling_strength": 0.05},
            id="addition",
        ),
    ],
)
def test_realistic_photon_operation_becomes_less_ideal_with_lower_detector_efficiency(
    realistic, operation_kwargs
):
    rho = _squeezed_vacuum(r=0.5)
    ideal = (
        FockGates.photon_subtraction(rho, N_cutoff=N_CUTOFF)
        if "tap_reflectivity" in operation_kwargs
        else FockGates.photon_addition(rho, N_cutoff=N_CUTOFF)
    )

    fidelities = []
    for efficiency in (0.2, 0.5, 0.95):
        realistic_state = realistic(
            rho,
            N_cutoff=N_CUTOFF,
            detector_efficiency=efficiency,
            ancilla_cutoff=8,
            **operation_kwargs,
        )
        fidelities.append(qt.fidelity(realistic_state, ideal))

    assert fidelities[0] < fidelities[1] < fidelities[2]


@pytest.mark.parametrize(
    ("realistic", "operation_kwargs"),
    [
        pytest.param(
            FockGates.realistic_photon_subtraction,
            {"tap_reflectivity": 0.15},
            id="subtraction",
        ),
        pytest.param(
            FockGates.realistic_photon_addition,
            {"coupling_strength": 0.15},
            id="addition",
        ),
    ],
)
@pytest.mark.parametrize(
    "rho",
    [
        pytest.param(qt.ket2dm(qt.fock(N_CUTOFF, 1)), id="fock-1"),
        pytest.param(qt.ket2dm(qt.coherent(N_CUTOFF, 0.8)), id="coherent"),
        pytest.param(_squeezed_vacuum(), id="squeezed"),
        pytest.param(qt.thermal_dm(N_CUTOFF, 0.4), id="thermal"),
    ],
)
def test_realistic_photon_operations_return_valid_density_matrices(
    realistic, operation_kwargs, rho
):
    result = realistic(
        rho,
        mode_idx=0,
        N_cutoff=N_CUTOFF,
        detector_efficiency=0.6,
        ancilla_cutoff=8,
        **operation_kwargs,
    )

    assert result.tr() == pytest.approx(1.0, abs=1e-8)
    assert np.min(result.eigenenergies()) > -1e-9
