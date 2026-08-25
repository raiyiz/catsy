import numpy as np
import qutip as qt

from catsy.fock import (
    FockGates,
    mean_photon_number,
    photon_addition,
    photon_number_measurement,
    photon_subtraction,
    realistic_photon_addition,
    realistic_photon_subtraction,
)


def test_fock_operations_are_exposed_as_module_functions():
    assert photon_subtraction is FockGates.photon_subtraction
    assert photon_addition is FockGates.photon_addition
    assert realistic_photon_subtraction is FockGates.realistic_photon_subtraction
    assert realistic_photon_addition is FockGates.realistic_photon_addition
    assert mean_photon_number is FockGates.mean_photon_number
    assert photon_number_measurement is FockGates.photon_number_measurement


def test_functional_api_matches_backwards_compatible_namespace():
    N_cutoff = 8
    rho = qt.ket2dm(qt.fock(N_cutoff, 2))

    functional = photon_subtraction(rho, N_cutoff=N_cutoff)
    compatibility = FockGates.photon_subtraction(rho, N_cutoff=N_cutoff)
    assert qt.fidelity(functional, compatibility) == 1.0
    assert mean_photon_number(rho, N_cutoff=N_cutoff) == FockGates.mean_photon_number(
        rho, N_cutoff=N_cutoff
    )


def test_functional_api_supports_realistic_operations():
    N_cutoff = 8
    rho = qt.ket2dm(qt.coherent(N_cutoff, 0.8))

    subtracted = realistic_photon_subtraction(
        rho,
        N_cutoff=N_cutoff,
        tap_reflectivity=0.05,
        detector_efficiency=0.8,
        ancilla_cutoff=5,
    )
    added = realistic_photon_addition(
        rho,
        N_cutoff=N_cutoff,
        coupling_strength=0.05,
        detector_efficiency=0.8,
        ancilla_cutoff=5,
    )

    assert np.isclose(subtracted.tr(), 1.0)
    assert np.isclose(added.tr(), 1.0)
