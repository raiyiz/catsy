"""State transformations used by :mod:`catsy.circuits`."""

from __future__ import annotations

from typing import cast

from catsy.fock import FockState
from catsy.gaussian import GaussianState, initial_state
from catsy.gaussian import beam_splitter as _gaussian_beam_splitter
from catsy.gaussian import displace as _gaussian_displace
from catsy.gaussian import loss as _gaussian_loss
from catsy.gaussian import rotate as _gaussian_rotate
from catsy.gaussian import squeeze as _gaussian_squeeze
from catsy.gaussian import thermal_loss as _gaussian_thermal_loss

from .circuits import CVState, Circuit
from .types import Modes, ParameterValue


def squeeze(state: CVState, modes: Modes, **kwargs: ParameterValue) -> CVState:
    if isinstance(state, FockState):
        return state.squeeze(mode=modes[0], r=cast(float, kwargs["r"]), theta=cast(float, kwargs.get("theta", 0.0)))
    return _gaussian_squeeze(state, modes, **kwargs)


def rotate(state: CVState, modes: Modes, **kwargs: ParameterValue) -> CVState:
    if isinstance(state, FockState):
        return state.rotate(mode=modes[0], phi=cast(float, kwargs["phi"]))
    return _gaussian_rotate(state, modes, **kwargs)


def displace(state: CVState, modes: Modes, **kwargs: ParameterValue) -> CVState:
    if isinstance(state, FockState):
        return state.displace(
            mode=modes[0],
            alpha=cast(complex, kwargs["alpha"]) if "alpha" in kwargs else None,
            x=cast(float, kwargs["x"]) if "x" in kwargs else None,
            p=cast(float, kwargs["p"]) if "p" in kwargs else None,
        )
    return _gaussian_displace(state, modes, **kwargs)


def beam_splitter(state: CVState, modes: Modes, **kwargs: ParameterValue) -> CVState:
    if isinstance(state, FockState):
        return state.beam_splitter(mode_a=modes[0], mode_b=modes[1], eta=cast(float, kwargs["eta"]))
    return _gaussian_beam_splitter(state, modes, **kwargs)


def loss(state: CVState, modes: Modes, **kwargs: ParameterValue) -> CVState:
    if isinstance(state, FockState):
        return state.loss(mode=modes[0], eta=cast(float, kwargs["eta"]))
    return _gaussian_loss(state, modes, **kwargs)


def thermal_loss(state: CVState, modes: Modes, **kwargs: ParameterValue) -> CVState:
    if isinstance(state, FockState):
        return state.thermal_loss(
            mode=modes[0],
            eta=cast(float, kwargs["eta"]),
            nbar=cast(float, kwargs.get("nbar", 0.0)),
            ancilla_cutoff=cast(int, kwargs["ancilla_cutoff"]) if "ancilla_cutoff" in kwargs else None,
        )
    return _gaussian_thermal_loss(state, modes, **kwargs)


def _ensure_fock(state: CVState, kwargs: dict[str, ParameterValue]) -> FockState:
    if isinstance(state, FockState):
        return state
    if "N_cutoff" not in kwargs:
        raise ValueError(
            "This gate is non-Gaussian and the circuit hasn't been promoted into Fock space yet; "
            "pass N_cutoff=... to this gate call."
        )
    return cast(GaussianState, state).to_fock(cast(int, kwargs["N_cutoff"]))


def photon_subtraction(state: CVState, modes: Modes, **kwargs: ParameterValue) -> FockState:
    return _ensure_fock(state, kwargs).photon_subtraction(mode=modes[0])


def photon_addition(state: CVState, modes: Modes, **kwargs: ParameterValue) -> FockState:
    return _ensure_fock(state, kwargs).photon_addition(mode=modes[0])


def realistic_photon_subtraction(state: CVState, modes: Modes, **kwargs: ParameterValue) -> FockState:
    return _ensure_fock(state, kwargs).realistic_photon_subtraction(
        mode=modes[0],
        tap_reflectivity=cast(float, kwargs.get("tap_reflectivity", 0.05)),
        detector_efficiency=cast(float, kwargs.get("detector_efficiency", 0.6)),
        ancilla_cutoff=cast(int, kwargs.get("ancilla_cutoff", 6)),
    )


def realistic_photon_addition(state: CVState, modes: Modes, **kwargs: ParameterValue) -> FockState:
    return _ensure_fock(state, kwargs).realistic_photon_addition(
        mode=modes[0],
        coupling_strength=cast(float, kwargs.get("coupling_strength", 0.05)),
        detector_efficiency=cast(float, kwargs.get("detector_efficiency", 0.6)),
        ancilla_cutoff=cast(int, kwargs.get("ancilla_cutoff", 6)),
    )


for _name, _transform in (
    ("Squeezer", squeeze),
    ("Rotator", rotate),
    ("Displacer", displace),
    ("BeamSplitter", beam_splitter),
    ("Noise", loss),
    ("ThermalLoss", thermal_loss),
    ("InitialState", initial_state),
    ("PhotonSubtraction", photon_subtraction),
    ("PhotonAddition", photon_addition),
    ("RealisticPhotonSubtraction", realistic_photon_subtraction),
    ("RealisticPhotonAddition", realistic_photon_addition),
):
    Circuit.register(_name, _transform)

__all__ = [
    "beam_splitter",
    "displace",
    "initial_state",
    "loss",
    "photon_addition",
    "photon_subtraction",
    "realistic_photon_addition",
    "realistic_photon_subtraction",
    "rotate",
    "squeeze",
    "thermal_loss",
]
