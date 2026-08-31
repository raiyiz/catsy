"""Backward-compatible imports for the pre-0.2 optics module.

Circuit construction lives in :mod:`catsy.circuits`, primitive state
transformations in :mod:`catsy.gates`, and higher-level physical models in
:mod:`catsy.experiments`.
"""

from .circuits import Circuit, CircuitState, CVState, Gate, GateTransform, Mode
from .experiments import KerrCavity, MachZehnderInterferometer, ObservableScanData
from .gates import (
    beam_splitter,
    displace,
    initial_state,
    loss,
    photon_addition,
    photon_subtraction,
    realistic_photon_addition,
    realistic_photon_subtraction,
    rotate,
    squeeze,
    thermal_loss,
)

__all__ = [
    "CVState",
    "Circuit",
    "CircuitState",
    "Gate",
    "GateTransform",
    "KerrCavity",
    "MachZehnderInterferometer",
    "Mode",
    "ObservableScanData",
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
