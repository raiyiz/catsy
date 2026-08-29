"""Small continuous-variable quantum-optics toolkit."""

from .fock import FockGates
from .gaussian import (
    GaussianChannel,
    GaussianMeasurements,
    GaussianState,
    LossChannels,
    beam_splitter,
    compute_duan_inseparability,
    compute_joint_correlation,
    compute_wigner_analytically,
    displace,
    loss,
    rotate,
    squeeze,
    thermal_loss,
)
from .journal import JournalEntry, SimulationJournal
from .optics import Circuit, Gate, KerrCavity, MachZehnderInterferometer
from . import optics as _optics
from .modes import Mode, ModeNamespace

# Keep the existing optics module API aligned with the canonical Mode class.
_optics.Mode = Mode

__all__ = [
    "Circuit",
    "FockGates",
    "Gate",
    "GaussianChannel",
    "GaussianMeasurements",
    "GaussianState",
    "JournalEntry",
    "KerrCavity",
    "LossChannels",
    "MachZehnderInterferometer",
    "Mode",
    "ModeNamespace",
    "SimulationJournal",
    "beam_splitter",
    "compute_duan_inseparability",
    "compute_joint_correlation",
    "compute_wigner_analytically",
    "displace",
    "loss",
    "rotate",
    "squeeze",
    "thermal_loss",
]
