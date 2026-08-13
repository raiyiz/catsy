"""Backward-compatible imports for the Fock and simulation APIs.

New code should import FockOperations and NonGaussianOperations from
``catst.fock`` and physical simulations from ``catst.simulations``.
"""

from .fock import FockOperations, NonGaussianOperations
from .simulations import KerrCavity, MachZehnderInterferometer

__all__ = [
    "FockOperations",
    "NonGaussianOperations",
    "KerrCavity",
    "MachZehnderInterferometer",
]
