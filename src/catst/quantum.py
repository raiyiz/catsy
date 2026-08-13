"""Compatibility imports for the Fock and simulation APIs.

New code should import FockOperations from ``catst.fock`` and physical
simulations from ``catst.simulations``.
"""

from .fock import FockOperations
from .simulations import KerrCavity, MachZehnderInterferometer

__all__ = [
    "FockOperations",
    "KerrCavity",
    "MachZehnderInterferometer",
]
