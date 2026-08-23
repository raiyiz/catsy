"""Runtime representation of a circuit/standalone mode."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .core import Circuit


@dataclass(slots=True, eq=False)
class Mode:
    """A named runtime mode.

    Equality is intentionally identity-based: two separately-created modes
    with the same name are distinct semantic objects. ``owner`` is populated
    only when a mode belongs to a :class:`~catsy.core.Circuit`.
    """

    name: str
    owner: Circuit | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("Mode name must be a non-empty string.")
