"""Runtime mode identity and ownership primitives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .core import Circuit


@dataclass(slots=True, eq=False)
class Mode:
    """A named runtime mode with identity semantics.

    ``Mode`` is deliberately distinct from :class:`str`: serialized circuit
    data uses mode names, while runtime code can use object identity and an
    explicit ownership relationship.
    """

    name: str
    owner: Circuit | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("Mode name must be a non-empty string.")

    def adopt(self, circuit: Circuit) -> None:
        """Assign this mode to ``circuit`` exactly once."""
        if self.owner is not None and self.owner is not circuit:
            raise ValueError(
                f"Mode '{self.name}' already belongs to another circuit."
            )
        self.owner = circuit

    def release(self, circuit: Circuit) -> None:
        """Release ownership, refusing to mutate another circuit's mode."""
        if self.owner is not circuit:
            raise ValueError(f"Mode '{self.name}' is not owned by this circuit.")
        self.owner = None
