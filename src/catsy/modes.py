"""Immutable mode identities used by universal optical circuits."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .optics import Circuit


@dataclass(frozen=True, eq=False, slots=True)
class Mode:
    """Immutable physical-mode identity.

    A standalone mode has no circuit owner and no canonical tensor index.
    Circuit-owned modes are created by :class:`catsy.optics.Circuit` and carry
    the circuit's immutable canonical ordering index.
    """

    name: str
    index: int | None = None
    owner: Circuit | None = None

    def __repr__(self) -> str:
        if self.owner is None:
            return f"Mode({self.name!r})"
        return f"Mode({self.name!r}, index={self.index})"


class ModeNamespace:
    """Read-only named access to a circuit's ordered mode objects."""

    __slots__ = ("_modes",)

    def __init__(self, modes: tuple[Mode, ...] = ()) -> None:
        self._modes = modes

    def __getattr__(self, name: str) -> Mode:
        try:
            return next(mode for mode in self._modes if mode.name == name)
        except StopIteration as exc:
            raise AttributeError(f"No mode named {name!r}.") from exc

    def __getitem__(self, key: int | str) -> Mode:
        if isinstance(key, int):
            return self._modes[key]
        return getattr(self, key)

    def __iter__(self):
        return iter(self._modes)

    def __len__(self) -> int:
        return len(self._modes)

    def __contains__(self, value: object) -> bool:
        return value in self._modes or any(
            isinstance(value, str) and mode.name == value for mode in self._modes
        )

    def __repr__(self) -> str:
        return f"ModeNamespace({self._modes!r})"
