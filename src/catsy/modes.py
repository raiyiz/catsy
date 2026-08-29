"""Immutable mode identities used by universal optical circuits."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .optics import Circuit


@dataclass(frozen=True, eq=False, slots=True)
class Mode:
    """Immutable physical-mode identity."""

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
        for mode in self._modes:
            if mode.name == name:
                return mode
        raise AttributeError(f"No mode named {name!r}.")

    def __getitem__(self, key: int | str) -> Mode:
        if isinstance(key, int):
            return self._modes[key]
        return self.__getattr__(key)

    def __iter__(self) -> Iterator[Mode]:
        return iter(self._modes)

    def __len__(self) -> int:
        return len(self._modes)

    def __contains__(self, value: object) -> bool:
        if isinstance(value, Mode):
            return value in self._modes
        return any(mode.name == value for mode in self._modes)

    def __repr__(self) -> str:
        return f"ModeNamespace({self._modes!r})"
