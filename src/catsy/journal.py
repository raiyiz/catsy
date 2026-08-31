"""Compatibility import for the simulation journal.

The canonical journal lives in ``simulations.journal``. The import is lazy so
installing and importing the reusable ``catsy`` package does not require the
repository's simulation application package to be installed as well.
"""

from __future__ import annotations

__all__ = ["JournalEntry", "SimulationJournal", "SimulationRun"]


def __getattr__(name: str):
    if name in __all__:
        from simulations.journal import JournalEntry, SimulationJournal, SimulationRun
        return {
            "JournalEntry": JournalEntry,
            "SimulationJournal": SimulationJournal,
            "SimulationRun": SimulationRun,
        }[name]
    raise AttributeError(name)
