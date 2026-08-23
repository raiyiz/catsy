"""A small set of shared types for catsy's public data structures."""

from __future__ import annotations

from typing import TypedDict

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
ModeName = str
Modes = tuple[ModeName, ...]
ParameterValue = int | float | complex | str
GateParameters = dict[str, ParameterValue]
JsonObject = dict[str, object]


class GaussianStateData(TypedDict):
    modes: list[ModeName]
    displacement: list[float]
    covariance: list[list[float]]


class GaussianChannelData(TypedDict):
    target_modes: list[ModeName]
    X: list[list[float]]
    Y: list[list[float]]
    d0: list[float]


class GateData(TypedDict):
    gate: str
    modes: list[ModeName]
    kwargs: GateParameters


class CircuitData(TypedDict):
    name: str
    modes: list[ModeName]
    gates: list[GateData]
