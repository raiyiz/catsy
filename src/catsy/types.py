"""A small set of shared types for catsy's public data structures."""

from __future__ import annotations

from typing import TypedDict

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
Mode = str
Modes = tuple[Mode, ...]
ParameterValue = int | float | complex
OperationParameters = dict[str, ParameterValue]
JsonObject = dict[str, object]


class GaussianStateData(TypedDict):
    modes: list[str]
    displacement: list[float]
    covariance: list[list[float]]


class GaussianChannelData(TypedDict):
    target_modes: list[str]
    X: list[list[float]]
    Y: list[list[float]]
    d0: list[float]


class OperationData(TypedDict):
    op: str
    modes: list[str]
    kwargs: OperationParameters


class CircuitData(TypedDict):
    modes: list[str]
    operations: list[OperationData]


class OpticalComponentData(TypedDict):
    name: str
    op: str
    ports: list[str]
    kwargs: OperationParameters


class OpticalSetupData(TypedDict):
    layout_name: str
    components: list[OpticalComponentData]
