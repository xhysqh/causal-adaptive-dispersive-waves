from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np
from numpy.typing import NDArray


class RepresentationKind(str, Enum):
    EULERIAN = "eulerian"
    PARTICLE = "particle"
    HYBRID = "hybrid"
    LOCAL_AMR = "local_amr"
    SPECTRAL = "spectral"


@dataclass(frozen=True, slots=True)
class ResolutionState:
    points: int
    active_modes: int | None = None
    active_fraction: float = 1.0

    def __post_init__(self) -> None:
        if self.points < 8:
            raise ValueError("resolution requires at least eight points")
        if self.active_modes is not None and not 0 < self.active_modes <= self.points:
            raise ValueError("active_modes must lie in (0, points]")
        if not 0.0 <= self.active_fraction <= 1.0:
            raise ValueError("active_fraction must lie in [0, 1]")


@dataclass(frozen=True, slots=True)
class WaveState:
    """Immutable solver-neutral state passed to feature and policy modules."""

    time: float
    field: NDArray[np.floating]
    dt: float
    domain_length: float
    representation: RepresentationKind
    resolution: ResolutionState
    positions: NDArray[np.floating] = field(default_factory=lambda: np.empty(0))
    amplitudes: NDArray[np.floating] = field(default_factory=lambda: np.empty(0))
    error_budget_remaining: float = np.inf
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        values = np.asarray(self.field, dtype=np.float64)
        positions = np.asarray(self.positions, dtype=np.float64)
        amplitudes = np.asarray(self.amplitudes, dtype=np.float64)
        if values.shape != (self.resolution.points,) or not np.all(np.isfinite(values)):
            raise ValueError("field must be a finite vector matching resolution.points")
        if positions.ndim != 1 or positions.shape != amplitudes.shape:
            raise ValueError("positions and amplitudes must be aligned vectors")
        if not np.all(np.isfinite(positions)) or not np.all(np.isfinite(amplitudes)):
            raise ValueError("particle coordinates and amplitudes must be finite")
        if not np.isfinite(self.time) or self.time < 0.0:
            raise ValueError("state time must be finite and nonnegative")
        if not np.isfinite(self.dt) or self.dt <= 0.0:
            raise ValueError("state dt must be positive and finite")
        if not np.isfinite(self.domain_length) or self.domain_length <= 0.0:
            raise ValueError("domain_length must be positive and finite")
        if self.error_budget_remaining < 0.0 or np.isnan(self.error_budget_remaining):
            raise ValueError("error budget must be nonnegative")
        values = values.copy()
        positions = positions.copy()
        amplitudes = amplitudes.copy()
        values.setflags(write=False)
        positions.setflags(write=False)
        amplitudes.setflags(write=False)
        object.__setattr__(self, "field", values)
        object.__setattr__(self, "positions", positions)
        object.__setattr__(self, "amplitudes", amplitudes)
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


@dataclass(frozen=True, slots=True)
class OperatorStepResult:
    state: WaveState
    action_id: str
    invariants_before: Mapping[str, float]
    invariants_after: Mapping[str, float]
    diagnostics: Mapping[str, float | int | bool | str]

    def maximum_relative_invariant_drift(self) -> float:
        common = self.invariants_before.keys() & self.invariants_after.keys()
        if not common:
            return 0.0
        return max(
            abs(self.invariants_after[key] - self.invariants_before[key])
            / max(abs(self.invariants_before[key]), np.finfo(float).eps)
            for key in common
        )
