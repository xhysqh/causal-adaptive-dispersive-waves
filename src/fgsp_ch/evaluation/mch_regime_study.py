from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
from numpy.typing import NDArray
from scipy.signal import resample

from fgsp_ch.discretization.grids import PeriodicGrid
from fgsp_ch.geometry.metric import a_norm
from fgsp_ch.discretization.helmholtz import HelmholtzOperator


@dataclass(frozen=True, slots=True)
class MCHRegimeDomain:
    """Pre-registered finite domain for an M6.1 claim."""

    alpha: tuple[float, float]
    gamma: tuple[float, float]
    length: tuple[float, float]
    points: tuple[int, ...]
    dt: tuple[float, float]
    families: tuple[str, ...]
    maximum_atoms: int = 8

    def __post_init__(self) -> None:
        intervals = (self.alpha, self.gamma, self.length, self.dt)
        if any(len(item) != 2 or item[0] > item[1] for item in intervals):
            raise ValueError("Regime-domain intervals must be ordered pairs.")
        if not all(np.all(np.isfinite(item)) for item in intervals):
            raise ValueError("Regime-domain bounds must be finite.")
        if self.length[0] <= 0.0 or self.dt[0] <= 0.0:
            raise ValueError("Length and time-step bounds must be positive.")
        if (
            not self.points
            or min(self.points) < 3
            or len(set(self.points)) != len(self.points)
            or not self.families
            or len(set(self.families)) != len(self.families)
        ):
            raise ValueError("Regime domain needs grids and solution families.")
        if self.maximum_atoms < 1:
            raise ValueError("maximum_atoms must be positive.")

    @classmethod
    def from_mapping(cls, values: dict[str, Any]) -> "MCHRegimeDomain":
        return cls(
            tuple(map(float, values["alpha"])),
            tuple(map(float, values["gamma"])),
            tuple(map(float, values["length"])),
            tuple(map(int, values["points"])),
            tuple(map(float, values["dt"])),
            tuple(map(str, values["families"])),
            int(values.get("maximum_atoms", 8)),
        )

    def contains(self, scenario: dict[str, Any], *, points: int, dt: float) -> bool:
        atoms = len(scenario.get("positions", ()))
        return bool(
            self.alpha[0] <= float(scenario["alpha"]) <= self.alpha[1]
            and self.gamma[0] <= float(scenario["gamma"]) <= self.gamma[1]
            and self.length[0] <= float(scenario["length"]) <= self.length[1]
            and points in self.points
            and self.dt[0] <= dt <= self.dt[1]
            and scenario["family"] in self.families
            and atoms <= self.maximum_atoms
        )


def periodic_resample(values: NDArray[np.floating], points: int) -> NDArray[np.floating]:
    """Fourier-resample a real periodic state to an arbitrary uniform grid."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or points < 3 or not np.all(np.isfinite(array)):
        raise ValueError("Periodic resampling needs a finite vector and >=3 points.")
    return np.asarray(resample(array, points), dtype=np.float64)


def cross_grid_relative_h1(
    left: NDArray[np.floating],
    right: NDArray[np.floating],
    *,
    length: float,
    comparison_points: int | None = None,
) -> float:
    """Compare arbitrary periodic grids in one declared discrete H1 metric."""
    points = comparison_points or max(left.size, right.size)
    left_common = periodic_resample(left, points)
    right_common = periodic_resample(right, points)
    operator = HelmholtzOperator(PeriodicGrid(points=points, length=length))
    return a_norm(left_common - right_common, operator) / max(
        a_norm(right_common, operator), np.finfo(float).eps
    )


@dataclass(frozen=True, slots=True)
class GroupedRiskSummary:
    coverage: float
    worst_group_coverage: float
    maximum_underprediction: float
    groups: dict[str, float]


def grouped_risk_summary(
    prediction: NDArray[np.floating],
    target: NDArray[np.floating],
    groups: Iterable[str],
) -> GroupedRiskSummary:
    """Report global and worst trajectory-group one-sided risk coverage."""
    predicted = np.asarray(prediction, dtype=np.float64)
    truth = np.asarray(target, dtype=np.float64)
    labels = np.asarray(list(groups), dtype=str)
    if predicted.shape != truth.shape or predicted.ndim != 1 or labels.shape != predicted.shape:
        raise ValueError("Risk arrays and group labels must be equal vectors.")
    if predicted.size == 0 or np.any(predicted < 0.0) or np.any(truth < 0.0):
        raise ValueError("Risk audit requires non-empty non-negative data.")
    if not np.all(np.isfinite(predicted)) or not np.all(np.isfinite(truth)):
        raise ValueError("Risk audit data must be finite.")
    covered = predicted + 1e-7 >= truth
    per_group = {
        label: float(np.mean(covered[labels == label]))
        for label in sorted(set(labels.tolist()))
    }
    return GroupedRiskSummary(
        float(np.mean(covered)),
        min(per_group.values()),
        float(np.max(np.maximum(truth - predicted, 0.0))),
        per_group,
    )
