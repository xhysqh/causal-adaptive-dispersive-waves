"""Trajectory-group statistics for the frozen CMAME-M2 comparison."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np


@dataclass(frozen=True, slots=True)
class MethodResult:
    case_id: str
    group_id: str
    family: str
    method: str
    relative_l2: float
    relative_h1: float
    relative_linf: float
    mass_drift: float
    h1_drift: float
    peak_position_error: float
    active_dof_time: float
    accepted_steps: int
    runtime_seconds: float
    finite: bool = True

    def validate(self) -> None:
        scalars = (
            self.relative_l2, self.relative_h1, self.relative_linf,
            self.mass_drift, self.h1_drift, self.peak_position_error,
            self.active_dof_time, self.runtime_seconds,
        )
        if not self.case_id or not self.group_id or not self.family or not self.method:
            raise ValueError("M2 result identifiers must be nonempty")
        if not self.finite or not all(math.isfinite(value) and value >= 0.0 for value in scalars):
            raise ValueError("M2 result metrics must be finite and nonnegative")
        if self.accepted_steps < 1:
            raise ValueError("accepted_steps must be positive")


def grouped_bootstrap_interval(
    values: Iterable[float],
    groups: Iterable[str],
    *,
    samples: int = 2000,
    confidence: float = 0.95,
    seed: int = 20260827,
) -> tuple[float, float]:
    values_array = np.asarray(list(values), dtype=float)
    groups_array = np.asarray(list(groups), dtype=str)
    if values_array.ndim != 1 or values_array.size == 0 or values_array.shape != groups_array.shape:
        raise ValueError("values and groups must be aligned nonempty vectors")
    if not np.all(np.isfinite(values_array)) or samples < 100 or not 0.0 < confidence < 1.0:
        raise ValueError("invalid grouped bootstrap input")
    unique = np.unique(groups_array)
    rng = np.random.default_rng(seed)
    estimates = np.empty(samples)
    for index in range(samples):
        selected = rng.choice(unique, size=len(unique), replace=True)
        draw = np.concatenate([values_array[groups_array == group] for group in selected])
        estimates[index] = np.median(draw)
    tail = 0.5 * (1.0 - confidence)
    return tuple(map(float, np.quantile(estimates, [tail, 1.0 - tail])))


def paired_log10_ratio(
    proposed: Iterable[float], baseline: Iterable[float]
) -> np.ndarray:
    left, right = np.asarray(list(proposed), dtype=float), np.asarray(list(baseline), dtype=float)
    if left.shape != right.shape or left.ndim != 1 or left.size == 0:
        raise ValueError("paired metrics must be aligned nonempty vectors")
    if np.any(left < 0.0) or np.any(right < 0.0) or not np.all(np.isfinite(left + right)):
        raise ValueError("paired metrics must be finite and nonnegative")
    floor = np.finfo(float).tiny
    return np.log10(np.maximum(left, floor) / np.maximum(right, floor))


def summarize_results(rows: Iterable[MethodResult]) -> dict[str, dict[str, float | list[float]]]:
    records = list(rows)
    for row in records:
        row.validate()
    summary: dict[str, dict[str, float | list[float]]] = {}
    for method in sorted({row.method for row in records}):
        subset = [row for row in records if row.method == method]
        errors = np.asarray([row.relative_h1 for row in subset])
        groups = [row.group_id for row in subset]
        summary[method] = {
            "cases": float(len(subset)),
            "median_relative_h1": float(np.median(errors)),
            "p95_relative_h1": float(np.quantile(errors, 0.95)),
            "worst_relative_h1": float(np.max(errors)),
            "median_h1_group_bootstrap_95": list(grouped_bootstrap_interval(errors, groups)),
            "median_mass_drift": float(np.median([row.mass_drift for row in subset])),
            "median_h1_drift": float(np.median([row.h1_drift for row in subset])),
            "median_active_dof_time": float(np.median([row.active_dof_time for row in subset])),
            "median_runtime_seconds": float(np.median([row.runtime_seconds for row in subset])),
        }
    return summary
