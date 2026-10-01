"""Common, representation-aware metrics used by the CMAME-P0 runner."""

from __future__ import annotations

import numpy as np

from fgsp_ch.discretization.grids import PeriodicGrid
from fgsp_ch.discretization.helmholtz import HelmholtzOperator
from fgsp_ch.evaluation.metrics import relative_h1_error, relative_l2_error, relative_linf_error
from fgsp_ch.geometry.metric import discrete_energy


def periodic_position_error(left: np.ndarray, right: np.ndarray, length: float) -> float:
    a, b = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    if a.shape != b.shape or a.ndim != 1 or length <= 0:
        raise ValueError("periodic positions must be aligned one-dimensional arrays")
    delta = (a - b + 0.5 * length) % length - 0.5 * length
    return float(np.max(np.abs(delta)))


def convergence_orders(errors: list[float], refinement_ratio: float = 2.0) -> list[float]:
    values = np.asarray(errors, dtype=float)
    if values.ndim != 1 or len(values) < 2 or np.any(values <= 0) or refinement_ratio <= 1:
        raise ValueError("positive errors and refinement_ratio > 1 are required")
    return [float(np.log(a / b) / np.log(refinement_ratio)) for a, b in zip(values[:-1], values[1:], strict=True)]


def field_metrics(state: np.ndarray, reference: np.ndarray, grid: PeriodicGrid, initial: np.ndarray) -> dict[str, float]:
    operator = HelmholtzOperator(grid)
    mass0 = float(grid.h * np.sum(initial))
    energy0 = discrete_energy(initial, operator)
    return {
        "relative_l2": relative_l2_error(state, reference, grid),
        "relative_h1": relative_h1_error(state, reference, operator),
        "relative_linf": relative_linf_error(state, reference),
        "mass_drift": abs(float(grid.h * np.sum(state)) - mass0),
        "h1_drift": abs(discrete_energy(state, operator) - energy0) / max(abs(energy0), np.finfo(float).eps),
    }
