"""Trajectory-level error ledger for CMAME-M1.3.

The ledger converts local compatible hierarchical defects into a terminal
budget.  Reference trajectories never enter this module.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fgsp_ch.discretization.difference import first_derivative
from fgsp_ch.discretization.grids import PeriodicGrid
from fgsp_ch.solvers.mch_hybrid_amr import HybridMCHState, hybrid_total_state
from fgsp_ch.solvers.mch_compatible_adaptive import minimum_periodic_separation


def computable_stability_rate(
    state: HybridMCHState, grid: PeriodicGrid, alpha: float, *, cap: float = 50.0
) -> float:
    """A finite registered-regime growth proxy using only the current state."""
    total = hybrid_total_state(state, grid)
    slope = first_derivative(total, grid)
    scale = abs(alpha) * (2.0 * np.max(np.abs(total)) + np.max(np.abs(slope)))
    if state.positions.size >= 2:
        separation = minimum_periodic_separation(state.positions, grid.length)
        scale += abs(alpha) * grid.h / max(separation, grid.h)
    return float(np.clip(scale, 0.0, cap))


@dataclass(slots=True)
class TrajectoryErrorLedger:
    tolerance: float
    final_time: float
    reserve_fraction: float = 0.1
    accumulated_bound: float = 0.0
    time: float = 0.0

    def __post_init__(self) -> None:
        if self.tolerance <= 0.0 or self.final_time <= 0.0:
            raise ValueError("trajectory tolerance and final time must be positive")
        if not 0.0 <= self.reserve_fraction < 1.0:
            raise ValueError("reserve_fraction must lie in [0,1)")

    @property
    def usable_tolerance(self) -> float:
        return (1.0 - self.reserve_fraction) * self.tolerance

    @property
    def remaining(self) -> float:
        return max(self.usable_tolerance - self.accumulated_bound, 0.0)

    def predict(self, dt: float, local_defect: float, migration_defect: float, stability_rate: float) -> float:
        if dt <= 0.0 or min(local_defect, migration_defect, stability_rate) < 0.0:
            raise ValueError("ledger inputs must be nonnegative and dt positive")
        return float(np.exp(stability_rate * dt) * (self.accumulated_bound + local_defect + migration_defect))

    def can_commit(self, dt: float, local_defect: float, migration_defect: float, stability_rate: float) -> bool:
        return self.predict(dt, local_defect, migration_defect, stability_rate) <= self.usable_tolerance

    def commit(self, dt: float, local_defect: float, migration_defect: float, stability_rate: float) -> float:
        self.accumulated_bound = self.predict(dt, local_defect, migration_defect, stability_rate)
        self.time += dt
        return self.accumulated_bound


def localized_residual(
    local_indicators: np.ndarray, active_coarse: np.ndarray, spatial_estimate: float
) -> float:
    """Scale unrefined Dörfler energy to the calibrated spatial estimate."""
    eta = np.asarray(local_indicators, dtype=np.float64)
    active = np.asarray(active_coarse, dtype=bool)
    if eta.shape != active.shape or np.any(eta < 0.0) or spatial_estimate < 0.0:
        raise ValueError("local residual inputs are incompatible")
    total = float(np.linalg.norm(eta))
    if total <= np.finfo(float).eps:
        return 0.0
    return float(spatial_estimate * np.linalg.norm(eta[~active]) / total)

