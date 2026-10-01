from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import nnls


@dataclass(frozen=True)
class ScaleErrorBudget:
    floor: float
    spatial_coefficient: float
    temporal_coefficient: float
    spatial_order: float
    temporal_order: float
    safety_factor: float

    def relative_budget(self, h, dt):
        return (
            self.floor
            + self.spatial_coefficient * np.asarray(h) ** self.spatial_order
            + self.temporal_coefficient * np.asarray(dt) ** self.temporal_order
        )

    def to_dict(self) -> dict:
        return {
            "floor": self.floor,
            "spatial_coefficient": self.spatial_coefficient,
            "temporal_coefficient": self.temporal_coefficient,
            "spatial_order": self.spatial_order,
            "temporal_order": self.temporal_order,
            "safety_factor": self.safety_factor,
        }


def fit_scale_error_budget(
    absolute_error: np.ndarray,
    state_norm: np.ndarray,
    reference_uncertainty: np.ndarray,
    h: np.ndarray,
    dt: np.ndarray,
    *,
    order_candidates: tuple[float, ...],
    safety_quantile: float = 0.9,
) -> ScaleErrorBudget:
    error = np.asarray(absolute_error, dtype=np.float64)
    scale = np.asarray(state_norm, dtype=np.float64)
    uncertainty = np.asarray(reference_uncertainty, dtype=np.float64)
    h = np.asarray(h, dtype=np.float64)
    dt = np.asarray(dt, dtype=np.float64)
    if not (error.shape == scale.shape == uncertainty.shape == h.shape == dt.shape):
        raise ValueError("scale audit arrays must have equal shape")
    if np.any(scale <= 0) or np.any(h <= 0) or np.any(dt <= 0):
        raise ValueError("state scale, h and dt must be positive")
    # Reference contamination is not attributed to the candidate discretization.
    target = np.maximum(error - uncertainty, 0.0) / scale
    best = None
    for spatial_order in order_candidates:
        for temporal_order in order_candidates:
            design = np.column_stack((
                np.ones(len(h)), h ** spatial_order, dt ** temporal_order,
            ))
            coefficients, _ = nnls(design, target)
            prediction = design @ coefficients
            loss = float(np.mean((np.log1p(target) - np.log1p(prediction)) ** 2))
            candidate = (loss, spatial_order, temporal_order, coefficients)
            if best is None or candidate[0] < best[0]:
                best = candidate
    assert best is not None
    _, spatial_order, temporal_order, coefficients = best
    base = (
        coefficients[0]
        + coefficients[1] * h ** spatial_order
        + coefficients[2] * dt ** temporal_order
    )
    denominator = uncertainty + scale * np.maximum(base, np.finfo(float).eps)
    ratio = error / np.maximum(denominator, np.finfo(float).eps)
    safety_factor = float(max(1.0, np.quantile(ratio, safety_quantile, method="higher")))
    return ScaleErrorBudget(
        floor=float(coefficients[0]),
        spatial_coefficient=float(coefficients[1]),
        temporal_coefficient=float(coefficients[2]),
        spatial_order=float(spatial_order),
        temporal_order=float(temporal_order),
        safety_factor=safety_factor,
    )


def scale_consistent_shadow_label(
    absolute_error: np.ndarray,
    state_norm: np.ndarray,
    reference_uncertainty: np.ndarray,
    h: np.ndarray,
    dt: np.ndarray,
    budget: ScaleErrorBudget,
) -> np.ndarray:
    physical = np.asarray(state_norm) * budget.relative_budget(h, dt)
    denominator = budget.safety_factor * (
        np.asarray(reference_uncertainty) + physical
    )
    return np.asarray(absolute_error) / np.maximum(denominator, np.finfo(float).eps)


def paired_dt_log_sensitivity(values, seeds, points, dt) -> float:
    """Median within-(seed, grid) endpoint slope; avoids cross-grid confounding."""
    values = np.asarray(values, dtype=np.float64)
    seeds = np.asarray(seeds)
    points = np.asarray(points)
    dt = np.asarray(dt, dtype=np.float64)
    slopes = []
    for seed in np.unique(seeds):
        for grid_points in np.unique(points):
            mask = (seeds == seed) & (points == grid_points)
            if np.count_nonzero(mask) < 2:
                continue
            local_dt = dt[mask]
            local_values = values[mask]
            low = int(np.argmin(local_dt))
            high = int(np.argmax(local_dt))
            if local_dt[high] <= local_dt[low]:
                continue
            numerator = abs(np.log(max(local_values[high], np.finfo(float).eps)) - np.log(
                max(local_values[low], np.finfo(float).eps)
            ))
            slopes.append(numerator / np.log(local_dt[high] / local_dt[low]))
    if not slopes:
        raise ValueError("paired dt sensitivity requires at least one two-dt group")
    return float(np.median(slopes))
