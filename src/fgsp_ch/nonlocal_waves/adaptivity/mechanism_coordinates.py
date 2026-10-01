"""Equation-neutral coordinates and support scores for CMAME-U1."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from fgsp_ch.nonlocal_waves.adaptivity.prospective_causal_operator import (
    FEATURE_ORDER as LEGACY_FEATURE_ORDER,
)


MECHANISM_FEATURE_ORDER = LEGACY_FEATURE_ORDER + (
    "localized_indicator_available",
    "spectral_indicator_available",
    "migration_indicator_available",
)

# These coordinates are all available before an action is committed.  The
# availability masks remain explicit in the base mechanism row; consequently a
# zero in an unavailable spectral channel is never interpreted as an observed
# zero by a downstream model.
HISTORY_SOURCE_FEATURE_ORDER = (
    "log_spatial_over_budget",
    "log_representation_over_budget",
    "log_stiffness",
    "localized_sharpness",
    "nonlocal_tail",
    "log_invariant_over_limit",
)
CAUSAL_HISTORY_FEATURE_ORDER = tuple(
    f"history_deviation_{name}" for name in HISTORY_SOURCE_FEATURE_ORDER
) + tuple(
    f"history_velocity_{name}" for name in HISTORY_SOURCE_FEATURE_ORDER
) + tuple(
    f"history_acceleration_{name}" for name in HISTORY_SOURCE_FEATURE_ORDER
)
CAUSAL_CONTROL_FEATURE_ORDER = (
    "history_remaining_budget_fraction",
    "history_remaining_time_fraction",
    "history_resolution_fraction",
    "history_log_current_dt",
    "history_cooldown_fraction",
)
V2_CAUSAL_FEATURE_ORDER = (
    MECHANISM_FEATURE_ORDER + CAUSAL_HISTORY_FEATURE_ORDER
    + CAUSAL_CONTROL_FEATURE_ORDER
)

_HISTORY_SOURCE_INDICES = tuple(
    LEGACY_FEATURE_ORDER.index(name) for name in HISTORY_SOURCE_FEATURE_ORDER
)


@dataclass(frozen=True, slots=True)
class CausalHistorySnapshot:
    """Pre-decision, non-anticipative physical-history coordinates."""

    physical: NDArray[np.float64]
    exponential_mean: NDArray[np.float64]
    deviation: NDArray[np.float64]
    velocity: NDArray[np.float64]
    acceleration: NDArray[np.float64]

    def features(self) -> NDArray[np.float64]:
        result = np.r_[self.deviation, self.velocity, self.acceleration]
        if result.shape != (len(CAUSAL_HISTORY_FEATURE_ORDER),):
            raise RuntimeError("V2 causal-history feature width mismatch")
        return np.asarray(result, dtype=np.float64)


@dataclass(slots=True)
class CausalHistoryState:
    """A fixed-memory causal filter for equation-neutral diagnostics.

    ``observe`` is called once per accepted-prefix state, before candidate
    actions are built. It never receives future labels, reference states or a
    candidate outcome.
    """

    rho: float = 0.7
    _previous: NDArray[np.float64] | None = None
    _previous_velocity: NDArray[np.float64] | None = None
    _mean: NDArray[np.float64] | None = None
    _has_velocity: bool = False

    def __post_init__(self):
        if not np.isfinite(self.rho) or not 0.0 < self.rho < 1.0:
            raise ValueError("V2 history rho must lie strictly between zero and one")

    def observe(self, physical) -> CausalHistorySnapshot:
        current = np.asarray(physical, dtype=np.float64)
        width = len(HISTORY_SOURCE_FEATURE_ORDER)
        if current.shape != (width,) or not np.all(np.isfinite(current)):
            raise ValueError("V2 physical-history coordinates must be finite")
        had_previous = self._previous is not None
        if not had_previous:
            mean = current.copy()
            velocity = np.zeros_like(current)
            acceleration = np.zeros_like(current)
        else:
            if self._mean is None or self._previous_velocity is None:
                raise RuntimeError("V2 causal-history state is internally inconsistent")
            mean = self.rho * self._mean + (1.0 - self.rho) * current
            velocity = current - self._previous
            acceleration = (
                velocity - self._previous_velocity
                if self._has_velocity else np.zeros_like(current)
            )
        snapshot = CausalHistorySnapshot(
            current.copy(), mean.copy(), current - mean,
            velocity.copy(), acceleration.copy(),
        )
        self._previous = current.copy()
        self._previous_velocity = velocity.copy()
        self._mean = mean.copy()
        self._has_velocity = had_previous
        return snapshot


def mechanism_history_coordinates(mechanism_row) -> NDArray[np.float64]:
    """Extract the six physical diagnostics that are permitted in history.

    The source row already includes explicit availability masks. Missing
    diagnostics retain their contract value here and remain distinguishable via
    those masks in the concatenated V2 row.
    """
    row = np.asarray(mechanism_row, dtype=np.float64)
    if row.shape != (len(MECHANISM_FEATURE_ORDER),) or not np.all(np.isfinite(row)):
        raise ValueError("invalid mechanism row for V2 causal history")
    return np.asarray(row[list(_HISTORY_SOURCE_INDICES)], dtype=np.float64)


def causal_control_coordinates(*, remaining_budget_fraction: float,
                               remaining_time_fraction: float,
                               resolution_fraction: float,
                               current_dt: float,
                               cooldown_fraction: float = 0.0) -> NDArray[np.float64]:
    """Build non-anticipative control context with explicit bounded fractions."""
    fractions = np.asarray(
        [remaining_budget_fraction, remaining_time_fraction,
         resolution_fraction, cooldown_fraction], dtype=np.float64,
    )
    if (np.any(~np.isfinite(fractions)) or np.any(fractions < 0.0)
            or np.any(fractions > 1.0) or not np.isfinite(current_dt)
            or current_dt <= 0.0):
        raise ValueError("invalid V2 causal control coordinates")
    return np.asarray(
        [remaining_budget_fraction, remaining_time_fraction,
         resolution_fraction, np.log(current_dt), cooldown_fraction],
        dtype=np.float64,
    )


def v2_causal_coordinates(mechanism_row, history: CausalHistorySnapshot,
                          control_row) -> NDArray[np.float64]:
    """Concatenate base candidate, causal history and control coordinates."""
    base = np.asarray(mechanism_row, dtype=np.float64)
    control = np.asarray(control_row, dtype=np.float64)
    if base.shape != (len(MECHANISM_FEATURE_ORDER),):
        raise ValueError("invalid V2 base mechanism row")
    if control.shape != (len(CAUSAL_CONTROL_FEATURE_ORDER),):
        raise ValueError("invalid V2 causal control row")
    result = np.r_[base, history.features(), control]
    if result.shape != (len(V2_CAUSAL_FEATURE_ORDER),) or not np.all(np.isfinite(result)):
        raise ValueError("invalid V2 causal feature row")
    return np.asarray(result, dtype=np.float64)


def mechanism_coordinates(
    legacy_features: NDArray[np.floating], *, localized_available: bool,
    spectral_available: bool, migration_available: bool,
) -> NDArray[np.float64]:
    """Map registered F1 features into a common, dimensionless mechanism row."""
    row = np.asarray(legacy_features, dtype=np.float64).copy()
    if row.shape != (len(LEGACY_FEATURE_ORDER),) or not np.all(np.isfinite(row)):
        raise ValueError("invalid legacy mechanism feature row")
    # Both quantities are concentration/energy fractions in the U1 contract.
    row[10] = np.clip(row[10], 0.0, 1.0)
    row[11] = max(row[11], 0.0)
    result = np.r_[row, float(localized_available), float(spectral_available),
                   float(migration_available)]
    if result.shape != (len(MECHANISM_FEATURE_ORDER),):
        raise RuntimeError("U1 mechanism coordinate width mismatch")
    return np.asarray(result, dtype=np.float64)


def coordinates_for_equation(
    legacy_features: NDArray[np.floating], equation: str,
) -> NDArray[np.float64]:
    if equation == "modified_camassa_holm":
        return mechanism_coordinates(
            legacy_features, localized_available=True,
            spectral_available=False, migration_available=True,
        )
    if equation == "benjamin_ono":
        return mechanism_coordinates(
            legacy_features, localized_available=True,
            spectral_available=True, migration_available=True,
        )
    if equation == "kdv":
        return mechanism_coordinates(
            legacy_features, localized_available=True,
            spectral_available=True, migration_available=True,
        )
    raise ValueError(f"unsupported mechanism coordinate source: {equation}")


@dataclass(frozen=True, slots=True)
class RobustMechanismScaler:
    center: NDArray[np.float64]
    scale: NDArray[np.float64]
    minimum_scale: float

    @classmethod
    def fit(cls, rows, *, minimum_scale: float = 5e-3):
        values = np.asarray(rows, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != len(MECHANISM_FEATURE_ORDER):
            raise ValueError("U1 robust scaler received an invalid matrix")
        center = np.median(values, axis=0)
        mad = np.median(np.abs(values - center), axis=0)
        scale = np.maximum(1.4826 * mad, float(minimum_scale))
        return cls(center, scale, float(minimum_scale))

    def transform(self, rows):
        values = np.asarray(rows, dtype=np.float64)
        if values.shape[-1] != len(MECHANISM_FEATURE_ORDER):
            raise ValueError("U1 robust transform width mismatch")
        return (values - self.center) / self.scale


def knn_support_score(rows, anchors, *, neighbors: int) -> NDArray[np.float64]:
    """RMS distance to k nearest frozen train anchors in robust coordinates."""
    query = np.atleast_2d(np.asarray(rows, dtype=np.float64))
    reference = np.asarray(anchors, dtype=np.float64)
    if query.shape[1:] != reference.shape[1:] or reference.ndim != 2:
        raise ValueError("U1 support matrices have incompatible shapes")
    k = min(max(int(neighbors), 1), len(reference))
    squared = np.mean((query[:, None, :] - reference[None, :, :])**2, axis=2)
    nearest = np.partition(squared, k - 1, axis=1)[:, :k]
    return np.sqrt(np.mean(nearest, axis=1))


def trajectory_conformal_radius(scores, groups, *, coverage: float) -> float:
    values = np.asarray(scores, dtype=np.float64)
    group_values = np.asarray(groups).astype(str)
    if values.shape != group_values.shape or not 0 < coverage < 1:
        raise ValueError("invalid U1 support calibration arrays")
    maxima = np.asarray([
        np.max(values[group_values == group]) for group in sorted(set(group_values))
    ])
    rank = min(len(maxima), int(np.ceil((len(maxima) + 1) * coverage)))
    return float(np.partition(maxima, rank - 1)[rank - 1])
