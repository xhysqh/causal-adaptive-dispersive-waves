"""Causal conditional features for the R2.4.2 target-cell repair.

R2.4.1-A showed that the two AMR bookkeeping channels reduce out-of-sample
identifiability, while the usefulness of the remaining local field channels
depends on peak geometry and rollout phase. R2.4.2 therefore uses

    60 frozen R2.3 + 8 stable local + 4 current geometry + 3 phase = 75

channels. Every channel is available before committing the candidate step.
"""

from __future__ import annotations

import numpy as np

from fgsp_ch.features.m63433524r241_local_risk import LOCAL_COUNTERFACTUAL_FEATURE_ORDER


R23_DIM = 60
FULL_LOCAL_DIM = len(LOCAL_COUNTERFACTUAL_FEATURE_ORDER)
STABLE_LOCAL_INDICES = np.arange(8, dtype=int)
STABLE_LOCAL_FEATURE_ORDER = tuple(
    LOCAL_COUNTERFACTUAL_FEATURE_ORDER[index] for index in STABLE_LOCAL_INDICES
)
GEOMETRY_FEATURE_ORDER = (
    "current_log1p_min_separation_over_h",
    "current_log1p_absolute_amplitude_ratio",
    "current_signed_amplitude_balance",
    "current_particle_count_fraction",
)
PHASE_FEATURE_ORDER = (
    "rollout_progress_fraction",
    "consecutive_fallback_fraction",
    "candidate_recency_fraction",
)
CONDITIONAL_LOCAL_RISK_FEATURE_ORDER = (
    *(f"r23_{index}" for index in range(R23_DIM)),
    *STABLE_LOCAL_FEATURE_ORDER,
    *GEOMETRY_FEATURE_ORDER,
    *PHASE_FEATURE_ORDER,
)

STABLE_LOCAL_DIM = len(STABLE_LOCAL_FEATURE_ORDER)
GEOMETRY_DIM = len(GEOMETRY_FEATURE_ORDER)
PHASE_DIM = len(PHASE_FEATURE_ORDER)
CONDITIONAL_LOCAL_RISK_FEATURE_DIM = R23_DIM + STABLE_LOCAL_DIM + GEOMETRY_DIM + PHASE_DIM

R23_SLICE = slice(0, R23_DIM)
LOCAL_SLICE = slice(R23_DIM, R23_DIM + STABLE_LOCAL_DIM)
GEOMETRY_SLICE = slice(LOCAL_SLICE.stop, LOCAL_SLICE.stop + GEOMETRY_DIM)
PHASE_SLICE = slice(GEOMETRY_SLICE.stop, CONDITIONAL_LOCAL_RISK_FEATURE_DIM)


def stable_local_features(local_features: np.ndarray) -> np.ndarray:
    """Remove the two R2.4.1 AMR bookkeeping channels."""
    values = np.asarray(local_features, dtype=np.float64)
    if values.shape != (FULL_LOCAL_DIM,) or not np.all(np.isfinite(values)):
        raise ValueError(f"full local features must be finite and {FULL_LOCAL_DIM}-dimensional")
    return values[STABLE_LOCAL_INDICES]


def current_peak_geometry_features(
    particle_state,
    *,
    grid_h: float,
    domain_length: float,
) -> np.ndarray:
    """Return four absolute, pre-commit periodic peak-geometry coordinates."""
    if not np.isfinite(grid_h) or grid_h <= 0.0:
        raise ValueError("grid_h must be positive and finite")
    if not np.isfinite(domain_length) or domain_length <= 0.0:
        raise ValueError("domain_length must be positive and finite")
    positions = np.mod(np.asarray(particle_state.positions, dtype=np.float64).ravel(), domain_length)
    amplitudes = np.asarray(particle_state.amplitudes, dtype=np.float64).ravel()
    if positions.shape != amplitudes.shape or not (
        np.all(np.isfinite(positions)) and np.all(np.isfinite(amplitudes))
    ):
        raise ValueError("particle positions and amplitudes must be aligned and finite")
    if positions.size < 2:
        log_separation = 0.0
        log_ratio = 0.0
    else:
        distance = np.abs(positions[:, None] - positions[None, :])
        distance = np.minimum(distance, domain_length - distance)
        distance += np.eye(len(positions)) * domain_length
        log_separation = float(
            np.log1p(float(np.min(distance)) / max(grid_h, np.finfo(float).eps))
        )
        absolute = np.abs(amplitudes)
        ratio = float(np.max(absolute)) / max(float(np.min(absolute)), np.finfo(float).eps)
        log_ratio = float(np.log1p(ratio))
    absolute_sum = max(float(np.sum(np.abs(amplitudes))), np.finfo(float).eps)
    balance = float(np.sum(amplitudes)) / absolute_sum if amplitudes.size else 0.0
    count_fraction = min(float(positions.size) / 4.0, 2.0)
    values = np.asarray((log_separation, log_ratio, balance, count_fraction), dtype=np.float64)
    if values.shape != (GEOMETRY_DIM,) or not np.all(np.isfinite(values)):
        raise ValueError("current geometry features must be finite")
    return values


def rollout_phase_features(
    *,
    rollout_step: int,
    maximum_rollout_steps: int,
    consecutive_fallback_fraction: float,
    last_candidate_step: int,
) -> np.ndarray:
    """Return causal phase coordinates without a behaviour-policy identifier."""
    if rollout_step < 0 or maximum_rollout_steps <= 0:
        raise ValueError("rollout indices must be non-negative with a positive maximum")
    denominator = max(int(maximum_rollout_steps) - 1, 1)
    progress = float(np.clip(rollout_step / denominator, 0.0, 1.0))
    fallback = float(np.clip(consecutive_fallback_fraction, 0.0, 1.0))
    if last_candidate_step < 0:
        recency = 1.0
    else:
        recency = float(
            np.clip((rollout_step - last_candidate_step) / max(maximum_rollout_steps, 1), 0.0, 1.0)
        )
    values = np.asarray((progress, fallback, recency), dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError("phase features must be finite")
    return values


def conditional_local_risk_features(
    *,
    r23_features: np.ndarray,
    local_features: np.ndarray,
    geometry_features: np.ndarray,
    phase_features: np.ndarray,
) -> np.ndarray:
    """Assemble the frozen 75-channel R2.4.2 feature vector."""
    r23 = np.asarray(r23_features, dtype=np.float64)
    geometry = np.asarray(geometry_features, dtype=np.float64)
    phase = np.asarray(phase_features, dtype=np.float64)
    if r23.shape != (R23_DIM,):
        raise ValueError(f"R2.3 features must have shape ({R23_DIM},)")
    if geometry.shape != (GEOMETRY_DIM,):
        raise ValueError(f"geometry features must have shape ({GEOMETRY_DIM},)")
    if phase.shape != (PHASE_DIM,):
        raise ValueError(f"phase features must have shape ({PHASE_DIM},)")
    values = np.concatenate((r23, stable_local_features(local_features), geometry, phase))
    if values.shape != (CONDITIONAL_LOCAL_RISK_FEATURE_DIM,) or not np.all(np.isfinite(values)):
        raise ValueError("conditional R2.4.2 features must be finite and 75-dimensional")
    return values


# Compatibility name for callers; the rejected six-channel differential is gone.
PEAK_GEOMETRY_FEATURE_ORDER = GEOMETRY_FEATURE_ORDER
peak_geometry_features = current_peak_geometry_features
