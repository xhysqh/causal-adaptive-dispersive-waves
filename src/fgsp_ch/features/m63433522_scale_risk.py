from __future__ import annotations

import numpy as np

from fgsp_ch.evaluation.m63433521_scale_audit import ScaleErrorBudget


SCALE_RISK_FEATURE_ORDER = (
    *(f"causal_{index}" for index in range(43)),
    "log_h",
    "log_dt",
    "log_dt_over_h",
    "log1p_causal_uncertainty_proxy_relative_h1",
    "log1p_scale_budget_relative_h1",
)

CLOSED_LOOP_SCALE_RISK_FEATURE_ORDER = (
    *SCALE_RISK_FEATURE_ORDER,
    "previous_action_was_candidate",
    "fallback_fraction_so_far",
    "consecutive_fallback_fraction",
    "log1p_previous_actual_relative_h1_increment",
)

# These quantities are evaluated before the controller commits either branch.
# They describe a candidate *relative to* the conservative fallback, never a
# behaviour-policy name or a post-hoc shadow/reference label.
ACTION_CONTRAST_FEATURE_ORDER = (
    "log1p_candidate_fallback_relative_h1",
    "log1p_candidate_fallback_relative_mass",
    "log1p_candidate_fallback_relative_energy",
    "candidate_cfl_over_limit",
    "candidate_patch_migration_ratio",
    "log1p_candidate_relative_h1_increment",
    "candidate_fraction_so_far",
    "time_since_last_candidate_fraction",
)

CLOSED_LOOP_ACTION_CONTRAST_FEATURE_ORDER = (
    *CLOSED_LOOP_SCALE_RISK_FEATURE_ORDER,
    *ACTION_CONTRAST_FEATURE_ORDER,
)


def scale_conditioned_features(
    causal_features: np.ndarray,
    *,
    h: float,
    dt: float,
    state_norm: float,
    causal_uncertainty_proxy: float,
    budget: ScaleErrorBudget,
) -> np.ndarray:
    base = np.asarray(causal_features, dtype=np.float64).copy()
    if base.shape != (43,):
        raise ValueError("causal_features must have 43 channels")
    if min(h, dt, state_norm) <= 0.0 or causal_uncertainty_proxy < 0.0:
        raise ValueError("invalid scale feature inputs")
    # Channel 34 is the old deployment risk.  It is deliberately removed so
    # the replacement certificate cannot learn a shortcut through its target.
    base[34] = 0.0
    relative_reference = causal_uncertainty_proxy / state_norm
    relative_budget = float(budget.relative_budget(h, dt))
    scale = np.asarray((
        np.log(h), np.log(dt), np.log(dt / h),
        np.log1p(relative_reference), np.log1p(relative_budget),
    ))
    result = np.concatenate((base, scale))
    if result.shape != (48,) or not np.all(np.isfinite(result)):
        raise ValueError("scale-conditioned features must be finite and 48-dimensional")
    return result


def closed_loop_scale_conditioned_features(
    causal_features: np.ndarray,
    *,
    h: float,
    dt: float,
    state_norm: float,
    causal_uncertainty_proxy: float,
    budget: ScaleErrorBudget,
    previous_action_was_candidate: bool,
    fallback_fraction_so_far: float,
    consecutive_fallback_fraction: float,
    previous_actual_relative_h1_increment: float,
) -> np.ndarray:
    """Append only controller history available before the current decision."""
    base = scale_conditioned_features(
        causal_features, h=h, dt=dt, state_norm=state_norm,
        causal_uncertainty_proxy=causal_uncertainty_proxy, budget=budget,
    )
    if not (0.0 <= fallback_fraction_so_far <= 1.0
            and 0.0 <= consecutive_fallback_fraction <= 1.0
            and previous_actual_relative_h1_increment >= 0.0):
        raise ValueError("closed-loop history features are outside their causal bounds")
    history = np.asarray((
        float(bool(previous_action_was_candidate)),
        float(fallback_fraction_so_far),
        float(consecutive_fallback_fraction),
        np.log1p(float(previous_actual_relative_h1_increment)),
    ))
    result = np.concatenate((base, history))
    if result.shape != (52,) or not np.all(np.isfinite(result)):
        raise ValueError("closed-loop scale features must be finite and 52-dimensional")
    return result


def action_contrast_features(
    *,
    candidate_fallback_relative_h1: float,
    candidate_fallback_relative_mass: float,
    candidate_fallback_relative_energy: float,
    candidate_cfl_over_limit: float,
    candidate_patch_migration_ratio: float,
    candidate_relative_h1_increment: float,
    candidate_fraction_so_far: float,
    time_since_last_candidate_fraction: float,
) -> np.ndarray:
    """Encode a pre-commit candidate-versus-fallback comparison.

    The function is intentionally scalar-only.  Its caller must provide
    values computed from the current state and two one-step previews; this
    prevents accidental use of later rollout outcomes or reference labels.
    """
    nonnegative = (
        candidate_fallback_relative_h1, candidate_fallback_relative_mass,
        candidate_fallback_relative_energy, candidate_cfl_over_limit,
        candidate_patch_migration_ratio, candidate_relative_h1_increment,
    )
    if (not np.all(np.isfinite(nonnegative)) or min(nonnegative) < 0.0
            or not 0.0 <= candidate_fraction_so_far <= 1.0
            or not 0.0 <= time_since_last_candidate_fraction <= 1.0):
        raise ValueError("invalid pre-commit action-contrast features")
    result = np.asarray((
        np.log1p(candidate_fallback_relative_h1),
        np.log1p(candidate_fallback_relative_mass),
        np.log1p(candidate_fallback_relative_energy),
        candidate_cfl_over_limit,
        candidate_patch_migration_ratio,
        np.log1p(candidate_relative_h1_increment),
        candidate_fraction_so_far,
        time_since_last_candidate_fraction,
    ), dtype=np.float64)
    if result.shape != (8,) or not np.all(np.isfinite(result)):
        raise ValueError("action-contrast features must be finite and 8-dimensional")
    return result


def action_conditioned_closed_loop_features(
    causal_features: np.ndarray,
    *,
    action_contrast: np.ndarray,
    **kwargs,
) -> np.ndarray:
    """Extend the legacy causal closed-loop representation to 60 channels."""
    base = closed_loop_scale_conditioned_features(causal_features, **kwargs)
    contrast = np.asarray(action_contrast, dtype=np.float64)
    if contrast.shape != (8,) or not np.all(np.isfinite(contrast)):
        raise ValueError("action contrast must be finite and 8-dimensional")
    result = np.concatenate((base, contrast))
    if result.shape != (60,):
        raise ValueError("action-conditioned closed-loop features must be 60-dimensional")
    return result
