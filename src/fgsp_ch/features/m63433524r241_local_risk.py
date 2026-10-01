"""Causal local candidate-risk features for the R2.4.1 repair.

The narrow ``(N, dt) = (32, 5e-5)`` cell could not be separated by the
global action-history margin alone.  This module therefore exposes a small
set of *pre-commit* field features.  Every quantity is computed from the
current state and the candidate/fallback one-step previews; no shadow label,
reference solution, future state, or behaviour-policy identifier is allowed.
"""

from __future__ import annotations

import numpy as np

from fgsp_ch.discretization.helmholtz import HelmholtzOperator
from fgsp_ch.geometry.metric import a_norm, discrete_energy


LOCAL_COUNTERFACTUAL_FEATURE_ORDER = (
    "log1p_candidate_fallback_field_rms_relative",
    "log1p_candidate_fallback_field_peak_relative",
    "log1p_candidate_fallback_gradient_rms_relative",
    "log1p_candidate_fallback_curvature_rms_relative",
    "signed_candidate_energy_increment_relative",
    "signed_fallback_energy_increment_relative",
    "log1p_candidate_increment_h1_relative",
    "log1p_fallback_increment_h1_relative",
    "candidate_patch_migration_ratio",
    "patch_entered_plus_exited_fraction",
)


def _periodic_gradient(field: np.ndarray, h: float) -> np.ndarray:
    return (np.roll(field, -1) - np.roll(field, 1)) / (2.0 * h)


def _periodic_curvature(field: np.ndarray, h: float) -> np.ndarray:
    return (np.roll(field, -1) - 2.0 * field + np.roll(field, 1)) / (h * h)


def local_counterfactual_features(
    *,
    state_total: np.ndarray,
    candidate_total: np.ndarray,
    fallback_total: np.ndarray,
    operator: HelmholtzOperator,
    grid_h: float,
    diagnostic,
) -> np.ndarray:
    """Return ten finite, causal local field features.

    ``candidate_total`` and ``fallback_total`` are previews for the same
    physical endpoint.  The field contrast is deliberately local in the
    Eulerian sense (RMS/peak/gradient/curvature), complementing the global
    H1 contrast already present in the 60-channel R2.3 representation.
    """
    state = np.asarray(state_total, dtype=np.float64)
    candidate = np.asarray(candidate_total, dtype=np.float64)
    fallback = np.asarray(fallback_total, dtype=np.float64)
    if state.ndim != 1 or candidate.shape != state.shape or fallback.shape != state.shape:
        raise ValueError("local counterfactual fields must be one-dimensional and aligned")
    if not np.isfinite(grid_h) or grid_h <= 0.0:
        raise ValueError("grid_h must be positive and finite")
    scale = max(float(np.sqrt(np.mean(state * state))), np.finfo(float).eps)
    h1_scale = max(float(a_norm(state, operator)), np.finfo(float).eps)
    energy_scale = max(abs(float(discrete_energy(state, operator))), np.finfo(float).eps)
    contrast = candidate - fallback
    gradient = _periodic_gradient(contrast, grid_h)
    curvature = _periodic_curvature(contrast, grid_h)
    values = np.asarray((
        np.log1p(float(np.sqrt(np.mean(contrast * contrast))) / scale),
        np.log1p(float(np.max(np.abs(contrast))) / scale),
        np.log1p(float(np.sqrt(np.mean(gradient * gradient))) / max(scale / grid_h, np.finfo(float).eps)),
        np.log1p(float(np.sqrt(np.mean(curvature * curvature))) / max(scale / (grid_h * grid_h), np.finfo(float).eps)),
        (float(discrete_energy(candidate, operator)) - float(discrete_energy(state, operator))) / energy_scale,
        (float(discrete_energy(fallback, operator)) - float(discrete_energy(state, operator))) / energy_scale,
        np.log1p(float(a_norm(candidate - state, operator)) / h1_scale),
        np.log1p(float(a_norm(fallback - state, operator)) / h1_scale),
        max(float(getattr(diagnostic, "patch_migration_to_step_increment", 0.0)), 0.0),
        max(float(getattr(diagnostic, "patch_entered_fraction", 0.0)), 0.0)
        + max(float(getattr(diagnostic, "patch_exited_fraction", 0.0)), 0.0),
    ), dtype=np.float64)
    if values.shape != (len(LOCAL_COUNTERFACTUAL_FEATURE_ORDER),) or not np.all(np.isfinite(values)):
        raise ValueError("local counterfactual features must be finite")
    return values
