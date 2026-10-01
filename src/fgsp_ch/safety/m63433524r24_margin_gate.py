"""Auditable non-negative risk-margin gate used by the R2.4 on-policy repair.

The preliminary R2.3 network is deliberately *not* retrained here.  R2.4 only
adds a small, causal post-processor to its frozen K=1 score.  This keeps the
repair interpretable and makes it possible to freeze the exact gate before the
two independent deployment audits.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np


# Indices in CLOSED_LOOP_ACTION_CONTRAST_FEATURE_ORDER.  The complement of the
# candidate fraction is used because a smaller candidate footprint is the
# conservative direction for the observed boundary failures.
MARGIN_CONTRAST_INDICES = (0, 4, 5, 6, 7)


def margin_coordinates(action_contrast: np.ndarray) -> np.ndarray:
    """Return five pre-commit, direction-normalised margin coordinates."""
    contrast = np.asarray(action_contrast, dtype=np.float64)
    if contrast.shape[-1] != 8:
        raise ValueError("R2.4 margin gate requires eight action-contrast channels")
    value = contrast[..., MARGIN_CONTRAST_INDICES].copy()
    value[..., 3] = 1.0 - value[..., 3]
    return value


def hinge_coordinates(action_contrast: np.ndarray, knots: np.ndarray, scales: np.ndarray) -> np.ndarray:
    value = margin_coordinates(action_contrast)
    knots = np.asarray(knots, dtype=np.float64)
    scales = np.asarray(scales, dtype=np.float64)
    if knots.shape != (5,) or scales.shape != (5,) or np.any(scales <= 0.0):
        raise ValueError("R2.4 margin knots/scales must be five positive coordinates")
    return np.clip((value - knots) / scales, 0.0, 1.0)


def nonnegative_margin(action_contrast: np.ndarray, specification: Mapping[str, object]) -> np.ndarray:
    """Evaluate ``b + sum_j w_j max(0, (v_j-q_j)/s_j)``.

    Parameters are validated at use time: a malformed or negative margin must
    fail closed rather than silently weaken the safety gate.
    """
    intercept = float(specification["intercept"])
    weights = np.asarray(specification["weights"], dtype=np.float64)
    if intercept < 0.0 or weights.shape != (5,) or np.any(weights < 0.0):
        raise ValueError("R2.4 margins must be non-negative")
    phi = hinge_coordinates(action_contrast, specification["knots"], specification["scales"])
    return intercept + np.einsum("...j,j->...", phi, weights)


@dataclass(frozen=True)
class MarginGateDecision:
    accepted: bool
    threshold: float
    base_score: float
    margin: float
    adjusted_score: float
    support_ratio: float
    hard_gate_passed: bool


def apply_margin_gate(base_decision, action_contrast: np.ndarray, specification: Mapping[str, object]) -> MarginGateDecision:
    """Apply one frozen cell-specific R2.4 margin to a preliminary decision."""
    margin = float(nonnegative_margin(np.asarray(action_contrast), specification))
    threshold = float(specification["threshold"])
    score = float(np.asarray(base_decision.joint_risk)[0])
    adjusted = score + margin
    accepted = bool(
        base_decision.hard_gate_passed
        and base_decision.support_ratio <= 1.0
        and adjusted <= threshold
    )
    return MarginGateDecision(
        accepted, threshold, score, margin, adjusted,
        float(base_decision.support_ratio), bool(base_decision.hard_gate_passed),
    )
