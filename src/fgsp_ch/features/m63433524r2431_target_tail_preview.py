"""Causal target-tail preview features for the R2.4.3.1 reachability head.

The four records are emitted by the *same* candidate-then-frozen-R2.3 branch
that later receives reference labels.  They contain only pre-commit solver and
frozen-controller quantities; reference/shadow quantities are deliberately
not accepted by this interface.
"""

from __future__ import annotations

import numpy as np


TARGET_TAIL_PREVIEW_FEATURE_ORDER = (
    "r23_k1_risk", "maximum_r23_k1_risk_to_k2", "maximum_r23_k1_risk_to_k4",
    "minimum_r23_margin_k1", "minimum_r23_margin_to_k2", "minimum_r23_margin_to_k4",
    "maximum_support_k1", "maximum_support_to_k2", "maximum_support_to_k4",
    "candidate_fraction_to_k2", "candidate_fraction_to_k4", "maximum_cfl_ratio_to_k4",
    "maximum_energy_ratio_to_k4", "maximum_patch_ratio_to_k4",
    "minimum_separation_over_h_to_k4", "cumulative_relative_h1_increment_to_k4",
)
TARGET_TAIL_PREVIEW_DIM = len(TARGET_TAIL_PREVIEW_FEATURE_ORDER)


def target_tail_preview_features(records: tuple[dict, ...] | list[dict]) -> np.ndarray:
    """Summarise a four-step causal target-policy tail into 16 finite channels."""
    if len(records) != 4:
        raise ValueError("R2.4.3.1 preview requires exactly four target-tail records")
    required = (
        "r23_k1_risk", "threshold", "support_ratio", "action", "cfl_ratio",
        "energy_ratio", "patch_ratio", "separation_over_h", "relative_h1_increment",
    )
    if any(any(key not in row for key in required) for row in records):
        raise ValueError("preview records are incomplete")
    risk = np.asarray([row["r23_k1_risk"] for row in records], dtype=np.float64)
    threshold = np.asarray([row["threshold"] for row in records], dtype=np.float64)
    support = np.asarray([row["support_ratio"] for row in records], dtype=np.float64)
    action = np.asarray([row["action"] for row in records], dtype=np.float64)
    cfl = np.asarray([row["cfl_ratio"] for row in records], dtype=np.float64)
    energy = np.asarray([row["energy_ratio"] for row in records], dtype=np.float64)
    patch = np.asarray([row["patch_ratio"] for row in records], dtype=np.float64)
    separation = np.asarray([row["separation_over_h"] for row in records], dtype=np.float64)
    increment = np.asarray([row["relative_h1_increment"] for row in records], dtype=np.float64)
    result = np.asarray((
        risk[0], risk[:2].max(), risk.max(),
        (threshold - risk)[:1].min(), (threshold - risk)[:2].min(), (threshold - risk).min(),
        support[:1].max(), support[:2].max(), support.max(),
        action[:2].mean(), action.mean(), cfl.max(), energy.max(), patch.max(),
        separation.min(), increment.sum(),
    ), dtype=np.float64)
    if result.shape != (TARGET_TAIL_PREVIEW_DIM,) or not np.all(np.isfinite(result)):
        raise ValueError("preview features must be finite 16-dimensional values")
    return result
