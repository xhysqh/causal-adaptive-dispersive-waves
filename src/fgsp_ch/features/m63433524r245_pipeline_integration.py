"""End-to-end R2.4 pipeline integration for deployment certification.

R2.4.5 wires together the full R2.4.x stack into a single callable that
takes raw solver outputs and returns a deployment decision:

    Raw solver outputs
        → R2.4.1  local risk features          (10-dim z_local)
        → R2.4.2  conditional local risk        (76-dim z_cond)
        → R2.4.3  joint conditional risk score  (4-dim z_joint → s_calibrated)
        → R2.4.4  deployment gate decision      (6-dim z_gate, GateDecision)

The integration layer is **read-only with respect to the frozen R2.3
certificate**: it accepts the precomputed base_r23_score as an input,
never re-trains or re-evaluates the R2.3 model.

Design contracts preserved from sub-milestones
-----------------------------------------------
* Conservative joint score: s_joint = max(base_r23_score, cond_local_upper).
* Zero unsafe acceptances at any selected threshold (hard constraint).
* No future state, reference label, or behaviour-policy identifier in inputs.
* All feature vectors are finite and lie in [0, 1] (scalar scores) or R^n.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from fgsp_ch.features.m63433524r244_deployment_gate import (
    GateDecision,
    evaluate_gate,
    batch_gate_decisions,
    GATE_FEATURE_DIM,
)
from fgsp_ch.features.m63433524r243_joint_conditional_risk import (
    joint_unsafe_score,
    joint_conditional_risk_features,
    JOINT_META_FEATURE_DIM,
)


# ── Pipeline result ───────────────────────────────────────────────────────────

@dataclass(frozen=True)
class PipelineResult:
    """Full audit record for one R2.4 pipeline evaluation."""

    gate_decision: GateDecision
    """Final accept/reject decision from the deployment gate."""

    s_joint: float
    """Conservative joint unsafe score = max(base_r23, cond_local_upper)."""

    calibrated_prob: float
    """Calibrated posterior unsafe probability from R2.4.3 (or proxy)."""

    joint_meta_features: np.ndarray
    """4-dimensional z_joint meta-feature vector (R2.4.3 input)."""

    gate_features: np.ndarray
    """6-dimensional z_gate audit vector (R2.4.4 output)."""


# ── Single-sample integration ─────────────────────────────────────────────────

def run_pipeline(
    *,
    base_r23_score: float,
    cond_local_mean: float,
    cond_local_upper: float,
    calibrated_prob: float,
    threshold: float,
) -> PipelineResult:
    """Evaluate the full R2.4 certification pipeline for one candidate action.

    Parameters
    ----------
    base_r23_score:
        Frozen R2.3 global reachability certificate score ∈ [0, 1].
    cond_local_mean:
        R2.4.2 conditional local risk posterior mean ∈ [0, 1].
    cond_local_upper:
        R2.4.2 conditional local risk upper bound ∈ [0, 1]; must be ≥ mean.
    calibrated_prob:
        R2.4.3 calibrated posterior unsafe probability ∈ [0, 1].
    threshold:
        Gate threshold ∈ [0, 1].  Action accepted iff s_joint ≤ threshold.

    Returns
    -------
    PipelineResult
        Immutable record with gate decision and full audit trail.
    """
    # Validate inputs eagerly so error messages point here, not deep in sub-modules.
    for name, val in (
        ("base_r23_score", base_r23_score),
        ("cond_local_mean", cond_local_mean),
        ("cond_local_upper", cond_local_upper),
        ("calibrated_prob", calibrated_prob),
        ("threshold", threshold),
    ):
        v = float(val)
        if not np.isfinite(v):
            raise ValueError(f"{name} must be finite, got {v}")
        if v < 0.0 or v > 1.0:
            raise ValueError(f"{name} must lie in [0, 1], got {v}")
    if float(cond_local_upper) < float(cond_local_mean):
        raise ValueError(
            f"cond_local_upper ({cond_local_upper}) must be >= "
            f"cond_local_mean ({cond_local_mean})"
        )

    # R2.4.3: compute joint meta-features and joint score
    z_joint = joint_conditional_risk_features(
        base_r23_score=base_r23_score,
        cond_local_mean=cond_local_mean,
        cond_local_upper=cond_local_upper,
    )
    s_joint = joint_unsafe_score(
        base_r23_score=base_r23_score,
        cond_local_upper=cond_local_upper,
    )

    # R2.4.4: deployment gate
    decision = evaluate_gate(
        s_joint=s_joint,
        base_r23_score=base_r23_score,
        cond_local_upper=cond_local_upper,
        calibrated_prob=calibrated_prob,
        threshold=threshold,
    )

    return PipelineResult(
        gate_decision=decision,
        s_joint=s_joint,
        calibrated_prob=float(calibrated_prob),
        joint_meta_features=z_joint,
        gate_features=decision.gate_features,
    )


# ── Batch integration ─────────────────────────────────────────────────────────

@dataclass(frozen=True)
class BatchPipelineResult:
    """Vectorised R2.4 pipeline output for N candidate actions."""

    accepted: np.ndarray
    """Boolean acceptance flags, shape ``(N,)``."""

    s_joint: np.ndarray
    """Conservative joint unsafe scores, shape ``(N,)``."""

    gate_features: np.ndarray
    """Gate audit feature matrix, shape ``(N, 6)``."""

    joint_meta_features: np.ndarray
    """Joint meta-feature matrix, shape ``(N, 4)``."""


def run_pipeline_batch(
    *,
    base_r23_score: np.ndarray,
    cond_local_mean: np.ndarray,
    cond_local_upper: np.ndarray,
    calibrated_prob: np.ndarray,
    threshold: float,
) -> BatchPipelineResult:
    """Vectorised R2.4 pipeline over N candidate actions.

    Parameters
    ----------
    base_r23_score, cond_local_mean, cond_local_upper, calibrated_prob:
        Shape ``(N,)`` arrays, each ∈ [0, 1].
    threshold:
        Scalar gate threshold ∈ [0, 1].

    Returns
    -------
    BatchPipelineResult
    """
    br = np.asarray(base_r23_score, dtype=np.float64)
    cm = np.asarray(cond_local_mean, dtype=np.float64)
    cu = np.asarray(cond_local_upper, dtype=np.float64)
    cp = np.asarray(calibrated_prob, dtype=np.float64)
    thr = float(threshold)

    if br.ndim != 1:
        raise ValueError("base_r23_score must be 1-D")
    n = len(br)
    for name, arr in (
        ("cond_local_mean", cm),
        ("cond_local_upper", cu),
        ("calibrated_prob", cp),
    ):
        if arr.shape != (n,):
            raise ValueError(f"{name} must have shape ({n},), got {arr.shape}")

    for name, arr in (
        ("base_r23_score", br),
        ("cond_local_mean", cm),
        ("cond_local_upper", cu),
        ("calibrated_prob", cp),
    ):
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{name} contains non-finite values")
        if np.any(arr < 0.0) or np.any(arr > 1.0):
            raise ValueError(f"{name} values must lie in [0, 1]")

    if np.any(cu < cm):
        raise ValueError("cond_local_upper must be >= cond_local_mean element-wise")

    if not np.isfinite(thr) or thr < 0.0 or thr > 1.0:
        raise ValueError(f"threshold must be in [0, 1], got {thr}")

    # R2.4.3: joint score (conservative max-pool)
    sj = np.maximum(br, cu)

    # R2.4.3: joint meta-features for all N samples
    disagreement_abs = np.abs(br - cm)
    disagreement_signed = cm - br
    joint_meta = np.stack([br, cu, disagreement_abs, disagreement_signed], axis=1)
    assert joint_meta.shape == (n, JOINT_META_FEATURE_DIM)

    # R2.4.4: batch gate
    accepted, gate_feats = batch_gate_decisions(
        s_joint=sj,
        base_r23_score=br,
        cond_local_upper=cu,
        calibrated_prob=cp,
        threshold=thr,
    )

    return BatchPipelineResult(
        accepted=accepted,
        s_joint=sj,
        gate_features=gate_feats,
        joint_meta_features=joint_meta,
    )


# ── Pipeline summary statistics ───────────────────────────────────────────────

def pipeline_summary(result: BatchPipelineResult) -> dict:
    """Compute scalar summary statistics from a batch pipeline result.

    Returns
    -------
    dict with keys:
        n_total, n_accepted, n_rejected, acceptance_rate,
        s_joint_mean, s_joint_max, s_joint_min,
        all_features_finite.
    """
    n = len(result.accepted)
    n_acc = int(result.accepted.sum())
    return {
        "n_total": n,
        "n_accepted": n_acc,
        "n_rejected": n - n_acc,
        "acceptance_rate": float(n_acc / n) if n > 0 else float("nan"),
        "s_joint_mean": float(result.s_joint.mean()) if n > 0 else float("nan"),
        "s_joint_max": float(result.s_joint.max()) if n > 0 else float("nan"),
        "s_joint_min": float(result.s_joint.min()) if n > 0 else float("nan"),
        "all_features_finite": bool(np.all(np.isfinite(result.gate_features))),
    }
