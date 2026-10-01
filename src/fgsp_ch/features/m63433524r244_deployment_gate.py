"""Deployment gate features for R2.4.4.

R2.4.3 produced a calibrated joint unsafe score combining the frozen R2.3
global reachability certificate with the R2.4.2 conditional local risk upper
bound.  R2.4.4 packages those scores into the final binary accept/reject
decision exposed to the deployment loop.

The gate applies a hard threshold on the joint unsafe score:

    accept  ←→  s_joint ≤ threshold
    reject  ←→  s_joint >  threshold

Additionally a 6-dimensional gate feature vector is produced for downstream
audit logging and offline analysis:

    z_gate = [
        s_joint,                  # conservative max-pooled score
        base_r23_score,           # frozen R2.3 certificate contribution
        cond_local_upper,         # R2.4.2 conditional upper bound
        calibrated_prob,          # R2.4.3 calibrated posterior probability
        accept_margin,            # threshold - s_joint  (negative → reject)
        is_accepted,              # 1.0 if accepted, 0.0 if rejected
    ]  ∈ R^6

No future state, reference label, or behaviour-policy identifier may appear.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# ── Dimension constants ───────────────────────────────────────────────────────

GATE_FEATURE_ORDER = (
    "s_joint",
    "base_r23_score",
    "cond_local_upper",
    "calibrated_prob",
    "accept_margin",
    "is_accepted",
)

GATE_FEATURE_DIM = len(GATE_FEATURE_ORDER)  # 6


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass(frozen=True)
class GateDecision:
    """Immutable result of a single deployment gate evaluation."""

    accepted: bool
    """True if the action was accepted (s_joint ≤ threshold)."""

    s_joint: float
    """Conservative joint unsafe score ∈ [0, 1]."""

    threshold: float
    """Gate threshold ∈ [0, 1]."""

    gate_features: np.ndarray
    """6-dimensional audit feature vector."""


# ── Public API ────────────────────────────────────────────────────────────────

def deployment_gate_features(
    *,
    s_joint: float,
    base_r23_score: float,
    cond_local_upper: float,
    calibrated_prob: float,
    threshold: float,
) -> np.ndarray:
    """Return the 6-dimensional gate audit feature vector z_gate.

    Parameters
    ----------
    s_joint:
        Conservative joint unsafe score = max(base_r23_score, cond_local_upper),
        clipped to [0, 1].
    base_r23_score:
        Frozen R2.3 reachability upper bound ∈ [0, 1].
    cond_local_upper:
        R2.4.2 conditional local upper bound ∈ [0, 1].
    calibrated_prob:
        R2.4.3 calibrated posterior probability ∈ [0, 1].
    threshold:
        Gate threshold ∈ [0, 1].  Action is accepted iff s_joint ≤ threshold.
    """
    for name, val in (
        ("s_joint", s_joint),
        ("base_r23_score", base_r23_score),
        ("cond_local_upper", cond_local_upper),
        ("calibrated_prob", calibrated_prob),
        ("threshold", threshold),
    ):
        val = float(val)
        if not np.isfinite(val):
            raise ValueError(f"{name} must be finite, got {val}")
        if val < 0.0 or val > 1.0:
            raise ValueError(f"{name} must lie in [0, 1], got {val}")

    s = float(s_joint)
    b = float(base_r23_score)
    u = float(cond_local_upper)
    c = float(calibrated_prob)
    thr = float(threshold)

    accepted = s <= thr
    margin = thr - s
    out = np.asarray(
        (s, b, u, c, margin, 1.0 if accepted else 0.0),
        dtype=np.float64,
    )
    assert out.shape == (GATE_FEATURE_DIM,)
    assert np.all(np.isfinite(out))
    return out


def evaluate_gate(
    *,
    s_joint: float,
    base_r23_score: float,
    cond_local_upper: float,
    calibrated_prob: float,
    threshold: float,
) -> GateDecision:
    """Evaluate the deployment gate and return a full GateDecision.

    The gate is **conservative**: it never accepts an action whose joint score
    exceeds the threshold, regardless of the calibrated probability.

    Parameters
    ----------
    s_joint, base_r23_score, cond_local_upper, calibrated_prob, threshold:
        Same semantics as :func:`deployment_gate_features`.

    Returns
    -------
    GateDecision
        Immutable decision record including the audit feature vector.
    """
    feats = deployment_gate_features(
        s_joint=s_joint,
        base_r23_score=base_r23_score,
        cond_local_upper=cond_local_upper,
        calibrated_prob=calibrated_prob,
        threshold=threshold,
    )
    s = float(s_joint)
    thr = float(threshold)
    return GateDecision(
        accepted=bool(s <= thr),
        s_joint=s,
        threshold=thr,
        gate_features=feats,
    )


def batch_gate_decisions(
    *,
    s_joint: np.ndarray,
    base_r23_score: np.ndarray,
    cond_local_upper: np.ndarray,
    calibrated_prob: np.ndarray,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised gate evaluation over N candidate actions.

    Parameters
    ----------
    s_joint, base_r23_score, cond_local_upper, calibrated_prob:
        Shape ``(N,)`` arrays, each ∈ [0, 1].
    threshold:
        Scalar gate threshold ∈ [0, 1].

    Returns
    -------
    accepted : np.ndarray of bool, shape ``(N,)``
    gate_features : np.ndarray of float64, shape ``(N, 6)``
    """
    sj = np.asarray(s_joint, dtype=np.float64)
    br = np.asarray(base_r23_score, dtype=np.float64)
    cu = np.asarray(cond_local_upper, dtype=np.float64)
    cp = np.asarray(calibrated_prob, dtype=np.float64)
    thr = float(threshold)

    if sj.ndim != 1:
        raise ValueError("s_joint must be 1-D")
    n = len(sj)
    for name, arr in (
        ("base_r23_score", br), ("cond_local_upper", cu), ("calibrated_prob", cp)
    ):
        if arr.shape != (n,):
            raise ValueError(f"{name} must have shape ({n},), got {arr.shape}")
    for name, arr in (
        ("s_joint", sj), ("base_r23_score", br),
        ("cond_local_upper", cu), ("calibrated_prob", cp),
    ):
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{name} contains non-finite values")
        if np.any(arr < 0.0) or np.any(arr > 1.0):
            raise ValueError(f"{name} values must lie in [0, 1]")
    if not np.isfinite(thr) or thr < 0.0 or thr > 1.0:
        raise ValueError(f"threshold must be in [0, 1], got {thr}")

    accepted = sj <= thr
    margin = thr - sj
    is_acc = accepted.astype(np.float64)
    gate_features = np.stack([sj, br, cu, cp, margin, is_acc], axis=1)
    return accepted, gate_features
