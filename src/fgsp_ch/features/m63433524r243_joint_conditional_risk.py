"""Joint conditional risk features for R2.4.3.

R2.4.2 produced a 76-dim conditional local risk score that augments the
frozen R2.3 global reachability certificate.  R2.4.3 fuses the two scores
into a single conservative joint upper bound via max-pooling, then appends
two scalar meta-features that capture the disagreement between the global
and local risk heads:

    z_joint = [
        base_r23_score,          # scalar, frozen R2.3 reachability upper bound
        cond_local_upper,        # scalar, R2.4.2 conditional local upper bound
        disagreement_abs,        # |base_r23 - cond_local_mean|
        disagreement_signed,     # cond_local_mean - base_r23
    ]  ∈ R^4

The joint unsafe score is:

    s_joint = max(base_r23_score, cond_local_upper)

This is the public interface consumed by the deployment gate.  No future
state, reference label, or behaviour-policy identifier may appear in
z_joint.
"""

from __future__ import annotations

import numpy as np

# ── Dimension constants ───────────────────────────────────────────────────────

JOINT_META_FEATURE_ORDER = (
    "base_r23_score",
    "cond_local_upper",
    "disagreement_abs",
    "disagreement_signed",
)

JOINT_META_FEATURE_DIM = len(JOINT_META_FEATURE_ORDER)  # 4


# ── Public API ────────────────────────────────────────────────────────────────

def joint_conditional_risk_features(
    *,
    base_r23_score: float,
    cond_local_mean: float,
    cond_local_upper: float,
) -> np.ndarray:
    """Return the 4-dim joint meta-feature vector z_joint.

    Parameters
    ----------
    base_r23_score:
        Scalar reachability upper bound from the frozen R2.3 certificate,
        clipped to [0, 1].
    cond_local_mean:
        Ensemble mean unsafe probability from the R2.4.2 conditional local
        risk head, clipped to [0, 1].
    cond_local_upper:
        Ensemble upper bound (mean + k * std) from R2.4.2, clipped to [0, 1].
    """
    b = float(base_r23_score)
    m = float(cond_local_mean)
    u = float(cond_local_upper)

    for name, val in (
        ("base_r23_score", b),
        ("cond_local_mean", m),
        ("cond_local_upper", u),
    ):
        if not np.isfinite(val):
            raise ValueError(f"{name} must be finite, got {val}")
        if val < 0.0 or val > 1.0:
            raise ValueError(f"{name} must lie in [0, 1], got {val}")

    if u < m - 1e-9:
        raise ValueError(
            f"cond_local_upper ({u:.6f}) must be >= cond_local_mean ({m:.6f})"
        )

    out = np.asarray(
        (b, u, abs(b - m), m - b),
        dtype=np.float64,
    )
    assert out.shape == (JOINT_META_FEATURE_DIM,)
    assert np.all(np.isfinite(out))
    return out


def joint_unsafe_score(
    *,
    base_r23_score: float,
    cond_local_upper: float,
) -> float:
    """Conservative joint score: max of the two upper bounds.

    This is guaranteed to be >= each individual score, so it cannot
    weaken either safety certificate in isolation.
    """
    b = float(base_r23_score)
    u = float(cond_local_upper)
    for name, val in (("base_r23_score", b), ("cond_local_upper", u)):
        if not np.isfinite(val):
            raise ValueError(f"{name} must be finite")
        if val < 0.0 or val > 1.0:
            raise ValueError(f"{name} must lie in [0, 1]")
    return float(max(b, u))
