from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import beta


@dataclass(frozen=True)
class FixedSequenceResult:
    threshold: float | None
    records: tuple[dict, ...]


def clopper_pearson_upper(failures: int, trials: int, delta: float) -> float:
    """One-sided exact binomial upper confidence bound."""
    if trials < 1 or not 0.0 < delta < 1.0:
        raise ValueError("trials must be positive and delta must lie in (0, 1)")
    if not 0 <= failures <= trials:
        raise ValueError("failures must lie between zero and trials")
    if failures == trials:
        return 1.0
    return float(beta.ppf(1.0 - delta, failures + 1, trials - failures))


def fixed_sequence_group_ltt(
    scores: np.ndarray,
    support_ratio: np.ndarray,
    unsafe: np.ndarray,
    groups: np.ndarray,
    candidates: list[float] | tuple[float, ...],
    *,
    alpha: float,
    delta: float,
) -> FixedSequenceResult:
    """Select the widest nonempty gate before the first rejected hypothesis.

    A trajectory group is a Bernoulli failure when any accepted state in that
    group is unsafe. Empty gates are valid risk hypotheses but are not useful
    deployment thresholds, so the fixed sequence continues through them.
    """
    scores = np.asarray(scores, dtype=float)
    support_ratio = np.asarray(support_ratio, dtype=float)
    unsafe = np.asarray(unsafe, dtype=bool)
    groups = np.asarray(groups)
    if not (scores.shape == support_ratio.shape == unsafe.shape == groups.shape):
        raise ValueError("scores, support, unsafe and groups must have equal shape")
    ordered = tuple(map(float, candidates))
    if any(right <= left for left, right in zip(ordered, ordered[1:])):
        raise ValueError("candidate thresholds must be strictly increasing")
    unique_groups = np.unique(groups)
    chosen = None
    records: list[dict] = []
    for threshold in ordered:
        accepted = (scores <= threshold) & (support_ratio <= 1.0)
        failures = sum(
            bool(np.any(accepted[groups == group] & unsafe[groups == group]))
            for group in unique_groups
        )
        risk_upper = clopper_pearson_upper(
            int(failures), len(unique_groups), delta,
        )
        records.append({
            "threshold": threshold,
            "failures": int(failures),
            "trials": int(len(unique_groups)),
            "risk_upper": risk_upper,
            "accepted": int(np.sum(accepted)),
        })
        if risk_upper > alpha:
            break
        if np.any(accepted):
            chosen = threshold
    return FixedSequenceResult(chosen, tuple(records))


def maximum_certifiable_failures(trials: int, alpha: float, delta: float) -> int:
    """Return the largest failure count whose exact upper bound is <= alpha."""
    certified = [
        failures for failures in range(trials + 1)
        if clopper_pearson_upper(failures, trials, delta) <= alpha
    ]
    return max(certified, default=-1)
