"""Resolution-aware paired statistics for the frozen CMAME-M2.5 study."""

from __future__ import annotations

from collections import defaultdict
from math import erf, sqrt
from typing import Iterable

import numpy as np


def resolved_log10_ratio(proposed: Iterable[float], baseline: Iterable[float], floors: Iterable[float]) -> np.ndarray:
    p = np.asarray(list(proposed), dtype=float)
    b = np.asarray(list(baseline), dtype=float)
    f = np.asarray(list(floors), dtype=float)
    if p.shape != b.shape or p.shape != f.shape or p.ndim != 1 or p.size == 0:
        raise ValueError("paired errors and floors must be aligned nonempty vectors")
    if np.any(p < 0.0) or np.any(b < 0.0) or np.any(f <= 0.0) or not np.isfinite(p + b + f).all():
        raise ValueError("errors/floors must be finite and nonnegative/positive")
    return np.log10(np.maximum(p, f) / np.maximum(b, f))


def grouped_bootstrap_median(values: Iterable[float], groups: Iterable[str], *, samples: int = 2000,
                             confidence: float = 0.95, seed: int = 20260828) -> tuple[float, float]:
    x = np.asarray(list(values), dtype=float); g = np.asarray(list(groups), dtype=str)
    if x.shape != g.shape or x.ndim != 1 or x.size == 0 or samples < 100:
        raise ValueError("invalid grouped bootstrap input")
    unique = np.unique(g); rng = np.random.default_rng(seed); draws = np.empty(samples)
    for index in range(samples):
        selected = rng.choice(unique, len(unique), replace=True)
        draws[index] = np.median(np.concatenate([x[g == item] for item in selected]))
    tail = (1.0 - confidence) / 2.0
    return tuple(map(float, np.quantile(draws, [tail, 1.0 - tail])))


def sign_test_pvalue(values: Iterable[float], tie: float = 0.02) -> float:
    x = np.asarray(list(values), dtype=float)
    wins = int(np.count_nonzero(x < -tie)); losses = int(np.count_nonzero(x > tie)); n = wins + losses
    if n == 0: return 1.0
    z = (abs(wins - n / 2.0) - 0.5) / sqrt(max(n / 4.0, np.finfo(float).eps))
    return float(min(1.0, 2.0 * (1.0 - 0.5 * (1.0 + erf(z / sqrt(2.0))))))


def holm_adjust(pvalues: dict[str, float]) -> dict[str, float]:
    ordered = sorted(pvalues, key=pvalues.get); total = len(ordered); adjusted = {}; running = 0.0
    for rank, name in enumerate(ordered):
        running = max(running, (total - rank) * float(pvalues[name]))
        adjusted[name] = min(1.0, running)
    return adjusted


def paired_family_summary(rows: list[dict], *, samples: int, confidence: float, tie: float) -> dict:
    buckets: dict[str, list[dict]] = defaultdict(list)
    for row in rows: buckets[str(row["family"])].append(row)
    result = {}
    for family, subset in sorted(buckets.items()):
        effects = np.asarray([float(row["resolved_log10_ratio"]) for row in subset])
        groups = [str(row["group_id"]) for row in subset]
        result[family] = {
            "pairs": len(subset), "trajectory_groups": len(set(groups)),
            "median_log10_ratio": float(np.median(effects)),
            "median_resolved_ratio": float(10.0 ** np.median(effects)),
            "group_bootstrap_log10_95": list(grouped_bootstrap_median(effects, groups, samples=samples, confidence=confidence)),
            "win_fraction": float(np.mean(effects < -tie)), "tie_fraction": float(np.mean(np.abs(effects) <= tie)),
            "loss_fraction": float(np.mean(effects > tie)), "sign_test_pvalue": sign_test_pvalue(effects, tie),
        }
    return result
