"""Pure metrics and protocol helpers for the R2.4.1-A read-only audit."""

from __future__ import annotations

import numpy as np


def binary_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Tie-aware ROC-AUC; positive means candidate-unsafe."""
    score = np.asarray(scores, dtype=float)
    label = np.asarray(labels, dtype=bool)
    positives, negatives = int(label.sum()), int((~label).sum())
    if positives == 0 or negatives == 0:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score), dtype=float)
    ranks[order] = np.arange(1, len(score) + 1, dtype=float)
    ordered = score[order]
    begin = 0
    while begin < len(score):
        end = begin + 1
        while end < len(score) and ordered[end] == ordered[begin]:
            end += 1
        ranks[order[begin:end]] = (begin + 1 + end) / 2.0
        begin = end
    return float((ranks[label].sum() - positives * (positives + 1) / 2.0) / (positives * negatives))


def average_precision(scores: np.ndarray, labels: np.ndarray) -> float:
    """Average precision for unsafe candidates, without an sklearn dependency."""
    score = np.asarray(scores, dtype=float)
    label = np.asarray(labels, dtype=bool)
    positives = int(label.sum())
    if positives == 0:
        return float("nan")
    order = np.argsort(-score, kind="mergesort")
    ordered = label[order]
    cumulative = np.cumsum(ordered, dtype=float)
    precision = cumulative / np.arange(1, len(ordered) + 1, dtype=float)
    return float(precision[ordered].sum() / positives)


def ablation_columns() -> dict[str, np.ndarray | None]:
    """Feature-column registry for the four diagnostic learned heads.

    The R2.4.1 archive stores 60 frozen R2.3 channels and ten local channels.
    ``None`` denotes the frozen scalar R2.3 score, so it entails no training.
    """
    r23 = np.arange(60, dtype=int)
    local = np.arange(60, 70, dtype=int)
    return {
        "frozen_r23_score": None,
        "r23_60_head": r23,
        "local_10_head": local,
        "fusion_70_head": np.arange(70, dtype=int),
        "fusion_without_curvature": np.r_[r23, np.array([60, 61, 64, 65, 66, 67, 68, 69])],
        "fusion_without_amr": np.r_[r23, np.arange(60, 68)],
    }


def group_count(values: np.ndarray, mask: np.ndarray) -> int:
    return int(len(set(np.asarray(values)[np.asarray(mask, dtype=bool)].tolist())))
