"""Training/calibration helpers for the M2.4 causal representation router."""

from __future__ import annotations

import numpy as np
import torch
from torch.nn import functional as F

from fgsp_ch.models.cmame_m24_causal_router import causal_horizon_loss


def grouped_action_ranking_loss(
    predicted_log_error: torch.Tensor, target_log_error: torch.Tensor,
    group: torch.Tensor, *, margin: float = 0.05,
) -> torch.Tensor:
    """Preserve the terminal-error ordering of candidate actions per state."""
    terms: list[torch.Tensor] = []
    score = predicted_log_error[:, -1]
    truth = target_log_error[:, -1]
    for value in torch.unique(group):
        index = torch.nonzero(group == value, as_tuple=False)[:, 0]
        if index.numel() < 2:
            continue
        for i in range(index.numel()):
            for j in range(i + 1, index.numel()):
                left, right = index[i], index[j]
                sign = torch.sign(truth[right] - truth[left])
                if sign != 0:
                    terms.append(F.relu(float(margin) - sign * (score[right] - score[left])))
    if not terms:
        return score.sum() * 0.0
    return torch.stack(terms).mean()


def fit_upper_margins(
    prediction_members: np.ndarray, targets: np.ndarray, *, quantile: float,
) -> np.ndarray:
    """Split-conformal horizon margins using calibration groups only."""
    values = np.asarray(prediction_members, dtype=np.float64)
    truth = np.asarray(targets, dtype=np.float64)
    if values.ndim != 3 or truth.shape != values.shape[1:]:
        raise ValueError("members must be [ensemble, rows, horizons]")
    if not 0.5 <= quantile < 1.0:
        raise ValueError("calibration quantile must lie in [0.5,1)")
    mean = values.mean(axis=0)
    std = values.std(axis=0)
    residual = truth - mean - std
    margins = np.quantile(residual, quantile, axis=0, method="higher")
    # A later horizon may never receive a less conservative margin.
    return np.maximum.accumulate(np.maximum(margins, 0.0))


def composite_training_loss(
    output, target_error: torch.Tensor, target_cost: torch.Tensor,
    target_budget: torch.Tensor, groups: torch.Tensor, *, epsilon: float,
    ranking_weight: float, cost_weight: float, budget_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    causal, weights = causal_horizon_loss(output.log_error_ratio, target_error, epsilon=epsilon)
    ranking = grouped_action_ranking_loss(output.log_error_ratio, target_error, groups)
    cost = F.smooth_l1_loss(output.log_cost, target_cost)
    budget = F.binary_cross_entropy(output.budget_fraction, target_budget)
    total = causal + ranking_weight * ranking + cost_weight * cost + budget_weight * budget
    metrics = {
        "causal": float(causal.detach()), "ranking": float(ranking.detach()),
        "cost": float(cost.detach()), "budget": float(budget.detach()),
        "minimum_causal_weight": float(weights.min().detach()),
    }
    return total, metrics

