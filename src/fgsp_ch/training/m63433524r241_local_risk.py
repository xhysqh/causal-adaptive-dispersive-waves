"""Training and conservative selection helpers for the R2.4.1 local head."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from fgsp_ch.models.m63433524r241_local_risk import TargetCellLocalRiskHead


def binary_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Tie-aware AUC without a sklearn dependency."""
    score = np.asarray(scores, dtype=float); label = np.asarray(labels, dtype=bool)
    positives, negatives = int(label.sum()), int((~label).sum())
    if positives == 0 or negatives == 0:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score), dtype=float)
    ranks[order] = np.arange(1, len(score) + 1, dtype=float)
    sorted_score = score[order]
    start = 0
    while start < len(score):
        end = start + 1
        while end < len(score) and sorted_score[end] == sorted_score[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    return float((ranks[label].sum() - positives * (positives + 1) / 2.0) / (positives * negatives))


@dataclass(frozen=True)
class LocalRiskEnsemble:
    mean: np.ndarray
    upper: np.ndarray
    member_scores: np.ndarray


def train_member(
    features: np.ndarray, labels: np.ndarray, *, seed: int, hidden_channels: int,
    epochs: int, learning_rate: float, device: torch.device,
) -> tuple[TargetCellLocalRiskHead, np.ndarray, np.ndarray]:
    """Fit one deterministic, class-balanced binary local-risk member."""
    x = np.asarray(features, dtype=np.float32); y = np.asarray(labels, dtype=np.float32)
    if x.ndim != 2 or y.shape != (len(x),):
        raise ValueError("invalid local-risk training arrays")
    mean, scale = x.mean(axis=0), x.std(axis=0)
    scale = np.maximum(scale, 1.0e-6)
    torch.manual_seed(int(seed))
    model = TargetCellLocalRiskHead(x.shape[1], hidden_channels).to(device)
    tensor_x = torch.as_tensor((x - mean) / scale, dtype=torch.float32, device=device)
    tensor_y = torch.as_tensor(y, dtype=torch.float32, device=device)
    positive = max(float(tensor_y.sum().item()), 1.0)
    negative = max(float(len(tensor_y) - positive), 1.0)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(negative / positive, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(learning_rate), weight_decay=1.0e-4)
    model.train()
    for _ in range(int(epochs)):
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model(tensor_x), tensor_y)
        loss.backward(); optimizer.step()
    return model.eval(), mean.astype(np.float64), scale.astype(np.float64)


@torch.no_grad()
def ensemble_score(
    members: list[tuple[TargetCellLocalRiskHead, np.ndarray, np.ndarray]], features: np.ndarray,
    *, upper_std_multiplier: float, device: torch.device,
) -> LocalRiskEnsemble:
    x = np.asarray(features, dtype=np.float32)
    scores = []
    for model, mean, scale in members:
        value = (x - mean) / scale
        logits = model(torch.as_tensor(value, dtype=torch.float32, device=device))
        scores.append(torch.sigmoid(logits).detach().cpu().numpy())
    member_scores = np.stack(scores, axis=0)
    mean = member_scores.mean(axis=0)
    upper = np.clip(mean + float(upper_std_multiplier) * member_scores.std(axis=0, ddof=0), 0.0, 1.0)
    return LocalRiskEnsemble(mean, upper, member_scores)


def largest_zero_unsafe_threshold(
    score: np.ndarray, unsafe: np.ndarray, eligible: np.ndarray, groups: np.ndarray,
    *, minimum_groups: int,
) -> dict | None:
    """Select a development-only threshold without weakening zero-error safety."""
    score = np.asarray(score, dtype=float); unsafe = np.asarray(unsafe, dtype=bool)
    eligible = np.asarray(eligible, dtype=bool); groups = np.asarray(groups)
    chosen = None
    for threshold in np.unique(np.sort(score[eligible])):
        accepted = eligible & (score <= threshold)
        failures = int(np.sum(accepted & unsafe))
        group_count = int(sum(np.any(accepted[groups == group]) for group in np.unique(groups)))
        record = {"threshold": float(threshold), "accepted": int(accepted.sum()),
                  "accepted_groups": group_count, "unsafe_acceptances": failures}
        if failures == 0 and group_count >= int(minimum_groups):
            if chosen is None or (record["accepted_groups"], record["accepted"]) > (chosen["accepted_groups"], chosen["accepted"]):
                chosen = record
    return chosen
