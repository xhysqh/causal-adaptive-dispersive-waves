from __future__ import annotations

import numpy as np
import torch
from torch.nn import functional as F

from fgsp_ch.models.m634334_bellman_risk import BellmanRiskOperator


def bellman_targets(step_risks: torch.Tensor) -> torch.Tensor:
    if step_risks.ndim != 2 or step_risks.shape[1] != 8:
        raise ValueError("step_risks must contain eight future risks.")
    return torch.stack([
        step_risks[:, :horizon].amax(dim=1) for horizon in (1, 2, 4, 8)
    ], dim=1)


def bellman_losses(
    model: BellmanRiskOperator,
    features: torch.Tensor,
    step_risks: torch.Tensor,
    sample_weight: torch.Tensor,
    pair_safe: torch.Tensor,
    pair_unsafe: torch.Tensor,
    pair_horizon: torch.Tensor,
):
    prediction = model(features).log_value
    target_raw = bellman_targets(step_risks)
    target = torch.log1p(target_raw)
    weight = sample_weight[:, None]
    supervised = torch.sum(
        weight * F.smooth_l1_loss(prediction, target, reduction="none")
    ) / (weight.sum() * target.shape[1]).clamp_min(1.0)
    predicted_raw = torch.expm1(prediction)
    segments = torch.stack([
        step_risks[:, :1].amax(dim=1),
        step_risks[:, 1:2].amax(dim=1),
        step_risks[:, 2:4].amax(dim=1),
        step_risks[:, 4:8].amax(dim=1),
    ], dim=1)
    backup = torch.stack((
        segments[:, 0],
        torch.maximum(predicted_raw[:, 0], segments[:, 1]),
        torch.maximum(predicted_raw[:, 1], segments[:, 2]),
        torch.maximum(predicted_raw[:, 2], segments[:, 3]),
    ), dim=1)
    bellman = F.smooth_l1_loss(predicted_raw, backup)
    unsafe = (target_raw > 1.0).float()
    false_safe = (
        unsafe * F.relu(target - prediction).square()
    ).sum() / unsafe.sum().clamp_min(1.0)
    if pair_safe.numel():
        column = pair_horizon + 1
        safe_value = prediction[pair_safe, column]
        unsafe_value = prediction[pair_unsafe, column]
        ranking = F.relu(0.05 + safe_value - unsafe_value).mean()
    else:
        ranking = prediction.new_zeros(())
    total = supervised + 0.3 * bellman + 0.8 * false_safe + 0.5 * ranking
    return total, (supervised, bellman, false_safe, ranking)


def train_bellman_operator(
    model: BellmanRiskOperator,
    features: torch.Tensor,
    step_risks: torch.Tensor,
    sample_weight: torch.Tensor,
    pair_safe: torch.Tensor,
    pair_unsafe: torch.Tensor,
    pair_horizon: torch.Tensor,
    *,
    epochs: int,
    learning_rate: float,
) -> dict[str, float]:
    model.set_normalization(features)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=1.0e-5,
    )
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        loss, _ = bellman_losses(
            model, features, step_risks, sample_weight, pair_safe, pair_unsafe,
            pair_horizon,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    model.eval()
    with torch.no_grad():
        total, parts = bellman_losses(
            model, features, step_risks, sample_weight, pair_safe, pair_unsafe,
            pair_horizon,
        )
    return {
        "total": float(total),
        **{key: float(value) for key, value in zip(
            ("supervised", "bellman", "false_safe", "ranking"),
            parts, strict=True,
        )},
    }
