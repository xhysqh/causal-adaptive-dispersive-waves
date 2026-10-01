from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.nn import functional as F

from fgsp_ch.models.m634331_horizon_risk import HorizonRiskToGoOperator


@dataclass(frozen=True, slots=True)
class HorizonTensorSplit:
    features: torch.Tensor
    log_risk: torch.Tensor
    first_exit: torch.Tensor
    boundary: torch.Tensor
    groups: np.ndarray
    families: np.ndarray
    points: np.ndarray


def tensor_split(archive, split: str, device: torch.device) -> HorizonTensorSplit:
    selected = np.asarray(archive["split"] == split)
    return HorizonTensorSplit(
        torch.as_tensor(archive["features"][selected], dtype=torch.float32, device=device),
        torch.as_tensor(np.log1p(archive["risk_to_go"][selected]), dtype=torch.float32, device=device),
        torch.as_tensor(archive["first_exit"][selected] - 1, dtype=torch.long, device=device),
        torch.as_tensor(archive["boundary"][selected], dtype=torch.float32, device=device),
        np.asarray(archive["group"][selected]),
        np.asarray(archive["family"][selected]),
        np.asarray(archive["points"][selected]),
    )


def trajectory_bootstrap(data: HorizonTensorSplit, seed: int) -> HorizonTensorSplit:
    rng = np.random.default_rng(seed)
    groups = np.unique(data.groups)
    sampled = rng.choice(groups, size=groups.size, replace=True)
    indices = np.concatenate([np.flatnonzero(data.groups == group) for group in sampled])
    index = torch.as_tensor(indices, dtype=torch.long, device=data.features.device)
    return HorizonTensorSplit(
        data.features[index], data.log_risk[index], data.first_exit[index],
        data.boundary[index], data.groups[indices], data.families[indices],
        data.points[indices],
    )


def horizon_losses(model: HorizonRiskToGoOperator, data: HorizonTensorSplit):
    output = model(data.features)
    regression = F.smooth_l1_loss(output.log_risk, data.log_risk)
    unsafe = (data.log_risk > np.log(2.0)).float()
    predicted_unsafe_logit = 5.0 * (output.log_risk - np.log(2.0))
    classification = F.binary_cross_entropy_with_logits(
        predicted_unsafe_logit, unsafe,
    )
    false_safe = (
        unsafe * F.relu(data.log_risk - output.log_risk).square()
    ).sum() / unsafe.sum().clamp_min(1.0)
    exit_loss = F.cross_entropy(output.exit_logits, data.first_exit)
    boundary_loss = F.binary_cross_entropy_with_logits(
        output.boundary_logit, data.boundary,
    )
    total = (
        regression + 0.5 * classification + 0.5 * false_safe
        + 0.2 * exit_loss + 0.1 * boundary_loss
    )
    return total, (regression, classification, false_safe, exit_loss, boundary_loss)


def train_horizon_operator(
    model: HorizonRiskToGoOperator,
    train: HorizonTensorSplit,
    *,
    seed: int,
    epochs: int,
    learning_rate: float,
) -> dict[str, float]:
    data = trajectory_bootstrap(train, seed)
    model.set_normalization(data.features)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1.0e-5)
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        loss, _ = horizon_losses(model, data)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    model.eval()
    with torch.no_grad():
        total, parts = horizon_losses(model, data)
    return {
        "total": float(total),
        **{key: float(value) for key, value in zip(
            ("regression", "classification", "false_safe", "first_exit", "boundary"),
            parts, strict=True,
        )},
    }
