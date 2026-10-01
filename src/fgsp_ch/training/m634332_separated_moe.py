from __future__ import annotations

import numpy as np
import torch
from torch.nn import functional as F

from fgsp_ch.models.m634331_horizon_risk import HorizonRiskToGoOperator
from fgsp_ch.models.m634332_separated_moe import SeparatedMoEResidual
from fgsp_ch.training.m634331_horizon_risk import (
    HorizonTensorSplit,
    trajectory_bootstrap,
)


def separated_only(data: HorizonTensorSplit) -> HorizonTensorSplit:
    indices = np.flatnonzero(data.families == "separated")
    index = torch.as_tensor(indices, dtype=torch.long, device=data.features.device)
    return HorizonTensorSplit(
        data.features[index], data.log_risk[index], data.first_exit[index],
        data.boundary[index], data.groups[indices], data.families[indices],
        data.points[indices],
    )


def moe_losses(
    base_model: HorizonRiskToGoOperator,
    model: SeparatedMoEResidual,
    data: HorizonTensorSplit,
):
    with torch.no_grad():
        base = base_model(data.features)
    output = model(data.features, base)
    prediction = output.prediction
    horizon_weight = torch.as_tensor(
        (1.0, 1.5, 2.0), device=data.features.device,
    )
    regression = torch.mean(
        F.smooth_l1_loss(prediction.log_risk, data.log_risk, reduction="none")
        * horizon_weight
    )
    unsafe = (data.log_risk > np.log(2.0)).float()
    classification = F.binary_cross_entropy_with_logits(
        5.0 * (prediction.log_risk - np.log(2.0)), unsafe,
    )
    false_safe = (
        unsafe * F.relu(data.log_risk - prediction.log_risk).square()
        * horizon_weight
    ).sum() / unsafe.sum().clamp_min(1.0)
    exit_loss = F.cross_entropy(prediction.exit_logits, data.first_exit)
    boundary = F.binary_cross_entropy_with_logits(
        prediction.boundary_logit, data.boundary,
    )
    usage = output.gate.mean(dim=0)
    balance = torch.mean((usage - 1.0 / model.expert_count).square())
    entropy = -torch.mean(torch.sum(
        output.gate * torch.log(output.gate.clamp_min(1.0e-8)), dim=1,
    ))
    # Balance prevents dead experts; the small positive entropy penalty makes
    # routing locally decisive instead of averaging identical experts.
    total = (
        regression + 0.5 * classification + 0.8 * false_safe
        + 0.2 * exit_loss + 0.1 * boundary
        + 0.5 * balance + 0.01 * entropy
    )
    return total, (
        regression, classification, false_safe, exit_loss, boundary,
        balance, entropy,
    )


def train_separated_moe(
    base_model: HorizonRiskToGoOperator,
    model: SeparatedMoEResidual,
    train: HorizonTensorSplit,
    *,
    seed: int,
    epochs: int,
    learning_rate: float,
) -> dict[str, float | list[float]]:
    data = trajectory_bootstrap(separated_only(train), seed)
    for parameter in base_model.parameters():
        parameter.requires_grad_(False)
    base_model.eval()
    model.set_normalization(data.features)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=1.0e-5,
    )
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        loss, _ = moe_losses(base_model, model, data)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    model.eval()
    with torch.no_grad():
        total, parts = moe_losses(base_model, model, data)
        gate = model(data.features, base_model(data.features)).gate
    return {
        "total": float(total),
        **{key: float(value) for key, value in zip(
            ("regression", "classification", "false_safe", "first_exit",
             "boundary", "balance", "gate_entropy"), parts, strict=True,
        )},
        "expert_usage": gate.mean(dim=0).cpu().tolist(),
    }
