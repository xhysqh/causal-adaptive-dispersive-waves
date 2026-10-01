from __future__ import annotations

import numpy as np
import torch
from torch.nn import functional as F

from fgsp_ch.models.m634333_joint_reachability import (
    JointReachabilityCertificate,
)
from fgsp_ch.training.m634331_horizon_risk import (
    HorizonTensorSplit,
    trajectory_bootstrap,
)


def reachability_labels(data: HorizonTensorSplit) -> torch.Tensor:
    return torch.stack([
        (data.first_exit < horizon).float() for horizon in (2, 4, 8)
    ], dim=1)


def joint_losses(
    model: JointReachabilityCertificate,
    context: torch.Tensor,
    frozen_log_risk: torch.Tensor,
    target_log_risk: torch.Tensor,
    first_exit: torch.Tensor,
):
    output = model(context, frozen_log_risk)
    labels = torch.stack([
        (first_exit < horizon).float() for horizon in (2, 4, 8)
    ], dim=1)
    horizon_weight = torch.as_tensor((1.0, 1.5, 2.0), device=context.device)
    regression = torch.mean(
        F.smooth_l1_loss(output.log_risk, target_log_risk, reduction="none")
        * horizon_weight
    )
    reach = F.binary_cross_entropy(
        output.reach_probability.clamp(1.0e-6, 1.0 - 1.0e-6), labels,
    )
    joint = F.binary_cross_entropy(
        output.joint_probability.clamp(1.0e-6, 1.0 - 1.0e-6), labels,
    )
    unsafe = (target_log_risk > np.log(2.0)).float()
    false_safe = (
        unsafe * F.relu(target_log_risk - output.log_risk).square()
        * horizon_weight
    ).sum() / unsafe.sum().clamp_min(1.0)
    risk_probability = torch.sigmoid(5.0 * (output.log_risk - np.log(2.0)))
    consistency = F.mse_loss(output.reach_probability, risk_probability)
    total = (
        regression + 0.7 * reach + 0.7 * joint
        + 0.8 * false_safe + 0.2 * consistency
    )
    return total, (regression, reach, joint, false_safe, consistency)


def train_joint_certificate(
    model: JointReachabilityCertificate,
    train: HorizonTensorSplit,
    context: torch.Tensor,
    frozen_log_risk: torch.Tensor,
    *,
    seed: int,
    epochs: int,
    learning_rate: float,
) -> dict[str, float]:
    boot = trajectory_bootstrap(train, seed)
    # Recover bootstrap indices through the group sequence so contexts remain
    # paired with labels, including repeated trajectory groups.
    rng = np.random.default_rng(seed)
    groups = np.unique(train.groups)
    sampled = rng.choice(groups, size=groups.size, replace=True)
    indices = np.concatenate([np.flatnonzero(train.groups == group) for group in sampled])
    index = torch.as_tensor(indices, dtype=torch.long, device=context.device)
    boot_context, boot_frozen = context[index], frozen_log_risk[index]
    model.set_normalization(boot_context)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=1.0e-5,
    )
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        loss, _ = joint_losses(
            model, boot_context, boot_frozen, boot.log_risk, boot.first_exit,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    model.eval()
    with torch.no_grad():
        total, parts = joint_losses(
            model, boot_context, boot_frozen, boot.log_risk, boot.first_exit,
        )
    return {
        "total": float(total),
        **{key: float(value) for key, value in zip(
            ("regression", "reachability", "joint", "false_safe", "consistency"),
            parts, strict=True,
        )},
    }
