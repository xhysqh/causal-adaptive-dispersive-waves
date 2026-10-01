from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.nn import functional as F

from fgsp_ch.models.m63432_deployment_safety import (
    DeploymentCandidateSafetyOperator,
)


@dataclass(frozen=True, slots=True)
class DeploymentTensorSplit:
    features: torch.Tensor
    unsafe: torch.Tensor
    log_defect: torch.Tensor
    direction: torch.Tensor
    groups: np.ndarray
    families: np.ndarray
    points: np.ndarray
    dts: np.ndarray


def tensor_split(archive, split: str, device: torch.device) -> DeploymentTensorSplit:
    selected = np.asarray(archive["split"] == split)
    return DeploymentTensorSplit(
        torch.as_tensor(archive["features"][selected], dtype=torch.float32, device=device),
        torch.as_tensor(~archive["safe"][selected], dtype=torch.float32, device=device),
        torch.as_tensor(
            np.log1p(archive["shadow_defect_ratio"][selected]),
            dtype=torch.float32, device=device,
        ),
        torch.as_tensor(archive["direction_cosine"][selected], dtype=torch.float32, device=device),
        np.asarray(archive["group"][selected]),
        np.asarray(archive["family"][selected]),
        np.asarray(archive["points"][selected]),
        np.asarray(archive["dt"][selected]),
    )


def trajectory_bootstrap(data: DeploymentTensorSplit, seed: int) -> DeploymentTensorSplit:
    rng = np.random.default_rng(seed)
    groups = np.unique(data.groups)
    sampled = rng.choice(groups, size=groups.size, replace=True)
    indices = np.concatenate([
        np.flatnonzero(data.groups == group) for group in sampled
    ])
    index = torch.as_tensor(indices, dtype=torch.long, device=data.features.device)
    return DeploymentTensorSplit(
        data.features[index], data.unsafe[index], data.log_defect[index],
        data.direction[index], data.groups[indices], data.families[indices],
        data.points[indices], data.dts[indices],
    )


def deployment_losses(model, data: DeploymentTensorSplit):
    output = model(data.features)
    unsafe_count = data.unsafe.sum().clamp_min(1.0)
    safe_count = (1.0 - data.unsafe).sum().clamp_min(1.0)
    class_weight = (
        2.0 * data.unsafe / unsafe_count
        + (1.0 - data.unsafe) / safe_count
    )
    classification = (
        F.binary_cross_entropy_with_logits(
            output.unsafe_logit, data.unsafe, reduction="none"
        ) * class_weight
    ).sum() / class_weight.sum()
    # False-safe predictions receive an additional asymmetric penalty.
    false_safe = (
        data.unsafe * F.relu(1.0 - output.unsafe_logit).square()
    ).sum() / unsafe_count
    defect = F.smooth_l1_loss(output.log_defect, data.log_defect)
    direction = F.smooth_l1_loss(output.direction, data.direction)
    total = classification + 0.5 * false_safe + 0.25 * defect + 0.1 * direction
    return total, (classification, false_safe, defect, direction)


def train_deployment_safety_operator(
    model: DeploymentCandidateSafetyOperator,
    train: DeploymentTensorSplit,
    *,
    seed: int,
    epochs: int,
    learning_rate: float,
) -> dict[str, float]:
    data = trajectory_bootstrap(train, seed)
    model.set_normalization(data.features)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=1.0e-5
    )
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        loss, _ = deployment_losses(model, data)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    model.eval()
    with torch.no_grad():
        total, parts = deployment_losses(model, data)
    return {
        "total": float(total),
        **{
            key: float(value) for key, value in zip(
                ("classification", "false_safe", "defect", "direction"),
                parts, strict=True,
            )
        },
    }
