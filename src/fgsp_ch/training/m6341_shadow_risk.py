from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.nn import functional as F

from fgsp_ch.models.m6341_shadow_risk import MonotoneShadowDefectRiskOperator


@dataclass(frozen=True, slots=True)
class M6341TensorSplit:
    context: torch.Tensor
    residual: torch.Tensor
    log_defect: torch.Tensor
    direction: torch.Tensor
    maximum_safe_weight: torch.Tensor
    groups: np.ndarray
    families: np.ndarray
    points: np.ndarray


def tensor_split(archive, split: str, device: torch.device) -> M6341TensorSplit:
    selected = np.asarray(archive["split"] == split)
    context = np.concatenate((
        archive["feature_summary"][selected], archive["parameters"][selected]
    ), axis=1)
    return M6341TensorSplit(
        torch.as_tensor(context, dtype=torch.float32, device=device),
        torch.as_tensor(archive["residual_features"][selected], dtype=torch.float32, device=device),
        torch.as_tensor(np.log1p(archive["shadow_defect_ratio"][selected]), dtype=torch.float32, device=device),
        torch.as_tensor(archive["direction_cosine"][selected], dtype=torch.float32, device=device),
        torch.as_tensor(archive["maximum_safe_weight"][selected], dtype=torch.float32, device=device),
        np.asarray(archive["group"][selected]),
        np.asarray(archive["family"][selected]),
        np.asarray(archive["points"][selected]),
    )


def _subset(data: M6341TensorSplit, indices: np.ndarray) -> M6341TensorSplit:
    tensor_indices = torch.as_tensor(indices, dtype=torch.long, device=data.context.device)
    return M6341TensorSplit(
        data.context[tensor_indices], data.residual[tensor_indices],
        data.log_defect[tensor_indices], data.direction[tensor_indices],
        data.maximum_safe_weight[tensor_indices], data.groups[indices],
        data.families[indices], data.points[indices],
    )


def trajectory_bootstrap(data: M6341TensorSplit, seed: int) -> M6341TensorSplit:
    rng = np.random.default_rng(seed)
    groups = np.unique(data.groups)
    sampled = rng.choice(groups, size=groups.size, replace=True)
    # Every member sees every trajectory once; group bootstrap observations
    # are appended as multiplicity weights. Pure replacement bootstrap is too
    # destructive for the current 12-group pilot archive.
    indices = np.concatenate((
        np.arange(data.groups.size),
        *[np.flatnonzero(data.groups == group) for group in sampled],
    ))
    return _subset(data, indices)


def _pairwise_ranking(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    truth_difference = target[:, None] - target[None, :]
    prediction_difference = prediction[:, None] - prediction[None, :]
    selected = truth_difference > 0.1
    if not torch.any(selected):
        return prediction.new_zeros(())
    return F.relu(0.02 - prediction_difference[selected]).mean()


def risk_losses(model: MonotoneShadowDefectRiskOperator, data: M6341TensorSplit):
    output = model(data.context, data.residual)
    error = data.log_defect - output.upper_log_defect
    median = F.smooth_l1_loss(output.median_log_defect, data.log_defect)
    quantile = torch.maximum(0.9 * error, -0.1 * error).mean()
    under = F.relu(error).square().mean()
    direction = F.smooth_l1_loss(output.direction, data.direction)
    unsafe = (data.maximum_safe_weight <= 0.0).to(output.unsafe_logit.dtype)
    count_unsafe = unsafe.sum().clamp_min(1.0)
    count_safe = (1.0 - unsafe).sum().clamp_min(1.0)
    weights = unsafe / count_unsafe + (1.0 - unsafe) / count_safe
    classification = (
        F.binary_cross_entropy_with_logits(
            output.unsafe_logit, unsafe, reduction="none"
        ) * weights
    ).sum() / weights.sum().clamp_min(1.0e-12)
    ranking = _pairwise_ranking(output.median_log_defect, data.log_defect)
    total = (
        median + 0.5 * quantile + 2.0 * under + 0.2 * direction
        + 0.2 * classification + 0.1 * ranking
    )
    return total, (median, quantile, under, direction, classification, ranking)


def train_shadow_risk_operator(
    model: MonotoneShadowDefectRiskOperator,
    train: M6341TensorSplit,
    *,
    epochs: int,
    learning_rate: float,
    bootstrap_seed: int,
) -> dict[str, float]:
    bootstrap = trajectory_bootstrap(train, bootstrap_seed)
    model.set_normalization(bootstrap.context, bootstrap.residual)
    # Train-only ridge initialization gives the tiny network the already
    # audited physical relation. Core coefficients are projected positive;
    # auxiliary residuals retain signed conditional corrections.
    with torch.no_grad():
        logged = torch.log1p(bootstrap.residual.clamp_min(0.0))
        normalized = (logged - model.residual_mean) / model.residual_scale
        design = torch.column_stack((
            torch.ones(normalized.shape[0], device=normalized.device), normalized
        ))
        penalty = torch.eye(design.shape[1], device=design.device) * 1.0
        penalty[0, 0] = 0.0
        coefficients = torch.linalg.solve(
            design.T @ design + penalty,
            design.T @ bootstrap.log_defect,
        )
        core_coefficients = coefficients[[1, 2, 5]].clamp_min(1.0e-4)
        model.residual_median.raw_weight.copy_(
            torch.log(torch.expm1(core_coefficients))[None]
        )
        model.residual_median.bias.copy_(coefficients[0:1])
        model.auxiliary_median.weight.copy_(coefficients[[3, 4, 6, 7]][None])
        model.auxiliary_median.bias.zero_()
        model.context_median.weight.zero_()
        model.context_median.bias.zero_()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=1.0e-5
    )
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        loss, _ = risk_losses(model, bootstrap)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    model.eval()
    with torch.no_grad():
        total, parts = risk_losses(model, bootstrap)
    names = ("median", "quantile", "under", "direction", "classification", "ranking")
    return {"total": float(total), **{
        name: float(value) for name, value in zip(names, parts, strict=True)
    }}


def derived_safe_weight(upper_log_defect: np.ndarray, tolerance: float) -> np.ndarray:
    defect = np.expm1(np.asarray(upper_log_defect, dtype=np.float64))
    result = np.zeros_like(defect)
    for weight in (1.0, 0.5, 0.25, 0.125, 0.0625):
        selected = (result == 0.0) & (weight * defect <= tolerance)
        result[selected] = weight
    return result
