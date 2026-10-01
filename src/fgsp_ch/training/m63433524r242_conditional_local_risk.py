"""Training, scoring and fixed-selection helpers for R2.4.2."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from fgsp_ch.features.m63433524r242_conditional_local_risk import (
    CONDITIONAL_LOCAL_RISK_FEATURE_DIM,
)
from fgsp_ch.models.m63433524r242_conditional_local_risk import (
    AblationRiskHead,
    TargetCellConditionalRiskHead,
)


def binary_auc(scores: np.ndarray, labels: np.ndarray) -> float:
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
    score = np.asarray(scores, dtype=float)
    label = np.asarray(labels, dtype=bool)
    positives = int(label.sum())
    if positives == 0:
        return float("nan")
    ordered = label[np.argsort(-score, kind="mergesort")]
    precision = np.cumsum(ordered, dtype=float) / np.arange(1, len(ordered) + 1)
    return float(precision[ordered].sum() / positives)


@dataclass(frozen=True)
class ConditionalRiskEnsemble:
    mean: np.ndarray
    upper: np.ndarray
    member_scores: np.ndarray


RiskMember = tuple[nn.Module, np.ndarray, np.ndarray]


def train_member(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    seed: int,
    hidden_channels: int,
    epochs: int,
    learning_rate: float,
    device: torch.device,
    architecture: str = "moe",
    residual_bound: float = 2.0,
    geometry_targets: np.ndarray | None = None,
    gate_loss_weight: float = 0.0,
) -> RiskMember:
    """Fit one deterministic class-balanced MoE or ablation member."""
    x = np.asarray(features, dtype=np.float32)
    y = np.asarray(labels, dtype=np.float32)
    if x.ndim != 2 or y.shape != (len(x),) or x.shape[1] <= 0:
        raise ValueError(f"invalid R2.4.2 training arrays: x={x.shape}, y={y.shape}")
    if architecture == "moe" and x.shape[1] != CONDITIONAL_LOCAL_RISK_FEATURE_DIM:
        raise ValueError(f"conditional MoE expects {CONDITIONAL_LOCAL_RISK_FEATURE_DIM} channels")
    if architecture not in {"moe", "mlp"}:
        raise ValueError(f"unknown R2.4.2 architecture: {architecture}")
    gate_target = None
    if geometry_targets is not None:
        values = np.asarray(geometry_targets, dtype=np.float32)
        if architecture != "moe" or values.shape != y.shape:
            raise ValueError("geometry targets are only valid for an aligned MoE batch")
        gate_target = torch.as_tensor(values, dtype=torch.float32, device=device)
    mean = x.mean(axis=0)
    scale = np.maximum(x.std(axis=0), 1.0e-6)
    torch.manual_seed(int(seed))
    if architecture == "moe":
        model: nn.Module = TargetCellConditionalRiskHead(
            input_channels=x.shape[1], hidden_channels=hidden_channels,
            residual_bound=residual_bound,
        )
    else:
        model = AblationRiskHead(x.shape[1], hidden_channels=hidden_channels)
    model = model.to(device)
    tensor_x = torch.as_tensor((x - mean) / scale, dtype=torch.float32, device=device)
    tensor_y = torch.as_tensor(y, dtype=torch.float32, device=device)
    positive = max(float(tensor_y.sum().item()), 1.0)
    negative = max(float(len(tensor_y) - positive), 1.0)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(negative / positive, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1.0e-4)
    model.train()
    for _ in range(int(epochs)):
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model(tensor_x), tensor_y)
        if gate_target is not None and gate_loss_weight > 0.0:
            gate = model.gate_probability(tensor_x)
            loss = loss + float(gate_loss_weight) * nn.functional.binary_cross_entropy(gate, gate_target)
        loss.backward()
        optimizer.step()
    return model.eval(), mean.astype(np.float64), scale.astype(np.float64)


@torch.no_grad()
def ensemble_score(
    members: list[RiskMember],
    features: np.ndarray,
    *,
    upper_std_multiplier: float,
    device: torch.device,
) -> ConditionalRiskEnsemble:
    x = np.asarray(features, dtype=np.float32)
    scores = []
    for model, mean, scale in members:
        if x.ndim != 2 or x.shape[1] != len(mean):
            raise ValueError("ensemble feature layout does not match member normalisation")
        tensor = torch.as_tensor((x - mean) / scale, dtype=torch.float32, device=device)
        scores.append(torch.sigmoid(model(tensor)).cpu().numpy())
    member_scores = np.stack(scores)
    mean_score = member_scores.mean(axis=0)
    upper = np.clip(
        mean_score + float(upper_std_multiplier) * member_scores.std(axis=0, ddof=0), 0.0, 1.0
    )
    return ConditionalRiskEnsemble(mean_score, upper, member_scores)


@torch.no_grad()
def ensemble_gate_probability(
    members: list[RiskMember], features: np.ndarray, *, device: torch.device,
) -> np.ndarray:
    """Mean geometry-gate probability for MoE interpretability audits."""
    x = np.asarray(features, dtype=np.float32)
    gates = []
    for model, mean, scale in members:
        if not isinstance(model, TargetCellConditionalRiskHead):
            raise ValueError("gate probabilities require conditional MoE members")
        tensor = torch.as_tensor((x - mean) / scale, dtype=torch.float32, device=device)
        gates.append(model.gate_probability(tensor).cpu().numpy())
    return np.stack(gates).mean(axis=0)


def largest_zero_unsafe_threshold(
    score: np.ndarray,
    unsafe: np.ndarray,
    eligible: np.ndarray,
    groups: np.ndarray,
    *,
    minimum_groups: int,
) -> dict | None:
    """Select a threshold on calibration only; never call this on audit labels."""
    score = np.asarray(score, dtype=float)
    unsafe = np.asarray(unsafe, dtype=bool)
    eligible = np.asarray(eligible, dtype=bool)
    groups = np.asarray(groups)
    chosen = None
    for threshold in np.unique(np.sort(score[eligible])):
        accepted = eligible & (score <= threshold)
        record = fixed_threshold_record(score, unsafe, eligible, groups, threshold=float(threshold))
        if record["unsafe_acceptances"] == 0 and record["accepted_groups"] >= int(minimum_groups):
            if chosen is None or (record["accepted_groups"], record["accepted"]) > (
                chosen["accepted_groups"], chosen["accepted"]
            ):
                chosen = record
    return chosen


def fixed_threshold_record(
    score: np.ndarray,
    unsafe: np.ndarray,
    eligible: np.ndarray,
    groups: np.ndarray,
    *,
    threshold: float,
) -> dict:
    score = np.asarray(score, dtype=float)
    unsafe = np.asarray(unsafe, dtype=bool)
    eligible = np.asarray(eligible, dtype=bool)
    groups = np.asarray(groups)
    accepted = eligible & (score <= float(threshold))
    accepted_groups = int(sum(np.any(accepted[groups == group]) for group in np.unique(groups)))
    return {
        "threshold": float(threshold),
        "accepted": int(accepted.sum()),
        "accepted_groups": accepted_groups,
        "unsafe_acceptances": int(np.sum(accepted & unsafe)),
    }
