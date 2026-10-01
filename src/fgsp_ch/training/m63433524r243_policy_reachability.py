"""Training helpers for the R2.4.3 policy-consistent reachable-risk head."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from fgsp_ch.models.m63433524r243_policy_reachability import PolicyReachabilityHead


@dataclass(frozen=True)
class ReachabilityEnsembleScore:
    mean: np.ndarray
    upper: np.ndarray
    members: np.ndarray


ReachabilityMember = tuple[PolicyReachabilityHead, np.ndarray, np.ndarray]


def conditional_hazard_loss(
    logits: torch.Tensor,
    exit_event: torch.Tensor,
    at_risk: torch.Tensor,
    *,
    positive_weight: float,
) -> torch.Tensor:
    if logits.shape != exit_event.shape or logits.shape != at_risk.shape:
        raise ValueError("hazard tensors must have identical shapes")
    raw = nn.functional.binary_cross_entropy_with_logits(
        logits,
        exit_event.float(),
        pos_weight=torch.as_tensor(positive_weight, device=logits.device),
        reduction="none",
    )
    mask = at_risk.float()
    return (raw * mask).sum() / mask.sum().clamp_min(1.0)


def train_reachability_member(
    features: np.ndarray,
    exit_event: np.ndarray,
    at_risk: np.ndarray,
    cumulative_severity: np.ndarray,
    *,
    seed: int,
    hidden_channels: int,
    epochs: int,
    learning_rate: float,
    severity_weight: float,
    device: torch.device,
) -> ReachabilityMember:
    x = np.asarray(features, dtype=np.float32)
    event = np.asarray(exit_event, dtype=np.float32)
    risk = np.asarray(at_risk, dtype=bool)
    severity = np.asarray(cumulative_severity, dtype=np.float32)
    if x.ndim != 2 or x.shape[1] != 68 or event.shape != (len(x), 4):
        raise ValueError("invalid R2.4.3 training arrays")
    if risk.shape != event.shape or severity.shape != event.shape:
        raise ValueError("R2.4.3 targets must align")
    mean = x.mean(axis=0)
    scale = np.maximum(x.std(axis=0), 1.0e-6)
    torch.manual_seed(int(seed))
    model = PolicyReachabilityHead(hidden_channels=hidden_channels).to(device)
    tx = torch.as_tensor((x - mean) / scale, device=device)
    te = torch.as_tensor(event, device=device)
    tr = torch.as_tensor(risk, device=device)
    ts = torch.as_tensor(np.log1p(np.maximum(severity, 0.0)), device=device)
    positives = max(float(event[risk].sum()), 1.0)
    negatives = max(float(risk.sum()) - positives, 1.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1.0e-4)
    model.train()
    for _ in range(int(epochs)):
        optimizer.zero_grad(set_to_none=True)
        output = model(tx)
        hazard = conditional_hazard_loss(
            output.hazard_logits, te, tr, positive_weight=negatives / positives,
        )
        severity_loss = nn.functional.smooth_l1_loss(output.cumulative_severity, ts)
        loss = hazard + float(severity_weight) * severity_loss
        loss.backward()
        optimizer.step()
    return model.eval(), mean.astype(np.float64), scale.astype(np.float64)


@torch.no_grad()
def ensemble_reachability_score(
    members: list[ReachabilityMember],
    features: np.ndarray,
    *,
    upper_std_multiplier: float,
    device: torch.device,
) -> ReachabilityEnsembleScore:
    x = np.asarray(features, dtype=np.float32)
    scores = []
    for model, mean, scale in members:
        tensor = torch.as_tensor(
            (x - mean) / scale, dtype=torch.float32, device=device,
        )
        scores.append(model(tensor).cumulative_exit.cpu().numpy())
    stacked = np.stack(scores)
    mean_score = stacked.mean(axis=0)
    upper = np.clip(
        mean_score + float(upper_std_multiplier) * stacked.std(axis=0, ddof=0),
        0.0,
        1.0,
    )
    # Individual cumulative risks are monotone, but a column-wise ensemble
    # standard-deviation allowance can otherwise violate that invariant.
    # Project only upward: this cannot turn an unsafe branch into an accepted
    # one and preserves the conservative interpretation of the upper score.
    upper = np.maximum.accumulate(upper, axis=1)
    return ReachabilityEnsembleScore(mean_score, upper, stacked)
