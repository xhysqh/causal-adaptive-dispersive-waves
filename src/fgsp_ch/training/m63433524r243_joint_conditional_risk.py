"""Training helpers for the R2.4.3 joint conditional risk calibrator.

The calibrator is trained on the 4-dim joint meta-feature vector z_joint
produced by combining the frozen R2.3 global reachability score with the
R2.4.2 conditional local risk ensemble upper bound.

Design contract
---------------
* Zero unsafe acceptances at any selected threshold (hard constraint).
* Conservative upper bound: joint score = max(base_r23, cond_local_upper).
* No future state, reference label, or behaviour-policy identifier in inputs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from fgsp_ch.models.m63433524r243_joint_conditional_risk import (
    JointConditionalRiskCalibrator,
    JOINT_META_FEATURE_DIM,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def binary_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Tie-aware AUC without a sklearn dependency."""
    score = np.asarray(scores, dtype=float)
    label = np.asarray(labels, dtype=bool)
    positives = int(label.sum())
    negatives = int((~label).sum())
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
    return float(
        (ranks[label].sum() - positives * (positives + 1) / 2.0)
        / (positives * negatives)
    )


@dataclass(frozen=True)
class JointCalibrationEnsemble:
    """Result of a multi-seed calibrator ensemble."""

    mean: np.ndarray
    """Element-wise mean unsafe probability, shape ``(N,)``."""
    upper: np.ndarray
    """Conservative upper bound (mean + k*std), clipped to [0, 1], shape ``(N,)``."""
    member_scores: np.ndarray
    """Per-member sigmoid probabilities, shape ``(M, N)``."""


# ── Training ──────────────────────────────────────────────────────────────────

def train_member(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    seed: int,
    hidden_dim: int = 16,
    epochs: int = 50,
    learning_rate: float = 1e-3,
    device: torch.device,
) -> tuple[JointConditionalRiskCalibrator, np.ndarray, np.ndarray]:
    """Fit one deterministic, class-balanced binary calibration member.

    Parameters
    ----------
    features:
        Shape ``(N, 4)`` – the z_joint meta-feature matrix.
    labels:
        Shape ``(N,)`` – binary float (1.0 = unsafe, 0.0 = safe).
    seed:
        RNG seed for reproducibility.
    hidden_dim:
        Hidden layer width for the calibrator MLP.
    epochs:
        Number of full-batch gradient steps.
    learning_rate:
        AdamW learning rate.
    device:
        Torch device for training.

    Returns
    -------
    model:
        Trained calibrator in eval mode.
    mean:
        Per-feature mean used for standardisation, shape ``(4,)``.
    scale:
        Per-feature std used for standardisation, shape ``(4,)``; minimum 1e-6.
    """
    x = np.asarray(features, dtype=np.float32)
    y = np.asarray(labels, dtype=np.float32)
    if x.ndim != 2 or x.shape[1] != JOINT_META_FEATURE_DIM:
        raise ValueError(
            f"R2.4.3 training expects (N, {JOINT_META_FEATURE_DIM}) features; "
            f"got {x.shape}"
        )
    if y.shape != (len(x),):
        raise ValueError(
            f"labels must be shape ({len(x)},); got {y.shape}"
        )

    mean = x.mean(axis=0)
    scale = np.maximum(x.std(axis=0), 1.0e-6)

    torch.manual_seed(int(seed))
    model = JointConditionalRiskCalibrator(
        input_dim=JOINT_META_FEATURE_DIM, hidden_dim=int(hidden_dim)
    ).to(device)

    tensor_x = torch.as_tensor((x - mean) / scale, dtype=torch.float32, device=device)
    tensor_y = torch.as_tensor(y, dtype=torch.float32, device=device)

    positive = max(float(tensor_y.sum().item()), 1.0)
    negative = max(float(len(tensor_y) - positive), 1.0)
    loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(negative / positive, device=device)
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(learning_rate), weight_decay=1.0e-4
    )

    model.train()
    for _ in range(int(epochs)):
        optimizer.zero_grad(set_to_none=True)
        logits = model(tensor_x)
        loss = loss_fn(logits, tensor_y)
        loss.backward()
        optimizer.step()

    return model.eval(), mean.astype(np.float64), scale.astype(np.float64)


# ── Ensemble inference ────────────────────────────────────────────────────────

@torch.no_grad()
def ensemble_score(
    members: list[tuple[JointConditionalRiskCalibrator, np.ndarray, np.ndarray]],
    features: np.ndarray,
    *,
    upper_std_multiplier: float,
    device: torch.device,
) -> JointCalibrationEnsemble:
    """Compute conservative ensemble unsafe probabilities.

    Parameters
    ----------
    members:
        List of ``(model, mean, scale)`` tuples from :func:`train_member`.
    features:
        Shape ``(N, 4)`` – z_joint meta-features for the query points.
    upper_std_multiplier:
        Multiplier ``k`` for the standard deviation in the upper-bound formula
        ``upper = mean + k * std``.
    device:
        Torch device for inference.

    Returns
    -------
    JointCalibrationEnsemble
    """
    x = np.asarray(features, dtype=np.float32)
    scores = []
    for model, mean, scale in members:
        normed = (x - mean) / scale
        logits = model(torch.as_tensor(normed, dtype=torch.float32, device=device))
        scores.append(torch.sigmoid(logits).detach().cpu().numpy())

    member_scores = np.stack(scores, axis=0)          # (M, N)
    mean_score = member_scores.mean(axis=0)
    upper = np.clip(
        mean_score + float(upper_std_multiplier) * member_scores.std(axis=0, ddof=0),
        0.0,
        1.0,
    )
    return JointCalibrationEnsemble(mean_score, upper, member_scores)


# ── Threshold selection ───────────────────────────────────────────────────────

def largest_zero_unsafe_threshold(
    score: np.ndarray,
    unsafe: np.ndarray,
    eligible: np.ndarray,
    groups: np.ndarray,
    *,
    minimum_groups: int,
) -> dict | None:
    """Select the most permissive threshold that still rejects every unsafe sample.

    Parameters
    ----------
    score:
        Joint unsafe probability estimate per sample, shape ``(N,)``.
    unsafe:
        Boolean ground-truth unsafe flag per sample.
    eligible:
        Boolean mask of samples eligible for acceptance.
    groups:
        Integer group identifier per sample (for coverage counting).
    minimum_groups:
        Minimum distinct groups that must be covered by accepted samples.

    Returns
    -------
    dict | None
        Best threshold record ``{threshold, accepted, accepted_groups,
        unsafe_acceptances}`` or ``None`` if no valid threshold exists.
    """
    score = np.asarray(score, dtype=float)
    unsafe = np.asarray(unsafe, dtype=bool)
    eligible = np.asarray(eligible, dtype=bool)
    groups = np.asarray(groups)

    chosen = None
    for threshold in np.unique(np.sort(score[eligible])):
        accepted = eligible & (score <= threshold)
        failures = int(np.sum(accepted & unsafe))
        group_count = int(
            sum(
                np.any(accepted[groups == g])
                for g in np.unique(groups)
            )
        )
        record = {
            "threshold": float(threshold),
            "accepted": int(accepted.sum()),
            "accepted_groups": group_count,
            "unsafe_acceptances": failures,
        }
        if failures == 0 and group_count >= int(minimum_groups):
            if chosen is None or (
                record["accepted_groups"],
                record["accepted"],
            ) > (chosen["accepted_groups"], chosen["accepted"]):
                chosen = record
    return chosen
