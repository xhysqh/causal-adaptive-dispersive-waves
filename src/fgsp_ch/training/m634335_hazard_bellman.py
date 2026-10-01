from __future__ import annotations

import copy

import torch
from torch.nn import functional as F

from fgsp_ch.models.m634335_hazard_bellman import (
    HazardBellmanOperator,
    bellman_residual,
)


def conditional_hazard_nll(
    logits: torch.Tensor,
    exit_event: torch.Tensor,
    at_risk: torch.Tensor,
    sample_weight: torch.Tensor | None = None,
    first_exit_weight: float = 1.0,
) -> torch.Tensor:
    event = exit_event.float()
    risk = at_risk.float()
    if not (logits.shape == event.shape == risk.shape):
        raise ValueError("hazard targets must match logits")
    if first_exit_weight < 1.0:
        raise ValueError("first_exit_weight must be at least one")
    horizon_weight = torch.ones((1, logits.shape[1]), device=logits.device)
    horizon_weight[:, 0] = float(first_exit_weight)
    loss = F.binary_cross_entropy_with_logits(
        logits, event, reduction="none",
    ) * risk * horizon_weight
    normalizer = risk * horizon_weight
    if sample_weight is not None:
        loss = loss * sample_weight[:, None]
        normalizer = normalizer * sample_weight[:, None]
    return loss.sum() / normalizer.sum().clamp_min(1.0)


def hazard_bellman_losses(
    model: HazardBellmanOperator,
    features: torch.Tensor,
    exit_event: torch.Tensor,
    at_risk: torch.Tensor,
    cumulative_exit: torch.Tensor,
    step_risk: torch.Tensor,
    direction_risk: torch.Tensor,
    severe_threshold: float,
    sample_weight: torch.Tensor | None = None,
    first_exit_weight: float = 1.0,
    cell_ids: torch.Tensor | None = None,
    boundary_ranking_strata: torch.Tensor | None = None,
    targeted_cell_ids: tuple[int, ...] = (),
    boundary_ranking_weight: float = 0.0,
    boundary_margin: float = 0.0,
):
    output = model(features)
    hazard = conditional_hazard_nll(
        output.hazard_logits, exit_event, at_risk, sample_weight, first_exit_weight,
    )
    horizons = torch.as_tensor((0, 1, 3, 7), device=features.device)
    horizon_classification = F.binary_cross_entropy(
        output.cumulative_exit[:, horizons].clamp(1e-6, 1 - 1e-6),
        cumulative_exit[:, horizons].float(),
    )
    log_step_target = torch.log1p(step_risk)
    log_cumulative_target = torch.cummax(log_step_target, dim=1).values
    severity = 0.5 * (
        F.smooth_l1_loss(output.log_step_severity, log_step_target)
        + F.smooth_l1_loss(output.log_cumulative_severity, log_cumulative_target)
    )
    direction = F.smooth_l1_loss(
        output.log_direction_risk, torch.log1p(direction_risk),
    )
    severe = (step_risk > severe_threshold).float()
    false_safe = (
        severe
        * F.relu(torch.log1p(step_risk) - output.log_step_severity).square()
    ).sum() / severe.sum().clamp_min(1.0)
    ranking = torch.zeros((), device=features.device)
    if boundary_ranking_weight:
        if cell_ids is None or len(cell_ids) != len(features):
            raise ValueError("boundary ranking requires one cell id per feature row")
        if boundary_ranking_strata is not None and len(boundary_ranking_strata) != len(features):
            raise ValueError("boundary-ranking strata must match feature rows")
        unsafe_boundary = cumulative_exit[:, 0].bool()
        terms = 0
        for cell in targeted_cell_ids:
            local = cell_ids == int(cell)
            strata = (torch.unique(boundary_ranking_strata[local])
                       if boundary_ranking_strata is not None else (None,))
            for stratum in strata:
                stratum_mask = local if stratum is None else local & (boundary_ranking_strata == stratum)
                unsafe_score = output.cumulative_exit[stratum_mask & unsafe_boundary, 0]
                safe_score = output.cumulative_exit[stratum_mask & ~unsafe_boundary, 0]
                if len(unsafe_score) and len(safe_score):
                    ranking = ranking + F.relu(
                        float(boundary_margin) - (unsafe_score[:, None] - safe_score[None, :])
                    ).mean()
                    terms += 1
        ranking = ranking / max(terms, 1)
    bellman = bellman_residual(output.hazard).square().mean()
    total = (
        hazard + 0.6 * horizon_classification + 0.5 * severity
        + 0.2 * direction + 0.8 * false_safe + bellman
        + float(boundary_ranking_weight) * ranking
    )
    return total, {
        "hazard": hazard,
        "horizon": horizon_classification,
        "severity": severity,
        "direction": direction,
        "false_safe": false_safe,
        "boundary_ranking": ranking,
        "bellman": bellman,
    }


def train_hazard_bellman_operator(
    model: HazardBellmanOperator,
    features: torch.Tensor,
    exit_event: torch.Tensor,
    at_risk: torch.Tensor,
    cumulative_exit: torch.Tensor,
    step_risk: torch.Tensor,
    direction_risk: torch.Tensor,
    *,
    severe_threshold: float,
    epochs: int,
    learning_rate: float,
    validation: tuple[torch.Tensor, ...] | None = None,
    patience: int = 20,
    evaluation_interval: int = 10,
    first_exit_weight: float = 1.0,
    sample_weight: torch.Tensor | None = None,
    cell_ids: torch.Tensor | None = None,
    boundary_ranking_strata: torch.Tensor | None = None,
    targeted_cell_ids: tuple[int, ...] = (),
    boundary_ranking_weight: float = 0.0,
    boundary_margin: float = 0.0,
) -> dict[str, float]:
    model.set_normalization(features)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=1e-5,
    )
    model.train()
    best_state = copy.deepcopy(model.state_dict())
    best_validation = float("inf")
    best_epoch = 0
    stale = 0
    completed_epoch = 0
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad(set_to_none=True)
        loss, _ = hazard_bellman_losses(
            model, features, exit_event, at_risk, cumulative_exit,
            step_risk, direction_risk, severe_threshold,
            sample_weight=sample_weight, cell_ids=cell_ids, targeted_cell_ids=targeted_cell_ids,
            boundary_ranking_strata=boundary_ranking_strata,
            boundary_ranking_weight=boundary_ranking_weight, boundary_margin=boundary_margin,
            first_exit_weight=first_exit_weight,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        completed_epoch = epoch
        if validation is not None and epoch % evaluation_interval == 0:
            model.eval()
            with torch.no_grad():
                validation_loss, _ = hazard_bellman_losses(
                    model, *validation, severe_threshold,
                    first_exit_weight=first_exit_weight,
                )
            value = float(validation_loss)
            if value < best_validation - 1e-6:
                best_validation = value
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
                stale = 0
            else:
                stale += 1
            model.train()
            if stale >= patience:
                break
    if validation is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        total, parts = hazard_bellman_losses(
            model, features, exit_event, at_risk, cumulative_exit,
            step_risk, direction_risk, severe_threshold,
            sample_weight=sample_weight, cell_ids=cell_ids, targeted_cell_ids=targeted_cell_ids,
            boundary_ranking_strata=boundary_ranking_strata,
            boundary_ranking_weight=boundary_ranking_weight, boundary_margin=boundary_margin,
            first_exit_weight=first_exit_weight,
        )
    return {
        "total": float(total),
        **{key: float(value) for key, value in parts.items()},
        "best_internal_validation": best_validation,
        "best_epoch": int(best_epoch),
        "completed_epoch": int(completed_epoch),
    }
