"""Unified causal error-budget operator for mCH and Benjamin--Ono.

The learned object in P3 is not a state surrogate.  Equation adapters map
their pre-commit diagnostics to the same dimensionless error coordinates.
An analytical envelope first removes inadmissible actions; the learned model
may only rank the remaining actions and may abstain.  Consequently it cannot
override conservation, introduce an action, or advance either PDE.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from numpy.typing import NDArray
import torch
from torch import nn
from torch.nn import functional as F


EQUATIONS = ("modified_camassa_holm", "benjamin_ono")
FEATURE_ORDER = (
    "log_temporal_over_budget", "log_spatial_over_budget",
    "log_representation_over_budget", "log_migration_over_budget",
    "log_invariant_over_limit", "log_stiffness",
    "remaining_budget_fraction", "remaining_time_fraction",
    "log_relative_work", "previous_rejected",
)


def _log_ratio(value: float, scale: float) -> float:
    return float(np.log(max(float(value), 1.0e-16) / max(float(scale), 1.0e-16)))


def causal_budget_features(
    *, temporal: float, spatial: float, representation: float,
    migration: float, invariant_drift: float, invariant_limit: float,
    stiffness: float, local_budget: float, remaining_budget_fraction: float,
    remaining_time_fraction: float, relative_work: float,
    previous_rejected: bool,
) -> NDArray[np.float64]:
    """Return solver-neutral, dimensionless information available pre-commit."""
    result = np.asarray([
        _log_ratio(temporal, local_budget),
        _log_ratio(spatial, local_budget),
        _log_ratio(representation, local_budget),
        _log_ratio(migration, local_budget),
        _log_ratio(invariant_drift, invariant_limit),
        float(np.log1p(max(stiffness, 0.0))),
        float(remaining_budget_fraction), float(remaining_time_fraction),
        float(np.log(max(relative_work, 1.0e-16))), float(previous_rejected),
    ], dtype=np.float64)
    if result.shape != (len(FEATURE_ORDER),) or not np.all(np.isfinite(result)):
        raise ValueError("invalid unified causal-budget feature vector")
    if not 0.0 <= result[6] <= 1.0 or not 0.0 <= result[7] <= 1.0:
        raise ValueError("remaining fractions must lie in [0,1]")
    return result


@dataclass(frozen=True, slots=True)
class CandidateEvidence:
    action_id: str
    predicted_budget: float
    remaining_budget: float
    invariant_drift: float
    invariant_limit: float
    estimated_work: float
    analytically_admissible: bool = True

    def __post_init__(self) -> None:
        values = (self.predicted_budget, self.remaining_budget,
                  self.invariant_drift, self.invariant_limit, self.estimated_work)
        if not self.action_id or not all(np.isfinite(values)):
            raise ValueError("candidate evidence must be named and finite")
        if self.remaining_budget < 0 or self.invariant_limit <= 0 or self.estimated_work <= 0:
            raise ValueError("invalid candidate budget, invariant limit, or work")


def analytical_safe_set(
    candidates: Iterable[CandidateEvidence], *, reserve_fraction: float,
) -> tuple[CandidateEvidence, ...]:
    """Filter actions before learned inference; rejected actions cannot re-enter."""
    if not 0.0 <= reserve_fraction < 1.0:
        raise ValueError("reserve_fraction must lie in [0,1)")
    safe = tuple(row for row in candidates if (
        row.analytically_admissible
        and row.predicted_budget <= (1.0 - reserve_fraction) * row.remaining_budget
        and row.invariant_drift <= row.invariant_limit
    ))
    return safe


@dataclass(frozen=True, slots=True)
class BudgetOperatorOutput:
    median_log_effectivity: torch.Tensor
    upper_log_effectivity: torch.Tensor
    log_cost_correction: torch.Tensor


class MonotoneCausalBudgetOperator(nn.Module):
    """Small shared operator with a monotone analytical-risk branch.

    The first six coordinates are defect/stiffness coordinates.  Softplus
    parameterisation makes their contribution non-negative.  Equation and
    contextual terms may shift calibration but cannot reverse defect order.
    """

    def __init__(self, *, hidden: int = 32) -> None:
        super().__init__()
        self.hidden = int(hidden)
        self.register_buffer("feature_mean", torch.zeros(1, len(FEATURE_ORDER)))
        self.register_buffer("feature_scale", torch.ones(1, len(FEATURE_ORDER)))
        self.danger_weight = nn.Parameter(torch.zeros(6))
        self.context = nn.Sequential(
            nn.Linear(4 + len(EQUATIONS), hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
        )
        self.median_head = nn.Linear(hidden, 1)
        self.upper_gap_head = nn.Linear(hidden, 1)
        self.cost_head = nn.Linear(hidden, 1)

    @torch.no_grad()
    def set_normalization(self, features: torch.Tensor) -> None:
        if features.ndim != 2 or features.shape[1] != len(FEATURE_ORDER):
            raise ValueError("P3 normalization shape mismatch")
        self.feature_mean.copy_(features.mean(0, keepdim=True))
        self.feature_scale.copy_(features.std(0, keepdim=True, unbiased=False).clamp_min(1e-8))

    def forward(self, features: torch.Tensor, equation_index: torch.Tensor) -> BudgetOperatorOutput:
        if features.ndim != 2 or features.shape[1] != len(FEATURE_ORDER):
            raise ValueError("P3 feature width mismatch")
        if equation_index.shape != (features.shape[0],):
            raise ValueError("P3 equation index must have shape [batch]")
        normalized = (features - self.feature_mean) / self.feature_scale
        danger = (normalized[:, :6] * F.softplus(self.danger_weight)).sum(-1, keepdim=True)
        equation = F.one_hot(equation_index.long(), len(EQUATIONS)).to(features.dtype)
        context = self.context(torch.cat((normalized[:, 6:], equation), dim=-1))
        median = danger + self.median_head(context)
        upper = median + F.softplus(self.upper_gap_head(context))
        return BudgetOperatorOutput(median[:, 0], upper[:, 0], self.cost_head(context)[:, 0])


def budget_operator_loss(
    output: BudgetOperatorOutput, target_log_effectivity: torch.Tensor,
    target_log_cost: torch.Tensor, *, upper_quantile: float = 0.95,
) -> torch.Tensor:
    error = target_log_effectivity - output.upper_log_effectivity
    pinball = torch.maximum(upper_quantile * error, (upper_quantile - 1.0) * error).mean()
    median = F.smooth_l1_loss(output.median_log_effectivity, target_log_effectivity)
    cost = F.smooth_l1_loss(output.log_cost_correction, target_log_cost)
    under = F.relu(error).mean()
    return median + 2.0 * pinball + 0.25 * cost + 3.0 * under


def equation_group_conformal_margins(
    predicted_upper: NDArray[np.floating], truth: NDArray[np.floating],
    equations: Sequence[str], groups: Sequence[str], *, coverage: float,
) -> dict[str, float]:
    """One-sided trajectory-maximum conformal margins, plus worst global margin."""
    pred = np.asarray(predicted_upper, dtype=np.float64)
    target = np.asarray(truth, dtype=np.float64)
    eq = np.asarray(equations).astype(str)
    gid = np.asarray(groups).astype(str)
    if not (pred.shape == target.shape == eq.shape == gid.shape):
        raise ValueError("P3 conformal arrays must be aligned")
    if not 0.5 < coverage < 1.0:
        raise ValueError("coverage must lie in (0.5,1)")
    result: dict[str, float] = {}
    for equation in EQUATIONS:
        scores = np.asarray([
            np.max(target[(eq == equation) & (gid == group)] - pred[(eq == equation) & (gid == group)])
            for group in sorted(set(gid[eq == equation].tolist()))
        ], dtype=np.float64)
        if not len(scores):
            raise ValueError(f"missing conformal groups for {equation}")
        rank = min(len(scores), int(np.ceil((len(scores) + 1) * coverage)))
        result[equation] = float(max(0.0, np.partition(scores, rank - 1)[rank - 1]))
    result["global"] = max(result.values())
    return result


@dataclass(frozen=True, slots=True)
class UnifiedAssessment:
    selected_action_id: str | None
    abstained: bool
    risk_upper: float
    support_distance: float
    safe_action_count: int


class FrozenUnifiedCausalMechanism:
    """Frozen ensemble that ranks only analytically safe actions."""

    def __init__(self, models: Sequence[MonotoneCausalBudgetOperator], manifest: dict,
                 device: torch.device) -> None:
        if len(models) < 3:
            raise ValueError("P3 requires three frozen members")
        self.models = tuple(model.eval() for model in models)
        self.manifest = manifest
        self.device = device

    @classmethod
    def load(cls, root: Path, manifest: dict, device: str | torch.device = "cpu"):
        target = torch.device(device)
        models = []
        for name in manifest["checkpoints"]:
            payload = torch.load(root / name, map_location=target, weights_only=False)
            model = MonotoneCausalBudgetOperator(hidden=payload["hidden"]).to(target)
            model.load_state_dict(payload["model"])
            models.append(model)
        return cls(models, manifest, target)

    @torch.no_grad()
    def assess(self, equation: str, features: NDArray[np.floating],
               candidates: Sequence[CandidateEvidence], *, reserve_fraction: float) -> UnifiedAssessment:
        safe = analytical_safe_set(candidates, reserve_fraction=reserve_fraction)
        if equation not in EQUATIONS or not safe:
            return UnifiedAssessment(None, True, np.inf, np.inf, len(safe))
        vector = np.asarray(features, dtype=np.float64)
        support = self.manifest["support"][equation]
        mean = np.asarray(support["mean"]); scale = np.asarray(support["scale"])
        distance = float(np.sqrt(np.mean(((vector - mean) / np.maximum(scale, 1e-8)) ** 2)))
        if distance > float(support["radius"]):
            return UnifiedAssessment(None, True, np.inf, distance, len(safe))
        x = torch.as_tensor(vector[None], dtype=torch.float32, device=self.device)
        e = torch.as_tensor([EQUATIONS.index(equation)], device=self.device)
        rows = [model(x, e) for model in self.models]
        upper = np.asarray([float(row.upper_log_effectivity[0].cpu()) for row in rows])
        correction = np.asarray([float(row.log_cost_correction[0].cpu()) for row in rows])
        log_upper = float(np.mean(upper) + np.std(upper)
                          + self.manifest["conformal_margins"][equation]
                          + self.manifest["conformal_margins"]["global"])
        risk_upper = float(np.exp(np.clip(log_upper, -32.0, 16.0)))
        if risk_upper > float(self.manifest["maximum_effectivity_upper"]):
            return UnifiedAssessment(None, True, risk_upper, distance, len(safe))
        # Conformal effectivity tightens, but can never relax, the analytical
        # envelope.  This is the actual learned-risk admission boundary.
        adjusted = tuple(row for row in safe if (
            risk_upper * row.predicted_budget
            <= (1.0 - reserve_fraction) * row.remaining_budget
        ))
        if not adjusted:
            return UnifiedAssessment(None, True, risk_upper, distance, len(safe))
        cost_factor = float(np.exp(np.clip(np.mean(correction), -4.0, 4.0)))
        selected = min(adjusted, key=lambda row: (row.estimated_work * cost_factor, row.action_id))
        return UnifiedAssessment(selected.action_id, False, risk_upper, distance, len(safe))
