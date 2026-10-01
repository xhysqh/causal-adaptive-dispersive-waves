"""Balanced prospective causal operator shared by mCH and Benjamin--Ono.

F1 learns neither solution values nor a time-step map.  It calibrates the
effectivity and cost of actions that have already passed an equation-specific
analytical safety filter.  The final one-sided conformal envelope can only
tighten that filter.  This separation is the central paper contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
from numpy.typing import NDArray
import torch
from torch import nn
from torch.nn import functional as F


EQUATIONS = ("modified_camassa_holm", "benjamin_ono")
ACTIONS = ("shrink_time", "hold", "refine_space")
FEATURE_ORDER = (
    "log_temporal_over_budget", "log_spatial_over_budget",
    "log_representation_over_budget", "log_migration_over_budget",
    "log_invariant_over_limit", "log_stiffness",
    "remaining_budget_fraction", "remaining_time_fraction",
    "log_relative_work", "previous_rejected", "localized_sharpness",
    "nonlocal_tail", "log_dt_factor", "log_resolution_factor",
    "probe_requested", "representation_change",
)


def _ratio(value: float, scale: float) -> float:
    return float(np.log(max(float(value), 1e-16) / max(float(scale), 1e-16)))


def prospective_features(
    *, temporal: float, spatial: float, representation: float,
    migration: float, invariant_drift: float, invariant_limit: float,
    stiffness: float, local_budget: float, remaining_budget_fraction: float,
    remaining_time_fraction: float, relative_work: float,
    previous_rejected: bool, localized_sharpness: float, nonlocal_tail: float,
    dt_factor: float, resolution_factor: float, probe_requested: bool,
    representation_change: bool,
) -> NDArray[np.float64]:
    """Dimensionless information available before an action is committed."""
    if dt_factor <= 0 or resolution_factor <= 0:
        raise ValueError("action scale factors must be positive")
    row = np.asarray((
        _ratio(temporal, local_budget), _ratio(spatial, local_budget),
        _ratio(representation, local_budget), _ratio(migration, local_budget),
        _ratio(invariant_drift, invariant_limit), np.log1p(max(stiffness, 0.0)),
        remaining_budget_fraction, remaining_time_fraction,
        np.log(max(relative_work, 1e-16)), float(previous_rejected),
        localized_sharpness, nonlocal_tail, np.log(dt_factor),
        np.log(resolution_factor), float(probe_requested),
        float(representation_change),
    ), dtype=np.float64)
    if row.shape != (len(FEATURE_ORDER),) or not np.all(np.isfinite(row)):
        raise ValueError("invalid F1 feature vector")
    if not 0 <= row[6] <= 1 or not 0 <= row[7] <= 1:
        raise ValueError("remaining fractions must lie in [0,1]")
    return row


@dataclass(frozen=True, slots=True)
class ActionEnvelope:
    action_id: str
    deterministic_error: float
    remaining_budget: float
    invariant_drift: float
    invariant_limit: float
    work: float
    analytically_admissible: bool = True

    def __post_init__(self) -> None:
        values = (self.deterministic_error, self.remaining_budget,
                  self.invariant_drift, self.invariant_limit, self.work)
        if self.action_id not in ACTIONS or not np.all(np.isfinite(values)):
            raise ValueError("invalid F1 action envelope")
        if min(self.deterministic_error, self.remaining_budget,
               self.invariant_drift) < 0 or self.invariant_limit <= 0 or self.work <= 0:
            raise ValueError("F1 action envelope values are outside their domain")


def analytical_safe_actions(
    actions: Iterable[ActionEnvelope], *, reserve_fraction: float,
) -> tuple[ActionEnvelope, ...]:
    if not 0 <= reserve_fraction < 1:
        raise ValueError("reserve_fraction must lie in [0,1)")
    return tuple(action for action in actions if (
        action.analytically_admissible
        and action.deterministic_error
        <= (1 - reserve_fraction) * action.remaining_budget
        and action.invariant_drift <= action.invariant_limit
    ))


@dataclass(frozen=True, slots=True)
class ProspectiveOutput:
    median_log_effectivity: torch.Tensor
    upper_log_effectivity: torch.Tensor
    log_cost: torch.Tensor
    probe_logit: torch.Tensor


class SharedProspectiveBudgetOperator(nn.Module):
    """Equation adapters followed by a shared monotone risk trunk."""

    def __init__(self, *, hidden: int = 48) -> None:
        super().__init__()
        self.hidden = int(hidden)
        width = len(FEATURE_ORDER)
        self.register_buffer("feature_mean", torch.zeros(1, width))
        self.register_buffer("feature_scale", torch.ones(1, width))
        self.danger_weight = nn.Parameter(torch.zeros(6))
        self.adapters = nn.ModuleList([
            nn.Sequential(nn.Linear(width, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
            for _ in EQUATIONS
        ])
        self.shared = nn.Sequential(
            nn.Linear(hidden + 4, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
        )
        self.median_head = nn.Linear(hidden, 1)
        self.upper_gap_head = nn.Linear(hidden, 1)
        self.cost_head = nn.Linear(hidden, 1)
        self.probe_head = nn.Linear(hidden, 1)

    @torch.no_grad()
    def set_normalization(self, features: torch.Tensor) -> None:
        if features.ndim != 2 or features.shape[1] != len(FEATURE_ORDER):
            raise ValueError("F1 normalization shape mismatch")
        self.feature_mean.copy_(features.mean(0, keepdim=True))
        self.feature_scale.copy_(features.std(0, keepdim=True, unbiased=False).clamp_min(1e-8))

    def forward(self, features: torch.Tensor, equation_index: torch.Tensor) -> ProspectiveOutput:
        if features.ndim != 2 or features.shape[1] != len(FEATURE_ORDER):
            raise ValueError("F1 feature width mismatch")
        if equation_index.shape != (features.shape[0],):
            raise ValueError("F1 equation index shape mismatch")
        x = (features - self.feature_mean) / self.feature_scale
        adapted = torch.empty((len(x), self.hidden), dtype=x.dtype, device=x.device)
        for index, adapter in enumerate(self.adapters):
            mask = equation_index == index
            if torch.any(mask):
                adapted[mask] = adapter(x[mask])
        context = self.shared(torch.cat((adapted, x[:, 6:10]), dim=-1))
        danger = (x[:, :6] * F.softplus(self.danger_weight)).sum(-1)
        median = danger + self.median_head(context)[:, 0]
        upper = median + F.softplus(self.upper_gap_head(context)[:, 0])
        return ProspectiveOutput(
            median, upper, self.cost_head(context)[:, 0], self.probe_head(context)[:, 0]
        )


def causal_prefix_weights(
    row_loss: torch.Tensor, groups: Sequence[str], prefixes: Sequence[int],
    *, epsilon: float,
) -> torch.Tensor:
    """Stop-gradient causal weights within each physical trajectory."""
    if row_loss.ndim != 1 or not (len(row_loss) == len(groups) == len(prefixes)):
        raise ValueError("causal prefix arrays must be aligned")
    weights = torch.ones_like(row_loss)
    for group in sorted(set(map(str, groups))):
        indices = [i for i, value in enumerate(groups) if str(value) == group]
        indices.sort(key=lambda i: int(prefixes[i]))
        accumulated = torch.zeros((), dtype=row_loss.dtype, device=row_loss.device)
        for index in indices:
            weights[index] = torch.exp(-epsilon * accumulated.detach())
            accumulated = accumulated + row_loss[index]
    return weights


def prospective_loss(
    output: ProspectiveOutput, target_effectivity: torch.Tensor,
    target_cost: torch.Tensor, target_probe: torch.Tensor,
    groups: Sequence[str], prefixes: Sequence[int], *, quantile: float,
    causal_epsilon: float,
) -> torch.Tensor:
    residual = target_effectivity - output.upper_log_effectivity
    pinball = torch.maximum(quantile * residual, (quantile - 1) * residual)
    rows = (
        F.smooth_l1_loss(output.median_log_effectivity, target_effectivity, reduction="none")
        + 2 * pinball + 3 * F.relu(residual)
        + .2 * F.smooth_l1_loss(output.log_cost, target_cost, reduction="none")
        + .2 * F.binary_cross_entropy_with_logits(output.probe_logit, target_probe, reduction="none")
    )
    return torch.mean(causal_prefix_weights(
        rows, groups, prefixes, epsilon=causal_epsilon
    ) * rows)


def grouped_conformal_margins(
    predicted: NDArray[np.floating], truth: NDArray[np.floating],
    equations: Sequence[str], groups: Sequence[str], *, coverage: float,
) -> dict[str, float]:
    pred, target = np.asarray(predicted), np.asarray(truth)
    eq, gid = np.asarray(equations).astype(str), np.asarray(groups).astype(str)
    if not (pred.shape == target.shape == eq.shape == gid.shape):
        raise ValueError("F1 conformal arrays must be aligned")
    result: dict[str, float] = {}
    for equation in EQUATIONS:
        scores = [float(np.max(target[(eq == equation) & (gid == group)]
                                   - pred[(eq == equation) & (gid == group)]))
                  for group in sorted(set(gid[eq == equation]))]
        if not scores:
            raise ValueError(f"missing F1 calibration groups for {equation}")
        rank = min(len(scores), int(np.ceil((len(scores) + 1) * coverage)))
        result[equation] = max(0.0, float(np.partition(scores, rank - 1)[rank - 1]))
    result["global"] = max(result.values())
    return result


@dataclass(frozen=True, slots=True)
class FrozenDecision:
    action_id: str | None
    abstained: bool
    upper_effectivity: float
    safe_action_count: int
    support_distance: float


class FrozenProspectiveMechanism:
    def __init__(self, models: Sequence[SharedProspectiveBudgetOperator], manifest: dict,
                 device: str | torch.device = "cpu") -> None:
        if len(models) != 3:
            raise ValueError("F1 requires exactly three frozen members")
        self.models = tuple(model.eval() for model in models)
        self.manifest = manifest
        self.device = torch.device(device)

    @classmethod
    def load(cls, root, manifest: dict, device="cpu"):
        root = __import__("pathlib").Path(root); target = torch.device(device); models = []
        for name in manifest["checkpoints"]:
            payload = torch.load(root / name, map_location=target, weights_only=False)
            model = SharedProspectiveBudgetOperator(hidden=payload["hidden"]).to(target)
            model.load_state_dict(payload["model"]); models.append(model)
        return cls(models, manifest, target)

    @torch.no_grad()
    def choose(self, equation: str, feature_rows: NDArray[np.floating],
               actions: Sequence[ActionEnvelope], *, reserve_fraction: float) -> FrozenDecision:
        safe = analytical_safe_actions(actions, reserve_fraction=reserve_fraction)
        if equation not in EQUATIONS or not safe:
            return FrozenDecision(None, True, np.inf, len(safe), np.inf)
        rows = np.asarray(feature_rows, dtype=np.float64)
        if rows.shape != (len(actions), len(FEATURE_ORDER)):
            raise ValueError("one F1 feature row is required per action")
        safe_indices = [i for i, action in enumerate(actions) if action in safe]
        support = self.manifest["support"][equation]
        mean, scale = np.asarray(support["mean"]), np.asarray(support["scale"])
        distances = np.sqrt(np.mean(((rows[safe_indices] - mean) / np.maximum(scale, 1e-8)) ** 2, axis=1))
        distance = float(np.max(distances))
        if distance > float(support["radius"]):
            return FrozenDecision(None, True, np.inf, len(safe), distance)
        x = torch.as_tensor(rows[safe_indices], dtype=torch.float32, device=self.device)
        e = torch.full((len(safe_indices),), EQUATIONS.index(equation), device=self.device)
        outputs = [model(x, e) for model in self.models]
        upper = torch.stack([row.upper_log_effectivity for row in outputs]).cpu().numpy()
        cost = torch.stack([row.log_cost for row in outputs]).cpu().numpy()
        log_upper = upper.mean(0) + upper.std(0) + self.manifest["margins"][equation] + self.manifest["margins"]["global"]
        effectivity = np.exp(np.clip(log_upper, -32, 16))
        admitted = [(j, actions[index]) for j, index in enumerate(safe_indices) if
                    effectivity[j] * actions[index].deterministic_error
                    <= (1 - reserve_fraction) * actions[index].remaining_budget]
        if not admitted:
            return FrozenDecision(None, True, float(np.max(effectivity)), len(safe), distance)
        selected_j, selected = min(admitted, key=lambda pair: (
            pair[1].work * float(np.exp(np.clip(cost[:, pair[0]].mean(), -4, 4))),
            pair[1].action_id,
        ))
        return FrozenDecision(selected.action_id, False, float(effectivity[selected_j]), len(safe), distance)
