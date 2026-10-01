"""Equation-neutral, restricted-authority causal operator for CMAME-U1."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from numpy.typing import NDArray
import torch
from torch import nn
from torch.nn import functional as F

from fgsp_ch.nonlocal_waves.adaptivity.mechanism_coordinates import (
    MECHANISM_FEATURE_ORDER,
    knn_support_score,
)
from fgsp_ch.nonlocal_waves.adaptivity.prospective_causal_operator import (
    ActionEnvelope,
    ProspectiveOutput,
    analytical_safe_actions,
    causal_prefix_weights,
)


class UniversalCausalBudgetOperator(nn.Module):
    """One mechanism encoder with no equation identifier or equation adapter."""

    def __init__(self, *, hidden: int = 64):
        super().__init__()
        self.hidden = int(hidden)
        width = len(MECHANISM_FEATURE_ORDER)
        self.register_buffer("feature_center", torch.zeros(1, width))
        self.register_buffer("feature_scale", torch.ones(1, width))
        self.danger_weight = nn.Parameter(torch.zeros(6))
        self.encoder = nn.Sequential(
            nn.Linear(width, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
        )
        self.shared = nn.Sequential(
            nn.Linear(hidden + 7, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
        )
        self.median_head = nn.Linear(hidden, 1)
        self.upper_gap_head = nn.Linear(hidden, 1)
        self.cost_head = nn.Linear(hidden, 1)
        self.probe_head = nn.Linear(hidden, 1)

    @torch.no_grad()
    def set_robust_scaler(self, center, scale):
        center_t = torch.as_tensor(center, dtype=self.feature_center.dtype,
                                   device=self.feature_center.device).reshape(1, -1)
        scale_t = torch.as_tensor(scale, dtype=self.feature_scale.dtype,
                                  device=self.feature_scale.device).reshape(1, -1)
        if center_t.shape != self.feature_center.shape or torch.any(scale_t <= 0):
            raise ValueError("invalid U1 robust scaler")
        self.feature_center.copy_(center_t); self.feature_scale.copy_(scale_t)

    def forward(self, features: torch.Tensor) -> ProspectiveOutput:
        if features.ndim != 2 or features.shape[1] != len(MECHANISM_FEATURE_ORDER):
            raise ValueError("U1 feature width mismatch")
        x = (features - self.feature_center) / self.feature_scale
        encoded = self.encoder(x)
        # Remaining budget/time/work/history plus the three availability masks.
        context = self.shared(torch.cat((encoded, x[:, 6:10], x[:, -3:]), dim=-1))
        danger = (x[:, :6] * F.softplus(self.danger_weight)).sum(-1)
        median = danger + self.median_head(context)[:, 0]
        upper = median + F.softplus(self.upper_gap_head(context)[:, 0])
        return ProspectiveOutput(
            median, upper, self.cost_head(context)[:, 0], self.probe_head(context)[:, 0]
        )


def universal_causal_loss(
    output: ProspectiveOutput, target_effectivity: torch.Tensor,
    target_cost: torch.Tensor, target_probe: torch.Tensor,
    groups, prefixes, *, quantile: float, causal_epsilon: float,
) -> torch.Tensor:
    residual = target_effectivity - output.upper_log_effectivity
    pinball = torch.maximum(quantile * residual, (quantile - 1) * residual)
    rows = (
        F.smooth_l1_loss(output.median_log_effectivity, target_effectivity, reduction="none")
        + 2 * pinball + 3 * F.relu(residual)
        + .2 * F.smooth_l1_loss(output.log_cost, target_cost, reduction="none")
        + .2 * F.binary_cross_entropy_with_logits(
            output.probe_logit, target_probe, reduction="none"
        )
    )
    return torch.mean(causal_prefix_weights(
        rows, groups, prefixes, epsilon=causal_epsilon
    ) * rows)


@dataclass(frozen=True, slots=True)
class UniversalDecision:
    action_id: str | None
    abstained: bool
    upper_effectivity: float
    safe_action_count: int
    support_distances: dict[str, float]
    donor_upper_effectivity: dict[str, float]


class FrozenUniversalCausalMechanism:
    feature_kind = "mechanism"

    def __init__(self, models: Sequence[UniversalCausalBudgetOperator], manifest: dict,
                 anchors, device="cpu"):
        if len(models) != 3:
            raise ValueError("U1 requires exactly three frozen members")
        self.models = tuple(model.eval() for model in models)
        self.manifest = manifest
        self.anchors = np.asarray(anchors, dtype=np.float64)
        self.device = torch.device(device)

    @classmethod
    def load(cls, root, manifest: dict, device="cpu"):
        root, target, models = Path(root), torch.device(device), []
        for checkpoint in manifest["checkpoints"]:
            payload = torch.load(root / checkpoint, map_location=target, weights_only=False)
            model = UniversalCausalBudgetOperator(hidden=payload["hidden"]).to(target)
            model.load_state_dict(payload["model"]); models.append(model)
        anchors = np.load(root / manifest["support_anchors"])["anchors"]
        return cls(models, manifest, anchors, target)

    @torch.no_grad()
    def raw_proposal(self, feature_rows: NDArray[np.floating],
                     actions: tuple[ActionEnvelope, ...], *, reserve_fraction: float):
        """Return an uncertified proposal, including outside frozen support.

        This method never grants authority.  It exists so an external analytical
        probe can decide whether a proposal from an unseen equation may be used.
        """
        safe = analytical_safe_actions(actions, reserve_fraction=reserve_fraction)
        rows = np.asarray(feature_rows, dtype=np.float64)
        if rows.shape != (len(actions), len(MECHANISM_FEATURE_ORDER)):
            raise ValueError("one U1 mechanism row is required per action")
        if not safe:
            return UniversalDecision(None, True, np.inf, 0, {}, {})
        safe_indices = [index for index, action in enumerate(actions) if action in safe]
        center = np.asarray(self.manifest["robust_center"], dtype=np.float64)
        scale = np.asarray(self.manifest["robust_scale"], dtype=np.float64)
        transformed = (rows[safe_indices] - center) / scale
        distance = float(np.max(knn_support_score(
            transformed, self.anchors,
            neighbors=int(self.manifest["support_neighbors"]),
        )))
        x = torch.as_tensor(rows[safe_indices], dtype=torch.float32, device=self.device)
        outputs = [model(x) for model in self.models]
        upper = torch.stack([row.upper_log_effectivity for row in outputs]).cpu().numpy()
        cost = torch.stack([row.log_cost for row in outputs]).cpu().numpy()
        log_upper = upper.mean(0) + upper.std(0) + float(self.manifest["conformal_margin"])
        effectivity = np.exp(np.clip(log_upper, -32, 16))
        admitted = [
            (local, action_index)
            for local, action_index in enumerate(safe_indices)
            if effectivity[local] * actions[action_index].deterministic_error
            <= (1 - reserve_fraction) * actions[action_index].remaining_budget
        ]
        pool = admitted if admitted else list(enumerate(safe_indices))
        local, index = min(pool, key=lambda pair: (
            0 if pair in admitted else 1,
            actions[pair[1]].work
            * float(np.exp(np.clip(cost[:, pair[0]].mean(), -4, 4))),
            effectivity[pair[0]] * actions[pair[1]].deterministic_error,
            actions[pair[1]].action_id,
        ))
        return UniversalDecision(
            actions[index].action_id, True, float(effectivity[local]), len(safe),
            {"universal": distance}, {"raw_prediction": float(effectivity[local])},
        )

    @torch.no_grad()
    def choose(self, feature_rows: NDArray[np.floating],
               actions: tuple[ActionEnvelope, ...], *, reserve_fraction: float):
        safe = analytical_safe_actions(actions, reserve_fraction=reserve_fraction)
        rows = np.asarray(feature_rows, dtype=np.float64)
        if rows.shape != (len(actions), len(MECHANISM_FEATURE_ORDER)):
            raise ValueError("one U1 mechanism row is required per action")
        if not safe:
            return UniversalDecision(None, True, np.inf, 0, {}, {})
        safe_indices = [index for index, action in enumerate(actions) if action in safe]
        center = np.asarray(self.manifest["robust_center"], dtype=np.float64)
        scale = np.asarray(self.manifest["robust_scale"], dtype=np.float64)
        transformed = (rows[safe_indices] - center) / scale
        scores = knn_support_score(
            transformed, self.anchors, neighbors=int(self.manifest["support_neighbors"])
        )
        distance = float(np.max(scores))
        if distance > float(self.manifest["support_radius"]):
            return UniversalDecision(
                None, True, np.inf, len(safe), {"universal": distance}, {}
            )
        x = torch.as_tensor(
            rows[safe_indices], dtype=torch.float32, device=self.device
        )
        outputs = [model(x) for model in self.models]
        upper = torch.stack([row.upper_log_effectivity for row in outputs]).cpu().numpy()
        cost = torch.stack([row.log_cost for row in outputs]).cpu().numpy()
        log_upper = upper.mean(0) + upper.std(0) + float(self.manifest["conformal_margin"])
        effectivity = np.exp(np.clip(log_upper, -32, 16))
        admitted = [
            (local, action_index)
            for local, action_index in enumerate(safe_indices)
            if effectivity[local] * actions[action_index].deterministic_error
            <= (1 - reserve_fraction) * actions[action_index].remaining_budget
        ]
        if not admitted:
            return UniversalDecision(
                None, True, float(np.max(effectivity)), len(safe),
                {"universal": distance}, {},
            )
        local, index = min(admitted, key=lambda pair: (
            actions[pair[1]].work
            * float(np.exp(np.clip(cost[:, pair[0]].mean(), -4, 4))),
            actions[pair[1]].action_id,
        ))
        return UniversalDecision(
            actions[index].action_id, False, float(effectivity[local]), len(safe),
            {"universal": distance}, {},
        )
