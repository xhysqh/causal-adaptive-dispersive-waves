"""Finite-horizon causal risk surrogate used by CMAME V2 stage 3.

The module does not select a discretization or advance a PDE state.  Given a
pre-decision V2 coordinate row, it estimates the logarithm of the maximum
two-step relative budget risk for an already enumerated candidate action.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from fgsp_ch.nonlocal_waves.adaptivity.mechanism_coordinates import V2_CAUSAL_FEATURE_ORDER


@dataclass(frozen=True, slots=True)
class PredictiveRiskOutput:
    """Member-level mean, one-sided spread, cost, and probe diagnostic."""

    mean_log_risk: torch.Tensor
    upper_log_risk: torch.Tensor
    log_risk_spread: torch.Tensor
    log_cost: torch.Tensor
    probe_logit: torch.Tensor


class V2PredictiveRiskOperator(nn.Module):
    """Equation-neutral, non-anticipative finite-horizon risk model.

    Equation labels are intentionally absent.  Explicit availability masks in
    the base mechanism coordinates remain part of the input contract.
    """

    def __init__(self, *, hidden: int = 64, input_width: int | None = None) -> None:
        super().__init__()
        self.hidden = int(hidden)
        width = len(V2_CAUSAL_FEATURE_ORDER) if input_width is None else int(input_width)
        if width < 1 or width > len(V2_CAUSAL_FEATURE_ORDER):
            raise ValueError("invalid V2 predictive-risk input width")
        self.input_width = width
        self.register_buffer("feature_center", torch.zeros(1, width))
        self.register_buffer("feature_scale", torch.ones(1, width))
        self.encoder = nn.Sequential(
            nn.Linear(width, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
        )
        self.mean_head = nn.Linear(hidden, 1)
        self.spread_head = nn.Linear(hidden, 1)
        self.cost_head = nn.Linear(hidden, 1)
        self.probe_head = nn.Linear(hidden, 1)

    @torch.no_grad()
    def set_robust_scaler(self, center, scale) -> None:
        center_t = torch.as_tensor(center, dtype=self.feature_center.dtype,
                                   device=self.feature_center.device).reshape(1, -1)
        scale_t = torch.as_tensor(scale, dtype=self.feature_scale.dtype,
                                  device=self.feature_scale.device).reshape(1, -1)
        if center_t.shape != self.feature_center.shape or torch.any(scale_t <= 0):
            raise ValueError("invalid V2 robust scaler")
        self.feature_center.copy_(center_t)
        self.feature_scale.copy_(scale_t)

    def forward(self, features: torch.Tensor) -> PredictiveRiskOutput:
        if features.ndim != 2 or features.shape[1] != self.input_width:
            raise ValueError("V2 predictive-risk feature width mismatch")
        encoded = self.encoder((features - self.feature_center) / self.feature_scale)
        mean = self.mean_head(encoded)[:, 0]
        log_spread = self.spread_head(encoded)[:, 0].clamp(-12.0, 8.0)
        # A member can only enlarge its mean forecast when proposing an upper
        # risk envelope; subsequent ensemble and conformal terms also enlarge it.
        upper = mean + F.softplus(log_spread)
        return PredictiveRiskOutput(
            mean_log_risk=mean,
            upper_log_risk=upper,
            log_risk_spread=log_spread,
            log_cost=self.cost_head(encoded)[:, 0],
            probe_logit=self.probe_head(encoded)[:, 0],
        )


def predictive_risk_loss(
    output: PredictiveRiskOutput, target_log_risk: torch.Tensor,
    target_log_cost: torch.Tensor, target_probe: torch.Tensor, *,
    upper_quantile: float, underestimate_weight: float,
    cost_weight: float, probe_weight: float, probe_positive_weight: float,
) -> torch.Tensor:
    """One-sided future-risk loss with an explicit penalty for underestimation."""
    if not 0.5 < upper_quantile < 1.0 or underestimate_weight < 1.0:
        raise ValueError("invalid V2 risk-loss parameters")
    residual = target_log_risk - output.upper_log_risk
    pinball = torch.maximum(upper_quantile * residual, (upper_quantile - 1.0) * residual)
    risk = (
        F.smooth_l1_loss(output.mean_log_risk, target_log_risk, reduction="none")
        + pinball
        + float(underestimate_weight) * F.relu(residual)
    )
    cost = F.smooth_l1_loss(output.log_cost, target_log_cost, reduction="none")
    positive_weight = torch.as_tensor(
        float(probe_positive_weight), dtype=target_probe.dtype, device=target_probe.device,
    )
    probe = F.binary_cross_entropy_with_logits(
        output.probe_logit, target_probe, reduction="none", pos_weight=positive_weight,
    )
    return torch.mean(risk + float(cost_weight) * cost + float(probe_weight) * probe)
