from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from fgsp_ch.models.m634331_horizon_risk import HorizonRiskOutput


@dataclass(frozen=True, slots=True)
class SeparatedMoEOutput:
    prediction: HorizonRiskOutput
    gate: torch.Tensor
    risk_correction: torch.Tensor
    exit_correction: torch.Tensor


class SeparatedMoEResidual(nn.Module):
    """Bounded residual experts for separated multipeakon risk.

    The gate only routes between experts.  It is deliberately not a safety
    decision; calibrated risk, exit and support gates remain downstream.
    """

    geometry_indices = (35, 36, 37, 38, 39, 40, 41, 42)

    def __init__(
        self,
        input_channels: int = 43,
        hidden: int = 48,
        experts: int = 3,
        correction_bound: float = 1.5,
        temperature: float = 0.7,
    ) -> None:
        super().__init__()
        if input_channels <= max(self.geometry_indices):
            raise ValueError("input_channels do not contain separated geometry.")
        if hidden < 8 or experts < 2 or correction_bound <= 0 or temperature <= 0:
            raise ValueError("invalid separated MoE configuration.")
        self.input_channels = int(input_channels)
        self.expert_count = int(experts)
        self.correction_bound = float(correction_bound)
        self.temperature = float(temperature)
        self.register_buffer("feature_mean", torch.zeros(input_channels))
        self.register_buffer("feature_scale", torch.ones(input_channels))
        geometry_channels = len(self.geometry_indices)
        self.gate_network = nn.Sequential(
            nn.Linear(geometry_channels, hidden // 2), nn.SiLU(),
            nn.Linear(hidden // 2, experts),
        )
        self.expert_networks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_channels, hidden), nn.SiLU(),
                nn.LayerNorm(hidden),
                nn.Linear(hidden, hidden), nn.SiLU(),
                nn.Linear(hidden, 13),
            )
            for _ in range(experts)
        ])

    @torch.no_grad()
    def set_normalization(self, features: torch.Tensor) -> None:
        if features.ndim != 2 or features.shape[1] != self.input_channels:
            raise ValueError("features have the wrong shape.")
        self.feature_mean.copy_(features.mean(dim=0))
        self.feature_scale.copy_(
            features.std(dim=0, unbiased=False).clamp_min(1.0e-6)
        )

    def forward(
        self, features: torch.Tensor, base: HorizonRiskOutput,
    ) -> SeparatedMoEOutput:
        if features.ndim != 2 or features.shape[1] != self.input_channels:
            raise ValueError("features have the wrong shape.")
        normalized = ((features - self.feature_mean) / self.feature_scale).clamp(
            -12.0, 12.0
        )
        geometry = normalized[:, self.geometry_indices]
        gate = torch.softmax(self.gate_network(geometry) / self.temperature, dim=1)
        expert = torch.stack(
            [network(normalized) for network in self.expert_networks], dim=1,
        )
        mixed = torch.sum(gate.unsqueeze(-1) * expert, dim=1)
        bounded = self.correction_bound * torch.tanh(mixed)
        risk_correction = bounded[:, :3]
        # cummax is the metric projection onto the ordered horizon cone used
        # here; it preserves causality and prevents a longer horizon being
        # certified safer than a shorter one.
        risk = torch.cummax(
            (base.log_risk + risk_correction).clamp_min(0.0), dim=1,
        ).values
        exit_correction = bounded[:, 3:12]
        prediction = HorizonRiskOutput(
            risk,
            base.exit_logits + exit_correction,
            base.boundary_logit + bounded[:, 12],
        )
        return SeparatedMoEOutput(
            prediction, gate, risk_correction, exit_correction,
        )

    def conditional_forward(
        self,
        features: torch.Tensor,
        base: HorizonRiskOutput,
        separated: torch.Tensor,
    ) -> HorizonRiskOutput:
        """Apply the residual only to explicitly separated states.

        Non-separated rows are selected directly from ``base`` so their
        floating-point values and gradients cannot be changed by this branch.
        """
        if separated.ndim != 1 or separated.shape[0] != features.shape[0]:
            raise ValueError("separated mask has the wrong shape.")
        residual = self(features, base).prediction
        mask = separated.to(dtype=torch.bool, device=features.device)
        return HorizonRiskOutput(
            torch.where(mask[:, None], residual.log_risk, base.log_risk),
            torch.where(mask[:, None], residual.exit_logits, base.exit_logits),
            torch.where(mask, residual.boundary_logit, base.boundary_logit),
        )
