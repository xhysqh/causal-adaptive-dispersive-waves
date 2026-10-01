"""Small conditional mixture-of-experts risk heads for R2.4.2."""

from __future__ import annotations

import torch
from torch import nn

from fgsp_ch.features.m63433524r242_conditional_local_risk import (
    CONDITIONAL_LOCAL_RISK_FEATURE_DIM,
    GEOMETRY_SLICE,
    LOCAL_SLICE,
    PHASE_SLICE,
    R23_SLICE,
)


CONDITIONAL_RISK_INPUT_DIM = CONDITIONAL_LOCAL_RISK_FEATURE_DIM


def _hidden(value: int) -> int:
    return max(int(value), 4)


class TargetCellConditionalRiskHead(nn.Module):
    """Geometry-gated balanced/skewed residual experts over R2.3 context."""

    def __init__(
        self,
        input_channels: int = CONDITIONAL_RISK_INPUT_DIM,
        hidden_channels: int = 64,
        residual_bound: float = 2.0,
    ) -> None:
        super().__init__()
        if input_channels != CONDITIONAL_RISK_INPUT_DIM:
            raise ValueError(f"conditional MoE requires {CONDITIONAL_RISK_INPUT_DIM} channels")
        if hidden_channels <= 0 or residual_bound <= 0.0:
            raise ValueError("network dimensions and residual bound must be positive")
        self.input_channels = int(input_channels)
        self.hidden_channels = int(hidden_channels)
        self.residual_bound = float(residual_bound)
        half = _hidden(hidden_channels // 2)
        self.context = nn.Sequential(
            nn.Linear(R23_SLICE.stop - R23_SLICE.start, hidden_channels),
            nn.SiLU(), nn.Linear(hidden_channels, 1),
        )
        expert_channels = (LOCAL_SLICE.stop - LOCAL_SLICE.start) + (PHASE_SLICE.stop - PHASE_SLICE.start)
        self.balanced_expert = nn.Sequential(
            nn.Linear(expert_channels, half), nn.SiLU(), nn.Linear(half, 1),
        )
        self.skewed_expert = nn.Sequential(
            nn.Linear(expert_channels, half), nn.SiLU(), nn.Linear(half, 1),
        )
        gate_channels = (GEOMETRY_SLICE.stop - GEOMETRY_SLICE.start) + (PHASE_SLICE.stop - PHASE_SLICE.start)
        self.geometry_gate = nn.Sequential(
            nn.Linear(gate_channels, half), nn.SiLU(), nn.Linear(half, 1),
        )

    def _validate(self, features: torch.Tensor) -> None:
        if features.ndim != 2 or features.shape[1] != self.input_channels:
            raise ValueError(
                f"conditional-risk features must be (N, {self.input_channels}), got {tuple(features.shape)}"
            )

    def gate_probability(self, features: torch.Tensor) -> torch.Tensor:
        self._validate(features)
        gate_input = torch.cat((features[:, GEOMETRY_SLICE], features[:, PHASE_SLICE]), dim=1)
        return torch.sigmoid(self.geometry_gate(gate_input).squeeze(-1))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        self._validate(features)
        context = self.context(features[:, R23_SLICE]).squeeze(-1)
        expert_input = torch.cat((features[:, LOCAL_SLICE], features[:, PHASE_SLICE]), dim=1)
        balanced = self.residual_bound * torch.tanh(self.balanced_expert(expert_input).squeeze(-1))
        skewed = self.residual_bound * torch.tanh(self.skewed_expert(expert_input).squeeze(-1))
        gate = self.gate_probability(features)
        return context + (1.0 - gate) * balanced + gate * skewed


class AblationRiskHead(nn.Module):
    """Compact MLP used only for forced development ablations."""

    def __init__(self, input_channels: int, hidden_channels: int = 64) -> None:
        super().__init__()
        if input_channels <= 0 or hidden_channels <= 0:
            raise ValueError("network dimensions must be positive")
        self.input_channels = int(input_channels)
        self.hidden_channels = int(hidden_channels)
        half = _hidden(hidden_channels // 2)
        self.network = nn.Sequential(
            nn.Linear(input_channels, hidden_channels), nn.SiLU(),
            nn.Linear(hidden_channels, half), nn.SiLU(), nn.Linear(half, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2 or features.shape[1] != self.input_channels:
            raise ValueError(f"ablation features must be (N, {self.input_channels})")
        return self.network(features).squeeze(-1)
