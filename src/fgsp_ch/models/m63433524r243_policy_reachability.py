"""Monotone multi-horizon reachable-risk head for R2.4.3."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from fgsp_ch.models.m634335_hazard_bellman import cumulative_exit_probability


POLICY_REACHABILITY_INPUT_DIM = 68
MAXIMUM_HORIZON = 4


@dataclass(frozen=True, slots=True)
class PolicyReachabilityOutput:
    hazard_logits: torch.Tensor
    hazard: torch.Tensor
    cumulative_exit: torch.Tensor
    cumulative_severity: torch.Tensor


class PolicyReachabilityHead(nn.Module):
    """R2.3 context plus stable local features; geometry is audit metadata only."""

    def __init__(self, input_channels: int = POLICY_REACHABILITY_INPUT_DIM, hidden_channels: int = 64):
        super().__init__()
        if input_channels != POLICY_REACHABILITY_INPUT_DIM or hidden_channels < 8:
            raise ValueError("invalid R2.4.3 policy-reachability dimensions")
        self.input_channels = int(input_channels)
        self.hidden_channels = int(hidden_channels)
        half = max(hidden_channels // 2, 8)
        self.encoder = nn.Sequential(
            nn.Linear(input_channels, hidden_channels),
            nn.SiLU(),
            nn.LayerNorm(hidden_channels),
            nn.Linear(hidden_channels, half),
            nn.SiLU(),
        )
        self.hazard_head = nn.Linear(half, MAXIMUM_HORIZON)
        self.severity_head = nn.Linear(half, MAXIMUM_HORIZON)

    def forward(self, features: torch.Tensor) -> PolicyReachabilityOutput:
        if features.ndim != 2 or features.shape[1] != self.input_channels:
            raise ValueError(f"policy-reachability features must be (N, {self.input_channels})")
        latent = self.encoder(features)
        hazard_logits = self.hazard_head(latent)
        hazard = torch.sigmoid(hazard_logits)
        severity = torch.cummax(F.softplus(self.severity_head(latent)), dim=1).values
        return PolicyReachabilityOutput(
            hazard_logits=hazard_logits,
            hazard=hazard,
            cumulative_exit=cumulative_exit_probability(hazard),
            cumulative_severity=severity,
        )
