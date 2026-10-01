from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


def cumulative_exit_probability(hazard: torch.Tensor) -> torch.Tensor:
    if hazard.ndim != 2:
        raise ValueError("hazard must have shape (batch, horizon)")
    survival = torch.cumprod((1.0 - hazard).clamp_min(1.0e-7), dim=1)
    return 1.0 - survival


def bellman_residual(hazard: torch.Tensor) -> torch.Tensor:
    """Exact first-step Bellman residual for every horizon >= 2."""
    cumulative = cumulative_exit_probability(hazard)
    tail = torch.stack([
        cumulative_exit_probability(hazard[:, 1:index + 1])[:, -1]
        for index in range(1, hazard.shape[1])
    ], dim=1)
    backup = hazard[:, :1] + (1.0 - hazard[:, :1]) * tail
    return cumulative[:, 1:] - backup


@dataclass(frozen=True, slots=True)
class HazardBellmanOutput:
    hazard_logits: torch.Tensor
    hazard: torch.Tensor
    cumulative_exit: torch.Tensor
    log_step_severity: torch.Tensor
    log_cumulative_severity: torch.Tensor
    log_direction_risk: torch.Tensor
    support_embedding: torch.Tensor


class HazardBellmanOperator(nn.Module):
    """Causal three-branch first-exit, severity and direction operator."""

    def __init__(
        self,
        input_channels: int = 43,
        hidden: int = 64,
        maximum_horizon: int = 8,
        embedding_channels: int = 8,
    ) -> None:
        super().__init__()
        if input_channels != 43 or hidden < 16 or maximum_horizon < 2:
            raise ValueError("invalid hazard-Bellman dimensions")
        self.input_channels = int(input_channels)
        self.maximum_horizon = int(maximum_horizon)
        self.register_buffer("feature_mean", torch.zeros(input_channels))
        self.register_buffer("feature_scale", torch.ones(input_channels))
        half = hidden // 2
        self.candidate_encoder = nn.Sequential(
            nn.Linear(32, hidden), nn.SiLU(), nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden), nn.SiLU(),
        )
        self.geometry_encoder = nn.Sequential(
            nn.Linear(4, half), nn.SiLU(), nn.Linear(half, half), nn.SiLU(),
        )
        self.control_encoder = nn.Sequential(
            nn.Linear(7, half), nn.SiLU(), nn.Linear(half, half), nn.SiLU(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(hidden + 2 * half, hidden), nn.SiLU(),
            nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.SiLU(),
        )
        self.hazard_head = nn.Linear(hidden, maximum_horizon)
        self.severity_head = nn.Linear(hidden, maximum_horizon)
        self.direction_head = nn.Linear(hidden, maximum_horizon)
        self.embedding_head = nn.Linear(hidden, embedding_channels)

    @torch.no_grad()
    def set_normalization(self, features: torch.Tensor) -> None:
        if features.ndim != 2 or features.shape[1] != self.input_channels:
            raise ValueError("features have the wrong shape")
        self.feature_mean.copy_(features.mean(dim=0))
        self.feature_scale.copy_(
            features.std(dim=0, unbiased=False).clamp_min(1.0e-6)
        )

    def forward(self, features: torch.Tensor) -> HazardBellmanOutput:
        if features.ndim != 2 or features.shape[1] != self.input_channels:
            raise ValueError("features have the wrong shape")
        normalized = ((features - self.feature_mean) / self.feature_scale).clamp(-12, 12)
        candidate = self.candidate_encoder(normalized[:, :32])
        geometry = self.geometry_encoder(normalized[:, 35:39])
        control = self.control_encoder(torch.cat((
            normalized[:, 32:35], normalized[:, 39:43],
        ), dim=1))
        latent = self.fusion(torch.cat((candidate, geometry, control), dim=1))
        hazard_logits = self.hazard_head(latent)
        hazard = torch.sigmoid(hazard_logits)
        step_severity = F.softplus(self.severity_head(latent))
        cumulative_severity = torch.cummax(step_severity, dim=1).values
        direction = F.softplus(self.direction_head(latent))
        embedding = F.normalize(self.embedding_head(latent), dim=1)
        return HazardBellmanOutput(
            hazard_logits=hazard_logits,
            hazard=hazard,
            cumulative_exit=cumulative_exit_probability(hazard),
            log_step_severity=step_severity,
            log_cumulative_severity=cumulative_severity,
            log_direction_risk=direction,
            support_embedding=embedding,
        )
