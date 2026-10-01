from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True, slots=True)
class HorizonRiskOutput:
    log_risk: torch.Tensor
    exit_logits: torch.Tensor
    boundary_logit: torch.Tensor


class HorizonRiskToGoOperator(nn.Module):
    """Monotone K=2,4,8 risk-to-go and first-exit predictor."""

    def __init__(self, input_channels: int = 43, hidden: int = 64) -> None:
        super().__init__()
        if input_channels < 1 or hidden < 8:
            raise ValueError("input_channels and hidden are too small.")
        self.input_channels = int(input_channels)
        self.hidden = int(hidden)
        self.register_buffer("feature_mean", torch.zeros(input_channels))
        self.register_buffer("feature_scale", torch.ones(input_channels))
        self.encoder = nn.Sequential(
            nn.Linear(input_channels, hidden), nn.SiLU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
        )
        self.risk_head = nn.Linear(hidden, 3)
        self.exit_head = nn.Linear(hidden, 9)
        self.boundary_head = nn.Linear(hidden, 1)

    @torch.no_grad()
    def set_normalization(self, features: torch.Tensor) -> None:
        if features.ndim != 2 or features.shape[1] != self.input_channels:
            raise ValueError("features have the wrong shape.")
        self.feature_mean.copy_(features.mean(dim=0))
        self.feature_scale.copy_(
            features.std(dim=0, unbiased=False).clamp_min(1.0e-6)
        )

    def encode(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2 or features.shape[1] != self.input_channels:
            raise ValueError("features have the wrong shape.")
        normalized = ((features - self.feature_mean) / self.feature_scale).clamp(
            -12.0, 12.0
        )
        return self.encoder(normalized)

    def forward(self, features: torch.Tensor) -> HorizonRiskOutput:
        latent = self.encode(features)
        raw = self.risk_head(latent)
        first = F.softplus(raw[:, :1])
        increments = F.softplus(raw[:, 1:])
        log_risk = torch.cat((first, first + torch.cumsum(increments, dim=1)), dim=1)
        return HorizonRiskOutput(
            log_risk,
            self.exit_head(latent),
            self.boundary_head(latent).squeeze(-1),
        )
