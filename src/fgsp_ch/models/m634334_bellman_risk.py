from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True, slots=True)
class BellmanRiskOutput:
    log_value: torch.Tensor


class BellmanRiskOperator(nn.Module):
    """Monotone V_1,V_2,V_4,V_8 discrete reachable-risk value function."""

    def __init__(self, input_channels: int = 43, hidden: int = 64) -> None:
        super().__init__()
        if input_channels < 1 or hidden < 8:
            raise ValueError("invalid Bellman risk dimensions.")
        self.input_channels = int(input_channels)
        self.register_buffer("feature_mean", torch.zeros(input_channels))
        self.register_buffer("feature_scale", torch.ones(input_channels))
        self.encoder = nn.Sequential(
            nn.Linear(input_channels, hidden), nn.SiLU(), nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
        )
        self.value_head = nn.Linear(hidden, 4)

    @torch.no_grad()
    def set_normalization(self, features: torch.Tensor) -> None:
        if features.ndim != 2 or features.shape[1] != self.input_channels:
            raise ValueError("features have the wrong shape.")
        self.feature_mean.copy_(features.mean(dim=0))
        self.feature_scale.copy_(
            features.std(dim=0, unbiased=False).clamp_min(1.0e-6)
        )

    def forward(self, features: torch.Tensor) -> BellmanRiskOutput:
        normalized = ((features - self.feature_mean) / self.feature_scale).clamp(
            -12.0, 12.0
        )
        raw = self.value_head(self.encoder(normalized))
        first = F.softplus(raw[:, :1])
        increments = F.softplus(raw[:, 1:])
        return BellmanRiskOutput(torch.cat((
            first, first + torch.cumsum(increments, dim=1),
        ), dim=1))
