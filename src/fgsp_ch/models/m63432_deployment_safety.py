from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True, slots=True)
class DeploymentSafetyOutput:
    unsafe_logit: torch.Tensor
    log_defect: torch.Tensor
    direction: torch.Tensor


class DeploymentCandidateSafetyOperator(nn.Module):
    """Small multi-task certificate for the fully assembled AMR candidate.

    Inputs are causal diagnostics available after proposal, conservative
    backbone, projection, and optional patch migration.  Classical shadows
    are labels only and are never inputs to this operator.
    """

    def __init__(self, input_channels: int = 32, hidden: int = 32) -> None:
        super().__init__()
        if input_channels < 1 or hidden < 4:
            raise ValueError("input_channels and hidden are too small.")
        self.input_channels = int(input_channels)
        self.hidden = int(hidden)
        self.register_buffer("feature_mean", torch.zeros(input_channels))
        self.register_buffer("feature_scale", torch.ones(input_channels))
        self.encoder = nn.Sequential(
            nn.Linear(input_channels, hidden), nn.SiLU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden // 2), nn.SiLU(),
        )
        width = hidden // 2
        self.unsafe_head = nn.Linear(width, 1)
        self.defect_head = nn.Linear(width, 1)
        self.direction_head = nn.Linear(width, 1)

    @torch.no_grad()
    def set_normalization(self, features: torch.Tensor) -> None:
        if features.ndim != 2 or features.shape[1] != self.input_channels:
            raise ValueError("features have the wrong shape.")
        self.feature_mean.copy_(features.mean(dim=0))
        self.feature_scale.copy_(
            features.std(dim=0, unbiased=False).clamp_min(1.0e-6)
        )

    def forward(self, features: torch.Tensor) -> DeploymentSafetyOutput:
        if features.ndim != 2 or features.shape[1] != self.input_channels:
            raise ValueError("features have the wrong shape.")
        normalized = ((features - self.feature_mean) / self.feature_scale).clamp(
            -12.0, 12.0
        )
        latent = self.encoder(normalized)
        unsafe = self.unsafe_head(latent).squeeze(-1)
        defect = F.softplus(self.defect_head(latent).squeeze(-1))
        direction = torch.tanh(self.direction_head(latent).squeeze(-1))
        return DeploymentSafetyOutput(unsafe, defect, direction)
