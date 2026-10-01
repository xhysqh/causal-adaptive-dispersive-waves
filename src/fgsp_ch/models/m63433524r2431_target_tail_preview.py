"""Three-branch monotone hazard network for R2.4.3.1."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from fgsp_ch.models.m634335_hazard_bellman import cumulative_exit_probability

TARGET_TAIL_INPUT_DIM = 84


class TargetTailPreviewHead(nn.Module):
    """Separate R2.3, local-state and target-tail encoders; no geometry input."""
    def __init__(self, input_channels: int = TARGET_TAIL_INPUT_DIM, hidden_channels: int = 64):
        super().__init__()
        if input_channels != TARGET_TAIL_INPUT_DIM or hidden_channels < 12:
            raise ValueError("R2.4.3.1 expects 84 inputs and at least 12 hidden channels")
        self.input_channels, self.hidden_channels = input_channels, hidden_channels
        branch = max(hidden_channels // 2, 8)
        def encoder(channels: int) -> nn.Sequential:
            return nn.Sequential(nn.Linear(channels, branch), nn.SiLU(), nn.LayerNorm(branch), nn.Linear(branch, branch), nn.SiLU())
        self.r23_encoder, self.local_encoder, self.preview_encoder = encoder(60), encoder(8), encoder(16)
        self.fusion = nn.Sequential(nn.Linear(branch * 3, hidden_channels), nn.SiLU(), nn.LayerNorm(hidden_channels))
        self.hazard_head, self.severity_head = nn.Linear(hidden_channels, 4), nn.Linear(hidden_channels, 4)

    def forward(self, features: torch.Tensor):
        if features.ndim != 2 or features.shape[1] != self.input_channels:
            raise ValueError("R2.4.3.1 features must have shape (N, 84)")
        latent = self.fusion(torch.cat((self.r23_encoder(features[:, :60]), self.local_encoder(features[:, 60:68]), self.preview_encoder(features[:, 68:])), dim=1))
        hazard_logits = self.hazard_head(latent)
        hazard = torch.sigmoid(hazard_logits)
        severity = torch.cummax(F.softplus(self.severity_head(latent)), dim=1).values
        return hazard_logits, cumulative_exit_probability(hazard), severity
