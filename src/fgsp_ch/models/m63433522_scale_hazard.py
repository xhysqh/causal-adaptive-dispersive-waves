from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from fgsp_ch.models.m634335_hazard_bellman import (
    HazardBellmanOutput,
    cumulative_exit_probability,
)


class ScaleConsistentHazardOperator(nn.Module):
    """First-exit hazard operator with causal, scale-conditioned FiLM.

    The physical branch is shared across all registered mesh/time-step cells.
    The final five frozen-scale features only modulate that branch; they cannot
    replace it with nine unrelated per-cell classifiers.
    """

    def __init__(
        self,
        input_channels: int = 48,
        hidden: int = 64,
        maximum_horizon: int = 8,
        embedding_channels: int = 8,
        conservative_scale_residual: bool = False,
        residual_bound: float = 2.0,
    ) -> None:
        super().__init__()
        if input_channels not in (48, 52, 60) or hidden < 16 or maximum_horizon < 2:
            raise ValueError("invalid scale-hazard dimensions")
        self.input_channels = int(input_channels)
        self.maximum_horizon = int(maximum_horizon)
        # 52-channel checkpoints remain byte-for-byte architecturally
        # compatible.  The optional final eight channels are a separate
        # pre-commit candidate/fallback contrast branch used only by R2.3.
        self.history_channels = min(int(input_channels - 48), 4)
        self.contrast_channels = max(int(input_channels - 52), 0)
        self.conservative_scale_residual = bool(conservative_scale_residual)
        if residual_bound <= 0.0:
            raise ValueError("residual_bound must be positive")
        self.residual_bound = float(residual_bound)
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
        # rollout, support, alpha, gamma, patch fraction, detail RMS.  The old
        # deployment-risk channel is intentionally not connected.
        self.control_encoder = nn.Sequential(
            nn.Linear(6, half), nn.SiLU(), nn.Linear(half, half), nn.SiLU(),
        )
        self.scale_modulator = nn.Sequential(
            nn.Linear(5, half), nn.SiLU(), nn.Linear(half, 2 * hidden),
        )
        self.history_encoder = None if not self.history_channels else nn.Sequential(
            nn.Linear(self.history_channels, half), nn.SiLU(),
            nn.Linear(half, half), nn.SiLU(),
        )
        self.contrast_encoder = None if not self.contrast_channels else nn.Sequential(
            nn.Linear(self.contrast_channels, half), nn.SiLU(),
            nn.Linear(half, half), nn.SiLU(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(
                hidden + 2 * half
                + (half if self.history_channels else 0)
                + (half if self.contrast_channels else 0),
                hidden,
            ),
            nn.SiLU(), nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden), nn.SiLU(),
        )
        self.hazard_head = nn.Linear(hidden, maximum_horizon)
        self.severity_head = nn.Linear(hidden, maximum_horizon)
        self.direction_head = nn.Linear(hidden, maximum_horizon)
        self.embedding_head = nn.Linear(hidden, embedding_channels)
        if self.conservative_scale_residual:
            residual_input = hidden + 5 + self.history_channels + self.contrast_channels
            self.scale_residual_head = nn.Sequential(
                nn.Linear(residual_input, half), nn.SiLU(),
                nn.Linear(half, maximum_horizon),
            )
            # The residual starts near zero and can only inflate risk as data
            # demonstrates an unresolved local scale boundary.
            nn.init.constant_(self.scale_residual_head[-1].bias, -6.0)
        else:
            self.scale_residual_head = None

    @torch.no_grad()
    def set_normalization(self, features: torch.Tensor) -> None:
        if features.ndim != 2 or features.shape[1] != self.input_channels:
            raise ValueError("features have the wrong shape")
        self.feature_mean.copy_(features.mean(dim=0))
        self.feature_scale.copy_(features.std(dim=0, unbiased=False).clamp_min(1.0e-6))

    def forward(self, features: torch.Tensor) -> HazardBellmanOutput:
        if features.ndim != 2 or features.shape[1] != self.input_channels:
            raise ValueError("features have the wrong shape")
        value = ((features - self.feature_mean) / self.feature_scale).clamp(-12, 12)
        candidate = self.candidate_encoder(value[:, :32])
        geometry = self.geometry_encoder(value[:, 35:39])
        control = self.control_encoder(torch.cat((
            value[:, 32:34], value[:, 39:43],
        ), dim=1))
        physical_features = [candidate, geometry, control]
        if self.history_encoder is not None:
            physical_features.append(self.history_encoder(value[:, 48:52]))
        if self.contrast_encoder is not None:
            physical_features.append(self.contrast_encoder(value[:, 52:]))
        physical = self.fusion(torch.cat(physical_features, dim=1))
        gamma, beta = self.scale_modulator(value[:, 43:48]).chunk(2, dim=1)
        # Bounded modulation makes interpolation across h/dt cells stable and
        # prevents a scale embedding from overwhelming the physical state.
        latent = physical * (1.0 + 0.25 * torch.tanh(gamma)) + 0.25 * beta
        residual_inputs = [latent, value[:, 43:48]]
        if self.history_channels:
            residual_inputs.append(value[:, 48:52])
        if self.contrast_channels:
            residual_inputs.append(value[:, 52:])
        base_logits = self.hazard_head(latent)
        if self.scale_residual_head is None:
            conservative_residual = torch.zeros_like(base_logits)
        else:
            raw_residual = self.scale_residual_head(torch.cat(residual_inputs, dim=1))
            conservative_residual = self.residual_bound * torch.tanh(F.softplus(raw_residual))
        logits = base_logits + conservative_residual
        hazard = torch.sigmoid(logits)
        step_severity = F.softplus(self.severity_head(latent)) + conservative_residual
        return HazardBellmanOutput(
            hazard_logits=logits,
            hazard=hazard,
            cumulative_exit=cumulative_exit_probability(hazard),
            log_step_severity=step_severity,
            log_cumulative_severity=torch.cummax(step_severity, dim=1).values,
            log_direction_risk=F.softplus(self.direction_head(latent)) + conservative_residual,
            support_embedding=F.normalize(self.embedding_head(latent), dim=1),
        )
