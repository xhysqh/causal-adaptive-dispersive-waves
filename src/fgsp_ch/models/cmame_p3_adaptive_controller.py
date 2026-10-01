"""Joint local mesh-action and global time-step controller for CMAME-P3."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from fgsp_ch.models.cmame_conservative_flux_operator import FluxResidualBlock, PeriodicConv1d


@dataclass(frozen=True, slots=True)
class P3ControllerOutput:
    action_logits: torch.Tensor
    log_local_error: torch.Tensor
    time_multiplier: torch.Tensor


class MCHSpaceTimeAdaptiveController(nn.Module):
    """Periodic controller; deterministic constraints remain outside the net."""

    def __init__(self, *, feature_channels: int = 24, parameter_channels: int = 10, hidden: int = 32) -> None:
        super().__init__()
        self.feature_channels = feature_channels
        self.parameter_channels = parameter_channels
        self.input = PeriodicConv1d(feature_channels, hidden, 5)
        self.condition = nn.Sequential(nn.Linear(parameter_channels, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.blocks = nn.ModuleList(FluxResidualBlock(hidden, dilation) for dilation in (1, 2, 4))
        self.action = PeriodicConv1d(hidden, 3, 3)
        self.error = PeriodicConv1d(hidden, 1, 3)
        # Channel 23 is the strictly pre-commit deterministic error proxy.  The
        # learned head predicts only its log-space discrepancy.  This keeps the
        # estimate positive, scale equivariant and substantially easier to
        # identify than an unconstrained absolute error field.
        self.error_log_gain = nn.Parameter(torch.ones(()))
        self.error_log_bias = nn.Parameter(torch.zeros(()))
        self.time = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 1))
        self.register_buffer("feature_mean", torch.zeros(1, feature_channels, 1))
        self.register_buffer("feature_scale", torch.ones(1, feature_channels, 1))
        self.register_buffer("parameter_mean", torch.zeros(1, parameter_channels))
        self.register_buffer("parameter_scale", torch.ones(1, parameter_channels))

    @torch.no_grad()
    def set_normalization(self, features: torch.Tensor, parameters: torch.Tensor) -> None:
        self.feature_mean.copy_(features.mean((0, 2), keepdim=True))
        self.feature_scale.copy_(features.std((0, 2), keepdim=True, unbiased=False).clamp_min(1e-8))
        self.parameter_mean.copy_(parameters.mean(0, keepdim=True))
        self.parameter_scale.copy_(parameters.std(0, keepdim=True, unbiased=False).clamp_min(1e-8))

    def forward(self, features: torch.Tensor, parameters: torch.Tensor, *, ablation: str = "full") -> P3ControllerOutput:
        normalized = (features - self.feature_mean) / self.feature_scale
        if ablation == "no_uncertainty":
            normalized = normalized.clone(); normalized[:, 21] = 0.0
        elif ablation == "no_particle_geometry":
            normalized = normalized.clone(); normalized[:, 10:20] = 0.0
        hidden = self.input(normalized)
        conditioned = (parameters - self.parameter_mean) / self.parameter_scale
        hidden = hidden + self.condition(conditioned)[:, :, None]
        for block in self.blocks:
            hidden = block(hidden)
        logits = self.action(hidden)
        proxy_log_error = torch.log(features[:, 23].abs().clamp_min(1.0e-8))
        log_error = (
            self.error_log_gain * proxy_log_error
            + self.error_log_bias
            + self.error(hidden)[:, 0]
        )
        pooled = hidden.mean(dim=-1)
        # Bounded proposal in [1/2, 2]; the deterministic controller may reduce it.
        multiplier = torch.exp(torch.tanh(self.time(pooled)[:, 0]) * torch.log(torch.tensor(2.0, device=hidden.device)))
        return P3ControllerOutput(logits, log_error, multiplier)
