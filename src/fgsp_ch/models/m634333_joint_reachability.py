from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True, slots=True)
class JointReachabilityOutput:
    log_risk: torch.Tensor
    reach_probability: torch.Tensor
    joint_probability: torch.Tensor
    interval_hazard: torch.Tensor


class JointReachabilityCertificate(nn.Module):
    """One monotone certificate for risk magnitude and first reachability.

    Input is the causal state followed by frozen MoE risk and cumulative exit
    coordinates.  Only ``joint_probability`` is a deployment decision score;
    the other outputs are coupled auxiliary predictions.
    """

    def __init__(
        self,
        input_channels: int = 49,
        hidden: int = 64,
        risk_correction_bound: float = 0.75,
        risk_temperature: float = 5.0,
    ) -> None:
        super().__init__()
        if input_channels < 7 or hidden < 8:
            raise ValueError("joint certificate dimensions are too small.")
        if risk_correction_bound <= 0 or risk_temperature <= 0:
            raise ValueError("joint certificate scales must be positive.")
        self.input_channels = int(input_channels)
        self.risk_correction_bound = float(risk_correction_bound)
        self.risk_temperature = float(risk_temperature)
        self.register_buffer("feature_mean", torch.zeros(input_channels))
        self.register_buffer("feature_scale", torch.ones(input_channels))
        self.encoder = nn.Sequential(
            nn.Linear(input_channels, hidden), nn.SiLU(), nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
        )
        self.risk_residual = nn.Linear(hidden, 3)
        self.hazard_head = nn.Linear(hidden + 3, 3)

    @torch.no_grad()
    def set_normalization(self, context: torch.Tensor) -> None:
        if context.ndim != 2 or context.shape[1] != self.input_channels:
            raise ValueError("context has the wrong shape.")
        self.feature_mean.copy_(context.mean(dim=0))
        self.feature_scale.copy_(
            context.std(dim=0, unbiased=False).clamp_min(1.0e-6)
        )

    def forward(
        self, context: torch.Tensor, frozen_log_risk: torch.Tensor,
    ) -> JointReachabilityOutput:
        if context.ndim != 2 or context.shape[1] != self.input_channels:
            raise ValueError("context has the wrong shape.")
        if frozen_log_risk.shape != (context.shape[0], 3):
            raise ValueError("frozen_log_risk has the wrong shape.")
        normalized = ((context - self.feature_mean) / self.feature_scale).clamp(
            -12.0, 12.0
        )
        latent = self.encoder(normalized)
        correction = self.risk_correction_bound * torch.tanh(
            self.risk_residual(latent)
        )
        log_risk = torch.cummax(
            (frozen_log_risk + correction).clamp_min(0.0), dim=1,
        ).values
        interval_hazard = torch.sigmoid(
            self.hazard_head(torch.cat((latent, log_risk), dim=1))
        )
        reach_probability = 1.0 - torch.cumprod(
            1.0 - interval_hazard.clamp(1.0e-6, 1.0 - 1.0e-6), dim=1,
        )
        # The cumulative hazard is already conditioned on the jointly learned
        # continuous risk coordinates through ``hazard_head``.  It is the one
        # deployment score.  Taking a probabilistic union with a second score
        # for the same exit event would double-count evidence and saturate both
        # safe and unsafe samples near one.
        joint_probability = reach_probability
        return JointReachabilityOutput(
            log_risk, reach_probability, joint_probability, interval_hazard,
        )
