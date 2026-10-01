"""Geometry-conditioned conservative interface-flux operator for CMAME-P2.2."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from fgsp_ch.models.cmame_conservative_flux_operator import FluxResidualBlock, PeriodicConv1d


@dataclass(frozen=True, slots=True)
class InterfaceFluxOutput:
    normalized_face_flux: torch.Tensor
    clustered_weight: torch.Tensor
    structural_gate: torch.Tensor


class MCHConservativeInterfaceFluxOperator(nn.Module):
    """Predict a local zero-mean flux, with exact smooth/particle reductions."""

    def __init__(
        self, *, feature_channels: int = 20, parameter_channels: int = 10,
        hidden: int = 32, dilations: tuple[int, ...] = (1, 2, 4),
        cluster_threshold: float = 0.16, cluster_temperature: float = 0.025,
    ) -> None:
        super().__init__()
        if hidden % 4:
            raise ValueError("hidden must be divisible by four")
        self.feature_channels = feature_channels
        self.parameter_channels = parameter_channels
        self.cluster_threshold = float(cluster_threshold)
        self.cluster_temperature = float(cluster_temperature)
        self.input = PeriodicConv1d(feature_channels, hidden, 5)
        self.condition = nn.Sequential(
            nn.Linear(parameter_channels, hidden), nn.SiLU(), nn.Linear(hidden, hidden)
        )
        self.blocks = nn.ModuleList(FluxResidualBlock(hidden, dilation) for dilation in dilations)
        self.separated_head = PeriodicConv1d(hidden, 1, 3)
        self.clustered_head = PeriodicConv1d(hidden, 1, 3)
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

    def forward(
        self, features: torch.Tensor, parameters: torch.Tensor, *, ablation: str = "full"
    ) -> InterfaceFluxOutput:
        if features.ndim != 3 or features.shape[1] != self.feature_channels:
            raise ValueError("features must be [batch, channels, points]")
        if parameters.shape != (features.shape[0], self.parameter_channels):
            raise ValueError("parameters have the wrong shape")
        values = (features - self.feature_mean) / self.feature_scale
        if ablation == "no_interaction_features":
            values = values.clone(); values[:, 10:] = 0.0
        hidden = self.input(values)
        conditioned = (parameters - self.parameter_mean) / self.parameter_scale
        hidden = hidden + self.condition(conditioned)[:, :, None]
        for block in self.blocks:
            hidden = block(hidden)
        separated = self.separated_head(hidden)[:, 0]
        clustered = self.clustered_head(hidden)[:, 0]
        minimum_separation_over_length = features[:, 18, 0]
        cluster_weight = torch.sigmoid(
            (self.cluster_threshold - minimum_separation_over_length)
            / self.cluster_temperature
        )
        if ablation == "single_expert":
            cluster_weight = torch.zeros_like(cluster_weight)
        raw = (1.0 - cluster_weight[:, None]) * separated + cluster_weight[:, None] * clustered
        particle_present = (parameters[:, 6] > 0.0).to(raw.dtype)
        # The dimensionless regular-momentum channel is exactly zero for a
        # pure atomic state and avoids dtype-dependent tiny-scale thresholds.
        regular_present = (features[:, 3].abs().amax(dim=-1) > 1.0e-10).to(raw.dtype)
        structural_gate = particle_present * regular_present
        interface_mask = features[:, 19].clamp(0.0, 1.0)
        if ablation == "no_mask":
            interface_mask = torch.ones_like(interface_mask)
        flux = structural_gate[:, None] * interface_mask * raw
        flux = flux - flux.mean(dim=-1, keepdim=True)
        return InterfaceFluxOutput(flux, cluster_weight, structural_gate)

    @staticmethod
    def conservative_rhs(normalized_flux: torch.Tensor, h: torch.Tensor | float) -> torch.Tensor:
        spacing = torch.as_tensor(h, dtype=normalized_flux.dtype, device=normalized_flux.device)
        if spacing.ndim == 1:
            spacing = spacing[:, None]
        return -(normalized_flux - torch.roll(normalized_flux, 1, dims=-1)) / spacing
