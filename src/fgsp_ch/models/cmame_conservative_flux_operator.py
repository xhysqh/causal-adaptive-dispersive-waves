"""mCH-specific periodic conservative neural flux operator."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


class PeriodicConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, *, dilation: int = 1) -> None:
        super().__init__()
        self.padding = dilation * (kernel_size // 2)
        self.layer = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.layer(F.pad(values, (self.padding, self.padding), mode="circular"))


class FluxResidualBlock(nn.Module):
    def __init__(self, channels: int, dilation: int) -> None:
        super().__init__()
        self.first = PeriodicConv1d(channels, channels, dilation=dilation)
        self.second = PeriodicConv1d(channels, channels, dilation=dilation)
        self.norm1 = nn.GroupNorm(4, channels)
        self.norm2 = nn.GroupNorm(4, channels)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        hidden = F.silu(self.norm1(self.first(values)))
        return F.silu(values + self.norm2(self.second(hidden)))


@dataclass(frozen=True, slots=True)
class ConservativeFluxOutput:
    face_correction: torch.Tensor
    defect_indicator: torch.Tensor


class MCHConservativeFluxOperator(nn.Module):
    """Predict only a bounded face-flux defect, never the next state."""

    def __init__(self, *, feature_channels: int = 10, parameter_channels: int = 6, hidden: int = 32, dilations: tuple[int, ...] = (1, 2, 4)) -> None:
        super().__init__()
        if hidden % 4:
            raise ValueError("hidden must be divisible by four")
        self.feature_channels = feature_channels
        self.parameter_channels = parameter_channels
        self.input = PeriodicConv1d(feature_channels, hidden)
        self.condition = nn.Sequential(nn.Linear(parameter_channels, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.blocks = nn.ModuleList(FluxResidualBlock(hidden, dilation) for dilation in dilations)
        self.flux_head = PeriodicConv1d(hidden, 1)
        self.defect_head = PeriodicConv1d(hidden, 1)
        self.register_buffer("feature_mean", torch.zeros(1, feature_channels, 1))
        self.register_buffer("feature_scale", torch.ones(1, feature_channels, 1))
        self.register_buffer("parameter_mean", torch.zeros(1, parameter_channels))
        self.register_buffer("parameter_scale", torch.ones(1, parameter_channels))
        self.register_buffer("target_scale", torch.ones(()))

    @torch.no_grad()
    def set_normalization(self, features: torch.Tensor, parameters: torch.Tensor, targets: torch.Tensor) -> None:
        self.feature_mean.copy_(features.mean((0, 2), keepdim=True))
        self.feature_scale.copy_(features.std((0, 2), keepdim=True, unbiased=False).clamp_min(1e-8))
        self.parameter_mean.copy_(parameters.mean(0, keepdim=True))
        self.parameter_scale.copy_(parameters.std(0, keepdim=True, unbiased=False).clamp_min(1e-8))
        self.target_scale.copy_(targets.square().mean().sqrt().clamp_min(1e-10))

    def forward(self, features: torch.Tensor, parameters: torch.Tensor, *, ablation: str = "full") -> ConservativeFluxOutput:
        if features.ndim != 3 or features.shape[1] != self.feature_channels:
            raise ValueError("features must be [batch, channels, points]")
        if parameters.shape != (features.shape[0], self.parameter_channels):
            raise ValueError("parameters have the wrong shape")
        normalized = (features - self.feature_mean) / self.feature_scale
        if ablation == "no_geometry":
            normalized = normalized.clone(); normalized[:, [1, 2, 5, 6]] = 0.0
        elif ablation == "no_curvature":
            normalized = normalized.clone(); normalized[:, [2, 5]] = 0.0
        elif ablation == "no_base_flux":
            normalized = normalized.clone(); normalized[:, 4] = 0.0
        values = self.input(normalized)
        normalized_parameters = (parameters - self.parameter_mean) / self.parameter_scale.clamp_min(1e-8)
        if ablation == "no_parameters":
            normalized_parameters = torch.zeros_like(normalized_parameters)
        conditioning = self.condition(normalized_parameters)[:, :, None]
        values = values + conditioning
        blocks = self.blocks[:1] if ablation == "single_scale" else self.blocks
        for block in blocks:
            values = block(values)
        raw = self.flux_head(values)[:, 0]
        # The zero Fourier mode of a flux is unidentifiable and dynamically
        # irrelevant. Removing it gives a unique conservative representation.
        raw = raw - raw.mean(dim=-1, keepdim=True)
        correction = self.target_scale * 4.0 * torch.tanh(0.25 * raw)
        correction = correction - correction.mean(dim=-1, keepdim=True)
        defect = F.softplus(self.defect_head(values)[:, 0])
        return ConservativeFluxOutput(correction, defect)

    @staticmethod
    def conservative_rhs(face_flux: torch.Tensor, *, h: torch.Tensor | float) -> torch.Tensor:
        spacing = torch.as_tensor(h, dtype=face_flux.dtype, device=face_flux.device)
        if spacing.ndim == 1:
            spacing = spacing[:, None]
        return -(face_flux - torch.roll(face_flux, 1, dims=-1)) / spacing
