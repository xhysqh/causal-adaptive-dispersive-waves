from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


class PeriodicConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int = 1):
        super().__init__()
        self.padding = dilation * (kernel_size // 2)
        self.convolution = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=0, dilation=dilation,
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        padded = F.pad(values, (self.padding, self.padding), mode="circular")
        return self.convolution(padded)


class PeriodicResidualBlock(nn.Module):
    def __init__(self, channels: int, dilation: int):
        super().__init__()
        self.first = PeriodicConv1d(channels, channels, 3, dilation)
        self.second = PeriodicConv1d(channels, channels, 3, dilation)
        self.norm1 = nn.GroupNorm(4, channels)
        self.norm2 = nn.GroupNorm(4, channels)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        residual = values
        values = F.silu(self.norm1(self.first(values)))
        values = self.norm2(self.second(values))
        return F.silu(values + residual)


@dataclass(frozen=True, slots=True)
class LocalAMROutput:
    refinement_score: torch.Tensor
    fine_detail: torch.Tensor
    face_flux_mismatch: torch.Tensor


class ConservativeLocalAMROperator(nn.Module):
    """Periodic three-head local operator; synchronization stays deterministic."""

    def __init__(
        self,
        *,
        feature_channels: int = 11,
        parameter_channels: int = 6,
        hidden: int = 32,
        dilations: tuple[int, ...] = (1, 2, 4, 8),
    ) -> None:
        super().__init__()
        if hidden % 4:
            raise ValueError("hidden must be divisible by four for GroupNorm.")
        self.feature_channels = feature_channels
        self.parameter_channels = parameter_channels
        self.input_projection = PeriodicConv1d(feature_channels, hidden, 3)
        self.parameter_projection = nn.Sequential(
            nn.Linear(parameter_channels, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        self.blocks = nn.ModuleList(
            PeriodicResidualBlock(hidden, dilation) for dilation in dilations
        )
        self.refinement_head = PeriodicConv1d(hidden, 1, 3)
        self.detail_head = PeriodicConv1d(hidden, 2, 3)
        self.face_head = PeriodicConv1d(hidden, 1, 3)
        self.refinement_proxy_gain = nn.Parameter(torch.tensor(-1.0))
        self.register_buffer("feature_mean", torch.zeros(1, feature_channels, 1))
        self.register_buffer("feature_scale", torch.ones(1, feature_channels, 1))
        self.register_buffer("parameter_mean", torch.zeros(1, parameter_channels))
        self.register_buffer("parameter_scale", torch.ones(1, parameter_channels))
        self.register_buffer("refinement_scale", torch.ones(()))
        self.register_buffer("detail_scale", torch.ones(()))
        self.register_buffer("face_scale", torch.ones(()))
        # Deployment-only compatibility switch. It is intentionally not a
        # checkpoint tensor so frozen M6.3.3/M6.3.4 baselines retain their
        # exact behavior unless the M6.3.4.3.1 controller enables it.
        self.bounded_degenerate_parameter_conditioning = False

    @torch.no_grad()
    def set_normalization(
        self,
        features: torch.Tensor,
        parameters: torch.Tensor,
        refinement: torch.Tensor,
        detail: torch.Tensor,
        face: torch.Tensor,
    ) -> None:
        feature_mean = features.mean(dim=(0, 2), keepdim=True)
        feature_scale = features.std(dim=(0, 2), keepdim=True, unbiased=False).clamp_min(1.0e-8)
        parameter_mean = parameters.mean(dim=0, keepdim=True)
        parameter_scale = parameters.std(dim=0, keepdim=True, unbiased=False).clamp_min(1.0e-8)
        self.feature_mean.copy_(feature_mean)
        self.feature_scale.copy_(feature_scale)
        self.parameter_mean.copy_(parameter_mean)
        self.parameter_scale.copy_(parameter_scale)
        self.refinement_scale.copy_(refinement.square().mean().sqrt().clamp_min(1.0e-10))
        self.detail_scale.copy_(detail.square().mean().sqrt().clamp_min(1.0e-10))
        face_residual = face - features[:, 9]
        self.face_scale.copy_(face_residual.square().mean().sqrt().clamp_min(1.0e-10))

    def forward(self, features: torch.Tensor, parameters: torch.Tensor) -> LocalAMROutput:
        if features.ndim != 3 or features.shape[1] != self.feature_channels:
            raise ValueError("features must have shape [batch, channels, points].")
        if parameters.shape != (features.shape[0], self.parameter_channels):
            raise ValueError("parameters have the wrong shape.")
        normalized = (features - self.feature_mean) / self.feature_scale
        conditioned = self.input_projection(normalized)
        normalized_parameters = (
            parameters - self.parameter_mean
        ) / self.parameter_scale
        if self.bounded_degenerate_parameter_conditioning:
            # A channel with zero train variance (notably dt and h in M6332)
            # carries no learned extrapolation law. Treat it as the training
            # reference instead of amplifying deployment shifts by 1e8.
            supported = self.parameter_scale > 1.0e-6
            normalized_parameters = torch.where(
                supported, normalized_parameters, torch.zeros_like(normalized_parameters)
            ).clamp(-6.0, 6.0)
        parameter_embedding = self.parameter_projection(
            normalized_parameters
        )[:, :, None]
        values = conditioned + parameter_embedding
        for block in self.blocks:
            values = block(values)
        curvature_proxy = features[:, 5].abs()
        curvature_proxy = curvature_proxy / curvature_proxy.square().mean(
            dim=1, keepdim=True
        ).sqrt().clamp_min(1.0e-8)
        proxy_gain = F.softplus(self.refinement_proxy_gain)
        refinement_residual = 0.25 * torch.tanh(self.refinement_head(values)[:, 0])
        refinement = torch.clamp_min(
            (proxy_gain * curvature_proxy + refinement_residual) * self.refinement_scale,
            0.0,
        )
        detail_pairs = self.detail_head(values) * self.detail_scale
        detail = detail_pairs.permute(0, 2, 1).reshape(features.shape[0], -1)
        face = features[:, 9] + self.face_head(values)[:, 0] * self.face_scale
        return LocalAMROutput(refinement, detail, face)

    @staticmethod
    def constrain_fine_detail(
        raw_detail: torch.Tensor,
        coarse_mask: torch.Tensor,
        target_patch_sum: torch.Tensor,
    ) -> torch.Tensor:
        if raw_detail.ndim != 2 or coarse_mask.shape != (
            raw_detail.shape[0], raw_detail.shape[1] // 2
        ):
            raise ValueError("detail and coarse mask shapes are inconsistent.")
        raw = raw_detail.to(torch.float64)
        fine_mask = torch.repeat_interleave(coarse_mask.to(torch.float64), 2, dim=1)
        count = fine_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        masked = raw * fine_mask
        centered = (masked - fine_mask * masked.sum(dim=1, keepdim=True) / count)
        closure = fine_mask * target_patch_sum.to(torch.float64)[:, None] / count
        return centered + closure

    @staticmethod
    def conservative_reflux_correction(
        face_flux_mismatch: torch.Tensor,
        boundary_face_mask: torch.Tensor,
        *,
        dt_over_h: torch.Tensor | float,
    ) -> torch.Tensor:
        selected = face_flux_mismatch * boundary_face_mask.to(face_flux_mismatch.dtype)
        factor = torch.as_tensor(
            dt_over_h, dtype=selected.dtype, device=selected.device
        )
        if factor.ndim == 1:
            factor = factor[:, None]
        return -factor * (selected - torch.roll(selected, 1, dims=1))
