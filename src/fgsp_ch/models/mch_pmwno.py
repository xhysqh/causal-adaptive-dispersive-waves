from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from fgsp_ch.models.peakon_green_attention import PeakonGreenAttention
from fgsp_ch.models.periodic_multiwavelet import PeriodicHaarTransform, PeriodicMultiwaveletBlock


@dataclass(frozen=True, slots=True)
class PMWNOOutput:
    peak_position_rate: torch.Tensor
    peak_amplitude_rate: torch.Tensor
    cluster_coordinate_rate: torch.Tensor
    field_momentum_rate: torch.Tensor
    spatial_detail_score: tuple[torch.Tensor, ...]
    temporal_error_ratio: torch.Tensor
    collision_horizon: torch.Tensor
    h1_uncertainty: torch.Tensor
    step_factor: torch.Tensor


class CertifiedPeakMeasureWaveletOperator(nn.Module):
    """M6.3 C-PMWNO structural prototype; training begins only in M6.3.1."""

    def __init__(
        self, *, field_channels: int = 4, parameter_channels: int = 11,
        hidden: int = 32, wavelet_levels: int = 2,
    ) -> None:
        super().__init__()
        self.peakon = PeakonGreenAttention(8, parameter_channels, hidden)
        self.cluster = nn.Sequential(
            # Periodic center sin/cos are decoder metadata, not dynamics
            # inputs, so cluster updates are exactly translation invariant.
            nn.Linear(7 + parameter_channels, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 6),
        )
        self.register_buffer("cluster_coordinate_mask", torch.ones(6))
        self.wavelet = PeriodicHaarTransform(wavelet_levels)
        self.field_lift = nn.Conv1d(field_channels, hidden, 1)
        self.field_block = PeriodicMultiwaveletBlock(hidden, hidden)
        self.field_readout = nn.Conv1d(hidden, 1, 1)
        self.detail_error_heads = nn.ModuleList(
            nn.Conv1d(hidden, 1, 1) for _ in range(wavelet_levels)
        )
        self.risk = nn.Sequential(nn.Linear(parameter_channels + 5, hidden), nn.GELU(), nn.Linear(hidden, 3))
        self.wavelet_levels = wavelet_levels

    def set_cluster_coordinate_mask(self, mask: torch.Tensor) -> None:
        if mask.shape != (6,):
            raise ValueError("cluster coordinate mask must have shape [6].")
        self.cluster_coordinate_mask.copy_(mask.to(self.cluster_coordinate_mask))

    def forward(
        self, peak_tokens: torch.Tensor, positions: torch.Tensor, amplitudes: torch.Tensor,
        peak_mask: torch.Tensor, field: torch.Tensor, parameters: torch.Tensor,
        length: torch.Tensor, cluster_tokens: torch.Tensor | None = None,
        cluster_mask: torch.Tensor | None = None,
    ) -> PMWNOOutput:
        q_rate, p_rate = self.peakon(
            peak_tokens, positions, amplitudes, peak_mask, parameters, length
        )
        if cluster_tokens is None:
            cluster_tokens = field.new_zeros(field.shape[0], 0, 9)
        if cluster_mask is None:
            cluster_mask = torch.ones(
                cluster_tokens.shape[:2], dtype=torch.bool, device=cluster_tokens.device
            )
        if cluster_tokens.ndim != 3 or cluster_tokens.shape[-1] != 9:
            raise ValueError("cluster_tokens must have shape [batch, clusters, 9].")
        cluster_parameters = parameters[:, None, :].expand(
            -1, cluster_tokens.shape[1], -1
        )
        cluster_rate = self.cluster(
            torch.cat((cluster_tokens[..., 2:], cluster_parameters), dim=-1)
        ) * cluster_mask[..., None] * self.cluster_coordinate_mask
        lifted = self.field_lift(field)
        pyramid = self.wavelet.encode(lifted)
        coarse = self.field_block(pyramid.coarse)
        details = tuple(self.field_block(detail) for detail in pyramid.details)
        updated = self.wavelet.decode(
            type(pyramid)(coarse, details, pyramid.original_points)
        )
        field_rate = self.field_readout(updated).squeeze(1)
        # A momentum-rate constant would change total mass; remove it exactly.
        field_rate = field_rate - field_rate.mean(dim=-1, keepdim=True)
        scores = tuple(
            torch.expm1(torch.nn.functional.softplus(head(detail))).squeeze(1)
            for head, detail in zip(self.detail_error_heads, details, strict=True)
        )
        pooled = torch.stack(
            (
                field_rate.square().mean(dim=-1).sqrt(),
                q_rate.square().mean(dim=-1).sqrt(),
                p_rate.square().mean(dim=-1).sqrt(),
                field[:, 1].square().mean(dim=-1).sqrt(),
                details[0].square().mean(dim=(1, 2)).sqrt(),
            ), dim=-1,
        )
        raw_risk = self.risk(torch.cat((parameters, pooled), dim=-1))
        risk = torch.nn.functional.softplus(raw_risk)
        temporal_log, collision_inverse, uncertainty_log = risk.unbind(dim=-1)
        temporal = torch.expm1(temporal_log)
        uncertainty = torch.expm1(uncertainty_log)
        collision_horizon = 1.0 / collision_inverse.clamp_min(1.0e-8)
        step = (0.9 * temporal.clamp_min(1.0e-8).pow(-1.0 / 3.0)).clamp(0.25, 1.0)
        return PMWNOOutput(
            q_rate, p_rate, cluster_rate, field_rate, scores, temporal,
            collision_horizon, uncertainty, step,
        )
