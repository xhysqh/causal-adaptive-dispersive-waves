from __future__ import annotations

import torch
from torch import nn

from fgsp_ch.models.cnn_corrector import PeriodicCNNCorrector


class _PeriodicSetEncoder(nn.Module):
    def __init__(self, node_channels: int, parameter_channels: int, hidden: int) -> None:
        super().__init__()
        self.edge = nn.Sequential(
            nn.Linear(2 * node_channels + 3, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        self.node = nn.Sequential(
            nn.Linear(node_channels + hidden + parameter_channels, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )

    def forward(
        self,
        positions: torch.Tensor,
        node_values: torch.Tensor,
        mask: torch.Tensor,
        parameters: torch.Tensor,
        length: torch.Tensor,
    ) -> torch.Tensor:
        mask = mask.bool()
        scale = torch.amax(torch.abs(node_values), dim=(1, 2), keepdim=True).clamp_min(1e-8)
        values = node_values / scale
        delta = (positions[:, None, :] - positions[:, :, None]) / length[:, None, None]
        delta = torch.remainder(delta + 0.5, 1.0) - 0.5
        receiver = values[:, :, None, :].expand(-1, -1, values.shape[1], -1)
        sender = values[:, None, :, :].expand(-1, values.shape[1], -1, -1)
        geometry = torch.stack(
            (
                torch.sin(2.0 * torch.pi * delta),
                torch.cos(2.0 * torch.pi * delta),
                torch.abs(delta),
            ),
            dim=-1,
        )
        edge_features = torch.cat((receiver, sender, geometry), dim=-1)
        edge_mask = mask[:, :, None] & mask[:, None, :]
        messages = self.edge(edge_features) * edge_mask[..., None]
        count = edge_mask.sum(dim=2, keepdim=True).clamp_min(1)
        aggregate = messages.sum(dim=2) / count
        expanded_parameters = parameters[:, None, :].expand(-1, positions.shape[1], -1)
        encoded = self.node(torch.cat((values, aggregate, expanded_parameters), dim=-1))
        return encoded * mask[..., None]


class ParticleGraphExpert(nn.Module):
    """Convex readout on fixed periodic Green-kernel particle features."""

    def __init__(
        self,
        *,
        parameter_channels: int,
        hidden: int = 16,
        conserve_amplitude: bool = True,
    ) -> None:
        super().__init__()
        self.conserve_amplitude = conserve_amplitude
        del hidden  # The M4 minimum deliberately has no learned particle encoder.
        feature_channels = parameter_channels + 8
        self.position_head = nn.Linear(feature_channels, 1)
        self.amplitude_head = (
            None if conserve_amplitude else nn.Linear(feature_channels, 1)
        )

    def forward(
        self,
        positions: torch.Tensor,
        amplitudes: torch.Tensor,
        mask: torch.Tensor,
        parameters: torch.Tensor,
        length: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mask = mask.bool()
        value_scale = torch.amax(torch.abs(amplitudes), dim=1, keepdim=True).clamp_min(1e-8)
        values = amplitudes / value_scale
        wrapped = positions[:, :, None] - positions[:, None, :]
        wrapped = torch.remainder(
            wrapped + 0.5 * length[:, None, None], length[:, None, None]
        ) - 0.5 * length[:, None, None]
        edge_mask = mask[:, :, None] & mask[:, None, :]
        kernel = torch.cosh(
            0.5 * length[:, None, None] - torch.abs(wrapped)
        ) / torch.cosh(0.5 * length[:, None, None])
        derivative = -torch.sign(wrapped) * torch.sinh(
            0.5 * length[:, None, None] - torch.abs(wrapped)
        ) / torch.cosh(0.5 * length[:, None, None])
        kernel = kernel * edge_mask
        derivative = derivative * edge_mask
        field = torch.sum(kernel * values[:, None, :], dim=2)
        slope = torch.sum(derivative * values[:, None, :], dim=2)
        count = mask.sum(dim=1, keepdim=True).clamp_min(1).to(values.dtype)
        total = torch.sum(values * mask, dim=1, keepdim=True)
        invariant = torch.stack(
            (
                values,
                values**2,
                field,
                slope,
                field**2,
                slope**2,
                total.expand_as(values),
                (count / 8.0).expand_as(values),
            ),
            dim=-1,
        )
        expanded_parameters = parameters[:, None, :].expand(-1, positions.shape[1], -1)
        features = torch.cat((invariant, expanded_parameters), dim=-1) * mask[..., None]
        position = self.position_head(features).squeeze(-1) * mask
        if self.amplitude_head is None:
            amplitude = torch.zeros_like(position)
        else:
            amplitude = self.amplitude_head(features).squeeze(-1) * mask
        return position, amplitude


class ClusterGraphExpert(nn.Module):
    """Jordan-cluster expert with only data-identifiable moment outputs."""

    def __init__(self, *, parameter_channels: int, hidden: int = 16) -> None:
        super().__init__()
        self.encoder = _PeriodicSetEncoder(4, parameter_channels, hidden)
        self.position_head = nn.Linear(hidden, 1)
        self.moment_head = nn.Linear(hidden, 4)

    def forward(
        self,
        centers: torch.Tensor,
        signs: torch.Tensor,
        moments: torch.Tensor,
        mask: torch.Tensor,
        parameters: torch.Tensor,
        length: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        values = torch.cat((signs[..., None], moments), dim=-1)
        encoded = self.encoder(centers, values, mask, parameters, length)
        position = self.position_head(encoded).squeeze(-1) * mask
        raw_moments = self.moment_head(encoded) * mask[..., None]
        conserved = torch.cat(
            (
                raw_moments[..., :1],
                # M3 finds no signal above the reference floor for the
                # conserved mass or the two higher within-cluster moments.
                torch.zeros_like(raw_moments[..., 1:]),
            ),
            dim=-1,
        )
        return position, conserved


class MinimalMCHAtlasNetwork(nn.Module):
    """Minimal separated/cluster/field mCH network used by the M4 audit."""

    def __init__(
        self,
        *,
        field_channels: int = 10,
        parameter_channels: int = 8,
        hidden: int = 16,
        conserve_amplitude: bool = True,
    ) -> None:
        super().__init__()
        context_channels = 8
        self.particle = ParticleGraphExpert(
            parameter_channels=parameter_channels + context_channels,
            hidden=hidden,
            conserve_amplitude=conserve_amplitude,
        )
        self.cluster = ClusterGraphExpert(
            parameter_channels=parameter_channels + context_channels, hidden=hidden
        )
        self.field = PeriodicCNNCorrector(
            field_channels=field_channels,
            parameter_channels=parameter_channels,
            hidden_channels=hidden,
            blocks=1,
            kernel_size=3,
            normalization="none",
        )
