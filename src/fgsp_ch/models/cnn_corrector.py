from __future__ import annotations

import torch
from torch import nn


class PeriodicResidualBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_size: int,
        normalization: str,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(
            channels,
            channels,
            kernel_size,
            padding=padding,
            padding_mode="circular",
        )
        self.conv2 = nn.Conv1d(
            channels,
            channels,
            kernel_size,
            padding=padding,
            padding_mode="circular",
        )
        if normalization == "group":
            groups = 4 if channels % 4 == 0 else 1
            self.norm1: nn.Module = nn.GroupNorm(groups, channels)
            self.norm2: nn.Module = nn.GroupNorm(groups, channels)
        elif normalization == "none":
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()
        else:
            raise ValueError(f"Unsupported normalization: {normalization}.")
        self.activation = nn.GELU()

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        residual = values
        values = self.activation(self.norm1(self.conv1(values)))
        values = self.norm2(self.conv2(values))
        return self.activation(values + residual)


class PeriodicCNNCorrector(nn.Module):
    """Small fully convolutional A1 defect-rate model."""

    def __init__(
        self,
        *,
        field_channels: int,
        parameter_channels: int,
        hidden_channels: int,
        blocks: int,
        kernel_size: int,
        normalization: str = "group",
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.input = nn.Conv1d(
            field_channels + parameter_channels,
            hidden_channels,
            kernel_size,
            padding=padding,
            padding_mode="circular",
        )
        self.blocks = nn.Sequential(
            *[
                PeriodicResidualBlock(
                    hidden_channels, kernel_size, normalization
                )
                for _ in range(blocks)
            ]
        )
        self.output = nn.Conv1d(hidden_channels, 1, kernel_size=1)
        self.activation = nn.GELU()

    def forward(
        self, fields: torch.Tensor, parameters: torch.Tensor
    ) -> torch.Tensor:
        expanded = parameters.unsqueeze(-1).expand(
            -1, -1, fields.shape[-1]
        )
        values = torch.cat((fields, expanded), dim=1)
        values = self.activation(self.input(values))
        return self.output(self.blocks(values)).squeeze(1)


class StepConsistentDualHeadCorrector(nn.Module):
    """A2 corrector with explicit spatial and temporal defect scales.

    The network heads predict scale-free components.  The only total defect
    exposed by this module is their configured, bounded-order combination.
    """

    def __init__(
        self,
        *,
        field_channels: int,
        parameter_channels: int,
        hidden_channels: int,
        blocks: int,
        kernel_size: int,
        normalization: str = "group",
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.input = nn.Conv1d(
            field_channels + parameter_channels,
            hidden_channels,
            kernel_size,
            padding=padding,
            padding_mode="circular",
        )
        self.blocks = nn.Sequential(
            *[
                PeriodicResidualBlock(
                    hidden_channels, kernel_size, normalization
                )
                for _ in range(blocks)
            ]
        )
        self.spatial_head = nn.Conv1d(hidden_channels, 1, 1)
        self.temporal_head = nn.Conv1d(hidden_channels, 1, 1)
        self.activation = nn.GELU()

    def components(
        self, fields: torch.Tensor, parameters: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        expanded = parameters.unsqueeze(-1).expand(
            -1, -1, fields.shape[-1]
        )
        encoded = self.activation(
            self.input(torch.cat((fields, expanded), dim=1))
        )
        encoded = self.blocks(encoded)
        return (
            self.spatial_head(encoded).squeeze(1),
            self.temporal_head(encoded).squeeze(1),
        )

    def forward(
        self,
        fields: torch.Tensor,
        parameters: torch.Tensor,
        h: torch.Tensor,
        dt: torch.Tensor,
        *,
        temporal_order: float,
        spatial_order: float,
        temporal_only: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        spatial, temporal = self.components(fields, parameters)
        dt_column = dt.reshape(-1, 1)
        h_column = h.reshape(-1, 1)
        spatial_scaled = dt_column * h_column.pow(spatial_order) * spatial
        if temporal_only:
            spatial_scaled = torch.zeros_like(spatial_scaled)
        temporal_scaled = dt_column.pow(temporal_order + 1.0) * temporal
        return spatial_scaled + temporal_scaled, spatial_scaled, temporal_scaled


class SequentialRichardsonCorrector(nn.Module):
    """Two-branch corrector for sequential Richardson supervision.

    The spatial and residual branches deliberately do not share an encoder.
    This makes freezing the spatial map during the second training stage a
    real identifiability constraint rather than freezing the representation
    needed by the residual branch as a side effect.
    """

    def __init__(
        self,
        *,
        field_channels: int,
        parameter_channels: int,
        hidden_channels: int,
        blocks: int,
        kernel_size: int,
        normalization: str = "group",
    ) -> None:
        super().__init__()
        arguments = {
            "field_channels": field_channels,
            "parameter_channels": parameter_channels,
            "hidden_channels": hidden_channels,
            "blocks": blocks,
            "kernel_size": kernel_size,
            "normalization": normalization,
        }
        self.spatial = PeriodicCNNCorrector(**arguments)
        self.residual = PeriodicCNNCorrector(**arguments)

    def freeze_spatial(self, frozen: bool = True) -> None:
        for parameter in self.spatial.parameters():
            parameter.requires_grad_(not frozen)

    def components(
        self, fields: torch.Tensor, parameters: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.spatial(fields, parameters), self.residual(fields, parameters)

    def forward(
        self, fields: torch.Tensor, parameters: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        spatial, residual = self.components(fields, parameters)
        return spatial + residual, spatial, residual
