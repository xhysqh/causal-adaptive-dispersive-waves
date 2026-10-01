from __future__ import annotations

import torch
from torch import nn


def periodic_green_features(
    positions: torch.Tensor, amplitudes: torch.Tensor, mask: torch.Tensor, length: torch.Tensor
) -> torch.Tensor:
    """Five exact periodic geometry channels for every directed particle pair."""
    if positions.shape != amplitudes.shape or positions.shape != mask.shape:
        raise ValueError("positions, amplitudes, and mask must have equal [batch, atoms] shapes.")
    displacement = positions[:, :, None] - positions[:, None, :]
    domain = length[:, None, None]
    wrapped = torch.remainder(displacement + 0.5 * domain, domain) - 0.5 * domain
    phase = 2.0 * torch.pi * wrapped / domain
    kernel = torch.cosh(0.5 * domain - torch.abs(wrapped)) / torch.cosh(0.5 * domain)
    derivative = -torch.sign(wrapped) * torch.sinh(
        0.5 * domain - torch.abs(wrapped)
    ) / torch.cosh(0.5 * domain)
    derivative = torch.where(wrapped == 0.0, torch.zeros_like(derivative), derivative)
    pair_mask = mask[:, :, None] & mask[:, None, :]
    features = torch.stack(
        (torch.sin(phase), torch.cos(phase), kernel, derivative, torch.abs(wrapped) / domain),
        dim=-1,
    )
    return features * pair_mask[..., None]


class PeakonGreenAttention(nn.Module):
    """Permutation-equivariant, periodic Green-aware particle update."""

    def __init__(self, node_channels: int = 8, parameter_channels: int = 11, hidden: int = 32) -> None:
        super().__init__()
        if node_channels < 3:
            raise ValueError("node_channels must include periodic position and invariant channels.")
        # Absolute sin/cos coordinates are retained in the public state codec for
        # decoding and cross-attention, but excluded from the dynamics readout.
        # This makes a global rotation of the periodic domain an exact symmetry.
        invariant_channels = node_channels - 2
        self.edge = nn.Sequential(
            nn.Linear(2 * invariant_channels + 5, hidden), nn.GELU(), nn.Linear(hidden, hidden)
        )
        self.node = nn.Sequential(
            nn.Linear(invariant_channels + hidden + parameter_channels, hidden),
            nn.GELU(),
            nn.Linear(hidden, 2),
        )

    def forward(
        self,
        tokens: torch.Tensor,
        positions: torch.Tensor,
        amplitudes: torch.Tensor,
        mask: torch.Tensor,
        parameters: torch.Tensor,
        length: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if tokens.ndim != 3 or tokens.shape[:2] != positions.shape:
            raise ValueError("tokens must have shape [batch, atoms, node_channels].")
        geometry = periodic_green_features(positions, amplitudes, mask, length)
        invariant = tokens[..., 2:]
        receiver = invariant[:, :, None, :].expand(-1, -1, tokens.shape[1], -1)
        sender = invariant[:, None, :, :].expand(-1, tokens.shape[1], -1, -1)
        pair_mask = mask[:, :, None] & mask[:, None, :]
        messages = self.edge(torch.cat((receiver, sender, geometry), dim=-1))
        messages = messages * pair_mask[..., None]
        aggregate = messages.sum(dim=2) / pair_mask.sum(dim=2, keepdim=True).clamp_min(1)
        global_values = parameters[:, None, :].expand(-1, tokens.shape[1], -1)
        raw = self.node(torch.cat((invariant, aggregate, global_values), dim=-1))
        raw = raw * mask[..., None]
        position_rate = raw[..., 0]
        amplitude_raw = raw[..., 1]
        active = mask.to(raw.dtype)
        mean = (amplitude_raw * active).sum(dim=1, keepdim=True) / active.sum(
            dim=1, keepdim=True
        ).clamp_min(1.0)
        amplitude_rate = (amplitude_raw - mean) * active
        return position_rate, amplitude_rate
