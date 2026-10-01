from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True, slots=True)
class WaveletPyramid:
    coarse: torch.Tensor
    details: tuple[torch.Tensor, ...]
    original_points: int | None = None


class PeriodicHaarTransform(nn.Module):
    """Exact orthonormal periodic Haar transform for M6.3.0 audits."""

    def __init__(self, levels: int = 2) -> None:
        super().__init__()
        if levels < 1:
            raise ValueError("levels must be positive.")
        self.levels = levels

    def encode(self, values: torch.Tensor) -> WaveletPyramid:
        if values.ndim not in (2, 3):
            raise ValueError("values must be [batch, points] or [batch, channels, points].")
        original_points = values.shape[-1]
        block = 2**self.levels
        padding = (-original_points) % block
        current = (
            torch.cat((values, values[..., :padding]), dim=-1)
            if padding else values
        )
        details: list[torch.Tensor] = []
        factor = torch.sqrt(torch.tensor(2.0, dtype=values.dtype, device=values.device))
        for _ in range(self.levels):
            even, odd = current[..., 0::2], current[..., 1::2]
            details.append((even - odd) / factor)
            current = (even + odd) / factor
        return WaveletPyramid(current, tuple(details), original_points)

    def decode(self, pyramid: WaveletPyramid) -> torch.Tensor:
        if len(pyramid.details) != self.levels:
            raise ValueError("pyramid level count does not match the transform.")
        current = pyramid.coarse
        factor = torch.sqrt(torch.tensor(2.0, dtype=current.dtype, device=current.device))
        for detail in reversed(pyramid.details):
            if detail.shape != current.shape:
                raise ValueError("detail and coarse shapes are inconsistent.")
            even = (current + detail) / factor
            odd = (current - detail) / factor
            reconstructed = torch.empty(
                *current.shape[:-1], 2 * current.shape[-1],
                dtype=current.dtype, device=current.device,
            )
            reconstructed[..., 0::2] = even
            reconstructed[..., 1::2] = odd
            current = reconstructed
        points = pyramid.original_points
        return current if points is None else current[..., :points]

    def forward(self, values: torch.Tensor) -> WaveletPyramid:
        return self.encode(values)


class PeriodicMultiwaveletBlock(nn.Module):
    """Shared local coefficient update; usable at every dyadic resolution."""

    def __init__(self, channels: int, hidden: int = 32) -> None:
        super().__init__()
        self.update = nn.Sequential(
            nn.Conv1d(channels, hidden, 3, padding=1, padding_mode="circular"),
            nn.GELU(),
            nn.Conv1d(hidden, channels, 3, padding=1, padding_mode="circular"),
        )

    def forward(self, coefficients: torch.Tensor) -> torch.Tensor:
        if coefficients.ndim != 3:
            raise ValueError("coefficients must have shape [batch, channels, points].")
        return coefficients + self.update(coefficients)
