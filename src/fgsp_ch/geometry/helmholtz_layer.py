from __future__ import annotations

import torch
from torch import nn


class PeriodicHelmholtzLayer(nn.Module):
    """Differentiable centered-difference A_h and its periodic inverse."""

    @staticmethod
    def apply(values: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        h2 = h.reshape(-1, 1).pow(2)
        second = (
            torch.roll(values, -1, dims=-1)
            - 2.0 * values
            + torch.roll(values, 1, dims=-1)
        ) / h2
        return values - second

    @staticmethod
    def solve(momentum: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        points = momentum.shape[-1]
        frequencies = torch.arange(
            points, device=momentum.device, dtype=momentum.dtype
        )
        eigenvalues = 1.0 + 4.0 * torch.sin(
            torch.pi * frequencies / points
        ).pow(2)[None, :] / h.reshape(-1, 1).pow(2)
        transformed = torch.fft.fft(momentum, dim=-1)
        return torch.fft.ifft(transformed / eigenvalues, dim=-1).real

    def forward(
        self, momentum: torch.Tensor, h: torch.Tensor
    ) -> torch.Tensor:
        return self.solve(momentum, h)
