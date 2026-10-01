"""Paper-facing learned baselines for the frozen CMAME-M2 protocol.

The models in this file are deliberately independent of the proposed M1.3
solver.  They share neither its estimator nor its local-AMR controller.
"""

from __future__ import annotations

import math

import torch
from torch import nn


def periodic_context(values: torch.Tensor, modes: int = 4) -> torch.Tensor:
    """Return translation-aware low-mode coordinates of an initial profile."""
    if values.ndim != 2:
        raise ValueError("values must have shape [batch, points]")
    transformed = torch.fft.rfft(values, dim=-1, norm="forward")
    usable = min(modes, transformed.shape[-1] - 1)
    low = transformed[:, 1 : usable + 1]
    context = [values.mean(-1, keepdim=True), values.std(-1, keepdim=True)]
    context.extend((low.real, low.imag))
    result = torch.cat(context, dim=-1)
    expected = 2 + 2 * modes
    if result.shape[-1] < expected:
        result = torch.nn.functional.pad(result, (0, expected - result.shape[-1]))
    return result


class CoordinatePINN(nn.Module):
    """Conditional periodic coordinate network with an exact initial trace.

    The network represents ``u(x,t)=u0(x)+tau*N(x,tau,alpha,u0)``.  Hence the
    initial condition is satisfied algebraically, not through a penalty.  The
    low Fourier context makes translations and peak locations identifiable.
    """

    def __init__(self, *, hidden: int = 64, layers: int = 4, modes: int = 4) -> None:
        super().__init__()
        self.modes = int(modes)
        width = 6 + 2 + 2 * self.modes
        blocks: list[nn.Module] = [nn.Linear(width, hidden), nn.Tanh()]
        for _ in range(layers - 1):
            blocks.extend((nn.Linear(hidden, hidden), nn.Tanh()))
        blocks.append(nn.Linear(hidden, 1))
        self.network = nn.Sequential(*blocks)

    def forward(
        self,
        coordinate: torch.Tensor,
        tau: torch.Tensor,
        alpha: torch.Tensor,
        initial_value: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        if coordinate.shape != tau.shape or coordinate.shape != initial_value.shape:
            raise ValueError("coordinate, tau and initial_value must be aligned")
        if alpha.shape != coordinate.shape or context.shape[:-1] != coordinate.shape:
            raise ValueError("PINN conditioning shapes are inconsistent")
        features = torch.cat(
            (
                torch.sin(coordinate).unsqueeze(-1),
                torch.cos(coordinate).unsqueeze(-1),
                torch.sin(2.0 * coordinate).unsqueeze(-1),
                torch.cos(2.0 * coordinate).unsqueeze(-1),
                tau.unsqueeze(-1),
                alpha.unsqueeze(-1),
                context,
            ),
            dim=-1,
        )
        return initial_value + tau * self.network(features).squeeze(-1)


class SpectralConvolution1d(nn.Module):
    """Resolution-independent complex Fourier multiplier."""

    def __init__(self, in_channels: int, out_channels: int, modes: int) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.modes = int(modes)
        scale = 1.0 / math.sqrt(in_channels * out_channels)
        self.weight = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes, dtype=torch.cfloat)
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        transformed = torch.fft.rfft(values, dim=-1, norm="ortho")
        count = min(self.modes, transformed.shape[-1])
        output = torch.zeros(
            values.shape[0], self.out_channels, transformed.shape[-1],
            dtype=transformed.dtype, device=values.device,
        )
        output[..., :count] = torch.einsum(
            "bim,iom->bom", transformed[..., :count], self.weight[..., :count]
        )
        return torch.fft.irfft(output, n=values.shape[-1], dim=-1, norm="ortho")


class FNO1dBaseline(nn.Module):
    """Time-conditional 1-D FNO with an exact zero-time increment."""

    def __init__(self, *, width: int = 32, modes: int = 12, layers: int = 4) -> None:
        super().__init__()
        self.lift = nn.Conv1d(5, width, 1)
        self.spectral = nn.ModuleList(
            SpectralConvolution1d(width, width, modes) for _ in range(layers)
        )
        self.local = nn.ModuleList(nn.Conv1d(width, width, 1) for _ in range(layers))
        self.project = nn.Sequential(nn.Conv1d(width, width, 1), nn.GELU(), nn.Conv1d(width, 1, 1))

    def forward(self, initial: torch.Tensor, tau: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        if initial.ndim != 2 or tau.shape != initial.shape[:1] or alpha.shape != tau.shape:
            raise ValueError("FNO expects initial [batch,points] and scalar batch conditions")
        points = initial.shape[-1]
        angle = 2.0 * torch.pi * torch.arange(points, device=initial.device, dtype=initial.dtype) / points
        channels = torch.stack(
            (
                initial,
                torch.sin(angle).expand_as(initial),
                torch.cos(angle).expand_as(initial),
                tau[:, None].expand_as(initial),
                alpha[:, None].expand_as(initial),
            ),
            dim=1,
        )
        hidden = self.lift(channels)
        for spectral, local in zip(self.spectral, self.local):
            hidden = torch.nn.functional.gelu(spectral(hidden) + local(hidden))
        return initial + tau[:, None] * self.project(hidden).squeeze(1)


def periodic_derivative(values: torch.Tensor, order: int, length: float) -> torch.Tensor:
    """Spectral derivative used only in the smooth-sector PINN residual."""
    if values.ndim != 2 or order < 0:
        raise ValueError("values must be [batch,points] and order nonnegative")
    wave = 2.0 * torch.pi * torch.fft.fftfreq(
        values.shape[-1], d=length / values.shape[-1], device=values.device
    ).to(values.dtype)
    return torch.fft.ifft((1j * wave) ** order * torch.fft.fft(values, dim=-1), dim=-1).real

