from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


class PositiveLinear(nn.Module):
    """Affine map with non-negative feature derivatives."""

    def __init__(self, inputs: int, outputs: int, *, bias: bool = True) -> None:
        super().__init__()
        self.raw_weight = nn.Parameter(torch.empty(outputs, inputs))
        self.bias = nn.Parameter(torch.zeros(outputs)) if bias else None
        nn.init.normal_(self.raw_weight, mean=-2.0, std=0.15)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return F.linear(values, F.softplus(self.raw_weight), self.bias)


@dataclass(frozen=True, slots=True)
class ShadowRiskOutput:
    median_log_defect: torch.Tensor
    upper_log_defect: torch.Tensor
    direction: torch.Tensor
    unsafe_logit: torch.Tensor


class MonotoneShadowDefectRiskOperator(nn.Module):
    """Small context/residual operator for shadow-defect upper bounds."""

    def __init__(
        self, context_channels: int = 48, residual_channels: int = 7,
        context_hidden: int = 24, residual_hidden: int = 12,
        context_correction_bound: float = 0.25,
    ) -> None:
        super().__init__()
        self.context_channels = context_channels
        self.residual_channels = residual_channels
        if context_correction_bound <= 0.0:
            raise ValueError("context_correction_bound must be positive.")
        self.context_correction_bound = float(context_correction_bound)
        self.register_buffer("context_mean", torch.zeros(context_channels))
        self.register_buffer("context_scale", torch.ones(context_channels))
        self.register_buffer("residual_mean", torch.zeros(residual_channels))
        self.register_buffer("residual_scale", torch.ones(residual_channels))
        self.context = nn.Sequential(
            nn.Linear(context_channels, context_hidden), nn.SiLU(),
            nn.LayerNorm(context_hidden),
            nn.Linear(context_hidden, context_hidden), nn.SiLU(),
        )
        self.residual = nn.Sequential(
            PositiveLinear(3, residual_hidden), nn.Softplus(),
        )
        self.auxiliary = nn.Sequential(
            nn.Linear(4, 8), nn.SiLU(),
        )
        self.context_median = nn.Linear(context_hidden, 1)
        self.residual_median = PositiveLinear(3, 1, bias=True)
        self.auxiliary_median = nn.Linear(4, 1)
        self.context_margin = nn.Linear(context_hidden, 1)
        self.residual_margin = PositiveLinear(3, 1, bias=False)
        self.auxiliary_margin = nn.Linear(4, 1)
        fused = context_hidden + residual_hidden + 8
        self.direction_head = nn.Sequential(
            nn.Linear(fused, 16), nn.SiLU(), nn.Linear(16, 1), nn.Tanh(),
        )
        self.unsafe_head = nn.Sequential(
            nn.Linear(fused + 1, 16), nn.SiLU(), nn.Linear(16, 1),
        )

    @torch.no_grad()
    def set_normalization(
        self, context: torch.Tensor, residual: torch.Tensor,
    ) -> None:
        self.context_mean.copy_(context.mean(dim=0))
        self.context_scale.copy_(context.std(dim=0, unbiased=False).clamp_min(1.0e-6))
        logged = torch.log1p(residual.clamp_min(0.0))
        self.residual_mean.copy_(logged.mean(dim=0))
        self.residual_scale.copy_(logged.std(dim=0, unbiased=False).clamp_min(1.0e-6))

    def forward(self, context: torch.Tensor, residual: torch.Tensor) -> ShadowRiskOutput:
        context_normalized = (context - self.context_mean) / self.context_scale
        residual_logged = torch.log1p(residual.clamp_min(0.0))
        residual_normalized = (residual_logged - self.residual_mean) / self.residual_scale
        context_latent = self.context(context_normalized)
        core = residual_normalized[:, [0, 1, 4]]
        auxiliary = residual_normalized[:, [2, 3, 5, 6]]
        residual_latent = self.residual(core)
        auxiliary_latent = self.auxiliary(auxiliary)
        # Physics residuals carry the dominant, monotone defect signal. The
        # high-dimensional context is only allowed a bounded correction so a
        # small trajectory set cannot override that signal out of sample.
        median = self.residual_median(core).squeeze(-1)
        median = median + self.context_correction_bound * torch.tanh(
            self.context_median(context_latent).squeeze(-1)
        )
        median = median + self.auxiliary_median(auxiliary).squeeze(-1)
        margin = self.residual_margin(core).squeeze(-1)
        margin = margin + self.context_correction_bound * torch.tanh(
            self.context_margin(context_latent).squeeze(-1)
        )
        margin = margin + self.context_correction_bound * torch.tanh(
            self.auxiliary_margin(auxiliary).squeeze(-1)
        )
        # Location and safety margin are optimized separately. Quantile
        # underprediction penalties must not bias the median location upward.
        upper = median.detach() + F.softplus(margin)
        fused = torch.cat((context_latent, residual_latent, auxiliary_latent), dim=1)
        direction = self.direction_head(fused).squeeze(-1)
        unsafe_logit = self.unsafe_head(
            torch.cat((fused, upper[:, None]), dim=1)
        ).squeeze(-1)
        return ShadowRiskOutput(median, upper, direction, unsafe_logit)
