"""Causal, representation-aware multi-horizon error model for CMAME-M2.4.

The network never advances the PDE.  It predicts the error-budget consumption
of a *pre-registered numerical action*.  Monotonicity in the forecast horizon
is structural: positive increments are accumulated in log-error space.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


HORIZONS = (1, 2, 4, 8)
ROUTES = ("smooth", "particle", "hybrid_amr")


@dataclass(frozen=True, slots=True)
class CausalRouteOutput:
    log_error_ratio: torch.Tensor
    log_cost: torch.Tensor
    budget_fraction: torch.Tensor


class CausalRepresentationRouter(nn.Module):
    """Small conditional MLP used only for error/cost prediction.

    ``state_features`` contain causal pre-step diagnostics and
    ``action_features`` describe the candidate route, mesh and time action.
    The first horizon is unconstrained; later horizons are non-negative
    increments, hence ``e_1 <= e_2 <= e_4 <= e_8`` by construction.
    """

    def __init__(
        self, *, state_channels: int = 16, action_channels: int = 8,
        hidden: int = 64, horizons: tuple[int, ...] = HORIZONS,
    ) -> None:
        super().__init__()
        if sorted(horizons) != list(horizons) or horizons[0] != 1:
            raise ValueError("horizons must be increasing and start at one")
        self.state_channels = int(state_channels)
        self.action_channels = int(action_channels)
        self.horizons = tuple(map(int, horizons))
        width = self.state_channels + self.action_channels
        self.encoder = nn.Sequential(
            nn.Linear(width, hidden), nn.SiLU(), nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden), nn.SiLU(), nn.LayerNorm(hidden),
        )
        self.first_error = nn.Linear(hidden, 1)
        self.error_increments = nn.Linear(hidden, len(self.horizons) - 1)
        self.cost = nn.Linear(hidden, 1)
        self.budget = nn.Linear(hidden, 1)
        self.register_buffer("state_mean", torch.zeros(1, self.state_channels))
        self.register_buffer("state_scale", torch.ones(1, self.state_channels))
        self.register_buffer("action_mean", torch.zeros(1, self.action_channels))
        self.register_buffer("action_scale", torch.ones(1, self.action_channels))

    @torch.no_grad()
    def set_normalization(self, state: torch.Tensor, action: torch.Tensor) -> None:
        if state.ndim != 2 or action.ndim != 2:
            raise ValueError("normalization arrays must be rank two")
        self.state_mean.copy_(state.mean(0, keepdim=True))
        self.state_scale.copy_(state.std(0, keepdim=True, unbiased=False).clamp_min(1.0e-8))
        self.action_mean.copy_(action.mean(0, keepdim=True))
        self.action_scale.copy_(action.std(0, keepdim=True, unbiased=False).clamp_min(1.0e-8))

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> CausalRouteOutput:
        if state.shape[-1] != self.state_channels or action.shape[-1] != self.action_channels:
            raise ValueError("M2.4 feature width does not match the frozen model")
        x = torch.cat((
            (state - self.state_mean) / self.state_scale,
            (action - self.action_mean) / self.action_scale,
        ), dim=-1)
        hidden = self.encoder(x)
        first = self.first_error(hidden)
        increments = F.softplus(self.error_increments(hidden))
        log_error = torch.cat((first, first + torch.cumsum(increments, dim=-1)), dim=-1)
        return CausalRouteOutput(
            log_error,
            self.cost(hidden)[:, 0],
            torch.sigmoid(self.budget(hidden)[:, 0]),
        )


def causal_horizon_loss(
    prediction: torch.Tensor, target: torch.Tensor, *, epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Causal Huber loss with detached prefix weights.

    A later horizon cannot dominate optimization while earlier predictions are
    still inaccurate.  Detaching the weights prevents the model from reducing
    the objective by manipulating the weights themselves.
    """
    if prediction.shape != target.shape or prediction.ndim != 2:
        raise ValueError("causal targets must have shape [batch, horizons]")
    raw = F.smooth_l1_loss(prediction, target, reduction="none")
    prefix = torch.cumsum(raw.detach(), dim=-1) - raw.detach()
    weights = torch.exp(-float(epsilon) * prefix)
    return torch.mean(weights * raw), weights

