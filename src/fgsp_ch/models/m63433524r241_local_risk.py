"""Small target-cell local risk head used only by R2.4.1."""

from __future__ import annotations

import torch
from torch import nn


class TargetCellLocalRiskHead(nn.Module):
    """A deliberately small MLP returning an unsafe logit.

    It is not an unrestricted solver surrogate.  It augments a frozen global
    certificate at the one unresolved grid/time cell and is later used only
    through an upper ensemble score.
    """

    def __init__(self, input_channels: int, hidden_channels: int = 32) -> None:
        super().__init__()
        if input_channels <= 0 or hidden_channels <= 0:
            raise ValueError("network dimensions must be positive")
        self.input_channels = int(input_channels)
        self.hidden_channels = int(hidden_channels)
        self.network = nn.Sequential(
            nn.Linear(self.input_channels, self.hidden_channels), nn.SiLU(),
            nn.Linear(self.hidden_channels, self.hidden_channels), nn.SiLU(),
            nn.Linear(self.hidden_channels, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2 or features.shape[1] != self.input_channels:
            raise ValueError("local-risk features have an unexpected shape")
        return self.network(features).squeeze(-1)
