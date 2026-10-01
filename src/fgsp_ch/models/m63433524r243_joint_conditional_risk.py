"""Joint conditional risk calibration head for R2.4.3.

Takes the 4-dimensional joint meta-feature vector z_joint produced by
``fgsp_ch.features.m63433524r243_joint_conditional_risk`` and outputs a
single scalar calibrated unsafe probability.  The model is deliberately
tiny: it is a post-hoc calibration layer, not a surrogate solver.

Architecture: Linear(4 → hidden) → SiLU → Linear(hidden → 1).
"""

from __future__ import annotations

import torch
from torch import nn

JOINT_META_FEATURE_DIM = 4  # matches features module


class JointConditionalRiskCalibrator(nn.Module):
    """Tiny MLP that calibrates the joint max-score into a posterior probability.

    Input:  z_joint ∈ R^4  (base_r23_score, cond_local_upper,
                             disagreement_abs, disagreement_signed)
    Output: scalar unsafe logit (apply sigmoid for probability).
    """

    def __init__(
        self,
        input_dim: int = JOINT_META_FEATURE_DIM,
        hidden_dim: int = 16,
    ) -> None:
        super().__init__()
        if input_dim <= 0 or hidden_dim <= 0:
            raise ValueError("input_dim and hidden_dim must be positive integers")
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.network = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Return scalar unsafe logit per sample.

        Parameters
        ----------
        features:
            Shape ``(N, input_dim)``.

        Returns
        -------
        torch.Tensor
            Shape ``(N,)`` – raw logits (not probabilities).
        """
        if features.ndim != 2 or features.shape[1] != self.input_dim:
            raise ValueError(
                f"Expected (N, {self.input_dim}) input, got {tuple(features.shape)}"
            )
        return self.network(features).squeeze(-1)
