"""Causality-respecting selective pruning of scheduled BO hierarchy probes.

The pruner is deliberately less powerful than the deterministic P1.1
controller.  It is queried only when P1.1 has already scheduled a regular
N--2N probe; it can cancel that probe, but can never add a probe, alter an
error estimate, advance the PDE, or select ``N``/``dt``.  Hard spectral
triggers and out-of-distribution inputs always follow P1.1.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from numpy.typing import NDArray
import torch
from torch import nn
from torch.nn import functional as F

from fgsp_ch.discretization.grids import PeriodicGrid
from fgsp_ch.nonlocal_waves.adaptivity.bo_classical import BOResolutionIndicators
from fgsp_ch.nonlocal_waves.adaptivity.bo_effectivity import spectral_band_features


FEATURE_ORDER = tuple([f"log_band_energy_{i}" for i in range(8)] + [
    "log_tail_ratio", "log_aliasing_defect", "log_nonlinear_dispersion_ratio",
    "log_maximum_linear_phase", "log_embedded_defect", "log_residual_defect",
    "log_dt", "log_points", "previous_rejected",
])


def _safe_log(value: float) -> float:
    return float(np.log(max(float(value), 1.0e-16)))


def build_probe_features(
    values: NDArray[np.floating],
    grid: PeriodicGrid,
    indicators: BOResolutionIndicators,
    *,
    embedded: float,
    residual: float,
    dt: float,
    previous_rejected: bool = False,
) -> NDArray[np.float64]:
    """Tolerance-independent, translation-invariant pre-step features."""
    result = np.concatenate((spectral_band_features(values, grid), np.asarray([
        _safe_log(indicators.tail_ratio),
        _safe_log(indicators.aliasing_defect),
        _safe_log(indicators.nonlinear_dispersion_ratio),
        _safe_log(indicators.maximum_linear_phase),
        _safe_log(embedded),
        _safe_log(residual),
        _safe_log(dt),
        _safe_log(grid.points),
        float(previous_rejected),
    ], dtype=np.float64)))
    if result.shape != (len(FEATURE_ORDER),) or not np.all(np.isfinite(result)):
        raise ValueError("invalid BO-P2.1 probe feature vector")
    return result


@dataclass(frozen=True, slots=True)
class ProbePrunerOutput:
    hidden_median: torch.Tensor
    hidden_upper: torch.Tensor
    decision_change_logit: torch.Tensor


class BOProbePruner(nn.Module):
    """Small quantile model for the hidden N--2N defect increment."""

    def __init__(self, *, channels: int = len(FEATURE_ORDER), hidden: int = 48) -> None:
        super().__init__()
        self.channels = int(channels)
        self.hidden = int(hidden)
        self.register_buffer("feature_mean", torch.zeros(1, channels))
        self.register_buffer("feature_scale", torch.ones(1, channels))
        self.backbone = nn.Sequential(
            nn.Linear(channels, hidden), nn.SiLU(), nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden), nn.SiLU(),
        )
        self.head = nn.Linear(hidden, 3)

    @torch.no_grad()
    def set_normalization(self, features: torch.Tensor) -> None:
        if features.ndim != 2 or features.shape[1] != self.channels:
            raise ValueError("BO-P2.1 normalization shape mismatch")
        self.feature_mean.copy_(features.mean(0, keepdim=True))
        self.feature_scale.copy_(
            features.std(0, keepdim=True, unbiased=False).clamp_min(1.0e-8)
        )

    def forward(self, features: torch.Tensor) -> ProbePrunerOutput:
        if features.ndim != 2 or features.shape[1] != self.channels:
            raise ValueError("BO-P2.1 feature width mismatch")
        hidden = self.backbone((features - self.feature_mean) / self.feature_scale)
        raw = self.head(hidden)
        median = raw[:, 0]
        upper = median + F.softplus(raw[:, 1])
        return ProbePrunerOutput(median, upper, raw[:, 2])


def pinball_loss(prediction: torch.Tensor, target: torch.Tensor, quantile: float) -> torch.Tensor:
    error = target - prediction
    return torch.maximum(quantile * error, (quantile - 1.0) * error).mean()


def probe_pruner_loss(
    output: ProbePrunerOutput,
    hidden_log_excess: torch.Tensor,
    decision_change: torch.Tensor,
    *,
    upper_quantile: float = 0.95,
    false_negative_weight: float = 4.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    median = pinball_loss(output.hidden_median, hidden_log_excess, 0.5)
    upper = pinball_loss(output.hidden_upper, hidden_log_excess, upper_quantile)
    positive_weight = torch.as_tensor(
        float(false_negative_weight), device=decision_change.device
    )
    classification = F.binary_cross_entropy_with_logits(
        output.decision_change_logit, decision_change, pos_weight=positive_weight
    )
    under = F.relu(hidden_log_excess - output.hidden_upper).mean()
    total = median + 2.0 * upper + 2.0 * classification + 4.0 * under
    return total, {
        "median": float(median.detach()), "upper": float(upper.detach()),
        "classification": float(classification.detach()), "under": float(under.detach()),
    }


def trajectory_group_conformal_margin(
    predicted_upper: NDArray[np.floating],
    truth: NDArray[np.floating],
    group_ids: Sequence[str],
    *,
    coverage: float,
) -> float:
    """One-sided split-conformal margin over trajectory-maximum residuals."""
    predicted = np.asarray(predicted_upper, dtype=np.float64).reshape(-1)
    target = np.asarray(truth, dtype=np.float64).reshape(-1)
    groups = np.asarray(group_ids).astype(str)
    if predicted.shape != target.shape or groups.shape != target.shape:
        raise ValueError("BO-P2.1 conformal arrays must have equal length")
    if not 0.5 < coverage < 1.0:
        raise ValueError("coverage must lie in (0.5,1)")
    scores = np.asarray([
        np.max(target[groups == group] - predicted[groups == group])
        for group in sorted(set(groups.tolist()))
    ])
    rank = min(len(scores), int(np.ceil((len(scores) + 1) * coverage)))
    return float(max(0.0, np.partition(scores, rank - 1)[rank - 1]))


@dataclass(frozen=True, slots=True)
class ProbePrunerAssessment:
    should_probe: bool
    skip_certified: bool
    hidden_excess_upper: float
    decision_change_probability: float
    uncertainty_width: float
    out_of_distribution: bool


class FrozenBOProbePruner:
    """Frozen ensemble whose only authority is cancelling a scheduled probe."""

    def __init__(self, models: Sequence[BOProbePruner], manifest: dict, device: torch.device):
        if len(models) < 3:
            raise ValueError("BO-P2.1 requires at least three frozen members")
        self.models = tuple(model.eval() for model in models)
        self.manifest = manifest
        self.device = device

    @classmethod
    def load(cls, root: Path, manifest: dict, device: str | torch.device = "cpu"):
        target = torch.device(device)
        models = []
        for name in manifest["checkpoints"]:
            payload = torch.load(root / name, map_location=target, weights_only=False)
            model = BOProbePruner(
                channels=payload["channels"], hidden=payload["hidden"]
            ).to(target)
            model.load_state_dict(payload["model"])
            models.append(model)
        return cls(models, manifest, target)

    @torch.no_grad()
    def assess(
        self,
        features: NDArray[np.floating],
        *,
        spatial_budget: float,
        baseline_due: bool,
        hard_trigger: bool,
    ) -> ProbePrunerAssessment:
        # The early return is the structural non-intrusion guarantee: the
        # learned component cannot create a probe that P1.1 did not schedule.
        if not baseline_due:
            return ProbePrunerAssessment(False, False, np.inf, 1.0, 0.0, False)
        if hard_trigger:
            return ProbePrunerAssessment(True, False, np.inf, 1.0, 0.0, False)
        tensor = torch.as_tensor(
            np.asarray(features, dtype=np.float32)[None], device=self.device
        )
        outputs = [model(tensor) for model in self.models]
        upper = np.asarray([float(row.hidden_upper[0].cpu()) for row in outputs])
        probability = float(np.mean([
            torch.sigmoid(row.decision_change_logit[0]).item() for row in outputs
        ]))
        width = float(np.std(upper))
        vector = np.asarray(features, dtype=np.float64)
        mean = np.asarray(self.manifest["support_mean"], dtype=np.float64)
        scale = np.asarray(self.manifest["support_scale"], dtype=np.float64)
        distance = float(np.sqrt(np.mean(((vector - mean) / np.maximum(scale, 1e-8)) ** 2)))
        ood = distance > float(self.manifest["support_radius"])
        log_upper = float(np.mean(upper) + np.std(upper) + self.manifest["group_conformal_margin"])
        hidden_upper = float(np.exp(np.clip(log_upper, -32.0, 8.0)))
        skip = bool(
            not ood
            and width <= float(self.manifest["maximum_uncertainty_width"])
            and probability <= float(self.manifest["maximum_decision_change_probability"])
            and hidden_upper <= float(self.manifest["skip_budget_fraction"]) * max(spatial_budget, 0.0)
        )
        return ProbePrunerAssessment(
            should_probe=not skip,
            skip_certified=skip,
            hidden_excess_upper=hidden_upper,
            decision_change_probability=probability,
            uncertainty_width=width,
            out_of_distribution=ood,
        )
