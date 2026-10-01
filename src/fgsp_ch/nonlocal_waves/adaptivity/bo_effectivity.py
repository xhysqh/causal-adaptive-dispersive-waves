"""Selective learned effectivity control for the periodic Benjamin--Ono solver.

The neural operator in this module never advances the PDE.  It maps causal,
translation-invariant pre-step diagnostics to conservative temporal/spatial
effectivity multipliers and to the value of an N--2N hierarchical probe.
Deterministic invariant gates and :mod:`bo_certified` remain authoritative.
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
from fgsp_ch.nonlocal_waves.operators.bo_fourier import BOFourierOperator


SPECTRAL_BANDS = 8
FEATURE_ORDER = tuple([f"log_band_energy_{i}" for i in range(SPECTRAL_BANDS)] + [
    "log_tail_ratio", "log_aliasing_defect", "log_nonlinear_dispersion_ratio",
    "log_maximum_linear_phase", "log_embedded_defect", "log_residual_defect",
    "log_dt", "log_points", "log_tolerance", "log_local_budget",
    "remaining_time_fraction", "remaining_budget_fraction",
    "mass_budget_fraction", "l2_budget_fraction", "hamiltonian_budget_fraction",
    "previous_rejected",
])
RISK_FEATURES = (
    "log_tail_ratio", "log_aliasing_defect", "log_nonlinear_dispersion_ratio",
    "log_maximum_linear_phase", "log_embedded_defect", "log_residual_defect",
    "mass_budget_fraction", "l2_budget_fraction", "hamiltonian_budget_fraction",
)
ACTION_FACTORS = tuple((n, dt) for n in (0.5, 1.0, 2.0) for dt in (0.5, 1.0, 1.5))


def _safe_log(value: float) -> float:
    return float(np.log(max(float(value), 1.0e-16)))


def spectral_band_features(values: NDArray[np.floating], grid: PeriodicGrid) -> NDArray[np.float64]:
    """Return phase/translation-invariant normalized Fourier-band energies."""
    spectrum = BOFourierOperator(grid).spectrum(np.asarray(values, dtype=np.float64))
    energy = (1.0 + BOFourierOperator(grid).wave_numbers**2) * np.abs(spectrum) ** 2
    total = max(float(np.sum(energy)), np.finfo(float).tiny)
    edges = np.linspace(0, energy.size, SPECTRAL_BANDS + 1, dtype=int)
    bands = [float(np.sum(energy[edges[i]:edges[i + 1]]) / total) for i in range(SPECTRAL_BANDS)]
    return np.log(np.maximum(np.asarray(bands, dtype=np.float64), 1.0e-16))


def build_effectivity_features(
    values: NDArray[np.floating], grid: PeriodicGrid, indicators: BOResolutionIndicators,
    *, embedded: float, residual: float, dt: float, tolerance: float,
    local_budget: float, remaining_time_fraction: float, remaining_budget_fraction: float,
    invariant_budget_fractions: Sequence[float] = (0.0, 0.0, 0.0),
    previous_rejected: bool = False,
) -> NDArray[np.float64]:
    """Build the frozen P2 causal feature vector; no reference values enter it."""
    invariant = np.asarray(invariant_budget_fractions, dtype=np.float64)
    if invariant.shape != (3,):
        raise ValueError("three invariant budget fractions are required")
    result = np.concatenate((spectral_band_features(values, grid), np.asarray([
        _safe_log(indicators.tail_ratio), _safe_log(indicators.aliasing_defect),
        _safe_log(indicators.nonlinear_dispersion_ratio), _safe_log(indicators.maximum_linear_phase),
        _safe_log(embedded), _safe_log(residual), _safe_log(dt), _safe_log(grid.points),
        _safe_log(tolerance), _safe_log(local_budget), float(remaining_time_fraction),
        float(remaining_budget_fraction), *invariant.tolist(), float(previous_rejected),
    ], dtype=np.float64)))
    if result.shape != (len(FEATURE_ORDER),) or not np.all(np.isfinite(result)):
        raise ValueError("invalid BO-P2 effectivity feature vector")
    return result


@dataclass(frozen=True, slots=True)
class EffectivityOutput:
    temporal_median: torch.Tensor
    temporal_upper: torch.Tensor
    spatial_median: torch.Tensor
    spatial_upper: torch.Tensor
    probe_logit: torch.Tensor
    action_log_risk: torch.Tensor


class BOEffectivityOperator(nn.Module):
    """Small monotone quantile operator over translation-invariant diagnostics."""

    def __init__(self, *, channels: int = len(FEATURE_ORDER), hidden: int = 64) -> None:
        super().__init__()
        self.channels = int(channels)
        self.hidden = int(hidden)
        risk_index = [FEATURE_ORDER.index(name) for name in RISK_FEATURES]
        context_index = [i for i in range(channels) if i not in risk_index]
        self.register_buffer("risk_index", torch.tensor(risk_index, dtype=torch.long))
        self.register_buffer("context_index", torch.tensor(context_index, dtype=torch.long))
        self.register_buffer("feature_mean", torch.zeros(1, channels))
        self.register_buffer("feature_scale", torch.ones(1, channels))
        self.context = nn.Sequential(
            nn.Linear(len(context_index), hidden), nn.SiLU(), nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden), nn.SiLU(),
        )
        # Positive risk weights make all risk heads coordinatewise monotone.
        self.risk_weight = nn.Parameter(torch.zeros(5, len(risk_index)))
        self.context_head = nn.Linear(hidden, 5)
        self.action_head = nn.Linear(hidden, len(ACTION_FACTORS))
        self.action_risk_weight = nn.Parameter(torch.zeros(len(ACTION_FACTORS), len(risk_index)))

    @torch.no_grad()
    def set_normalization(self, features: torch.Tensor) -> None:
        if features.ndim != 2 or features.shape[1] != self.channels:
            raise ValueError("BO-P2 normalization shape mismatch")
        self.feature_mean.copy_(features.mean(0, keepdim=True))
        self.feature_scale.copy_(features.std(0, keepdim=True, unbiased=False).clamp_min(1.0e-8))

    def forward(self, features: torch.Tensor) -> EffectivityOutput:
        if features.ndim != 2 or features.shape[1] != self.channels:
            raise ValueError("BO-P2 feature width mismatch")
        normalized = (features - self.feature_mean) / self.feature_scale
        context = self.context(normalized.index_select(1, self.context_index))
        # Softplus-normalized risk coordinates remain monotone after positive scaling.
        risk = normalized.index_select(1, self.risk_index)
        raw = self.context_head(context) + risk @ F.softplus(self.risk_weight).T
        temporal_median = raw[:, 0]
        temporal_upper = temporal_median + F.softplus(raw[:, 1])
        spatial_median = raw[:, 2]
        spatial_upper = spatial_median + F.softplus(raw[:, 3])
        action = self.action_head(context) + risk @ F.softplus(self.action_risk_weight).T
        return EffectivityOutput(
            temporal_median, temporal_upper, spatial_median, spatial_upper, raw[:, 4], action,
        )


def pinball_loss(prediction: torch.Tensor, target: torch.Tensor, quantile: float) -> torch.Tensor:
    error = target - prediction
    return torch.maximum(float(quantile) * error, (float(quantile) - 1.0) * error).mean()


def effectivity_training_loss(
    output: EffectivityOutput, temporal_log_effectivity: torch.Tensor,
    spatial_log_effectivity: torch.Tensor, probe_required: torch.Tensor,
    action_log_risk: torch.Tensor, *, upper_quantile: float = 0.95,
) -> tuple[torch.Tensor, dict[str, float]]:
    median = pinball_loss(output.temporal_median, temporal_log_effectivity, 0.5) + pinball_loss(
        output.spatial_median, spatial_log_effectivity, 0.5
    )
    upper = pinball_loss(output.temporal_upper, temporal_log_effectivity, upper_quantile) + pinball_loss(
        output.spatial_upper, spatial_log_effectivity, upper_quantile
    )
    probe = F.binary_cross_entropy_with_logits(output.probe_logit, probe_required)
    action = F.smooth_l1_loss(output.action_log_risk, action_log_risk)
    # Extra asymmetric penalty makes under-estimation materially more expensive.
    under = F.relu(temporal_log_effectivity - output.temporal_upper).mean() + F.relu(
        spatial_log_effectivity - output.spatial_upper
    ).mean()
    total = median + 2.0 * upper + 1.5 * probe + 0.5 * action + 4.0 * under
    return total, {"median": float(median.detach()), "upper": float(upper.detach()),
                   "probe": float(probe.detach()), "action": float(action.detach()),
                   "under": float(under.detach())}


def conformal_upper_margins(
    member_upper: NDArray[np.floating], truth: NDArray[np.floating], *, coverage: float,
) -> NDArray[np.float64]:
    """Split-conformal margins for temporal and spatial log-effectivities."""
    members = np.asarray(member_upper, dtype=np.float64)
    target = np.asarray(truth, dtype=np.float64)
    if members.ndim != 3 or members.shape[2] != 2 or target.shape != members.shape[1:]:
        raise ValueError("upper members must be [ensemble, rows, 2]")
    if not 0.5 < coverage < 1.0:
        raise ValueError("coverage must lie in (0.5,1)")
    envelope = members.mean(0) + members.std(0)
    return np.maximum(0.0, np.quantile(target - envelope, coverage, axis=0, method="higher"))


@dataclass(frozen=True, slots=True)
class SelectiveAssessment:
    temporal_error_upper: float
    spatial_error_upper: float
    probe_required: bool
    probe_probability: float
    uncertainty_width: float
    out_of_distribution: bool
    action_log_risk: tuple[float, ...]


class FrozenBOEffectivityEnsemble:
    """Frozen ensemble with conformal bounds and explicit abstention."""

    def __init__(self, models: Sequence[BOEffectivityOperator], manifest: dict, device: torch.device):
        if len(models) < 3:
            raise ValueError("BO-P2 requires at least three frozen members")
        self.models = tuple(model.eval() for model in models)
        self.manifest = manifest
        self.device = device

    @classmethod
    def load(cls, root: Path, manifest: dict, device: str | torch.device = "cpu"):
        target = torch.device(device); models = []
        for name in manifest["checkpoints"]:
            payload = torch.load(root / name, map_location=target, weights_only=False)
            model = BOEffectivityOperator(channels=payload["channels"], hidden=payload["hidden"]).to(target)
            model.load_state_dict(payload["model"]); models.append(model)
        return cls(models, manifest, target)

    @torch.no_grad()
    def assess(self, features: NDArray[np.floating], *, embedded: float, residual: float) -> SelectiveAssessment:
        tensor = torch.as_tensor(np.asarray(features, dtype=np.float32)[None], device=self.device)
        outputs = [model(tensor) for model in self.models]
        temporal = np.asarray([float(row.temporal_upper[0].cpu()) for row in outputs])
        spatial = np.asarray([float(row.spatial_upper[0].cpu()) for row in outputs])
        probe_probability = float(np.mean([torch.sigmoid(row.probe_logit[0]).item() for row in outputs]))
        actions = np.mean([row.action_log_risk[0].cpu().numpy() for row in outputs], axis=0)
        margins = np.asarray(self.manifest["conformal_margins"], dtype=np.float64)
        upper = np.asarray([temporal.mean() + temporal.std(), spatial.mean() + spatial.std()]) + margins
        width = float(max(temporal.std(), spatial.std()))
        mean = np.asarray(self.manifest["support_mean"], dtype=np.float64)
        scale = np.asarray(self.manifest["support_scale"], dtype=np.float64)
        distance = float(np.sqrt(np.mean(((np.asarray(features) - mean) / np.maximum(scale, 1e-8)) ** 2)))
        ood = distance > float(self.manifest["support_radius"])
        required = (probe_probability >= float(self.manifest["probe_probability_threshold"]) or
                    width > float(self.manifest["maximum_uncertainty_width"]) or ood)
        return SelectiveAssessment(
            float(np.exp(np.clip(upper[0], -20.0, 20.0)) * embedded),
            float(np.exp(np.clip(upper[1], -20.0, 20.0)) * residual), required,
            probe_probability, width, ood, tuple(map(float, actions)),
        )
