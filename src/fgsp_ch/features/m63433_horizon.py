from __future__ import annotations

import numpy as np
import torch

from fgsp_ch.models.m63432_deployment_safety import DeploymentCandidateSafetyOperator


HORIZON_FEATURE_ORDER = (
    *(f"candidate_{index}" for index in range(32)),
    "rollout_step_fraction",
    "legacy_log_support_ratio",
    "legacy_deployment_risk",
    "log_minimum_separation_over_h",
    "log_amplitude_ratio",
    "signed_amplitude_balance",
    "particle_count_fraction",
    "equation_alpha",
    "equation_gamma",
    "patch_fraction",
    "detail_relative_rms",
)


@torch.no_grad()
def deployment_risk_coordinates(
    models: list[DeploymentCandidateSafetyOperator],
    features: np.ndarray,
    *,
    sigma: float,
    defect_correction: float,
    direction_correction: float,
    device: torch.device,
) -> tuple[float, float, float, float]:
    tensor = torch.as_tensor(features[None], dtype=torch.float32, device=device)
    outputs = [model(tensor) for model in models]
    probability = np.asarray([
        float(torch.sigmoid(row.unsafe_logit[0]).cpu()) for row in outputs
    ])
    defect = np.asarray([float(row.log_defect[0].cpu()) for row in outputs])
    direction = np.asarray([float(row.direction[0].cpu()) for row in outputs])
    probability_upper = float(probability.mean() + sigma * probability.std())
    defect_upper = float(
        defect.mean() + sigma * defect.std() + defect_correction
    )
    direction_lower = float(
        direction.mean() - sigma * direction.std() - direction_correction
    )
    defect_vote = float(1.0 / (1.0 + np.exp(-8.0 * (
        defect_upper - np.log1p(0.1)
    ))))
    direction_vote = float(1.0 / (1.0 + np.exp(8.0 * (
        direction_lower + 0.05
    ))))
    risk = max(probability_upper, defect_vote, direction_vote)
    return probability_upper, defect_upper, direction_lower, risk


def particle_geometry_features(state, *, length: float, h: float) -> np.ndarray:
    positions = np.asarray(state.positions, dtype=np.float64)
    amplitudes = np.asarray(state.amplitudes, dtype=np.float64)
    if positions.size < 2:
        separation = 0.0
        amplitude_ratio = 0.0
    else:
        pairwise = np.abs(positions[:, None] - positions[None, :])
        pairwise = np.minimum(pairwise, length - pairwise)
        pairwise += np.eye(positions.size) * length
        separation = float(np.min(pairwise))
        absolute = np.abs(amplitudes)
        amplitude_ratio = float(
            np.max(absolute) / max(np.min(absolute), np.finfo(float).eps)
        )
    balance = float(
        np.sum(amplitudes) / max(np.sum(np.abs(amplitudes)), np.finfo(float).eps)
    ) if amplitudes.size else 0.0
    return np.asarray((
        np.log1p(separation / max(h, np.finfo(float).eps)),
        np.log1p(amplitude_ratio),
        balance,
        min(float(positions.size) / 4.0, 2.0),
    ))


def horizon_causal_features(
    candidate_features: np.ndarray,
    state,
    *,
    rollout_step: int,
    maximum_rollout_steps: int,
    support_ratio: float,
    deployment_risk: float,
    length: float,
    h: float,
    alpha: float,
    gamma: float,
) -> np.ndarray:
    candidate = np.asarray(candidate_features, dtype=np.float64)
    if candidate.shape != (32,):
        raise ValueError("candidate_features must have 32 channels.")
    support = np.log1p(min(max(float(support_ratio), 0.0), 1.0e6))
    patch_fraction = float(np.mean(state.composite_field.patch.coarse_mask))
    detail = np.asarray(state.composite_field.fine_detail_momentum, dtype=np.float64)
    coarse = np.asarray(state.composite_field.coarse_momentum, dtype=np.float64)
    detail_relative_rms = float(
        np.sqrt(np.mean(detail**2))
        / max(np.sqrt(np.mean(coarse**2)), np.finfo(float).eps)
    )
    result = np.concatenate((
        candidate,
        np.asarray((
            float(rollout_step) / max(int(maximum_rollout_steps), 1),
            support,
            float(deployment_risk),
        )),
        particle_geometry_features(state, length=length, h=h),
        np.asarray((float(alpha), float(gamma), patch_fraction, detail_relative_rms)),
    ))
    if result.shape != (43,) or not np.all(np.isfinite(result)):
        raise ValueError("horizon causal features must be finite and 43-dimensional.")
    return result
