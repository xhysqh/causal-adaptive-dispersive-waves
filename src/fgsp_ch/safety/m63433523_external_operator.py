from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from fgsp_ch.safety.m6343352_gram_support import GramSupportCertificate
from fgsp_ch.safety.m63433522_scale_certificate import (
    joint_reachability_score,
    temperature_scaled_hazards,
    temperature_scaled_cumulative,
)


@dataclass(frozen=True)
class FrozenScaleDecision:
    accepted: bool
    cell: str
    threshold: float
    joint_risk: np.ndarray
    cumulative_risk_upper: np.ndarray
    log_severity_upper: np.ndarray
    log_direction_upper: np.ndarray
    support_ratio: float
    hard_gate_passed: bool


@torch.no_grad()
def evaluate_frozen_scale_risk(
    models,
    features: np.ndarray,
    *,
    cell: str,
    calibration: dict,
    support: GramSupportCertificate,
    hard_gate_passed: bool,
    device: torch.device,
) -> FrozenScaleDecision:
    tensor = torch.as_tensor(
        np.asarray(features, dtype=np.float32)[None], device=device,
    )
    outputs = [model(tensor) for model in models]
    hazards = np.stack([row.hazard.cpu().numpy() for row in outputs])
    scaled_hazards = temperature_scaled_hazards(
        hazards, float(calibration["temperature"]),
    )
    shrinkage = float(calibration.get("hazard_shrinkage", 0.0))
    base = np.asarray(
        calibration.get("hazard_base_by_cell", {}).get(cell, np.zeros(hazards.shape[-1])),
        dtype=np.float64,
    )
    calibrated_hazards = (1.0 - shrinkage) * scaled_hazards + shrinkage * base[None, None, :]
    cumulative_members = 1.0 - np.cumprod(1.0 - calibrated_hazards, axis=2)
    cumulative_upper = np.clip(
        cumulative_members.mean(axis=0)[0] + cumulative_members.std(axis=0)[0],
        0.0, 1.0,
    )
    cumulative_upper = np.maximum.accumulate(cumulative_upper)
    severity_members = np.stack([
        row.log_cumulative_severity[0].cpu().numpy() for row in outputs
    ])
    direction_members = np.stack([
        row.log_direction_risk[0].cpu().numpy() for row in outputs
    ])
    severity_upper = (
        severity_members.mean(axis=0) + severity_members.std(axis=0)
        + np.asarray(calibration["cell_margins"][cell], dtype=np.float64)
    )
    direction_upper = (
        direction_members.mean(axis=0) + direction_members.std(axis=0)
        + np.asarray(calibration["direction_cell_margins"][cell], dtype=np.float64)
    )
    embedding = np.stack([
        row.support_embedding[0].cpu().numpy() for row in outputs
    ]).mean(axis=0, keepdims=True)
    support_ratio = float(support.support_ratio(embedding)[0])
    joint = np.maximum.accumulate(joint_reachability_score(
        cumulative_upper, severity_upper, direction_upper,
    ))
    deployment_column = int(calibration["deployment_horizon"]) - 1
    threshold = float(calibration["selected_cell_thresholds"][cell])
    accepted = bool(
        hard_gate_passed and support_ratio <= 1.0
        and joint[deployment_column] <= threshold
    )
    return FrozenScaleDecision(
        accepted, str(cell), threshold, joint, cumulative_upper,
        severity_upper, direction_upper, support_ratio, bool(hard_gate_passed),
    )
