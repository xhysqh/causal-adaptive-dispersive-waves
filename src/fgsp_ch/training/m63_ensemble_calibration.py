from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from fgsp_ch.datasets.m63_branch_dataset import FieldBranchSample, ParticleBranchSample
from fgsp_ch.models.mch_pmwno import CertifiedPeakMeasureWaveletOperator


@dataclass(frozen=True, slots=True)
class M63Calibration:
    temporal_scale: float
    spatial_scale: float
    local_scale: float
    particle_scale: float
    particle_floor: float
    requested_coverage: float


@dataclass(frozen=True, slots=True)
class M63Coverage:
    temporal: float
    spatial: float
    local: float
    particle: float


def _quantile_scale(target: np.ndarray, predictor: np.ndarray, coverage: float) -> float:
    ratios = np.asarray(target) / np.maximum(np.asarray(predictor), np.finfo(np.float64).eps)
    rank = min(ratios.size - 1, int(np.ceil((ratios.size + 1) * coverage)) - 1)
    return max(1.0, float(np.sort(ratios.reshape(-1))[rank]))


def field_ensemble_prediction(
    models: list[CertifiedPeakMeasureWaveletOperator], samples: list[FieldBranchSample],
    device: torch.device,
) -> dict[str, object]:
    field = torch.as_tensor(np.stack([row.field for row in samples]), device=device, dtype=torch.float32)
    parameters = torch.as_tensor(np.stack([row.parameters for row in samples]), device=device, dtype=torch.float32)
    batch = len(samples)
    tokens = torch.zeros(batch, 1, 8, device=device)
    positions = torch.zeros(batch, 1, device=device)
    amplitudes = torch.zeros(batch, 1, device=device)
    mask = torch.zeros(batch, 1, dtype=torch.bool, device=device)
    length = parameters[:, 2] * (2.0 * torch.pi)
    outputs = []
    with torch.no_grad():
        for model in models:
            outputs.append(model(tokens, positions, amplitudes, mask, field, parameters, length))
    temporal_members = torch.stack([row.temporal_error_ratio for row in outputs])
    spatial_members = torch.stack([row.h1_uncertainty for row in outputs])
    local_members = [
        torch.stack([row.spatial_detail_score[level] for row in outputs])
        for level in range(models[0].wavelet_levels)
    ]
    return {
        "temporal_mean": temporal_members.mean(0).cpu().numpy(),
        "temporal_std": temporal_members.std(0, unbiased=False).cpu().numpy(),
        "spatial_mean": spatial_members.mean(0).cpu().numpy(),
        "spatial_std": spatial_members.std(0, unbiased=False).cpu().numpy(),
        "local_mean": tuple(value.mean(0).cpu().numpy() for value in local_members),
        "local_std": tuple(value.std(0, unbiased=False).cpu().numpy() for value in local_members),
    }


def particle_ensemble_prediction(
    models: list[CertifiedPeakMeasureWaveletOperator], samples: list[ParticleBranchSample],
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    stack = lambda name, dtype=torch.float32: torch.as_tensor(
        np.stack([getattr(row, name) for row in samples]), device=device, dtype=dtype
    )
    tokens, positions, amplitudes = stack("tokens"), stack("positions"), stack("amplitudes")
    mask = stack("mask", torch.bool)
    parameters = stack("parameters")
    length = torch.as_tensor([row.length for row in samples], device=device, dtype=torch.float32)
    with torch.no_grad():
        members = torch.stack([
            model.peakon(tokens, positions, amplitudes, mask, parameters, length)[0]
            for model in models
        ])
    return members.mean(0).cpu().numpy(), members.std(0, unbiased=False).cpu().numpy(), mask.cpu().numpy()


def fit_m63_ensemble_calibration(
    models: list[CertifiedPeakMeasureWaveletOperator],
    field_calibration: list[FieldBranchSample],
    particle_calibration: list[ParticleBranchSample],
    *, device: torch.device, coverage: float = 0.9,
) -> M63Calibration:
    if len(models) < 3 or not 0.0 < coverage < 1.0:
        raise ValueError("M6.3.2 requires three models and valid coverage.")
    field = field_ensemble_prediction(models, field_calibration, device)
    temporal_target = np.asarray([row.temporal_error_ratio for row in field_calibration])
    spatial_target = np.asarray([row.spatial_error_ratio for row in field_calibration])
    temporal_base = field["temporal_mean"] + field["temporal_std"]
    spatial_base = field["spatial_mean"] + field["spatial_std"]
    local_target = np.concatenate([
        np.stack([row.spatial_error_levels[level] for row in field_calibration]).reshape(-1)
        for level in range(models[0].wavelet_levels)
    ])
    local_base = np.concatenate([
        (field["local_mean"][level] + field["local_std"][level]).reshape(-1)
        for level in range(models[0].wavelet_levels)
    ])
    particle_mean, particle_std, particle_mask = particle_ensemble_prediction(
        models, particle_calibration, device
    )
    particle_target = np.stack([row.position_rate for row in particle_calibration])
    active_target = particle_target[particle_mask]
    particle_floor = max(0.05 * float(np.std(active_target)), 1.0e-6)
    particle_error = np.abs(particle_mean - particle_target)[particle_mask]
    particle_base = particle_std[particle_mask] + particle_floor
    return M63Calibration(
        _quantile_scale(temporal_target, temporal_base, coverage),
        _quantile_scale(spatial_target, spatial_base, coverage),
        _quantile_scale(local_target, local_base, coverage),
        _quantile_scale(particle_error, particle_base, coverage),
        particle_floor,
        coverage,
    )


def audit_m63_coverage(
    models: list[CertifiedPeakMeasureWaveletOperator], calibration: M63Calibration,
    fields: list[FieldBranchSample], particles: list[ParticleBranchSample],
    *, device: torch.device,
) -> M63Coverage:
    field = field_ensemble_prediction(models, fields, device)
    temporal_upper = calibration.temporal_scale * (field["temporal_mean"] + field["temporal_std"])
    spatial_upper = calibration.spatial_scale * (field["spatial_mean"] + field["spatial_std"])
    temporal_target = np.asarray([row.temporal_error_ratio for row in fields])
    spatial_target = np.asarray([row.spatial_error_ratio for row in fields])
    local_upper = np.concatenate([
        calibration.local_scale * (field["local_mean"][level] + field["local_std"][level]).reshape(-1)
        for level in range(models[0].wavelet_levels)
    ])
    local_target = np.concatenate([
        np.stack([row.spatial_error_levels[level] for row in fields]).reshape(-1)
        for level in range(models[0].wavelet_levels)
    ])
    mean, std, mask = particle_ensemble_prediction(models, particles, device)
    target = np.stack([row.position_rate for row in particles])
    upper = calibration.particle_scale * (std + calibration.particle_floor)
    return M63Coverage(
        float(np.mean(temporal_upper + 1e-7 >= temporal_target)),
        float(np.mean(spatial_upper + 1e-7 >= spatial_target)),
        float(np.mean(local_upper + 1e-7 >= local_target)),
        float(np.mean(upper[mask] + 1e-7 >= np.abs(mean - target)[mask])),
    )
