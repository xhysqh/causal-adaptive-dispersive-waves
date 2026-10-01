from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch

from fgsp_ch.datasets.m63_branch_dataset import (
    ClusterBranchSample,
    FieldBranchSample,
    ParticleBranchSample,
)
from fgsp_ch.models.mch_pmwno import CertifiedPeakMeasureWaveletOperator


@dataclass(frozen=True, slots=True)
class BranchMetric:
    branch: str
    validation_mse: float
    constant_baseline_mse: float
    ratio_to_baseline: float
    finite: bool


def _best_constant_mse(train: np.ndarray, target: np.ndarray) -> float:
    mean = np.mean(train, axis=0, keepdims=True)
    mean_mse = float(np.mean((target - mean) ** 2))
    zero_mse = float(np.mean(target**2))
    return min(mean_mse, zero_mse)


def _optimize(loss_fn, parameters: Iterable[torch.nn.Parameter], *, epochs: int, lr: float) -> None:
    values = list(parameters)
    optimizer = torch.optim.AdamW(values, lr=lr, weight_decay=1.0e-6)
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn()
        if not torch.isfinite(loss):
            raise FloatingPointError("M6.3.1 branch loss became non-finite.")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(values, 5.0)
        optimizer.step()


def _particle_tensors(samples: list[ParticleBranchSample], device: torch.device):
    tensor = lambda values, dtype=torch.float32: torch.as_tensor(np.stack(values), device=device, dtype=dtype)
    return (
        tensor([row.tokens for row in samples]), tensor([row.positions for row in samples]),
        tensor([row.amplitudes for row in samples]), tensor([row.mask for row in samples], torch.bool),
        tensor([row.parameters for row in samples]), tensor([row.length for row in samples]),
        tensor([row.position_rate for row in samples]),
    )


def train_particle_branch(
    model: CertifiedPeakMeasureWaveletOperator,
    train: list[ParticleBranchSample], validation: list[ParticleBranchSample],
    *, epochs: int, learning_rate: float, device: torch.device,
) -> tuple[BranchMetric, float]:
    train_batch = _particle_tensors(train, device)
    target_values = np.concatenate([row.position_rate[row.mask] for row in train])
    scale = max(float(np.std(target_values)), 1.0e-8)

    def loss_fn() -> torch.Tensor:
        predicted, amplitude = model.peakon(*train_batch[:6])
        active = train_batch[3]
        return torch.mean(((predicted[active] - train_batch[6][active]) / scale) ** 2) + torch.mean(amplitude[active] ** 2)

    _optimize(loss_fn, model.peakon.parameters(), epochs=epochs, lr=learning_rate)
    batch = _particle_tensors(validation, device)
    with torch.no_grad():
        prediction = model.peakon(*batch[:6])[0].cpu().numpy()
    target = np.stack([row.position_rate for row in validation])
    mask = np.stack([row.mask for row in validation])
    mse = float(np.mean((prediction[mask] - target[mask]) ** 2))
    train_padded = np.stack([row.position_rate for row in train])
    train_mask = np.stack([row.mask for row in train])
    baseline = _best_constant_mse(train_padded[train_mask][:, None], target[mask][:, None])
    return BranchMetric("particle_position", mse, baseline, mse / max(baseline, 1e-30), bool(np.isfinite(mse))), scale


def _cluster_tensors(samples: list[ClusterBranchSample], device: torch.device):
    tensor = lambda values: torch.as_tensor(np.stack(values), device=device, dtype=torch.float32)
    return tensor([row.token for row in samples]), tensor([row.parameters for row in samples]), tensor([row.target for row in samples])


def train_cluster_branch(
    model: CertifiedPeakMeasureWaveletOperator,
    train: list[ClusterBranchSample], validation: list[ClusterBranchSample],
    *, epochs: int, learning_rate: float, device: torch.device,
) -> tuple[BranchMetric, np.ndarray, np.ndarray]:
    tokens, parameters, targets = _cluster_tensors(train, device)
    train_targets = np.stack([row.target for row in train])
    coordinate_mask_np = np.var(train_targets, axis=0) >= 1.0e-12
    if not np.any(coordinate_mask_np):
        raise RuntimeError("No identifiable cluster coordinate exceeds the reference floor.")
    model.set_cluster_coordinate_mask(torch.as_tensor(coordinate_mask_np, device=device))
    scale_np = np.maximum(np.std(train_targets, axis=0), 1.0e-8)
    scale = torch.as_tensor(scale_np, device=device, dtype=torch.float32)
    coordinate_mask = torch.as_tensor(coordinate_mask_np, device=device)

    def loss_fn() -> torch.Tensor:
        expanded = parameters[:, None, :]
        prediction = model.cluster(torch.cat((tokens[:, None, 2:], expanded), dim=-1)).squeeze(1)
        return torch.mean(((prediction[:, coordinate_mask] - targets[:, coordinate_mask]) / scale[coordinate_mask]) ** 2)

    _optimize(loss_fn, model.cluster.parameters(), epochs=epochs, lr=learning_rate)
    tokens, parameters, targets = _cluster_tensors(validation, device)
    with torch.no_grad():
        prediction = model.cluster(torch.cat((tokens[:, None, 2:], parameters[:, None, :]), dim=-1)).squeeze(1).cpu().numpy()
    target_np = targets.cpu().numpy()
    normalized_prediction = prediction[:, coordinate_mask_np] / scale_np[coordinate_mask_np]
    normalized_target = target_np[:, coordinate_mask_np] / scale_np[coordinate_mask_np]
    mse = float(np.mean((normalized_prediction - normalized_target) ** 2))
    train_normalized = train_targets[:, coordinate_mask_np] / scale_np[coordinate_mask_np]
    baseline = _best_constant_mse(train_normalized, normalized_target)
    return BranchMetric("cluster_coordinates", mse, baseline, mse / max(baseline, 1e-30), bool(np.isfinite(mse))), scale_np, coordinate_mask_np


def _field_tensors(samples: list[FieldBranchSample], device: torch.device):
    tensor = lambda values: torch.as_tensor(np.stack(values), device=device, dtype=torch.float32)
    return tensor([row.field for row in samples]), tensor([row.parameters for row in samples]), tensor([row.target_momentum_rate for row in samples])


def _field_forward(model: CertifiedPeakMeasureWaveletOperator, field: torch.Tensor) -> torch.Tensor:
    lifted = model.field_lift(field)
    pyramid = model.wavelet.encode(lifted)
    transformed = type(pyramid)(
        model.field_block(pyramid.coarse),
        tuple(model.field_block(detail) for detail in pyramid.details),
        pyramid.original_points,
    )
    rate = model.field_readout(model.wavelet.decode(transformed)).squeeze(1)
    return rate - rate.mean(dim=-1, keepdim=True)


def train_field_branch(
    model: CertifiedPeakMeasureWaveletOperator,
    train: list[FieldBranchSample], validation: list[FieldBranchSample],
    *, epochs: int, learning_rate: float, device: torch.device,
) -> tuple[BranchMetric, float]:
    fields, _, targets = _field_tensors(train, device)
    scale = max(float(np.std(np.concatenate([row.target_momentum_rate for row in train]))), 1.0e-8)

    def loss_fn() -> torch.Tensor:
        return torch.mean(((_field_forward(model, fields) - targets) / scale) ** 2)

    parameters = list(model.field_lift.parameters()) + list(model.field_block.parameters()) + list(model.field_readout.parameters())
    _optimize(loss_fn, parameters, epochs=epochs, lr=learning_rate)
    fields, _, targets = _field_tensors(validation, device)
    with torch.no_grad():
        prediction = _field_forward(model, fields).cpu().numpy()
    target_np = targets.cpu().numpy()
    mse = float(np.mean(((prediction - target_np) / scale) ** 2))
    train_normalized = np.stack([row.target_momentum_rate for row in train]) / scale
    baseline = _best_constant_mse(train_normalized, target_np / scale)
    return BranchMetric("field_momentum", mse, baseline, mse / max(baseline, 1e-30), bool(np.isfinite(mse))), scale


def _error_forward(
    model: CertifiedPeakMeasureWaveletOperator, field: torch.Tensor, parameters: torch.Tensor
) -> tuple[tuple[torch.Tensor, ...], torch.Tensor, torch.Tensor]:
    lifted = model.field_lift(field)
    pyramid = model.wavelet.encode(lifted)
    coarse = model.field_block(pyramid.coarse)
    details = tuple(model.field_block(detail) for detail in pyramid.details)
    updated = model.wavelet.decode(
        type(pyramid)(coarse, details, pyramid.original_points)
    )
    rate = model.field_readout(updated).squeeze(1)
    rate = rate - rate.mean(dim=-1, keepdim=True)
    spatial = tuple(
        torch.expm1(torch.nn.functional.softplus(head(detail))).squeeze(1)
        for head, detail in zip(model.detail_error_heads, details, strict=True)
    )
    zeros = torch.zeros(field.shape[0], device=field.device, dtype=field.dtype)
    pooled = torch.stack((
        rate.square().mean(dim=-1).sqrt(), zeros, zeros,
        field[:, 1].square().mean(dim=-1).sqrt(),
        details[0].square().mean(dim=(1, 2)).sqrt(),
    ), dim=-1)
    raw = torch.nn.functional.softplus(model.risk(torch.cat((parameters, pooled), dim=-1)))
    temporal = torch.expm1(raw[:, 0])
    uncertainty = torch.expm1(raw[:, 2])
    return spatial, temporal, uncertainty


def train_error_branches(
    model: CertifiedPeakMeasureWaveletOperator,
    train: list[FieldBranchSample], validation: list[FieldBranchSample],
    *, epochs: int, learning_rate: float, device: torch.device,
    initialize_from_train_scale: bool = True,
) -> tuple[BranchMetric, BranchMetric, BranchMetric]:
    fields, parameters, _ = _field_tensors(train, device)
    spatial_targets = tuple(
        torch.as_tensor(np.stack([row.spatial_error_levels[level] for row in train]), device=device, dtype=torch.float32)
        for level in range(model.wavelet_levels)
    )
    temporal_target = torch.as_tensor([row.temporal_error_ratio for row in train], device=device, dtype=torch.float32)
    spatial_scalar = torch.as_tensor([row.spatial_error_ratio for row in train], device=device, dtype=torch.float32)

    def inverse_softplus(value: float) -> float:
        return float(np.log(np.expm1(max(value, 1.0e-6))))

    # Initialize in the physical Richardson scale.  This is train-only target
    # normalization, not validation leakage.
    if initialize_from_train_scale:
        with torch.no_grad():
            for level, head in enumerate(model.detail_error_heads):
                desired = float(torch.log1p(spatial_targets[level]).mean().cpu())
                head.bias.fill_(inverse_softplus(desired))
                head.weight.zero_()
            final = model.risk[-1]
            final.bias[0] = inverse_softplus(float(torch.log1p(temporal_target).mean().cpu()))
            final.bias[2] = inverse_softplus(float(torch.log1p(spatial_scalar).mean().cpu()))

    def loss_fn() -> torch.Tensor:
        spatial, temporal, uncertainty = _error_forward(model, fields, parameters)
        loss = torch.mean((torch.log1p(temporal) - torch.log1p(temporal_target)) ** 2)
        loss = loss + torch.mean((torch.log1p(uncertainty) - torch.log1p(spatial_scalar)) ** 2)
        for prediction, target in zip(spatial, spatial_targets, strict=True):
            loss = loss + torch.mean((torch.log1p(prediction) - torch.log1p(target)) ** 2)
        return loss

    parameters_to_train = list(model.detail_error_heads.parameters()) + list(model.risk.parameters())
    _optimize(loss_fn, parameters_to_train, epochs=epochs, lr=learning_rate)
    fields, parameters, _ = _field_tensors(validation, device)
    with torch.no_grad():
        spatial, temporal, uncertainty = _error_forward(model, fields, parameters)
    temporal_pred = np.log1p(temporal.cpu().numpy())
    temporal_target_np = np.log1p(np.asarray([row.temporal_error_ratio for row in validation]))
    spatial_pred = np.log1p(uncertainty.cpu().numpy())
    spatial_target_np = np.log1p(np.asarray([row.spatial_error_ratio for row in validation]))
    local_prediction = np.concatenate(
        [np.log1p(value.cpu().numpy()).reshape(len(validation), -1) for value in spatial], axis=1
    )
    local_target = np.concatenate(
        [
            np.log1p(np.stack([row.spatial_error_levels[level] for row in validation]))
            for level in range(model.wavelet_levels)
        ], axis=1,
    )
    temporal_mse = float(np.mean((temporal_pred - temporal_target_np) ** 2))
    spatial_mse = float(np.mean((spatial_pred - spatial_target_np) ** 2))
    train_temporal = np.log1p(np.asarray([row.temporal_error_ratio for row in train]))
    train_spatial = np.log1p(np.asarray([row.spatial_error_ratio for row in train]))
    train_local = np.concatenate(
        [
            np.log1p(np.stack([row.spatial_error_levels[level] for row in train]))
            for level in range(model.wavelet_levels)
        ], axis=1,
    )
    temporal_baseline = _best_constant_mse(train_temporal[:, None], temporal_target_np[:, None])
    spatial_baseline = _best_constant_mse(train_spatial[:, None], spatial_target_np[:, None])
    local_mse = float(np.mean((local_prediction - local_target) ** 2))
    local_baseline = _best_constant_mse(train_local, local_target)
    return (
        BranchMetric("temporal_richardson", temporal_mse, temporal_baseline, temporal_mse / max(temporal_baseline, 1e-30), bool(np.isfinite(temporal_mse))),
        BranchMetric("spatial_richardson", spatial_mse, spatial_baseline, spatial_mse / max(spatial_baseline, 1e-30), bool(np.isfinite(spatial_mse))),
        BranchMetric("local_spatial_richardson", local_mse, local_baseline, local_mse / max(local_baseline, 1e-30), bool(np.isfinite(local_mse))),
    )
