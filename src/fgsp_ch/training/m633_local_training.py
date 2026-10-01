from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.nn import functional as F

from fgsp_ch.discretization.adaptive_mesh import dorfler_mark
from fgsp_ch.models.m633_local_amr_operator import ConservativeLocalAMROperator


@dataclass(frozen=True, slots=True)
class M633TensorSplit:
    features: torch.Tensor
    parameters: torch.Tensor
    refinement: torch.Tensor
    mask: torch.Tensor
    face: torch.Tensor
    boundary: torch.Tensor
    detail: torch.Tensor


def tensor_split(archive, split: str, device: torch.device) -> M633TensorSplit:
    selected = np.asarray(archive["split"] == split)
    def tensor(name: str, dtype=torch.float32):
        return torch.as_tensor(archive[name][selected], dtype=dtype, device=device)
    return M633TensorSplit(
        tensor("features"), tensor("parameters"), tensor("refinement_score"),
        tensor("refinement_mask", torch.bool), tensor("reflux_face_target"),
        tensor("boundary_face_mask", torch.bool), tensor("fine_detail_target"),
    )


def _set_trainable(model: ConservativeLocalAMROperator, stage: str) -> None:
    for parameter in model.parameters():
        parameter.requires_grad_(stage != "local")
    if stage == "local":
        for module in (model.detail_head, model.face_head):
            for parameter in module.parameters():
                parameter.requires_grad_(True)


def _predicted_support(
    refinement_score: torch.Tensor,
    oracle_mask: torch.Tensor,
    *,
    theta: float,
    maximum_fraction: float,
    halo_cells: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build detached deployment masks while retaining only labelled detail cells."""
    masks = []
    for score in refinement_score.detach().cpu().numpy():
        masks.append(dorfler_mark(
            np.maximum(score, 0.0), theta=theta,
            maximum_fraction=maximum_fraction, halo_cells=halo_cells,
        ).coarse_mask)
    predicted = torch.as_tensor(np.asarray(masks), dtype=torch.bool, device=oracle_mask.device)
    # The archive intentionally stores local detail labels only on the oracle
    # patch. Intersecting prevents an unlabelled cell from becoming a false
    # zero target, while still exposing the heads to deployment-time masks.
    detail_mask = predicted & oracle_mask
    boundary = predicted != torch.roll(predicted, shifts=-1, dims=1)
    return detail_mask, boundary


def _losses(
    model: ConservativeLocalAMROperator,
    data: M633TensorSplit,
    *,
    deployment_masks: bool = False,
    theta: float = 0.6,
    maximum_fraction: float = 0.4,
    halo_cells: int = 1,
):
    output = model(data.features, data.parameters)
    refinement_scale = model.refinement_scale.clamp_min(1.0e-10)
    detail_scale = model.detail_scale.clamp_min(1.0e-10)
    face_scale = model.face_scale.clamp_min(1.0e-10)
    refinement_loss = F.smooth_l1_loss(
        output.refinement_score / refinement_scale,
        data.refinement / refinement_scale,
    )
    mask, boundary = data.mask, data.boundary
    if deployment_masks:
        mask, boundary = _predicted_support(
            output.refinement_score, data.mask, theta=theta,
            maximum_fraction=maximum_fraction, halo_cells=halo_cells,
        )
    fine_mask = torch.repeat_interleave(mask, 2, dim=1)
    patch_sum = torch.where(fine_mask, data.detail, 0.0).to(torch.float64).sum(dim=1)
    detail = model.constrain_fine_detail(output.fine_detail, mask, patch_sum)
    detail_error = (detail - data.detail) / detail_scale
    detail_loss = (detail_error.square() * fine_mask).sum() / fine_mask.sum().clamp_min(1)
    face_error = (output.face_flux_mismatch - data.face) / face_scale
    face_loss = (face_error.square() * boundary).sum() / boundary.sum().clamp_min(1)
    # Low periodic moments stabilize moving sparse patch boundaries.
    points = data.face.shape[1]
    phase = 2.0 * torch.pi * torch.arange(points, device=data.face.device) / points
    selected_error = face_error * boundary
    weak = torch.zeros((), device=data.face.device)
    for mode in range(1, 6):
        weak = weak + (selected_error * torch.cos(mode * phase)).mean(dim=1).square().mean()
        weak = weak + (selected_error * torch.sin(mode * phase)).mean(dim=1).square().mean()
    return refinement_loss, detail_loss, face_loss, weak


def train_local_amr_operator(
    model: ConservativeLocalAMROperator,
    train: M633TensorSplit,
    *,
    refinement_epochs: int,
    local_epochs: int,
    joint_epochs: int,
    learning_rate: float,
    theta: float = 0.6,
    maximum_fraction: float = 0.4,
    halo_cells: int = 1,
) -> list[dict[str, float]]:
    model.set_normalization(
        train.features, train.parameters, train.refinement,
        train.detail, train.face,
    )
    history = []
    stages = (
        ("refinement", refinement_epochs, (1.0, 0.0, 0.0, 0.0), learning_rate),
        ("local", local_epochs, (0.0, 1.0, 1.0, 0.1), learning_rate),
        ("joint", joint_epochs, (1.0, 0.5, 0.5, 0.05), 0.3 * learning_rate),
    )
    for stage, epochs, weights, rate in stages:
        _set_trainable(model, stage)
        optimizer = torch.optim.AdamW(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=rate, weight_decay=1.0e-6,
        )
        for _ in range(epochs):
            optimizer.zero_grad(set_to_none=True)
            branch_losses = _losses(model, train)
            if stage == "joint":
                deployment_losses = _losses(
                    model, train, deployment_masks=True, theta=theta,
                    maximum_fraction=maximum_fraction, halo_cells=halo_cells,
                )
                branch_losses = tuple(
                    0.8 * oracle + 0.2 * deployment
                    for oracle, deployment in zip(
                        branch_losses, deployment_losses, strict=True
                    )
                )
            loss = sum(weight * value for weight, value in zip(weights, branch_losses, strict=True))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        with torch.no_grad():
            values = _losses(model, train)
            if stage == "joint":
                deployment_values = _losses(
                    model, train, deployment_masks=True, theta=theta,
                    maximum_fraction=maximum_fraction, halo_cells=halo_cells,
                )
                values = tuple(
                    0.8 * oracle + 0.2 * deployment
                    for oracle, deployment in zip(
                        values, deployment_values, strict=True
                    )
                )
        history.append({
            "stage": stage,
            "refinement_loss": float(values[0]),
            "detail_loss": float(values[1]),
            "face_loss": float(values[2]),
            "weak_loss": float(values[3]),
        })
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    return history


def _ratio(prediction: np.ndarray, target: np.ndarray, train_mean: float) -> dict[str, float]:
    mse = float(np.mean((prediction - target) ** 2))
    mean_mse = float(np.mean((target - train_mean) ** 2))
    zero_mse = float(np.mean(target**2))
    baseline = min(mean_mse, zero_mse)
    return {
        "mse": mse,
        "strong_baseline_mse": baseline,
        "ratio_to_strong_baseline": mse / max(baseline, np.finfo(float).eps),
    }


@torch.no_grad()
def evaluate_local_amr_operator(
    model: ConservativeLocalAMROperator,
    train: M633TensorSplit,
    evaluation: M633TensorSplit,
    *,
    theta: float,
    maximum_fraction: float,
    halo_cells: int,
) -> dict:
    model.eval()
    output = model(evaluation.features, evaluation.parameters)
    detail = model.constrain_fine_detail(
        output.fine_detail, evaluation.mask,
        evaluation.detail.to(torch.float64).sum(dim=1),
    )
    refinement_pred = output.refinement_score.cpu().numpy()
    refinement_true = evaluation.refinement.cpu().numpy()
    face_mask = evaluation.boundary.cpu().numpy()
    detail_mask = torch.repeat_interleave(evaluation.mask, 2, dim=1).cpu().numpy()
    face_pred = output.face_flux_mismatch.cpu().numpy()[face_mask]
    face_true = evaluation.face.cpu().numpy()[face_mask]
    detail_pred = detail.cpu().numpy()[detail_mask]
    detail_true = evaluation.detail.cpu().numpy()[detail_mask]
    recalls, energy_recalls = [], []
    true_masks = evaluation.mask.cpu().numpy()
    for prediction, target, true_mask in zip(
        refinement_pred, refinement_true, true_masks, strict=True
    ):
        predicted_mask = dorfler_mark(
            np.maximum(prediction, 0.0), theta=theta,
            maximum_fraction=maximum_fraction, halo_cells=halo_cells,
        ).coarse_mask
        recalls.append(
            np.count_nonzero(predicted_mask & true_mask)
            / max(np.count_nonzero(true_mask), 1)
        )
        energy_recalls.append(
            np.sum(target[predicted_mask & true_mask] ** 2)
            / max(np.sum(target[true_mask] ** 2), np.finfo(float).eps)
        )
    return {
        "refinement": _ratio(
            refinement_pred.ravel(), refinement_true.ravel(),
            float(train.refinement.mean().cpu()),
        ),
        "detail": _ratio(
            detail_pred, detail_true,
            float(train.detail.mean().cpu()),
        ),
        "face": _ratio(
            face_pred, face_true,
            float(train.face[train.boundary].mean().cpu()),
        ),
        "mask_recall": float(np.mean(recalls)),
        "marked_energy_recall": float(np.mean(energy_recalls)),
        "finite": bool(
            torch.all(torch.isfinite(output.refinement_score))
            and torch.all(torch.isfinite(detail))
            and torch.all(torch.isfinite(output.face_flux_mismatch))
        ),
        "maximum_detail_sum_error": float(torch.max(torch.abs(
            detail.sum(dim=1) - evaluation.detail.to(torch.float64).sum(dim=1)
        )).cpu()),
    }
