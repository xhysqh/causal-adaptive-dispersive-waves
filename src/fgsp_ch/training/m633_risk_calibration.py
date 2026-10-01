from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from fgsp_ch.models.m633_local_amr_operator import ConservativeLocalAMROperator
from fgsp_ch.training.m633_local_training import M633TensorSplit


BRANCHES = ("refinement", "detail", "face")


@dataclass(frozen=True, slots=True)
class M633RiskCalibration:
    requested_coverage: float
    global_scales: dict[str, float]
    family_scales: dict[str, dict[str, float]]
    raw_floors: dict[str, float]

    def scale(self, branch: str, family: str | None = None) -> float:
        value = self.global_scales[branch]
        if family is not None and family in self.family_scales:
            value = max(value, self.family_scales[family][branch])
        return float(value)


def _relative_rms(error: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weight = mask.to(error.dtype)
    numerator = (error.square() * weight).sum(dim=1) / weight.sum(dim=1).clamp_min(1.0)
    denominator = (target.square() * weight).sum(dim=1) / weight.sum(dim=1).clamp_min(1.0)
    return numerator.sqrt() / denominator.sqrt().clamp_min(1.0e-12)


@torch.no_grad()
def ensemble_error_and_raw_risk(
    models: list[ConservativeLocalAMROperator], data: M633TensorSplit,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    refinement, detail, face = [], [], []
    detail_sum = data.detail.to(torch.float64).sum(dim=1)
    for model in models:
        output = model(data.features, data.parameters)
        refinement.append(output.refinement_score)
        detail.append(model.constrain_fine_detail(output.fine_detail, data.mask, detail_sum))
        face.append(output.face_flux_mismatch)
    predictions = {
        "refinement": torch.stack(refinement),
        "detail": torch.stack(detail),
        "face": torch.stack(face),
    }
    targets = {
        "refinement": data.refinement,
        "detail": data.detail.to(torch.float64),
        "face": data.face,
    }
    masks = {
        "refinement": torch.ones_like(data.refinement, dtype=torch.bool),
        "detail": torch.repeat_interleave(data.mask, 2, dim=1),
        "face": data.boundary,
    }
    errors, risks = {}, {}
    for branch in BRANCHES:
        values = predictions[branch]
        mean = values.mean(dim=0)
        standard_deviation = values.var(dim=0, unbiased=False).sqrt()
        errors[branch] = _relative_rms(
            mean - targets[branch], targets[branch], masks[branch]
        ).cpu().numpy()
        # Dimensionless disagreement normalized by the ensemble mean. A small
        # floor is later calibrated and prevents identical members from
        # reporting zero risk.
        risks[branch] = _relative_rms(
            standard_deviation, mean, masks[branch]
        ).cpu().numpy()
    return errors, risks


def _conformal_quantile(values: np.ndarray, coverage: float) -> float:
    ordered = np.sort(np.asarray(values, dtype=np.float64))
    if ordered.size == 0:
        raise ValueError("Conformal calibration requires nonempty groups.")
    rank = min(int(np.ceil((ordered.size + 1) * coverage)) - 1, ordered.size - 1)
    return float(ordered[max(rank, 0)])


def fit_group_risk_calibration(
    errors: dict[str, np.ndarray],
    raw_risks: dict[str, np.ndarray],
    groups: np.ndarray,
    families: np.ndarray,
    *,
    coverage: float = 0.9,
    raw_floor: float = 1.0e-4,
    safety_factor: float = 1.25,
) -> M633RiskCalibration:
    if not 0.5 < coverage < 1.0 or raw_floor <= 0.0 or safety_factor < 1.0:
        raise ValueError("Invalid calibration coverage or raw floor.")
    groups, families = np.asarray(groups), np.asarray(families)
    unique_groups = np.unique(groups)
    global_scales, family_scales = {}, {}
    floors = {branch: raw_floor for branch in BRANCHES}
    for branch in BRANCHES:
        ratios = errors[branch] / np.maximum(raw_risks[branch], raw_floor)
        group_scores = np.asarray([
            np.max(ratios[groups == group]) for group in unique_groups
        ])
        global_scales[branch] = safety_factor * max(
            1.0, _conformal_quantile(group_scores, coverage)
        )
    for family in np.unique(families):
        family_scales[str(family)] = {}
        family_groups = np.unique(groups[families == family])
        for branch in BRANCHES:
            ratios = errors[branch] / np.maximum(raw_risks[branch], raw_floor)
            scores = np.asarray([
                np.max(ratios[groups == group]) for group in family_groups
            ])
            family_scales[str(family)][branch] = safety_factor * max(
                1.0, _conformal_quantile(scores, coverage)
            )
    return M633RiskCalibration(
        coverage, global_scales, family_scales, floors
    )


def calibrated_upper_bounds(
    calibration: M633RiskCalibration,
    raw_risks: dict[str, np.ndarray],
    families: np.ndarray,
) -> dict[str, np.ndarray]:
    result = {}
    for branch in BRANCHES:
        result[branch] = np.asarray([
            calibration.scale(branch, str(family))
            * max(float(raw), calibration.raw_floors[branch])
            for raw, family in zip(raw_risks[branch], families, strict=True)
        ])
    return result


def coverage_audit(
    errors: dict[str, np.ndarray],
    upper_bounds: dict[str, np.ndarray],
    groups: np.ndarray,
    families: np.ndarray,
) -> dict:
    rows = {}
    for branch in BRANCHES:
        covered = errors[branch] <= upper_bounds[branch] * (1.0 + 1.0e-7) + 1.0e-12
        group_coverage = {
            str(group): float(np.mean(covered[groups == group]))
            for group in np.unique(groups)
        }
        family_coverage = {
            str(family): float(np.mean(covered[families == family]))
            for family in np.unique(families)
        }
        rows[branch] = {
            "coverage": float(np.mean(covered)),
            "worst_group_coverage": min(group_coverage.values()),
            "group_coverage": group_coverage,
            "family_coverage": family_coverage,
            "maximum_underprediction": float(np.max(
                np.maximum(errors[branch] - upper_bounds[branch], 0.0)
            )),
        }
    return rows
