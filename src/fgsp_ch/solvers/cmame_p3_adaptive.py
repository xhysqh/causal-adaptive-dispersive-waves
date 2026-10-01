"""Deterministic admissibility and conservative synchronization for CMAME-P3."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from fgsp_ch.discretization.adaptive_mesh import PeriodicRefinementPatch, dorfler_mark, periodic_halo


@dataclass(frozen=True, slots=True)
class P3AdaptiveDecision:
    patch: PeriodicRefinementPatch
    proposed_dt: float
    accepted_dt: float
    collision_bound: float
    cfl_bound: float
    forced_refinement_cells: int


@dataclass(frozen=True, slots=True)
class P3PatchMigration:
    fine_detail: NDArray[np.floating]
    activated_cells: int
    deactivated_cells: int
    mass_defect: float


def bounded_error_indicator(
    predicted_log_error: NDArray[np.floating],
    deterministic_proxy: NDArray[np.floating], *, maximum_log_correction: float = 4.0,
) -> NDArray[np.floating]:
    """Project a learned log-error onto a causal, finite physics envelope.

    The projection is part of the numerical method rather than a learned
    threshold.  Out-of-support ``NaN``/``Inf`` predictions are mapped to the
    conservative upper envelope and can therefore only request more work.
    """
    prediction = np.asarray(predicted_log_error, dtype=np.float64)
    proxy = np.asarray(deterministic_proxy, dtype=np.float64)
    if prediction.shape != proxy.shape or maximum_log_correction <= 0.0:
        raise ValueError("P3 error-envelope inputs are inconsistent")
    base = np.log(np.maximum(np.abs(proxy), 1.0e-8))
    lower, upper = base - maximum_log_correction, base + maximum_log_correction
    clean = np.nan_to_num(prediction, nan=np.inf, posinf=np.inf, neginf=-np.inf)
    clean = np.where(np.isposinf(clean), upper, clean)
    clean = np.where(np.isneginf(clean), lower, clean)
    return np.exp(np.clip(clean, lower, upper))


def migrate_fine_detail(
    fine_detail: NDArray[np.floating], old_patch: PeriodicRefinementPatch,
    new_patch: PeriodicRefinementPatch, *, fine_h: float,
) -> P3PatchMigration:
    """Move hierarchical detail between periodic 2:1 patches conservatively.

    Existing detail is retained only on the overlap.  Newly activated cells
    start from the prolongated coarse state (zero hierarchical detail).  The
    disappearing detail's zero mode is redistributed over the new patch, so a
    regrid event cannot create or destroy composite momentum.
    """
    detail = np.asarray(fine_detail, dtype=np.float64)
    old = np.asarray(old_patch.fine_mask, dtype=bool)
    new = np.asarray(new_patch.fine_mask, dtype=bool)
    if detail.ndim != 1 or old.shape != detail.shape or new.shape != detail.shape:
        raise ValueError("P3 patch migration shapes are inconsistent")
    if fine_h <= 0.0 or not np.any(new):
        raise ValueError("fine_h must be positive and the new patch nonempty")
    old_mass = float(fine_h * np.sum(detail[old]))
    migrated = np.where(old & new, detail, 0.0)
    defect = old_mass - float(fine_h * np.sum(migrated[new]))
    migrated[new] += defect / (fine_h * np.count_nonzero(new))
    return P3PatchMigration(
        migrated,
        int(np.count_nonzero(new & ~old)),
        int(np.count_nonzero(old & ~new)),
        abs(old_mass - float(fine_h * np.sum(migrated[new]))),
    )


def conservative_patch_synchronize(
    coarse: NDArray[np.floating], fine_restricted: NDArray[np.floating], mask: NDArray[np.bool_]
) -> NDArray[np.floating]:
    """Replace a local patch while preserving the coarse-grid zero mode exactly."""
    base = np.asarray(coarse, dtype=np.float64)
    fine = np.asarray(fine_restricted, dtype=np.float64)
    active = np.asarray(mask, dtype=bool)
    if base.shape != fine.shape or active.shape != base.shape or not np.any(active):
        return base.copy()
    correction = np.where(active, fine - base, 0.0)
    correction[active] -= np.sum(correction) / np.count_nonzero(active)
    result = base + correction
    if abs(np.sum(result) - np.sum(base)) > 1e-12 * max(1.0, abs(np.sum(base))):
        raise FloatingPointError("P3 patch synchronization lost mass")
    return result


def deterministic_adaptive_decision(
    indicators: NDArray[np.floating], action_logits: NDArray[np.floating],
    interface_weight: NDArray[np.floating], *, current_dt: float,
    neural_multiplier: float, h: float, characteristic_speed: float,
    minimum_separation: float, relative_particle_speed: float,
    theta: float = 0.65, maximum_fraction: float = 0.5,
    maximum_cfl: float = 0.4, collision_safety: float = 0.2,
    minimum_dt: float | None = None, maximum_dt: float | None = None,
) -> P3AdaptiveDecision:
    eta = np.asarray(indicators, dtype=np.float64)
    logits = np.asarray(action_logits, dtype=np.float64)
    interface = np.asarray(interface_weight, dtype=np.float64)
    if logits.shape != (3, eta.size):
        raise ValueError("action logits must have shape [3, cells]")
    learned_refine = np.argmax(logits, axis=0) == 2
    forced = interface >= 0.35
    combined = np.maximum(eta, np.where(learned_refine | forced, np.max(eta) + 1.0, 0.0))
    patch = dorfler_mark(combined, theta=theta, maximum_fraction=maximum_fraction, halo_cells=1)
    required = periodic_halo(forced, 1)
    if np.count_nonzero(required) <= int(maximum_fraction * eta.size):
        coarse_mask = patch.coarse_mask | required
        patch = PeriodicRefinementPatch(coarse_mask, np.repeat(coarse_mask, 2), patch.marked_energy_fraction, float(np.mean(coarse_mask)))
    cfl_bound = maximum_cfl * h / max(abs(characteristic_speed), np.finfo(float).eps)
    collision_bound = collision_safety * minimum_separation / max(abs(relative_particle_speed), np.finfo(float).eps)
    lower = 0.25 * current_dt if minimum_dt is None else float(minimum_dt)
    upper = 2.0 * current_dt if maximum_dt is None else float(maximum_dt)
    if lower <= 0.0 or upper < lower:
        raise ValueError("registered time-step bounds are inconsistent")
    proposed = float(np.clip(
        current_dt * float(np.clip(neural_multiplier, 0.5, 2.0)), lower, upper
    ))
    accepted = max(lower, min(proposed, cfl_bound, collision_bound, upper))
    return P3AdaptiveDecision(patch, proposed, accepted, collision_bound, cfl_bound, int(np.count_nonzero(required)))
