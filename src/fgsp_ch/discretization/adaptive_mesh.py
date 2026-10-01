from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class PeriodicRefinementPatch:
    coarse_mask: NDArray[np.bool_]
    fine_mask: NDArray[np.bool_]
    marked_energy_fraction: float
    active_fraction: float


def periodic_halo(mask: NDArray[np.bool_], cells: int) -> NDArray[np.bool_]:
    values = np.asarray(mask, dtype=bool)
    if values.ndim != 1 or cells < 0:
        raise ValueError("mask must be one-dimensional and cells non-negative.")
    expanded = values.copy()
    for offset in range(1, cells + 1):
        expanded |= np.roll(values, offset) | np.roll(values, -offset)
    return expanded


def dorfler_mark(
    indicators: NDArray[np.floating], *, theta: float = 0.6,
    maximum_fraction: float = 0.4, halo_cells: int = 1,
) -> PeriodicRefinementPatch:
    """Minimum-cardinality periodic marking before a bounded halo is added."""
    eta = np.asarray(indicators, dtype=np.float64)
    if eta.ndim != 1 or eta.size < 2 or not np.all(np.isfinite(eta)) or np.any(eta < 0.0):
        raise ValueError("indicators must be a finite non-negative vector.")
    if not 0.0 < theta <= 1.0 or not 0.0 < maximum_fraction <= 1.0:
        raise ValueError("theta and maximum_fraction must lie in (0, 1].")
    energy = eta**2
    total = float(np.sum(energy))
    maximum = max(1, int(np.floor(maximum_fraction * eta.size)))
    minimum_patch = min(eta.size, 1 + 2 * halo_cells)
    if maximum < minimum_patch:
        raise ValueError(
            "maximum_fraction is too small for one marked cell and its periodic halo."
        )
    selected = np.zeros(eta.size, dtype=bool)
    if total > 0.0:
        order = np.argsort(energy)[::-1]
        for index in order:
            candidate = selected.copy()
            candidate[index] = True
            if np.count_nonzero(periodic_halo(candidate, halo_cells)) > maximum:
                continue
            selected = candidate
            if np.sum(energy[selected]) >= theta * total:
                break
    coarse = periodic_halo(selected, halo_cells)
    fine = np.repeat(coarse, 2)
    captured = 1.0 if total == 0.0 else float(np.sum(energy[selected]) / total)
    return PeriodicRefinementPatch(
        coarse, fine, captured, float(np.mean(fine))
    )


def finite_volume_divergence(face_flux: NDArray[np.floating], h: float) -> NDArray[np.floating]:
    flux = np.asarray(face_flux, dtype=np.float64)
    if flux.ndim != 1 or h <= 0.0:
        raise ValueError("face_flux must be one-dimensional and h positive.")
    return np.asarray((flux - np.roll(flux, 1)) / h)


def reflux_two_to_one(
    coarse_provisional: NDArray[np.floating],
    fine_final: NDArray[np.floating],
    coarse_face_flux: NDArray[np.floating],
    fine_time_averaged_face_flux: NDArray[np.floating],
    covered_coarse: NDArray[np.bool_],
    *, dt: float, coarse_h: float,
) -> NDArray[np.floating]:
    """Synchronize a periodic 2:1 finite-volume patch conservatively.

    Face ``j`` lies between cells ``j`` and ``j+1``.  Fine face ``2*j+1``
    is the geometrically coincident patch-boundary face.
    """
    coarse = np.asarray(coarse_provisional, dtype=np.float64).copy()
    fine = np.asarray(fine_final, dtype=np.float64)
    coarse_flux = np.asarray(coarse_face_flux, dtype=np.float64)
    fine_flux = np.asarray(fine_time_averaged_face_flux, dtype=np.float64)
    covered = np.asarray(covered_coarse, dtype=bool)
    points = coarse.size
    if (
        coarse.ndim != 1 or fine.shape != (2 * points,)
        or coarse_flux.shape != (points,) or fine_flux.shape != (2 * points,)
        or covered.shape != (points,) or dt <= 0.0 or coarse_h <= 0.0
    ):
        raise ValueError("Inconsistent periodic 2:1 reflux data.")
    coarse[covered] = 0.5 * (
        fine[0::2][covered] + fine[1::2][covered]
    )
    for face in range(points):
        left = face
        right = (face + 1) % points
        if covered[left] == covered[right]:
            continue
        coincident_fine_face = 2 * face + 1
        mismatch = fine_flux[coincident_fine_face] - coarse_flux[face]
        if not covered[left] and covered[right]:
            coarse[left] -= dt * mismatch / coarse_h
        elif covered[left] and not covered[right]:
            coarse[right] += dt * mismatch / coarse_h
    return coarse


def composite_mass(
    synchronized_coarse: NDArray[np.floating],
    fine_values: NDArray[np.floating],
    covered_coarse: NDArray[np.bool_],
    *, coarse_h: float,
) -> float:
    coarse = np.asarray(synchronized_coarse)
    fine = np.asarray(fine_values)
    covered = np.asarray(covered_coarse, dtype=bool)
    if fine.shape != (2 * coarse.size,) or covered.shape != coarse.shape:
        raise ValueError("Composite mass shapes are inconsistent.")
    uncovered_mass = coarse_h * np.sum(coarse[~covered])
    fine_mask = np.repeat(covered, 2)
    fine_mass = 0.5 * coarse_h * np.sum(fine[fine_mask])
    return float(uncovered_mass + fine_mass)
