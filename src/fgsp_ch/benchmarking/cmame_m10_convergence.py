"""Diagnostics for the CMAME-M1.0 mathematical baseline.

This module deliberately contains no learned-model dependency.  It separates
regular-field, atomic-particle and total-field errors on one evaluation grid.
"""

from __future__ import annotations

import numpy as np

from fgsp_ch.discretization.grids import PeriodicGrid
from fgsp_ch.discretization.helmholtz import HelmholtzOperator
from fgsp_ch.geometry.metric import a_norm


def observed_orders(scales: list[float], errors: list[float], *, floor: float = 1.0e-15) -> list[float]:
    """Return pairwise orders for arbitrary strictly decreasing scales."""
    h = np.asarray(scales, dtype=np.float64)
    e = np.asarray(errors, dtype=np.float64)
    if h.ndim != 1 or e.shape != h.shape or h.size < 2:
        raise ValueError("scales and errors must be aligned one-dimensional arrays")
    if np.any(~np.isfinite(h)) or np.any(~np.isfinite(e)) or np.any(h <= 0.0):
        raise ValueError("scales and errors must be finite and scales positive")
    if np.any(h[:-1] <= h[1:]):
        raise ValueError("scales must be strictly decreasing")
    safe = np.maximum(e, floor)
    return [
        float(np.log(safe[i] / safe[i + 1]) / np.log(h[i] / h[i + 1]))
        for i in range(h.size - 1)
    ]


def monotone_refinement_fraction(errors: list[float], *, relative_slack: float = 1.0e-10) -> float:
    values = np.asarray(errors, dtype=np.float64)
    if values.ndim != 1 or values.size < 2 or np.any(~np.isfinite(values)):
        raise ValueError("at least two finite errors are required")
    return float(np.mean(values[1:] <= values[:-1] * (1.0 + relative_slack)))


def periodic_particle_position_error(left: np.ndarray, right: np.ndarray, length: float) -> float:
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    if a.ndim != 1 or a.shape != b.shape or length <= 0.0:
        raise ValueError("particle arrays must be aligned and length positive")
    if not a.size:
        return 0.0
    delta = (a - b + 0.5 * length) % length - 0.5 * length
    return float(np.max(np.abs(delta)))


def component_error_decomposition(
    candidate_regular: np.ndarray,
    candidate_particle: np.ndarray,
    reference_regular: np.ndarray,
    reference_particle: np.ndarray,
    grid: PeriodicGrid,
) -> dict[str, float | str]:
    """Decompose total H1 error without asserting orthogonality.

    The alignment ratio is one for perfectly aligned component errors, near
    zero for cancellation, and may exceed one only by roundoff.
    """
    arrays = [
        np.asarray(value, dtype=np.float64)
        for value in (candidate_regular, candidate_particle, reference_regular, reference_particle)
    ]
    for value in arrays:
        grid.validate_state(value)
    cr, cp, rr, rp = arrays
    operator = HelmholtzOperator(grid)
    regular_abs = a_norm(cr - rr, operator)
    particle_abs = a_norm(cp - rp, operator)
    total_abs = a_norm((cr + cp) - (rr + rp), operator)
    reference_norm = max(a_norm(rr + rp, operator), np.finfo(float).eps)
    component_sum = regular_abs + particle_abs
    if regular_abs > 1.25 * particle_abs:
        dominant = "regular"
    elif particle_abs > 1.25 * regular_abs:
        dominant = "particle"
    else:
        dominant = "balanced"
    return {
        "regular_h1_absolute": regular_abs,
        "particle_h1_absolute": particle_abs,
        "total_h1_absolute": total_abs,
        "regular_h1_relative_to_total_reference": regular_abs / reference_norm,
        "particle_h1_relative_to_total_reference": particle_abs / reference_norm,
        "total_h1_relative": total_abs / reference_norm,
        "component_alignment_ratio": 0.0 if component_sum == 0.0 else total_abs / component_sum,
        "triangle_closure_defect": max(0.0, total_abs - component_sum),
        "dominant_error_source": dominant,
    }
