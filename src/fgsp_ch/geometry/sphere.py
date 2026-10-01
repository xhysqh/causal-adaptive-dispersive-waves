from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from fgsp_ch.discretization.helmholtz import HelmholtzOperator
from fgsp_ch.geometry.metric import a_norm, discrete_energy
from fgsp_ch.solvers.energy_projection import project_to_energy


def a_inner(
    left: NDArray[np.floating],
    right: NDArray[np.floating],
    helmholtz: HelmholtzOperator,
) -> float:
    helmholtz.grid.validate_state(left)
    helmholtz.grid.validate_state(right)
    dtype = np.result_type(left.dtype, right.dtype, helmholtz.grid.dtype)
    promoted_left = np.asarray(left, dtype=dtype)
    promoted_right = np.asarray(right, dtype=dtype)
    return float(
        helmholtz.grid.h
        * np.dot(promoted_left, helmholtz.apply(promoted_right))
    )


def tangent_project(
    base: NDArray[np.floating],
    vector: NDArray[np.floating],
    helmholtz: HelmholtzOperator,
    *,
    epsilon: float = 1.0e-15,
) -> NDArray[np.floating]:
    """A_h-orthogonally project ``vector`` onto the energy-shell tangent."""
    denominator = a_inner(base, base, helmholtz)
    if denominator <= epsilon:
        if a_norm(vector, helmholtz) <= epsilon:
            return np.zeros_like(vector)
        raise ValueError("A nonzero tangent is undefined at the zero-energy state.")
    coefficient = a_inner(base, vector, helmholtz) / denominator
    result = np.asarray(
        vector - coefficient * base,
        dtype=np.result_type(base.dtype, vector.dtype),
    )
    if not np.all(np.isfinite(result)):
        raise FloatingPointError("Tangent projection produced NaN or Inf.")
    return result


def radial_align(
    candidate: NDArray[np.floating],
    base: NDArray[np.floating],
    helmholtz: HelmholtzOperator,
) -> NDArray[np.floating]:
    """Put ``candidate`` on the discrete energy shell through ``base``."""
    return project_to_energy(
        candidate, discrete_energy(base, helmholtz), helmholtz
    ).state


def exp_map(
    base: NDArray[np.floating],
    tangent: NDArray[np.floating],
    helmholtz: HelmholtzOperator,
    *,
    epsilon: float = 1.0e-15,
) -> NDArray[np.floating]:
    """Exponential map on the A_h sphere, with a stable small-angle sinc."""
    radius = a_norm(base, helmholtz)
    tangent_norm = a_norm(tangent, helmholtz)
    if radius <= epsilon:
        if tangent_norm <= epsilon:
            return np.zeros_like(base)
        raise ValueError("The exponential map is undefined at zero energy.")
    orthogonality = abs(a_inner(base, tangent, helmholtz))
    roundoff_floor = (
        64.0
        * np.finfo(base.dtype).eps
        * max(a_inner(base, base, helmholtz), 1.0)
    )
    if orthogonality > (
        1.0e-10 * radius * max(tangent_norm, epsilon) + roundoff_floor
    ):
        raise ValueError("exp_map requires an A_h-tangent vector.")
    angle = tangent_norm / radius
    if angle < 1.0e-7:
        angle2 = angle * angle
        cosine = 1.0 - 0.5 * angle2 + angle2 * angle2 / 24.0
        sinc = 1.0 - angle2 / 6.0 + angle2 * angle2 / 120.0
    else:
        cosine = float(np.cos(angle))
        sinc = float(np.sin(angle) / angle)
    result = np.asarray(cosine * base + sinc * tangent, dtype=base.dtype)
    if not np.all(np.isfinite(result)):
        raise FloatingPointError("Exponential map produced NaN or Inf.")
    return result


def log_map(
    base: NDArray[np.floating],
    target: NDArray[np.floating],
    helmholtz: HelmholtzOperator,
    *,
    antipodal_tolerance: float = 1.0e-8,
    epsilon: float = 1.0e-15,
) -> NDArray[np.floating]:
    """Shortest A_h-sphere logarithm after radial target alignment."""
    radius = a_norm(base, helmholtz)
    if radius <= epsilon:
        if a_norm(target, helmholtz) <= epsilon:
            return np.zeros_like(base)
        raise ValueError("The logarithm map is undefined at zero energy.")
    aligned = radial_align(target, base, helmholtz)
    cosine = float(np.clip(a_inner(base, aligned, helmholtz) / radius**2, -1.0, 1.0))
    angle = float(np.arccos(cosine))
    if np.pi - angle <= antipodal_tolerance:
        raise ValueError("The logarithm map is ambiguous near the antipode.")
    if angle < 1.0e-7:
        angle2 = angle * angle
        factor = 1.0 + angle2 / 6.0 + 7.0 * angle2 * angle2 / 360.0
    else:
        factor = float(angle / np.sin(angle))
    result = factor * (aligned - cosine * base)
    result = tangent_project(base, result, helmholtz)
    if not np.all(np.isfinite(result)):
        raise FloatingPointError("Logarithm map produced NaN or Inf.")
    return np.asarray(result, dtype=base.dtype)


def angular_trust_region(
    base: NDArray[np.floating],
    tangent: NDArray[np.floating],
    helmholtz: HelmholtzOperator,
    *,
    maximum_angle: float,
    epsilon: float = 1.0e-15,
) -> tuple[NDArray[np.floating], float, float]:
    if not 0.0 < maximum_angle < np.pi / 2.0:
        raise ValueError("maximum_angle must lie in (0, pi/2).")
    radius = a_norm(base, helmholtz)
    norm = a_norm(tangent, helmholtz)
    if radius <= epsilon:
        if norm <= epsilon:
            return np.zeros_like(tangent), 1.0, 0.0
        raise ValueError("Angular trust region is undefined at zero energy.")
    angle = norm / radius
    scale = min(1.0, maximum_angle / max(angle, epsilon))
    return np.asarray(scale * tangent, dtype=tangent.dtype), float(scale), float(angle)
