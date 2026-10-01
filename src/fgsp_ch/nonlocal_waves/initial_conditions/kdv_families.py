"""Pre-registered periodic KdV challenge families."""

from __future__ import annotations

import numpy as np
from scipy.special import ellipj, ellipk

from fgsp_ch.discretization.grids import PeriodicGrid


KDV_FAMILIES = (
    "single_soliton",
    "two_soliton_collision",
    "cnoidal_periodic",
    "smooth_to_oscillatory",
)


def _sech_squared(argument):
    clipped = np.clip(np.asarray(argument, dtype=np.float64), -40, 40)
    return 1.0 / np.cosh(clipped)**2


def kdv_initial_condition(family: str, grid: PeriodicGrid, *, seed: int):
    """Return an initial state without using any F1/F2 outcome."""
    if family not in KDV_FAMILIES:
        raise ValueError(f"unknown KdV family: {family}")
    rng = np.random.default_rng(int(seed))
    x, length = grid.x, grid.length
    center = grid.x_min + 0.5 * length
    if family == "single_soliton":
        speed = 0.9 + 0.2 * rng.random()
        distance = grid.periodic_distance(x, center + rng.uniform(-0.05, 0.05) * length)
        values = 0.5 * speed * _sech_squared(0.5 * np.sqrt(speed) * distance)
    elif family == "two_soliton_collision":
        fast, slow = 1.25 + 0.1 * rng.random(), 0.45 + 0.08 * rng.random()
        q_fast, q_slow = center - 0.22 * length, center + 0.12 * length
        values = (
            0.5 * fast * _sech_squared(0.5 * np.sqrt(fast) * grid.periodic_distance(x, q_fast))
            + 0.5 * slow * _sech_squared(0.5 * np.sqrt(slow) * grid.periodic_distance(x, q_slow))
        )
    elif family == "cnoidal_periodic":
        modulus = 0.82 + 0.08 * rng.random()
        phase = rng.uniform(0, length)
        argument = 2 * ellipk(modulus) * (x - phase) / length
        cn = ellipj(argument, modulus)[1]
        values = 0.08 + (0.32 + 0.04 * rng.random()) * cn**2
    else:
        distance = grid.periodic_distance(x, center)
        width = (0.09 + 0.015 * rng.random()) * length
        carrier = 1 + 0.12 * np.cos(4 * np.pi * distance / length)
        values = 0.08 + (0.36 + 0.04 * rng.random()) * np.exp(-(distance / width)**2) * carrier
    result = np.asarray(values, dtype=np.float64)
    grid.validate_state(result)
    return result
