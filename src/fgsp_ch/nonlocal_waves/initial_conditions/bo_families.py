from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from fgsp_ch.discretization.grids import PeriodicGrid


def _distance(grid: PeriodicGrid, center: float) -> NDArray[np.floating]:
    return grid.periodic_distance(grid.x, center)


def bo_initial_condition(
    family: str,
    grid: PeriodicGrid,
    *,
    amplitude: float = 1.0,
    phase_shift: float = 0.0,
) -> NDArray[np.floating]:
    """Registered smooth periodic BO reference families.

    Lorentzian profiles are controls for algebraic-tail resolution; P0 does
    not claim that their periodic superpositions are exact BO solutions.
    """
    if not np.isfinite(phase_shift):
        raise ValueError("BO family phase_shift must be finite")
    x = np.mod(grid.x - phase_shift, grid.length)
    if amplitude <= 0.0 or not np.isfinite(amplitude):
        raise ValueError("BO family amplitude must be positive and finite")
    if family == "smooth_periodic":
        values = 0.15 + amplitude * (
            0.22 * np.cos(x) - 0.08 * np.sin(2.0 * x) + 0.035 * np.cos(5.0 * x)
        )
    elif family == "single_lorentzian":
        width = 0.30
        center = np.mod(0.35 * grid.length + phase_shift, grid.length)
        values = 0.05 + amplitude * 0.42 / (1.0 + (_distance(grid, center) / width) ** 2)
    elif family == "two_lorentzian":
        left_center = np.mod(0.28 * grid.length + phase_shift, grid.length)
        right_center = np.mod(0.68 * grid.length + phase_shift, grid.length)
        left = 0.36 / (1.0 + (_distance(grid, left_center) / 0.26) ** 2)
        right = 0.25 / (1.0 + (_distance(grid, right_center) / 0.38) ** 2)
        values = 0.04 + amplitude * (left + right)
    elif family == "smooth_to_oscillatory":
        center = np.mod(0.46 * grid.length + phase_shift, grid.length)
        envelope = np.exp(-(_distance(grid, center) / 0.75) ** 2)
        values = 0.08 + amplitude * (
            0.28 * envelope + 0.045 * np.cos(7.0 * x) + 0.025 * np.sin(11.0 * x)
        )
    else:
        raise KeyError(f"unknown BO initial-condition family: {family}")
    result = np.asarray(values, dtype=np.float64)
    grid.validate_state(result)
    return result
