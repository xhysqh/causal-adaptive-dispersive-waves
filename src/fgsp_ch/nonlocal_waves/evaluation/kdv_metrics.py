"""Invariant and error metrics for periodic KdV trajectories."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from fgsp_ch.discretization.grids import PeriodicGrid
from fgsp_ch.nonlocal_waves.operators.kdv_fourier import KdVFourierOperator
from fgsp_ch.nonlocal_waves.evaluation.spectral_norms import relative_rfft_h1


@dataclass(frozen=True, slots=True)
class KdVInvariants:
    mass: float
    half_l2_squared: float
    hamiltonian: float

    def as_dict(self) -> dict[str, float]:
        return {
            "mass": self.mass,
            "half_l2_squared": self.half_l2_squared,
            "hamiltonian": self.hamiltonian,
        }


def kdv_invariants(values: NDArray[np.floating], grid: PeriodicGrid) -> KdVInvariants:
    data = np.asarray(values, dtype=np.float64)
    grid.validate_state(data)
    derivative = KdVFourierOperator(grid).derivative(data)
    return KdVInvariants(
        mass=float(grid.h * np.sum(data)),
        half_l2_squared=float(0.5 * grid.h * np.sum(data**2)),
        # For u_t + 6 u u_x + u_xxx = 0.
        hamiltonian=float(grid.h * np.sum(0.5 * derivative**2 - data**3)),
    )


def relative_kdv_h1(
    approximation: NDArray[np.floating], reference: NDArray[np.floating],
    grid: PeriodicGrid,
) -> float:
    approx = np.asarray(approximation, dtype=np.float64)
    ref = np.asarray(reference, dtype=np.float64)
    grid.validate_state(approx)
    grid.validate_state(ref)
    return relative_rfft_h1(approx, ref, spacing=grid.h)
