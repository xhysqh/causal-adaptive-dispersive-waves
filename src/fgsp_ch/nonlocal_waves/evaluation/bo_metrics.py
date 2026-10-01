from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from fgsp_ch.discretization.grids import PeriodicGrid
from fgsp_ch.nonlocal_waves.operators.bo_fourier import BOFourierOperator
from fgsp_ch.nonlocal_waves.evaluation.spectral_norms import relative_rfft_h1


@dataclass(frozen=True, slots=True)
class BOInvariants:
    mass: float
    half_l2_squared: float
    hamiltonian: float

    def as_dict(self) -> dict[str, float]:
        return {
            "mass": self.mass,
            "half_l2_squared": self.half_l2_squared,
            "hamiltonian": self.hamiltonian,
        }


def bo_invariants(values: NDArray[np.floating], grid: PeriodicGrid) -> BOInvariants:
    data = np.asarray(values, dtype=np.float64)
    grid.validate_state(data)
    abs_d_u = BOFourierOperator(grid).abs_d(data)
    return BOInvariants(
        mass=float(grid.h * np.sum(data)),
        half_l2_squared=float(0.5 * grid.h * np.sum(data * data)),
        hamiltonian=float(
            grid.h * np.sum(data**3 / 6.0 - 0.5 * data * abs_d_u)
        ),
    )


def relative_spectral_h1(
    approximation: NDArray[np.floating],
    reference: NDArray[np.floating],
    grid: PeriodicGrid,
) -> float:
    approx = np.asarray(approximation, dtype=np.float64)
    ref = np.asarray(reference, dtype=np.float64)
    grid.validate_state(approx)
    grid.validate_state(ref)
    return relative_rfft_h1(approx, ref, spacing=grid.h)
