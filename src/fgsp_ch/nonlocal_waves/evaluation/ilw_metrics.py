"""Invariant and error metrics for the periodic ILW equation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fgsp_ch.nonlocal_waves.operators.ilw_fourier import ILWFourierOperator
from fgsp_ch.nonlocal_waves.evaluation.spectral_norms import relative_rfft_h1


@dataclass(frozen=True, slots=True)
class ILWInvariants:
    mass: float
    half_l2_squared: float
    hamiltonian: float

    def as_dict(self):
        return {
            "mass": self.mass,
            "half_l2_squared": self.half_l2_squared,
            "hamiltonian": self.hamiltonian,
        }


def ilw_invariants(values, grid, depth, nonlinear_transport=1.0):
    data = np.asarray(values, dtype=np.float64)
    grid.validate_state(data)
    operator = ILWFourierOperator(grid, depth, nonlinear_transport)
    dispersive = operator.physical(operator.dispersion_multiplier * operator.spectrum(data))
    return ILWInvariants(
        float(grid.h * np.sum(data)),
        float(0.5 * grid.h * np.sum(data**2)),
        float(grid.h * np.sum(
            0.5 * data * dispersive - nonlinear_transport * data**3 / 6.0
        )),
    )


def relative_ilw_h1(approximation, reference, grid):
    approx, ref = np.asarray(approximation), np.asarray(reference)
    grid.validate_state(approx)
    grid.validate_state(ref)
    return relative_rfft_h1(approx, ref, spacing=grid.h)
