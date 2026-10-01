"""Structure and spectral H1 metrics for power-law dispersive waves."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fgsp_ch.nonlocal_waves.operators.power_law_fourier import PowerLawFourierOperator
from fgsp_ch.nonlocal_waves.evaluation.spectral_norms import relative_rfft_h1


@dataclass(frozen=True, slots=True)
class PowerLawInvariants:
    mass: float
    half_l2_squared: float
    hamiltonian: float

    def as_dict(self):
        return {
            "mass": self.mass,
            "half_l2_squared": self.half_l2_squared,
            "hamiltonian": self.hamiltonian,
        }


def power_law_invariants(values, grid, traits):
    data = np.asarray(values, dtype=np.float64); grid.validate_state(data)
    operator = PowerLawFourierOperator(grid, traits)
    fractional = operator.physical(
        operator.wave_numbers**((traits.dispersion_order - 1) / 2)
        * operator.spectrum(data)
    )
    return PowerLawInvariants(
        float(grid.h * np.sum(data)),
        float(0.5 * grid.h * np.sum(data**2)),
        float(grid.h * np.sum(
            0.5 * fractional**2 - traits.nonlinear_transport * data**3 / 6
        )),
    )


def relative_power_law_h1(approximation, reference, grid):
    approx, ref = np.asarray(approximation), np.asarray(reference)
    grid.validate_state(approx); grid.validate_state(ref)
    return relative_rfft_h1(approx, ref, spacing=grid.h)
