"""Operator-derived traits for equation-name-free adaptive control."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class DispersiveWaveTraits:
    """Traits are computed from the PDE symbol, not an equation identifier."""

    dispersion_order: float
    nonlinear_transport: float
    nonlocal_symbol: bool
    quadratic_flux: bool = True

    def __post_init__(self):
        if not 1.0 < self.dispersion_order <= 4.0:
            raise ValueError("dispersion order must lie in the registered wave range")
        if self.nonlinear_transport <= 0 or not np.isfinite(self.nonlinear_transport):
            raise ValueError("nonlinear transport coefficient must be positive")

    def linear_symbol(self, wave_numbers: NDArray[np.floating]):
        k = np.asarray(wave_numbers, dtype=np.float64)
        if np.any(k < 0.0) or not np.all(np.isfinite(k)):
            raise ValueError("power-law traits require finite nonnegative rFFT modes")
        return np.asarray(1j * np.power(k, self.dispersion_order), dtype=np.complex128)

    def maximum_phase(self, wave_numbers, dt: float) -> float:
        return float(abs(dt) * np.max(np.abs(self.linear_symbol(wave_numbers))))

    def mechanism_vector(self) -> NDArray[np.float64]:
        return np.asarray([
            self.dispersion_order / 4.0,
            np.log1p(self.nonlinear_transport) / np.log(7.0),
            float(self.nonlocal_symbol),
            float(self.quadratic_flux),
        ], dtype=np.float64)
