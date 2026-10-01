"""Dealiased Fourier operator for an unseen power-law dispersive wave."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from fgsp_ch.discretization.grids import PeriodicGrid
from fgsp_ch.nonlocal_waves.operators.wave_traits import DispersiveWaveTraits


@dataclass(frozen=True, slots=True)
class PowerLawFourierOperator:
    r"""Semidiscretize the registered symbol family

    .. math::
       \partial_t \hat u_k = i |k|^p \hat u_k
       - i\,\beta k\widehat{u^2}_k/2.

    Neither ``p`` nor ``beta`` is converted to an equation name.
    """

    grid: PeriodicGrid
    traits: DispersiveWaveTraits
    padding_factor: float = 1.5
    wave_numbers: NDArray[np.float64] = field(init=False, repr=False)
    linear_symbol: NDArray[np.complex128] = field(init=False, repr=False)

    def __post_init__(self):
        if self.grid.points % 2 or self.padding_factor < 1.5:
            raise ValueError("power-law Fourier grids require even N and 3/2 padding")
        k = 2 * np.pi * np.fft.rfftfreq(self.grid.points, d=self.grid.h)
        object.__setattr__(self, "wave_numbers", np.asarray(k, dtype=np.float64))
        object.__setattr__(self, "linear_symbol", self.traits.linear_symbol(k))

    @property
    def padded_points(self):
        points = int(np.ceil(self.padding_factor * self.grid.points))
        return points if points % 2 == 0 else points + 1

    def spectrum(self, values):
        data = np.asarray(values, dtype=np.float64); self.grid.validate_state(data)
        result = np.asarray(np.fft.rfft(data), dtype=np.complex128)
        result[-1] = 0.0
        return result

    def physical(self, coefficients):
        spectrum = np.asarray(coefficients, dtype=np.complex128)
        if spectrum.shape != (self.grid.points // 2 + 1,):
            raise ValueError("power-law real spectrum has an incompatible shape")
        values = np.asarray(np.fft.irfft(spectrum, n=self.grid.points), dtype=np.float64)
        if not np.all(np.isfinite(values)):
            raise FloatingPointError("power-law inverse transform is non-finite")
        return values

    def dealiased_square_spectrum(self, values):
        coefficients = self.spectrum(values)
        n, m = self.grid.points, self.padded_points
        padded = np.zeros(m // 2 + 1, dtype=np.complex128)
        padded[:n // 2] = (m / n) * coefficients[:n // 2]
        square = np.fft.rfft(np.fft.irfft(padded, n=m)**2)
        result = np.zeros(n // 2 + 1, dtype=np.complex128)
        result[:n // 2] = (n / m) * square[:n // 2]
        result[-1] = 0.0
        return result

    def nonlinear_spectrum(self, coefficients):
        square = self.dealiased_square_spectrum(self.physical(coefficients))
        result = -0.5j * self.traits.nonlinear_transport * self.wave_numbers * square
        result[0] = result[-1] = 0.0
        return np.asarray(result, dtype=np.complex128)

    def rhs_spectrum(self, coefficients):
        result = self.linear_symbol * coefficients + self.nonlinear_spectrum(coefficients)
        result[0] = result[-1] = 0.0
        return result

    def derivative(self, values):
        return self.physical(1j * self.wave_numbers * self.spectrum(values))

    def aliasing_defect(self, values):
        data = np.asarray(values, dtype=np.float64)
        aliased = np.fft.rfft(data * data); aliased[-1] = 0.0
        dealiased = self.dealiased_square_spectrum(data)
        return float(np.linalg.norm(aliased - dealiased)
                     / max(np.linalg.norm(dealiased), np.finfo(float).eps))

    def unresolved_nonlinear_increment(self, values, dt: float):
        if dt <= 0.0 or not np.isfinite(dt):
            raise ValueError("unresolved power-law increment requires positive finite dt")
        n, m = self.grid.points, self.padded_points
        coefficients = self.spectrum(values)
        padded = np.zeros(m // 2 + 1, dtype=np.complex128)
        padded[:n // 2] = (m / n) * coefficients[:n // 2]
        square = np.fft.rfft(np.fft.irfft(padded, n=m)**2)
        k = 2 * np.pi * np.fft.rfftfreq(m, d=self.grid.length / m)
        nonlinear = -0.5j * self.traits.nonlinear_transport * k * (n / m) * square
        numerator = abs(dt) * np.sqrt(np.sum((1 + k[n // 2:]**2)
                                             * np.abs(nonlinear[n // 2:])**2))
        denominator = np.sqrt(np.sum((1 + self.wave_numbers**2)
                                     * np.abs(coefficients)**2))
        return float(numerator / max(denominator, np.finfo(float).eps))


def project_power_law_spectrum(values, source, target_points, traits):
    if target_points < 16 or target_points % 2:
        raise ValueError("target power-law grid must be even")
    target = PeriodicGrid(int(target_points), source.length, source.x_min)
    source_hat = PowerLawFourierOperator(source, traits).spectrum(values)
    target_hat = np.zeros(target.points // 2 + 1, dtype=np.complex128)
    copied = min(source.points // 2, target.points // 2)
    target_hat[:copied] = (target.points / source.points) * source_hat[:copied]
    target_hat[-1] = 0.0
    return PowerLawFourierOperator(target, traits).physical(target_hat), target
