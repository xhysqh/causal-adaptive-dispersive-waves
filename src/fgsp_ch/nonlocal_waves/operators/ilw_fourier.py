"""Stable dealiased Fourier operator for the periodic ILW equation."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from fgsp_ch.discretization.grids import PeriodicGrid


def ilw_dispersion_multiplier(wave_numbers, depth: float):
    r"""Return ``k*coth(depth*k) - 1/depth`` without cancellation.

    The Bernoulli expansion is used near the origin.  This is essential in
    the shallow-water regime, where direct subtraction loses most digits.
    """
    if depth <= 0.0 or not np.isfinite(depth):
        raise ValueError("ILW depth must be positive and finite")
    k = np.asarray(wave_numbers, dtype=np.float64)
    if np.any(k < 0.0) or not np.all(np.isfinite(k)):
        raise ValueError("ILW rFFT wave numbers must be finite and nonnegative")
    z = depth * k
    result = np.empty_like(k)
    small = np.abs(z) < 2.0e-2
    z2 = z[small] ** 2
    result[small] = (
        z2 / 3.0 - z2**2 / 45.0 + 2.0 * z2**3 / 945.0
        - z2**4 / 4725.0
    ) / depth
    large = ~small
    result[large] = k[large] / np.tanh(z[large]) - 1.0 / depth
    result[0] = 0.0
    return np.asarray(result, dtype=np.float64)


@dataclass(frozen=True, slots=True)
class ILWFourierOperator:
    r"""Semidiscretize ``u_t + beta*u*u_x - L_delta*u_x = 0``."""

    grid: PeriodicGrid
    depth: float
    nonlinear_transport: float = 1.0
    padding_factor: float = 1.5
    wave_numbers: NDArray[np.float64] = field(init=False, repr=False)
    dispersion_multiplier: NDArray[np.float64] = field(init=False, repr=False)
    linear_symbol: NDArray[np.complex128] = field(init=False, repr=False)

    def __post_init__(self):
        if self.grid.points % 2 or self.padding_factor < 1.5:
            raise ValueError("ILW requires an even grid and at least 3/2 padding")
        if self.nonlinear_transport <= 0.0 or not np.isfinite(self.nonlinear_transport):
            raise ValueError("ILW nonlinear transport must be positive and finite")
        k = 2.0 * np.pi * np.fft.rfftfreq(self.grid.points, d=self.grid.h)
        multiplier = ilw_dispersion_multiplier(k, self.depth)
        object.__setattr__(self, "wave_numbers", np.asarray(k, dtype=np.float64))
        object.__setattr__(self, "dispersion_multiplier", multiplier)
        object.__setattr__(self, "linear_symbol", np.asarray(1j * k * multiplier, dtype=np.complex128))

    @property
    def padded_points(self):
        points = int(np.ceil(self.padding_factor * self.grid.points))
        return points if points % 2 == 0 else points + 1

    def spectrum(self, values):
        data = np.asarray(values, dtype=np.float64)
        self.grid.validate_state(data)
        result = np.asarray(np.fft.rfft(data), dtype=np.complex128)
        result[-1] = 0.0
        return result

    def physical(self, coefficients):
        spectrum = np.asarray(coefficients, dtype=np.complex128)
        if spectrum.shape != (self.grid.points // 2 + 1,):
            raise ValueError("ILW real spectrum has an incompatible shape")
        values = np.asarray(np.fft.irfft(spectrum, n=self.grid.points), dtype=np.float64)
        if not np.all(np.isfinite(values)):
            raise FloatingPointError("ILW inverse transform is non-finite")
        return values

    def dealiased_square_spectrum(self, values):
        coefficients = self.spectrum(values)
        n, m = self.grid.points, self.padded_points
        padded = np.zeros(m // 2 + 1, dtype=np.complex128)
        padded[: n // 2] = (m / n) * coefficients[: n // 2]
        square = np.fft.rfft(np.fft.irfft(padded, n=m) ** 2)
        result = np.zeros(n // 2 + 1, dtype=np.complex128)
        result[: n // 2] = (n / m) * square[: n // 2]
        result[-1] = 0.0
        return result

    def nonlinear_spectrum(self, coefficients):
        square = self.dealiased_square_spectrum(self.physical(coefficients))
        result = -0.5j * self.nonlinear_transport * self.wave_numbers * square
        result[0] = result[-1] = 0.0
        return np.asarray(result, dtype=np.complex128)

    def rhs_spectrum(self, coefficients):
        result = self.linear_symbol * np.asarray(coefficients) + self.nonlinear_spectrum(coefficients)
        result[0] = result[-1] = 0.0
        return result

    def derivative(self, values):
        return self.physical(1j * self.wave_numbers * self.spectrum(values))

    def aliasing_defect(self, values):
        data = np.asarray(values, dtype=np.float64)
        aliased = np.fft.rfft(data * data)
        aliased[-1] = 0.0
        dealiased = self.dealiased_square_spectrum(data)
        return float(np.linalg.norm(aliased - dealiased) / max(
            np.linalg.norm(dealiased), np.finfo(float).eps
        ))

    def unresolved_nonlinear_increment(self, values, dt: float):
        if dt <= 0.0 or not np.isfinite(dt):
            raise ValueError("unresolved ILW increment requires positive finite dt")
        n, m = self.grid.points, self.padded_points
        coefficients = self.spectrum(values)
        padded = np.zeros(m // 2 + 1, dtype=np.complex128)
        padded[: n // 2] = (m / n) * coefficients[: n // 2]
        square = np.fft.rfft(np.fft.irfft(padded, n=m) ** 2)
        k = 2.0 * np.pi * np.fft.rfftfreq(m, d=self.grid.length / m)
        nonlinear = -0.5j * self.nonlinear_transport * k * (n / m) * square
        high = slice(n // 2, None)
        numerator = abs(dt) * np.sqrt(np.sum((1.0 + k[high] ** 2) * np.abs(nonlinear[high]) ** 2))
        denominator = np.sqrt(np.sum((1.0 + self.wave_numbers**2) * np.abs(coefficients) ** 2))
        return float(numerator / max(denominator, np.finfo(float).eps))


def project_ilw_spectrum(values, source, target_points, depth, nonlinear_transport=1.0):
    if target_points < 16 or target_points % 2:
        raise ValueError("target ILW grid must be even and contain at least 16 points")
    target = PeriodicGrid(int(target_points), source.length, source.x_min)
    source_hat = ILWFourierOperator(source, depth, nonlinear_transport).spectrum(values)
    target_hat = np.zeros(target.points // 2 + 1, dtype=np.complex128)
    copied = min(source.points // 2, target.points // 2)
    target_hat[:copied] = (target.points / source.points) * source_hat[:copied]
    target_hat[-1] = 0.0
    return ILWFourierOperator(target, depth, nonlinear_transport).physical(target_hat), target
