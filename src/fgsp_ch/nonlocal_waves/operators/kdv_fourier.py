"""Dealiased Fourier operator for the periodic Korteweg--de Vries equation."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from fgsp_ch.discretization.grids import PeriodicGrid


ComplexArray = NDArray[np.complexfloating]
RealArray = NDArray[np.floating]


@dataclass(frozen=True, slots=True)
class KdVFourierOperator:
    """Semidiscretize ``u_t + 6 u u_x + u_xxx = 0``.

    The linear interaction-picture generator is ``L(k)=i k^3`` and the
    quadratic term is evaluated with 3/2 padding.  The real-grid Nyquist mode
    is removed because it has no signed partner on an even grid.
    """

    grid: PeriodicGrid
    padding_factor: float = 1.5
    wave_numbers: RealArray = field(init=False, repr=False)
    linear_symbol: ComplexArray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.grid.points % 2:
            raise ValueError("KdV Fourier grids require an even point count")
        if self.padding_factor < 1.5:
            raise ValueError("KdV quadratic flux requires at least 3/2 padding")
        k = 2.0 * np.pi * np.fft.rfftfreq(self.grid.points, d=self.grid.h)
        object.__setattr__(self, "wave_numbers", np.asarray(k, dtype=np.float64))
        object.__setattr__(self, "linear_symbol", np.asarray(1j * k**3, dtype=np.complex128))

    @property
    def padded_points(self) -> int:
        points = int(np.ceil(self.padding_factor * self.grid.points))
        return points if points % 2 == 0 else points + 1

    def spectrum(self, values: RealArray) -> ComplexArray:
        data = np.asarray(values, dtype=np.float64)
        self.grid.validate_state(data)
        result = np.asarray(np.fft.rfft(data), dtype=np.complex128)
        result[-1] = 0.0
        return result

    def physical(self, coefficients: ComplexArray) -> RealArray:
        spectrum = np.asarray(coefficients, dtype=np.complex128)
        if spectrum.shape != (self.grid.points // 2 + 1,):
            raise ValueError("KdV real spectrum has an incompatible shape")
        result = np.asarray(np.fft.irfft(spectrum, n=self.grid.points), dtype=np.float64)
        if not np.all(np.isfinite(result)):
            raise FloatingPointError("KdV inverse transform is non-finite")
        return result

    def derivative(self, values: RealArray) -> RealArray:
        return self.physical(1j * self.wave_numbers * self.spectrum(values))

    def dealiased_square_spectrum(self, values: RealArray) -> ComplexArray:
        coefficients = self.spectrum(values)
        n, m = self.grid.points, self.padded_points
        padded = np.zeros(m // 2 + 1, dtype=np.complex128)
        padded[: n // 2] = (m / n) * coefficients[: n // 2]
        fine = np.fft.irfft(padded, n=m)
        square = np.fft.rfft(fine * fine)
        result = np.zeros(n // 2 + 1, dtype=np.complex128)
        result[: n // 2] = (n / m) * square[: n // 2]
        result[-1] = 0.0
        if not np.all(np.isfinite(result)):
            raise FloatingPointError("dealiased KdV quadratic spectrum is non-finite")
        return result

    def nonlinear_spectrum(self, coefficients: ComplexArray) -> ComplexArray:
        values = self.physical(coefficients)
        result = -3j * self.wave_numbers * self.dealiased_square_spectrum(values)
        result[0] = 0.0
        result[-1] = 0.0
        return np.asarray(result, dtype=np.complex128)

    def rhs_spectrum(self, coefficients: ComplexArray) -> ComplexArray:
        spectrum = np.asarray(coefficients, dtype=np.complex128)
        result = self.linear_symbol * spectrum + self.nonlinear_spectrum(spectrum)
        result[0] = 0.0
        result[-1] = 0.0
        return result

    def aliasing_defect(self, values: RealArray) -> float:
        data = np.asarray(values, dtype=np.float64)
        aliased = np.fft.rfft(data * data)
        aliased[-1] = 0.0
        dealiased = self.dealiased_square_spectrum(data)
        return float(
            np.linalg.norm(aliased - dealiased)
            / max(np.linalg.norm(dealiased), np.finfo(float).eps)
        )

    def unresolved_nonlinear_increment(self, values: RealArray, dt: float) -> float:
        """H1-weighted part of the quadratic increment above the N-grid cutoff."""
        if dt <= 0.0 or not np.isfinite(dt):
            raise ValueError("unresolved KdV increment requires positive finite dt")
        data = np.asarray(values, dtype=np.float64)
        self.grid.validate_state(data)
        n, m = self.grid.points, self.padded_points
        coefficients = self.spectrum(data)
        padded = np.zeros(m // 2 + 1, dtype=np.complex128)
        padded[: n // 2] = (m / n) * coefficients[: n // 2]
        fine = np.fft.irfft(padded, n=m)
        square = np.fft.rfft(fine * fine)
        k_fine = 2.0 * np.pi * np.fft.rfftfreq(m, d=self.grid.length / m)
        nonlinear = -3j * k_fine * (n / m) * square
        high = nonlinear[n // 2 :]
        high_k = k_fine[n // 2 :]
        numerator = abs(dt) * float(np.sqrt(np.sum((1 + high_k**2) * np.abs(high)**2)))
        denominator = float(np.sqrt(np.sum(
            (1 + self.wave_numbers**2) * np.abs(coefficients)**2
        )))
        return numerator / max(denominator, np.finfo(float).eps)
