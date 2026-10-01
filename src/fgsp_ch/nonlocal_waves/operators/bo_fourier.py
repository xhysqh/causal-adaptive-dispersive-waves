"""Fourier--Hilbert operators for the periodic Benjamin--Ono equation."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from fgsp_ch.discretization.grids import PeriodicGrid


ComplexArray = NDArray[np.complexfloating]
RealArray = NDArray[np.floating]


@dataclass(frozen=True, slots=True)
class BOFourierOperator:
    """Dealiased semidiscretization of

    ``u_t + 0.5 (u^2)_x - |D| u_x = 0``.

    The real FFT convention fixes ``H_hat(k)=-i sign(k)`` and therefore the
    linear evolution symbol is ``L(k)=i k |k|``.
    """

    grid: PeriodicGrid
    padding_factor: float = 1.5
    wave_numbers: RealArray = field(init=False, repr=False)
    linear_symbol: ComplexArray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.grid.points % 2:
            raise ValueError("BO Fourier reference grids must have an even point count")
        if self.padding_factor < 1.5:
            raise ValueError("reference nonlinear evaluation requires at least 3/2 padding")
        k = 2.0 * np.pi * np.fft.rfftfreq(self.grid.points, d=self.grid.h)
        linear = 1j * k * np.abs(k)
        object.__setattr__(self, "wave_numbers", np.asarray(k, dtype=np.float64))
        object.__setattr__(self, "linear_symbol", np.asarray(linear, dtype=np.complex128))

    @property
    def padded_points(self) -> int:
        points = int(np.ceil(self.padding_factor * self.grid.points))
        return points if points % 2 == 0 else points + 1

    def spectrum(self, values: RealArray) -> ComplexArray:
        data = np.asarray(values, dtype=np.float64)
        self.grid.validate_state(data)
        result = np.asarray(np.fft.rfft(data), dtype=np.complex128)
        # The even-grid Nyquist coefficient has no signed partner and is not
        # retained by the 3/2 quadratic product.
        result[-1] = 0.0
        return result

    def physical(self, coefficients: ComplexArray) -> RealArray:
        spectrum = np.asarray(coefficients, dtype=np.complex128)
        if spectrum.shape != (self.grid.points // 2 + 1,):
            raise ValueError("BO real spectrum has an incompatible shape")
        result = np.asarray(np.fft.irfft(spectrum, n=self.grid.points), dtype=np.float64)
        if not np.all(np.isfinite(result)):
            raise FloatingPointError("BO inverse transform is non-finite")
        return result

    def hilbert(self, values: RealArray) -> RealArray:
        coefficients = self.spectrum(values)
        multiplier = np.full(coefficients.shape, -1j, dtype=np.complex128)
        multiplier[0] = 0.0
        multiplier[-1] = 0.0
        return self.physical(multiplier * coefficients)

    def abs_d(self, values: RealArray) -> RealArray:
        return self.physical(self.wave_numbers * self.spectrum(values))

    def dealiased_square_spectrum(self, values: RealArray) -> ComplexArray:
        """Return the N-grid spectrum of ``values**2`` via 3/2 padding."""
        coefficients = self.spectrum(values)
        n = self.grid.points
        m = self.padded_points
        padded = np.zeros(m // 2 + 1, dtype=np.complex128)
        # Exclude the N-grid Nyquist coefficient; all signed low modes are
        # represented in the larger real FFT.  M/N compensates IFFT scaling.
        padded[: n // 2] = (m / n) * coefficients[: n // 2]
        fine = np.fft.irfft(padded, n=m)
        square = np.fft.rfft(fine * fine)
        result = np.zeros(n // 2 + 1, dtype=np.complex128)
        result[: n // 2] = (n / m) * square[: n // 2]
        result[-1] = 0.0
        if not np.all(np.isfinite(result)):
            raise FloatingPointError("dealiased BO quadratic spectrum is non-finite")
        return result

    def nonlinear_spectrum(self, coefficients: ComplexArray) -> ComplexArray:
        values = self.physical(coefficients)
        square = self.dealiased_square_spectrum(values)
        result = -0.5j * self.wave_numbers * square
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
        """Relative difference between aliased and 3/2-padded quadratic flux."""
        data = np.asarray(values, dtype=np.float64)
        aliased = np.fft.rfft(data * data)
        aliased[-1] = 0.0
        dealiased = self.dealiased_square_spectrum(data)
        return float(
            np.linalg.norm(aliased - dealiased)
            / max(np.linalg.norm(dealiased), np.finfo(float).eps)
        )

    def unresolved_nonlinear_increment(self, values: RealArray, dt: float) -> float:
        """H1-weighted nonlinear increment carried by modes above N/2.

        The modes are already available on the 3/2 product grid, so this is a
        cheap residual indicator rather than an additional reference solve.
        """
        data = np.asarray(values, dtype=np.float64)
        self.grid.validate_state(data)
        if dt <= 0.0 or not np.isfinite(dt):
            raise ValueError("unresolved BO increment requires positive finite dt")
        n = self.grid.points
        m = self.padded_points
        coefficients = self.spectrum(data)
        padded = np.zeros(m // 2 + 1, dtype=np.complex128)
        padded[: n // 2] = (m / n) * coefficients[: n // 2]
        fine = np.fft.irfft(padded, n=m)
        square = np.fft.rfft(fine * fine)
        k_fine = 2.0 * np.pi * np.fft.rfftfreq(m, d=self.grid.length / m)
        nonlinear = -0.5j * k_fine * (n / m) * square
        high = nonlinear[n // 2 :]
        high_k = k_fine[n // 2 :]
        numerator = abs(dt) * float(np.sqrt(np.sum((1.0 + high_k**2) * np.abs(high) ** 2)))
        state_norm = float(np.sqrt(np.sum((1.0 + self.wave_numbers**2) * np.abs(coefficients) ** 2)))
        return numerator / max(state_norm, np.finfo(float).eps)
