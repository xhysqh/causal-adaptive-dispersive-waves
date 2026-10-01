from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from scipy.signal import resample

from .grids import PeriodicGrid


@dataclass(slots=True)
class FourierOperators:
    """Fourier pseudospectral operators on a uniform periodic grid."""

    grid: PeriodicGrid
    filter_alpha: float = 18.0
    filter_order: int = 16
    _kappa: NDArray[np.floating] = field(init=False, repr=False)
    _mode_numbers: NDArray[np.floating] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._kappa = 2.0 * np.pi * np.fft.fftfreq(
            self.grid.points, d=self.grid.h
        )
        self._mode_numbers = np.fft.fftfreq(self.grid.points) * self.grid.points
        if self.filter_alpha < 0.0:
            raise ValueError("filter_alpha must be non-negative.")
        if self.filter_order < 2 or self.filter_order % 2:
            raise ValueError("filter_order must be an even integer >= 2.")

    @property
    def wavenumbers(self) -> NDArray[np.floating]:
        return self._kappa.copy()

    @property
    def dealias_mask(self) -> NDArray[np.bool_]:
        """Two-thirds-rule mask retaining ``|mode| <= floor(N/3)``."""
        return np.abs(self._mode_numbers) <= np.floor(self.grid.points / 3)

    @property
    def exponential_filter(self) -> NDArray[np.floating]:
        maximum = max(float(np.max(np.abs(self._mode_numbers))), 1.0)
        eta = np.abs(self._mode_numbers) / maximum
        weights = np.exp(-self.filter_alpha * eta**self.filter_order)
        weights[0] = 1.0
        return weights

    def first_derivative(
        self, values: NDArray[np.floating]
    ) -> NDArray[np.floating]:
        self.grid.validate_state(values)
        return np.asarray(
            np.fft.ifft(1j * self._kappa * np.fft.fft(values)).real,
            dtype=values.dtype,
        )

    def second_derivative(
        self, values: NDArray[np.floating]
    ) -> NDArray[np.floating]:
        self.grid.validate_state(values)
        return np.asarray(
            np.fft.ifft(-(self._kappa**2) * np.fft.fft(values)).real,
            dtype=values.dtype,
        )

    def helmholtz_apply(
        self, values: NDArray[np.floating]
    ) -> NDArray[np.floating]:
        """Apply spectral ``I-d_xx``."""
        self.grid.validate_state(values)
        transformed = (1.0 + self._kappa**2) * np.fft.fft(values)
        return np.asarray(np.fft.ifft(transformed).real, dtype=values.dtype)

    def helmholtz_solve(
        self, momentum: NDArray[np.floating]
    ) -> NDArray[np.floating]:
        """Invert spectral ``I-d_xx`` exactly mode by mode."""
        self.grid.validate_state(momentum)
        transformed = np.fft.fft(momentum) / (1.0 + self._kappa**2)
        return np.asarray(np.fft.ifft(transformed).real, dtype=momentum.dtype)

    def dealias(
        self, values: NDArray[np.floating]
    ) -> NDArray[np.floating]:
        """Apply the two-thirds Fourier truncation."""
        self.grid.validate_state(values)
        transformed = np.fft.fft(values)
        transformed[~self.dealias_mask] = 0.0
        return np.asarray(np.fft.ifft(transformed).real, dtype=values.dtype)

    def filter(
        self, values: NDArray[np.floating]
    ) -> NDArray[np.floating]:
        """Apply the configured exponential spectral filter."""
        self.grid.validate_state(values)
        transformed = np.fft.fft(values) * self.exponential_filter
        return np.asarray(np.fft.ifft(transformed).real, dtype=values.dtype)

    def nonlinear_product(
        self,
        *factors: NDArray[np.floating],
        dealias: bool,
    ) -> NDArray[np.floating]:
        if not factors:
            raise ValueError("At least one factor is required.")
        product = np.ones(self.grid.points, dtype=factors[0].dtype)
        for factor in factors:
            self.grid.validate_state(factor)
            product *= factor
        return self.dealias(product) if dealias else product

    def h1_norm(self, values: NDArray[np.floating]) -> float:
        derivative = self.first_derivative(values)
        return float(
            np.sqrt(self.grid.h * np.sum(values**2 + derivative**2))
        )

    def energy(self, values: NDArray[np.floating]) -> float:
        return 0.5 * self.h1_norm(values) ** 2

    def high_frequency_tail(
        self, values: NDArray[np.floating], *, fraction: float
    ) -> float:
        if not 0.0 < fraction < 1.0:
            raise ValueError("fraction must lie in (0, 1).")
        power = np.abs(np.fft.fft(values)) ** 2
        cutoff = (1.0 - fraction) * np.max(np.abs(self._mode_numbers))
        selected = np.abs(self._mode_numbers) >= cutoff
        total = float(np.sum(power))
        return 0.0 if total == 0.0 else float(np.sum(power[selected]) / total)


def fourier_restrict(
    fine_values: NDArray[np.floating], coarse_points: int
) -> NDArray[np.floating]:
    """Fourier low-pass restriction from a fine periodic grid."""
    if coarse_points < 3 or coarse_points > fine_values.shape[-1]:
        raise ValueError("Invalid coarse point count.")
    return np.asarray(resample(fine_values, coarse_points, axis=-1)).real

