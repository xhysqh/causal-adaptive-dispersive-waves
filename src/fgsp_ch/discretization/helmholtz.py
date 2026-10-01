from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from scipy import sparse

from .difference import second_derivative, second_derivative_matrix
from .grids import PeriodicGrid


@dataclass(slots=True)
class HelmholtzOperator:
    """Discrete Helmholtz operator ``A_h = I - D_2`` and its FFT inverse."""

    grid: PeriodicGrid
    _eigenvalues: NDArray[np.floating] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        modes = np.arange(self.grid.points, dtype=self.grid.dtype)
        self._eigenvalues = (
            1.0
            + 4.0
            * np.sin(np.pi * modes / self.grid.points) ** 2
            / self.grid.h**2
        )

    @property
    def matrix(self) -> sparse.csr_matrix:
        identity = sparse.identity(
            self.grid.points, dtype=self.grid.dtype, format="csr"
        )
        return identity - second_derivative_matrix(self.grid)

    def apply(
        self, values: NDArray[np.floating]
    ) -> NDArray[np.floating]:
        """Compute momentum ``M = A_h U``."""
        return values - second_derivative(values, self.grid)

    def solve(
        self, momentum: NDArray[np.floating]
    ) -> NDArray[np.floating]:
        """Solve ``A_h U = M`` using the exact circulant FFT diagonalization."""
        self.grid.validate_state(momentum)
        transformed = np.fft.fft(momentum)
        values = np.fft.ifft(transformed / self._eigenvalues).real
        result = np.asarray(values, dtype=momentum.dtype)
        if not np.all(np.isfinite(result)):
            raise FloatingPointError("Helmholtz inversion produced NaN or Inf.")
        return result

