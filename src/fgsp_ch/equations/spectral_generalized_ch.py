from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from fgsp_ch.discretization.grids import PeriodicGrid
from fgsp_ch.discretization.spectral import FourierOperators

from .generalized_ch import GeneralizedCHParameters


@dataclass(slots=True)
class SpectralGeneralizedCH:
    """Fourier pseudospectral generalized CH equation in momentum form."""

    grid: PeriodicGrid
    parameters: GeneralizedCHParameters
    use_dealiasing: bool = True
    use_filter: bool = False
    filter_alpha: float = 18.0
    filter_order: int = 16
    operators: FourierOperators = field(init=False)

    def __post_init__(self) -> None:
        self.operators = FourierOperators(
            self.grid,
            filter_alpha=self.filter_alpha,
            filter_order=self.filter_order,
        )

    def momentum_rhs(
        self, momentum: NDArray[np.floating]
    ) -> NDArray[np.floating]:
        """Evaluate the pseudospectral momentum right-hand side."""
        self.grid.validate_state(momentum)
        p = self.parameters
        ops = self.operators
        u = ops.helmholtz_solve(momentum)
        ux = ops.first_derivative(u)
        mx = ops.first_derivative(momentum)

        u_squared = ops.nonlinear_product(
            u, u, dealias=self.use_dealiasing
        )
        ux_squared = ops.nonlinear_product(
            ux, ux, dealias=self.use_dealiasing
        )
        flux = ops.nonlinear_product(
            u_squared - ux_squared,
            momentum,
            dealias=self.use_dealiasing,
        )
        term_one = p.k1 * ops.first_derivative(flux)

        u_power = np.asarray(u**p.k, dtype=u.dtype)
        if self.use_dealiasing:
            u_power = ops.dealias(u_power)
        advection = ops.nonlinear_product(
            u_power, mx, dealias=self.use_dealiasing
        )
        stretching = ops.nonlinear_product(
            np.asarray(u ** (p.k - 1), dtype=u.dtype),
            ux,
            momentum,
            dealias=self.use_dealiasing,
        )
        result = -term_one - p.k2 * (
            advection + (p.k + 1) * stretching
        )
        if self.use_dealiasing:
            result = ops.dealias(result)
        if self.use_filter:
            result = ops.filter(result)
        if not np.all(np.isfinite(result)):
            raise FloatingPointError("Spectral CH RHS contains NaN or Inf.")
        return np.asarray(result, dtype=momentum.dtype)

