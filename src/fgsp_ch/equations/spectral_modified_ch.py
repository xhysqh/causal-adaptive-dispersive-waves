from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from fgsp_ch.discretization.grids import PeriodicGrid
from fgsp_ch.discretization.spectral import FourierOperators
from fgsp_ch.equations.modified_ch import ModifiedCHParameters


@dataclass(slots=True)
class SpectralModifiedCH:
    """Independent Fourier pseudospectral smooth cubic/FORQ mCH reference."""

    grid: PeriodicGrid
    parameters: ModifiedCHParameters = field(default_factory=ModifiedCHParameters)
    use_dealiasing: bool = True
    operators: FourierOperators = field(init=False)

    def __post_init__(self) -> None:
        self.operators = FourierOperators(self.grid)

    def momentum_rhs(self, momentum: NDArray[np.floating]) -> NDArray[np.floating]:
        self.grid.validate_state(momentum)
        ops = self.operators
        state = ops.helmholtz_solve(momentum)
        ux = ops.first_derivative(state)
        flux = self.parameters.alpha * (state**2 - ux**2) * momentum
        if self.use_dealiasing:
            flux = ops.dealias(np.asarray(flux, dtype=state.dtype))
        result = -ops.first_derivative(flux) - self.parameters.gamma * ux
        if self.use_dealiasing:
            result = ops.dealias(result)
        if not np.all(np.isfinite(result)):
            raise FloatingPointError("Spectral mCH momentum RHS contains NaN or Inf.")
        return np.asarray(result, dtype=momentum.dtype)

    def state_rhs(self, state: NDArray[np.floating]) -> NDArray[np.floating]:
        self.grid.validate_state(state)
        momentum = self.operators.helmholtz_apply(state)
        return self.operators.helmholtz_solve(self.momentum_rhs(momentum))
