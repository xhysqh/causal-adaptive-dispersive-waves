from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from fgsp_ch.discretization.difference import first_derivative
from fgsp_ch.discretization.grids import PeriodicGrid
from fgsp_ch.discretization.helmholtz import HelmholtzOperator


@dataclass(frozen=True, slots=True)
class ModifiedCHParameters:
    """Parameters of ``m_t + alpha[(u^2-u_x^2)m]_x + gamma u_x = 0``."""

    alpha: float = 1.0
    gamma: float = 0.0

    def __post_init__(self) -> None:
        if not np.isfinite(self.alpha) or not np.isfinite(self.gamma):
            raise ValueError("mCH parameters must be finite.")
        if self.alpha == 0.0:
            raise ValueError("alpha=0 removes the cubic mCH transport.")


@dataclass(slots=True)
class ModifiedCH:
    """Smooth-grid conservative-flux semidiscretization of cubic/FORQ mCH.

    This operator is a smooth Eulerian model. At peak singularities the product
    ``u_x^2 m`` must instead be interpreted by :class:`MCHWeakFormContract`.
    """

    grid: PeriodicGrid
    parameters: ModifiedCHParameters = field(default_factory=ModifiedCHParameters)
    helmholtz: HelmholtzOperator = field(init=False)

    def __post_init__(self) -> None:
        self.helmholtz = HelmholtzOperator(self.grid)

    def flux(self, state: NDArray[np.floating]) -> NDArray[np.floating]:
        self.grid.validate_state(state)
        ux = first_derivative(state, self.grid)
        momentum = self.helmholtz.apply(state)
        result = self.parameters.alpha * (state**2 - ux**2) * momentum
        if not np.all(np.isfinite(result)):
            raise FloatingPointError("mCH flux contains NaN or Inf.")
        return np.asarray(result, dtype=state.dtype)

    def momentum_rhs(self, state: NDArray[np.floating]) -> NDArray[np.floating]:
        ux = first_derivative(state, self.grid)
        rhs = -first_derivative(self.flux(state), self.grid) - self.parameters.gamma * ux
        if not np.all(np.isfinite(rhs)):
            raise FloatingPointError("mCH momentum RHS contains NaN or Inf.")
        return np.asarray(rhs, dtype=state.dtype)

    def state_rhs(self, state: NDArray[np.floating]) -> NDArray[np.floating]:
        return self.helmholtz.solve(self.momentum_rhs(state))
