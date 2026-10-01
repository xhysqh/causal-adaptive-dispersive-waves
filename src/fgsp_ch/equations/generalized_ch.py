from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from scipy import sparse

from fgsp_ch.discretization.difference import (
    first_derivative,
    first_derivative_matrix,
)
from fgsp_ch.discretization.grids import PeriodicGrid
from fgsp_ch.discretization.helmholtz import HelmholtzOperator


@dataclass(frozen=True, slots=True)
class GeneralizedCHParameters:
    """Parameters ``(k1, k2, k)`` of the generalized CH family."""

    k1: float
    k2: float
    k: int

    def __post_init__(self) -> None:
        if not np.isfinite(self.k1) or not np.isfinite(self.k2):
            raise ValueError("k1 and k2 must be finite.")
        if isinstance(self.k, bool) or int(self.k) != self.k or self.k < 1:
            raise ValueError("k must be an integer greater than or equal to one.")


@dataclass(slots=True)
class GeneralizedCH:
    """Centered spatial discretization of the generalized CH momentum PDE."""

    grid: PeriodicGrid
    parameters: GeneralizedCHParameters
    helmholtz: HelmholtzOperator = field(init=False)
    _d1: sparse.csr_matrix = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.helmholtz = HelmholtzOperator(self.grid)
        self._d1 = first_derivative_matrix(self.grid)

    def linearized_transport(
        self, coefficient_state: NDArray[np.floating]
    ) -> sparse.csr_matrix:
        """Build ``L(u_bar)`` such that frozen PDE is ``m_t + L m = 0``."""
        self.grid.validate_state(coefficient_state)
        p = self.parameters
        ux = first_derivative(coefficient_state, self.grid)
        q = coefficient_state**2 - ux**2
        term1 = p.k1 * (self._d1 @ sparse.diags(q, format="csr"))
        term2 = p.k2 * (
            sparse.diags(coefficient_state**p.k, format="csr") @ self._d1
            + sparse.diags(
                (p.k + 1)
                * coefficient_state ** (p.k - 1)
                * ux,
                format="csr",
            )
        )
        operator = (term1 + term2).tocsr()
        if not np.all(np.isfinite(operator.data)):
            raise FloatingPointError("Transport operator contains NaN or Inf.")
        return operator

    def momentum_rhs(
        self, state: NDArray[np.floating]
    ) -> NDArray[np.floating]:
        """Return ``m_t = -L(u)m`` for the discrete generalized CH PDE."""
        momentum = self.helmholtz.apply(state)
        rhs = -(self.linearized_transport(state) @ momentum)
        result = np.asarray(rhs, dtype=state.dtype)
        if not np.all(np.isfinite(result)):
            raise FloatingPointError("Generalized CH RHS contains NaN or Inf.")
        return result

    def state_rhs(
        self, state: NDArray[np.floating]
    ) -> NDArray[np.floating]:
        """Return ``u_t = A_h^{-1} m_t``."""
        return self.helmholtz.solve(self.momentum_rhs(state))

