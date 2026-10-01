from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import sparse
from scipy.sparse.linalg import spsolve

from fgsp_ch.equations.generalized_ch import GeneralizedCH
from fgsp_ch.geometry.metric import discrete_energy

from .base import StepResult
from .energy_projection import project_to_energy


@dataclass(slots=True)
class SemiImplicitCHSolver:
    """Second-order linearly implicit baseline with energy projection.

    For steps after startup, coefficients are extrapolated to the temporal
    midpoint, ``u_bar = 3/2 u^n - 1/2 u^(n-1)``, and momentum obeys

    ``(I + dt/2 L(u_bar)) m^(n+1) =
      (I - dt/2 L(u_bar)) m^n``.

    The first step uses explicit second-order Heun before the same projection.
    """

    equation: GeneralizedCH
    dt: float
    energy_atol: float = 1.0e-14
    finite_check: bool = True

    def __post_init__(self) -> None:
        if not np.isfinite(self.dt) or self.dt <= 0.0:
            raise ValueError("Time step must be positive and finite.")

    def provisional_step(
        self,
        current: NDArray[np.floating],
        previous: NDArray[np.floating] | None = None,
    ) -> NDArray[np.floating]:
        self.equation.grid.validate_state(current)
        if previous is None:
            first = self.equation.state_rhs(current)
            predictor = current + self.dt * first
            second = self.equation.state_rhs(predictor)
            provisional = current + 0.5 * self.dt * (first + second)
        else:
            self.equation.grid.validate_state(previous)
            coefficient_state = 1.5 * current - 0.5 * previous
            transport = self.equation.linearized_transport(coefficient_state)
            identity = sparse.identity(
                self.equation.grid.points,
                dtype=current.dtype,
                format="csr",
            )
            momentum = self.equation.helmholtz.apply(current)
            lhs = identity + 0.5 * self.dt * transport
            rhs = (identity - 0.5 * self.dt * transport) @ momentum
            next_momentum = spsolve(lhs.tocsc(), rhs)
            provisional = self.equation.helmholtz.solve(
                np.asarray(next_momentum, dtype=current.dtype)
            )
        result = np.asarray(provisional, dtype=current.dtype)
        if self.finite_check and not np.all(np.isfinite(result)):
            raise FloatingPointError("Baseline step produced NaN or Inf.")
        return result

    def step(
        self,
        current: NDArray[np.floating],
        previous: NDArray[np.floating] | None = None,
    ) -> StepResult:
        energy_before = discrete_energy(current, self.equation.helmholtz)
        provisional = self.provisional_step(current, previous)
        projection = project_to_energy(
            provisional,
            energy_before,
            self.equation.helmholtz,
            atol=self.energy_atol,
        )
        energy_after = discrete_energy(
            projection.state, self.equation.helmholtz
        )
        return StepResult(
            state=projection.state,
            provisional_state=provisional,
            energy_before=energy_before,
            provisional_energy=projection.source_energy,
            energy_after=energy_after,
            projection_scale=projection.scale,
        )

