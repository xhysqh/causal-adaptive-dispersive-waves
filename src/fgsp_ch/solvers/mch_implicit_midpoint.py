from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import root

from fgsp_ch.equations.conservative_modified_ch import InvariantTangentModifiedCH
from fgsp_ch.geometry.metric import discrete_energy


@dataclass(frozen=True, slots=True)
class ImplicitMidpointMCHStep:
    state: NDArray[np.floating]
    nonlinear_residual: float
    function_evaluations: int
    mass_drift: float
    energy_drift: float


@dataclass(slots=True)
class ImplicitMidpointInvariantMCHSolver:
    """Implicit midpoint discretization of the invariant-tangent mCH ODE.

    Midpoint preserves the linear mass and quadratic discrete H1 invariant of
    the tangent semidiscretization up to the nonlinear solve tolerance.
    """

    equation: InvariantTangentModifiedCH
    dt: float
    nonlinear_tolerance: float = 1.0e-11
    max_function_evaluations: int = 500

    def __post_init__(self) -> None:
        if self.dt <= 0.0 or not np.isfinite(self.dt):
            raise ValueError("dt must be positive and finite.")
        if self.nonlinear_tolerance <= 0.0 or self.max_function_evaluations < 1:
            raise ValueError("Nonlinear solver controls must be positive.")

    def step(self, state: NDArray[np.floating]) -> ImplicitMidpointMCHStep:
        self.equation.grid.validate_state(state)
        initial_mass = float(self.equation.grid.h * np.sum(state))
        initial_energy = discrete_energy(state, self.equation.helmholtz)
        guess = state + self.dt * self.equation.state_rhs(state)

        def residual(candidate: NDArray[np.floating]) -> NDArray[np.floating]:
            midpoint = 0.5 * (state + candidate)
            return candidate - state - self.dt * self.equation.state_rhs(midpoint)

        solution = root(
            residual,
            guess,
            method="hybr",
            options={
                "xtol": self.nonlinear_tolerance,
                "maxfev": self.max_function_evaluations,
            },
        )
        candidate = np.asarray(solution.x, dtype=state.dtype)
        residual_norm = float(np.linalg.norm(residual(candidate), ord=np.inf))
        if (not solution.success and residual_norm > 10 * self.nonlinear_tolerance) or (
            residual_norm > 100 * self.nonlinear_tolerance
        ):
            raise RuntimeError(
                f"Implicit midpoint solve failed: {solution.message}; "
                f"residual={residual_norm:.3e}."
            )
        if not np.all(np.isfinite(candidate)):
            raise FloatingPointError("Implicit midpoint step contains NaN or Inf.")
        final_mass = float(self.equation.grid.h * np.sum(candidate))
        final_energy = discrete_energy(candidate, self.equation.helmholtz)
        return ImplicitMidpointMCHStep(
            state=candidate,
            nonlinear_residual=residual_norm,
            function_evaluations=int(solution.nfev),
            mass_drift=final_mass - initial_mass,
            energy_drift=final_energy - initial_energy,
        )
