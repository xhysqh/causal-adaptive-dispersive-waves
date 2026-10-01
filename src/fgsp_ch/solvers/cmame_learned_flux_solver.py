"""Structure-preserving rollout wrapper for a CMAME-P1 flux predictor."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from numpy.typing import NDArray

from fgsp_ch.datasets.cmame_p1_flux import cell_to_face_flux, dimensionless_mch_features
from fgsp_ch.discretization.adaptive_mesh import finite_volume_divergence
from fgsp_ch.discretization.difference import first_derivative
from fgsp_ch.equations.modified_ch import ModifiedCH
from fgsp_ch.geometry.metric import discrete_inner
from fgsp_ch.solvers.mch_energy_projected import project_to_mass_h1


FluxPredictor = Callable[[NDArray[np.floating], NDArray[np.floating]], NDArray[np.floating]]


def h1_tangent_momentum_rhs(equation: ModifiedCH, state: NDArray[np.floating], momentum_rhs: NDArray[np.floating]) -> NDArray[np.floating]:
    """Project a zero-mean momentum RHS onto the discrete H1 tangent."""
    equation.grid.validate_state(state)
    equation.grid.validate_state(momentum_rhs)
    rhs = momentum_rhs - np.mean(momentum_rhs)
    centered = state - np.mean(state)
    direction = equation.helmholtz.apply(centered)
    denominator = discrete_inner(state, direction, equation.grid)
    numerator = discrete_inner(state, rhs, equation.grid)
    if abs(denominator) <= 100 * np.finfo(float).eps:
        if abs(numerator) > 1e-12:
            raise FloatingPointError("degenerate H1 tangent direction")
        return rhs
    projected = rhs - (numerator / denominator) * direction
    return np.asarray(projected)


@dataclass(frozen=True, slots=True)
class LearnedFluxStep:
    state: NDArray[np.floating]
    correction_rms: float
    mass_drift: float
    relative_energy_drift: float
    tangent_defect: float


@dataclass(slots=True)
class CMAMELearnedFluxSolver:
    equation: ModifiedCH
    dt: float
    predictor: FluxPredictor

    def _rhs(self, state: NDArray[np.floating]) -> tuple[NDArray[np.floating], float, float]:
        features, parameters = dimensionless_mch_features(state, self.equation.grid, self.equation.parameters, self.dt)
        correction = np.asarray(self.predictor(features, parameters), dtype=np.float64)
        if correction.shape != state.shape or not np.all(np.isfinite(correction)):
            raise FloatingPointError("learned face flux must be finite and match the grid")
        base = cell_to_face_flux(self.equation.flux(state))
        ux = first_derivative(state, self.equation.grid)
        raw = -finite_volume_divergence(base + correction, self.equation.grid.h) - self.equation.parameters.gamma * ux
        projected = h1_tangent_momentum_rhs(self.equation, state, raw)
        tangent_defect = abs(discrete_inner(state, projected, self.equation.grid))
        return self.equation.helmholtz.solve(projected), float(np.sqrt(np.mean(correction**2))), tangent_defect

    def step(self, state: NDArray[np.floating]) -> LearnedFluxStep:
        if self.dt <= 0.0:
            raise ValueError("dt must be positive")
        grid = self.equation.grid
        mass0 = float(grid.h * np.sum(state))
        energy0 = discrete_inner(state, self.equation.helmholtz.apply(state), grid) / 2.0
        k1, correction1, defect1 = self._rhs(state)
        predictor = state + self.dt * k1
        k2, correction2, defect2 = self._rhs(predictor)
        candidate = state + 0.5 * self.dt * (k1 + k2)
        projected = project_to_mass_h1(candidate, mass0, energy0, self.equation)
        return LearnedFluxStep(
            projected.state,
            max(correction1, correction2),
            abs(projected.mass - mass0),
            abs(projected.energy - energy0) / max(abs(energy0), np.finfo(float).eps),
            max(defect1, defect2),
        )
