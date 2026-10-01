from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from fgsp_ch.equations.modified_ch import ModifiedCH
from fgsp_ch.geometry.metric import discrete_energy


@dataclass(frozen=True, slots=True)
class MassH1ProjectionResult:
    state: NDArray[np.floating]
    scale: float
    mass: float
    energy: float


def project_to_mass_h1(
    candidate: NDArray[np.floating],
    target_mass: float,
    target_energy: float,
    equation: ModifiedCH,
    *,
    atol: float = 1.0e-14,
) -> MassH1ProjectionResult:
    """Project onto the intersection of a mass hyperplane and an H1 sphere."""
    grid = equation.grid
    grid.validate_state(candidate)
    if not np.isfinite(target_mass) or not np.isfinite(target_energy) or target_energy < 0:
        raise ValueError("Projection targets must be finite and energy non-negative.")
    mean = target_mass / grid.length
    constant_energy = 0.5 * grid.length * mean**2
    available = target_energy - constant_energy
    if available < -atol:
        raise ValueError("Target mass is incompatible with target H1 energy.")
    centered = candidate - np.mean(candidate)
    centered_energy = discrete_energy(centered, equation.helmholtz)
    if available <= atol:
        state = np.full_like(candidate, mean)
        scale = 0.0
    else:
        if centered_energy <= atol:
            raise FloatingPointError("Candidate has no zero-mean component to rescale.")
        scale = float(np.sqrt(available / centered_energy))
        state = np.asarray(mean + scale * centered, dtype=candidate.dtype)
    mass = float(grid.h * np.sum(state))
    energy = discrete_energy(state, equation.helmholtz)
    if not np.all(np.isfinite(state)):
        raise FloatingPointError("Mass-H1 projection produced NaN or Inf.")
    return MassH1ProjectionResult(state, scale, mass, energy)


@dataclass(frozen=True, slots=True)
class MCHStepResult:
    state: NDArray[np.floating]
    mass: float
    energy: float
    projection_scale: float


@dataclass(slots=True)
class MassH1ProjectedMCHSolver:
    """Second-order Heun baseline projected onto mass and H1 invariants."""

    equation: ModifiedCH
    dt: float

    def __post_init__(self) -> None:
        if self.dt <= 0.0 or not np.isfinite(self.dt):
            raise ValueError("dt must be positive and finite.")

    def step(self, state: NDArray[np.floating]) -> MCHStepResult:
        self.equation.grid.validate_state(state)
        mass = float(self.equation.grid.h * np.sum(state))
        energy = discrete_energy(state, self.equation.helmholtz)
        k1 = self.equation.state_rhs(state)
        predictor = state + self.dt * k1
        k2 = self.equation.state_rhs(predictor)
        candidate = state + 0.5 * self.dt * (k1 + k2)
        projected = project_to_mass_h1(candidate, mass, energy, self.equation)
        return MCHStepResult(
            projected.state, projected.mass, projected.energy, projected.scale
        )
