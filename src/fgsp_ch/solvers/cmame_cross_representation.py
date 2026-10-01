"""Deterministic CMAME-P2.0 Eulerian--particle coupling foundation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from fgsp_ch.discretization.adaptive_mesh import finite_volume_divergence
from fgsp_ch.discretization.difference import first_derivative
from fgsp_ch.discretization.grids import PeriodicGrid
from fgsp_ch.discretization.helmholtz import HelmholtzOperator
from fgsp_ch.equations.modified_ch import ModifiedCH, ModifiedCHParameters
from fgsp_ch.geometry.metric import discrete_inner
from fgsp_ch.representations.cmame_p2_hybrid import (
    CMAMEHybridState,
    atomic_measure_weight,
    hybrid_h1_squared,
    hybrid_mass,
    periodic_interpolate,
    regular_state,
    total_state,
)
from fgsp_ch.solvers.mch_energy_projected import MassH1ProjectedMCHSolver
from fgsp_ch.solvers.mch_multipeakon import (
    ConservativePeriodicMCHMultipeakonSolver,
    conservative_periodic_mch_velocity,
    mch_particle_h1,
)
from fgsp_ch.solvers.multipeakon import (
    periodic_peakon_kernel,
    periodic_peakon_kernel_derivative,
)


def wrap_positions(grid: PeriodicGrid, positions: NDArray[np.floating]) -> NDArray[np.floating]:
    return np.asarray((np.asarray(positions) - grid.x_min) % grid.length + grid.x_min)


def minimum_periodic_separation(grid: PeriodicGrid, positions: NDArray[np.floating]) -> float:
    q = np.asarray(positions, dtype=np.float64)
    if q.size < 2:
        return grid.length
    distance = np.abs(
        (q[:, None] - q[None, :] + 0.5 * grid.length) % grid.length
        - 0.5 * grid.length
    )
    distance[np.eye(q.size, dtype=bool)] = np.inf
    return float(np.min(distance))


def coupled_particle_velocity(
    state: CMAMEHybridState,
    grid: PeriodicGrid,
    *,
    alpha: float,
) -> NDArray[np.floating]:
    """Weak mCH characteristic velocity evaluated on the atomic support."""
    state.validate(grid)
    if not state.positions.size:
        return np.empty(0, dtype=np.float64)
    q = np.asarray(state.positions)
    a = np.asarray(state.amplitudes)
    field = regular_state(state, grid)
    field_slope = first_derivative(field, grid)
    delta = q[:, None] - q[None, :]
    value = periodic_peakon_kernel(delta, grid.length) @ a
    value += periodic_interpolate(grid, field, q)
    regular_slope = periodic_peakon_kernel_derivative(delta, grid.length) @ a
    regular_slope += periodic_interpolate(grid, field_slope, q)
    self_slope = a * np.tanh(0.5 * grid.length)
    velocity = alpha * (value**2 - regular_slope**2 - self_slope**2)
    if not np.all(np.isfinite(velocity)):
        raise FloatingPointError("coupled particle velocity contains NaN or Inf")
    return np.asarray(velocity)


def regular_momentum_rhs(
    state: CMAMEHybridState,
    grid: PeriodicGrid,
    parameters: ModifiedCHParameters,
) -> NDArray[np.floating]:
    """Conservative regular-measure RHS driven by the total hybrid velocity."""
    state.validate(grid)
    if state.positions.size and parameters.gamma != 0.0:
        raise ValueError("P2.0 mixed atomic states require gamma=0")
    total = total_state(state, grid)
    slope = first_derivative(total, grid)
    cell_flux = parameters.alpha * (total**2 - slope**2) * state.regular_momentum
    face_flux = 0.5 * (cell_flux + np.roll(cell_flux, -1))
    rhs = -finite_volume_divergence(face_flux, grid.h) - parameters.gamma * slope
    if not np.all(np.isfinite(rhs)):
        raise FloatingPointError("regular hybrid RHS contains NaN or Inf")
    return np.asarray(rhs)


@dataclass(frozen=True, slots=True)
class HybridProjection:
    state: CMAMEHybridState
    scale: float
    mass: float
    h1_squared: float


def project_regular_component_to_hybrid_invariants(
    candidate: CMAMEHybridState,
    grid: PeriodicGrid,
    target_mass: float,
    target_h1_squared: float,
    *,
    atol: float = 1.0e-12,
) -> HybridProjection:
    """Adjust only the regular field to close total mass and exact hybrid H1."""
    candidate.validate(grid)
    if not np.isfinite(target_mass) or not np.isfinite(target_h1_squared):
        raise ValueError("projection targets must be finite")
    weight = atomic_measure_weight(grid.length)
    atomic_mass = weight * np.sum(candidate.amplitudes)
    regular_mean = (target_mass - atomic_mass) / grid.length
    field = regular_state(candidate, grid)
    centered = field - np.mean(field)
    helmholtz = HelmholtzOperator(grid)
    coefficient2 = discrete_inner(centered, helmholtz.apply(centered), grid)
    if candidate.positions.size:
        centered_at_q = periodic_interpolate(grid, centered, candidate.positions)
        coefficient1 = 2.0 * weight * np.dot(candidate.amplitudes, centered_at_q)
        particle = mch_particle_h1(
            candidate.positions, candidate.amplitudes, grid.length
        )
        cross_constant = 2.0 * weight * np.sum(candidate.amplitudes) * regular_mean
    else:
        coefficient1 = 0.0
        particle = 0.0
        cross_constant = 0.0
    constant = grid.length * regular_mean**2 + cross_constant + particle
    residual = constant - target_h1_squared
    if coefficient2 <= atol:
        if abs(coefficient1) <= atol:
            if abs(residual) > 10.0 * atol:
                raise FloatingPointError("hybrid invariant projection is degenerate")
            scale = 0.0
        else:
            scale = float(-residual / coefficient1)
    else:
        discriminant = coefficient1**2 - 4.0 * coefficient2 * residual
        tolerance = 100.0 * np.finfo(float).eps * max(
            coefficient1**2, abs(4.0 * coefficient2 * residual), 1.0
        )
        if discriminant < -tolerance:
            raise FloatingPointError("hybrid H1 target is unreachable along projection ray")
        root = np.sqrt(max(discriminant, 0.0))
        roots = (
            (-coefficient1 + root) / (2.0 * coefficient2),
            (-coefficient1 - root) / (2.0 * coefficient2),
        )
        scale = float(min(roots, key=lambda value: abs(value - 1.0)))
    projected_field = np.asarray(regular_mean + scale * centered)
    projected = CMAMEHybridState(
        helmholtz.apply(projected_field),
        np.asarray(candidate.positions).copy(),
        np.asarray(candidate.amplitudes).copy(),
    )
    mass = hybrid_mass(projected, grid)
    h1 = hybrid_h1_squared(projected, grid)
    if abs(mass - target_mass) > 20.0 * atol:
        raise FloatingPointError("hybrid mass projection did not close")
    if abs(h1 - target_h1_squared) > 100.0 * atol * max(abs(target_h1_squared), 1.0):
        raise FloatingPointError("hybrid H1 projection did not close")
    return HybridProjection(projected, scale, mass, h1)


@dataclass(frozen=True, slots=True)
class CMAMECrossRepresentationStep:
    state: CMAMEHybridState
    branch: str
    mass_drift: float
    relative_h1_drift: float
    amplitude_drift: float
    projection_scale: float
    minimum_separation: float


@dataclass(slots=True)
class CMAMECrossRepresentationSolver:
    """Second-order deterministic P2.0 coupler with exact branch reductions."""

    grid: PeriodicGrid
    parameters: ModifiedCHParameters
    dt: float
    collision_tolerance: float = 1.0e-8

    def __post_init__(self) -> None:
        if not np.isfinite(self.dt) or self.dt <= 0.0:
            raise ValueError("dt must be positive and finite")
        if self.collision_tolerance <= 0.0:
            raise ValueError("collision_tolerance must be positive")

    def _validate_contract(self, state: CMAMEHybridState) -> None:
        state.validate(self.grid)
        if state.positions.size and self.parameters.gamma != 0.0:
            raise ValueError("P2.0 singular coupling is registered only for gamma=0")
        if minimum_periodic_separation(self.grid, state.positions) <= self.collision_tolerance:
            raise RuntimeError("state is outside the pre-collision contract")

    def step(self, state: CMAMEHybridState) -> CMAMECrossRepresentationStep:
        self._validate_contract(state)
        mass0 = hybrid_mass(state, self.grid)
        h10 = hybrid_h1_squared(state, self.grid)
        amplitudes0 = np.asarray(state.amplitudes).copy()
        field0 = regular_state(state, self.grid)
        if not state.positions.size:
            smooth = MassH1ProjectedMCHSolver(
                ModifiedCH(self.grid, self.parameters), self.dt
            ).step(field0)
            next_state = CMAMEHybridState(
                HelmholtzOperator(self.grid).apply(smooth.state),
                np.empty(0),
                np.empty(0),
            )
            scale = smooth.projection_scale
            branch = "smooth"
        elif np.linalg.norm(state.regular_momentum) <= 100.0 * np.finfo(float).eps:
            particle = ConservativePeriodicMCHMultipeakonSolver(
                self.grid.length,
                alpha=self.parameters.alpha,
                rtol=1.0e-12,
                atol=1.0e-14,
                collision_tolerance=self.collision_tolerance,
            ).solve(
                state.positions,
                state.amplitudes,
                final_time=self.dt,
                output_dt=self.dt,
            )
            next_state = CMAMEHybridState(
                np.zeros(self.grid.points),
                wrap_positions(self.grid, particle.positions[-1]),
                particle.amplitudes[-1].copy(),
            )
            scale = 1.0
            branch = "particle"
        else:
            rhs0 = regular_momentum_rhs(state, self.grid, self.parameters)
            velocity0 = coupled_particle_velocity(
                state, self.grid, alpha=self.parameters.alpha
            )
            predictor = CMAMEHybridState(
                state.regular_momentum + self.dt * rhs0,
                wrap_positions(self.grid, state.positions + self.dt * velocity0),
                amplitudes0.copy(),
            )
            self._validate_contract(predictor)
            rhs1 = regular_momentum_rhs(predictor, self.grid, self.parameters)
            velocity1 = coupled_particle_velocity(
                predictor, self.grid, alpha=self.parameters.alpha
            )
            candidate = CMAMEHybridState(
                state.regular_momentum + 0.5 * self.dt * (rhs0 + rhs1),
                wrap_positions(
                    self.grid,
                    state.positions + 0.5 * self.dt * (velocity0 + velocity1),
                ),
                amplitudes0.copy(),
            )
            projection = project_regular_component_to_hybrid_invariants(
                candidate, self.grid, mass0, h10
            )
            next_state = projection.state
            scale = projection.scale
            branch = "mixed"
        mass1 = hybrid_mass(next_state, self.grid)
        h11 = hybrid_h1_squared(next_state, self.grid)
        amplitude_drift = (
            float(np.max(np.abs(next_state.amplitudes - amplitudes0)))
            if amplitudes0.size else 0.0
        )
        return CMAMECrossRepresentationStep(
            next_state,
            branch,
            abs(mass1 - mass0),
            abs(h11 - h10) / max(abs(h10), np.finfo(float).eps),
            amplitude_drift,
            scale,
            minimum_periodic_separation(self.grid, next_state.positions),
        )


def pure_particle_velocity_defect(
    state: CMAMEHybridState, grid: PeriodicGrid, *, alpha: float
) -> float:
    """Convenience audit for exact reduction to the independent particle ODE."""
    if np.linalg.norm(state.regular_momentum) > 100.0 * np.finfo(float).eps:
        raise ValueError("pure-particle reduction requires zero regular momentum")
    coupled = coupled_particle_velocity(state, grid, alpha=alpha)
    reference = conservative_periodic_mch_velocity(
        state.positions, state.amplitudes, grid.length, alpha=alpha
    )
    return float(np.max(np.abs(coupled - reference))) if coupled.size else 0.0

