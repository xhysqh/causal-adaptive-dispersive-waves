"""Compatible invariant integrator for periodic cubic mCH hybrid states.

The method applies a Gonzalez discrete gradient to the grid mass and H1
functionals, projects the *vector field* (not the accepted state) onto their
common tangent space, and advances with a symmetric midpoint fixed point.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from fgsp_ch.discretization.adaptive_mesh import finite_volume_divergence
from fgsp_ch.discretization.grids import PeriodicGrid
from fgsp_ch.discretization.helmholtz import HelmholtzOperator
from fgsp_ch.equations.modified_ch import ModifiedCHParameters
from fgsp_ch.representations.cmame_p2_hybrid import (
    CMAMEHybridState,
    atomic_measure_weight,
    hybrid_h1_squared,
    hybrid_mass,
)
from fgsp_ch.solvers.mch_hybrid_amr import (
    HybridMCHState,
    hybrid_flux_and_source,
    hybrid_particle_velocity,
    hybrid_total_state,
)
from fgsp_ch.solvers.multipeakon import periodic_peakon_kernel_derivative


@dataclass(frozen=True, slots=True)
class CompatibleStepDiagnostics:
    iterations: int
    residual: float
    raw_constraint_defect: float
    corrected_constraint_defect: float
    relative_vector_correction: float
    mass_drift: float
    h1_drift: float


@dataclass(slots=True)
class CompatibleInvariantHybridMCHSolver:
    grid: PeriodicGrid
    parameters: ModifiedCHParameters
    dt: float
    nonlinear_tolerance: float = 1.0e-11
    maximum_iterations: int = 80
    damping: float = 1.0
    rank_tolerance: float = 1.0e-12

    def __post_init__(self) -> None:
        if self.dt <= 0.0 or self.nonlinear_tolerance <= 0.0:
            raise ValueError("dt and nonlinear tolerance must be positive")
        if self.maximum_iterations < 2 or not 0.0 < self.damping <= 1.0:
            raise ValueError("invalid compatible midpoint iteration settings")

    def _pack(self, state: HybridMCHState) -> NDArray[np.floating]:
        self.grid.validate_state(np.asarray(state.field_momentum))
        if state.positions.shape != state.amplitudes.shape:
            raise ValueError("particle positions and amplitudes must align")
        return np.concatenate((np.asarray(state.field_momentum, dtype=np.float64),
                               np.asarray(state.positions, dtype=np.float64)))

    def _unpack(self, values: NDArray[np.floating], amplitudes: NDArray[np.floating], *, wrap: bool) -> HybridMCHState:
        data = np.asarray(values, dtype=np.float64)
        if data.shape != (self.grid.points + amplitudes.size,):
            raise ValueError("packed hybrid state has the wrong shape")
        positions = data[self.grid.points:].copy()
        if wrap:
            positions = np.remainder(positions - self.grid.x_min, self.grid.length) + self.grid.x_min
        return HybridMCHState(positions, amplitudes.copy(), data[:self.grid.points].copy())

    def invariants(self, values: NDArray[np.floating], amplitudes: NDArray[np.floating]) -> NDArray[np.floating]:
        state = self._unpack(values, amplitudes, wrap=False)
        represented = CMAMEHybridState(state.field_momentum, state.positions, state.amplitudes)
        return np.asarray([
            hybrid_mass(represented, self.grid),
            0.5 * hybrid_h1_squared(represented, self.grid),
        ], dtype=np.float64)

    def raw_rhs(self, values: NDArray[np.floating], amplitudes: NDArray[np.floating]) -> NDArray[np.floating]:
        state = self._unpack(values, amplitudes, wrap=False)
        flux, source = hybrid_flux_and_source(state, self.grid, self.parameters)
        dm = -finite_volume_divergence(flux, self.grid.h) + source
        dq = hybrid_particle_velocity(state, self.grid, alpha=self.parameters.alpha)
        result = np.concatenate((dm, dq))
        if not np.all(np.isfinite(result)):
            raise FloatingPointError("compatible mCH vector field is non-finite")
        return result

    def midpoint_gradients(self, values: NDArray[np.floating], amplitudes: NDArray[np.floating]) -> NDArray[np.floating]:
        """Analytic gradients of representation-exact mass and H1 energy."""
        state = self._unpack(values, amplitudes, wrap=False)
        field = HelmholtzOperator(self.grid).solve(state.field_momentum)
        count = amplitudes.size
        mass_gradient = np.concatenate((
            np.full(self.grid.points, self.grid.h), np.zeros(count, dtype=np.float64)
        ))
        energy_momentum_gradient = self.grid.h * field
        energy_gradient = np.concatenate((energy_momentum_gradient, np.zeros(count, dtype=np.float64)))
        if count:
            weight = atomic_measure_weight(self.grid.length)
            adjoint = np.zeros(self.grid.points, dtype=np.float64)
            coordinate = (state.positions - self.grid.x_min) / self.grid.h
            left = np.floor(coordinate).astype(int) % self.grid.points
            fraction = coordinate - np.floor(coordinate)
            np.add.at(adjoint, left, amplitudes * (1.0 - fraction))
            np.add.at(adjoint, (left + 1) % self.grid.points, amplitudes * fraction)
            energy_gradient[:self.grid.points] += weight * HelmholtzOperator(self.grid).solve(adjoint)
            # ``periodic_interpolate`` is piecewise linear, so its exact
            # position derivative is the corresponding cell slope (not D1).
            slope_at_q = (
                field[(left + 1) % self.grid.points] - field[left]
            ) / self.grid.h
            delta = state.positions[:, None] - state.positions[None, :]
            particle_gradient = (
                2.0 * np.tanh(0.5 * self.grid.length) * amplitudes
                * (periodic_peakon_kernel_derivative(delta, self.grid.length) @ amplitudes)
            )
            energy_gradient[self.grid.points:] = weight * amplitudes * slope_at_q + particle_gradient
        return np.column_stack((mass_gradient, energy_gradient))

    def discrete_gradients(
        self,
        initial: NDArray[np.floating],
        final: NDArray[np.floating],
        amplitudes: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """Symmetric Gonzalez gradients satisfying both discrete chain rules."""
        delta = np.asarray(final) - np.asarray(initial)
        midpoint = 0.5 * (np.asarray(initial) + np.asarray(final))
        gradients = self.midpoint_gradients(midpoint, amplitudes)
        denominator = float(delta @ delta)
        if denominator <= np.finfo(float).eps:
            return gradients
        invariant_delta = self.invariants(final, amplitudes) - self.invariants(initial, amplitudes)
        chain_defect = invariant_delta - gradients.T @ delta
        return gradients + np.outer(delta, chain_defect / denominator)

    def tangent_vector(
        self,
        raw: NDArray[np.floating],
        gradients: NDArray[np.floating],
    ) -> tuple[NDArray[np.floating], float, float, float]:
        """Return the minimum-Euclidean-norm correction to both tangent constraints."""
        matrix = gradients.T @ gradients
        multiplier = np.linalg.pinv(matrix, rcond=self.rank_tolerance) @ (gradients.T @ raw)
        corrected = raw - gradients @ multiplier
        raw_defect = float(np.linalg.norm(gradients.T @ raw))
        corrected_defect = float(np.linalg.norm(gradients.T @ corrected))
        correction = float(np.linalg.norm(corrected - raw) / max(np.linalg.norm(raw), np.finfo(float).eps))
        return corrected, raw_defect, corrected_defect, correction

    def instantaneous_diagnostics(self, state: HybridMCHState) -> dict[str, float]:
        values = self._pack(state)
        raw = self.raw_rhs(values, state.amplitudes)
        gradients = self.midpoint_gradients(values, state.amplitudes)
        _, raw_defect, corrected_defect, correction = self.tangent_vector(raw, gradients)
        return {
            "raw_constraint_defect": raw_defect,
            "corrected_constraint_defect": corrected_defect,
            "relative_vector_correction": correction,
        }

    def step(self, state: HybridMCHState) -> tuple[HybridMCHState, CompatibleStepDiagnostics]:
        amplitudes = np.asarray(state.amplitudes, dtype=np.float64)
        initial = self._pack(state)
        initial_invariants = self.invariants(initial, amplitudes)
        guess = initial + self.dt * self.raw_rhs(initial, amplitudes)
        raw_defect = corrected_defect = correction = np.inf
        residual = np.inf
        for iteration in range(1, self.maximum_iterations + 1):
            midpoint = 0.5 * (initial + guess)
            raw = self.raw_rhs(midpoint, amplitudes)
            gradients = self.discrete_gradients(initial, guess, amplitudes)
            tangent, raw_defect, corrected_defect, correction = self.tangent_vector(raw, gradients)
            target = initial + self.dt * tangent
            update = target - guess
            residual = float(np.linalg.norm(update) / max(1.0, np.linalg.norm(target)))
            guess = guess + self.damping * update
            if residual <= self.nonlinear_tolerance:
                break
        else:
            raise RuntimeError(f"compatible midpoint failed to converge: residual={residual:.3e}")
        final_invariants = self.invariants(guess, amplitudes)
        scale = np.maximum(np.abs(initial_invariants), np.finfo(float).eps)
        drift = np.abs(final_invariants - initial_invariants) / scale
        result = self._unpack(guess, amplitudes, wrap=True)
        return result, CompatibleStepDiagnostics(
            iteration, residual, raw_defect, corrected_defect, correction,
            float(drift[0]), float(drift[1]),
        )

    def solve(self, initial: HybridMCHState, *, final_time: float) -> tuple[tuple[HybridMCHState, ...], tuple[CompatibleStepDiagnostics, ...]]:
        steps = round(final_time / self.dt)
        if steps < 1 or not np.isclose(steps * self.dt, final_time, atol=1.0e-12):
            raise ValueError("final_time must be a positive multiple of dt")
        states = [initial]
        diagnostics = []
        for _ in range(steps):
            state, row = self.step(states[-1])
            states.append(state); diagnostics.append(row)
        return tuple(states), tuple(diagnostics)
