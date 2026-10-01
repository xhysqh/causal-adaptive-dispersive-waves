"""Independent Fourier--DOP853 reference for CMAME-P2.1 labels."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp
from scipy.signal import resample

from fgsp_ch.discretization.grids import PeriodicGrid
from fgsp_ch.discretization.helmholtz import HelmholtzOperator
from fgsp_ch.discretization.spectral import FourierOperators, fourier_restrict
from fgsp_ch.equations.modified_ch import ModifiedCHParameters
from fgsp_ch.representations.cmame_p2_hybrid import (
    CMAMEHybridState,
    periodic_interpolate,
)
from fgsp_ch.solvers.cmame_cross_representation import wrap_positions
from fgsp_ch.solvers.multipeakon import (
    periodic_peakon_kernel,
    periodic_peakon_kernel_derivative,
)


@dataclass(frozen=True, slots=True)
class P21ReferenceIncrement:
    regular_momentum_rate: NDArray[np.floating]
    particle_position_rate: NDArray[np.floating]
    minimum_separation: float


@dataclass(frozen=True, slots=True)
class P21ReferenceTrajectory:
    grid: PeriodicGrid
    state: CMAMEHybridState
    minimum_separation: float
    internal_steps: int


def spectral_hybrid_rhs(
    regular_momentum: NDArray[np.floating],
    positions: NDArray[np.floating],
    amplitudes: NDArray[np.floating],
    grid: PeriodicGrid,
    parameters: ModifiedCHParameters,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Independent semidiscrete RHS using Fourier regular-field operators."""
    if positions.size and parameters.gamma != 0.0:
        raise ValueError("P2.1 mixed references require gamma=0")
    ops = FourierOperators(grid)
    field = ops.helmholtz_solve(np.asarray(regular_momentum))
    field_slope = ops.first_derivative(field)
    if positions.size:
        delta_grid = grid.x[:, None] - positions[None, :]
        atomic = periodic_peakon_kernel(delta_grid, grid.length) @ amplitudes
        atomic_slope = periodic_peakon_kernel_derivative(delta_grid, grid.length) @ amplitudes
    else:
        atomic = np.zeros(grid.points)
        atomic_slope = np.zeros(grid.points)
    total = field + atomic
    slope = field_slope + atomic_slope
    flux = parameters.alpha * (total**2 - slope**2) * regular_momentum
    momentum_rate = -ops.first_derivative(ops.dealias(flux)) - parameters.gamma * slope
    if positions.size:
        delta = positions[:, None] - positions[None, :]
        value_q = periodic_peakon_kernel(delta, grid.length) @ amplitudes
        value_q += periodic_interpolate(grid, field, positions)
        slope_q = periodic_peakon_kernel_derivative(delta, grid.length) @ amplitudes
        slope_q += periodic_interpolate(grid, field_slope, positions)
        self_slope = amplitudes * np.tanh(0.5 * grid.length)
        position_rate = parameters.alpha * (value_q**2 - slope_q**2 - self_slope**2)
    else:
        position_rate = np.empty(0)
    return np.asarray(momentum_rate), np.asarray(position_rate)


def _minimum_separation(positions: NDArray[np.floating], length: float) -> float:
    if positions.size < 2:
        return length
    distance = np.abs(
        (positions[:, None] - positions[None, :] + 0.5 * length) % length
        - 0.5 * length
    )
    distance[np.eye(positions.size, dtype=bool)] = np.inf
    return float(np.min(distance))


def reference_increment(
    coarse_state: CMAMEHybridState,
    coarse_grid: PeriodicGrid,
    parameters: ModifiedCHParameters,
    dt: float,
    *,
    refinement: int,
    rtol: float,
    atol: float,
    collision_tolerance: float,
) -> P21ReferenceIncrement:
    """Advance a refined joint ODE and restrict its increment to the coarse grid."""
    if refinement < 2 or dt <= 0.0:
        raise ValueError("reference refinement must be >=2 and dt positive")
    coarse_state.validate(coarse_grid)
    fine = PeriodicGrid(
        refinement * coarse_grid.points,
        coarse_grid.length,
        coarse_grid.x_min,
    )
    coarse_field = HelmholtzOperator(coarse_grid).solve(coarse_state.regular_momentum)
    # Fourier resampling transfers the regular field, never the atomic measure.
    fine_field = np.asarray(resample(coarse_field, fine.points)).real
    fine_ops = FourierOperators(fine)
    initial_momentum = fine_ops.helmholtz_apply(fine_field)
    q0 = np.asarray(coarse_state.positions, dtype=np.float64)
    a = np.asarray(coarse_state.amplitudes, dtype=np.float64)
    initial = np.concatenate((initial_momentum, q0))

    def rhs(_time: float, values: NDArray[np.floating]) -> NDArray[np.floating]:
        momentum = values[: fine.points]
        positions = wrap_positions(fine, values[fine.points :])
        dm, dq = spectral_hybrid_rhs(momentum, positions, a, fine, parameters)
        return np.concatenate((dm, dq))

    solution = solve_ivp(
        rhs,
        (0.0, dt),
        initial,
        method="DOP853",
        rtol=rtol,
        atol=atol,
        t_eval=[dt],
    )
    if not solution.success:
        raise RuntimeError(f"P2.1 reference failed: {solution.message}")
    final = solution.y[:, -1]
    final_momentum = final[: fine.points]
    final_q = wrap_positions(fine, final[fine.points :])
    separation = _minimum_separation(final_q, fine.length)
    if separation <= collision_tolerance:
        raise RuntimeError("P2.1 reference reached the collision guard")
    rate = fourier_restrict(final_momentum - initial_momentum, coarse_grid.points) / dt
    displacement = (
        (final_q - q0 + 0.5 * fine.length) % fine.length - 0.5 * fine.length
    )
    return P21ReferenceIncrement(
        np.asarray(rate),
        np.asarray(displacement / dt),
        separation,
    )


def spectral_hybrid_rollout(
    coarse_state: CMAMEHybridState,
    coarse_grid: PeriodicGrid,
    parameters: ModifiedCHParameters,
    final_time: float,
    *,
    refinement: int = 4,
    rtol: float = 1.0e-10,
    atol: float = 1.0e-12,
    collision_tolerance: float = 1.0e-8,
) -> P21ReferenceTrajectory:
    """Independent refined Fourier--DOP853 trajectory for paper benchmarks.

    Unlike :func:`reference_increment`, this returns the complete refined
    terminal state.  No P1/P2/P3 model, learned threshold or AMR decision is
    evaluated anywhere in the reference chain.
    """
    if refinement < 2 or final_time <= 0.0:
        raise ValueError("reference refinement must be >=2 and final_time positive")
    coarse_state.validate(coarse_grid)
    fine = PeriodicGrid(
        refinement * coarse_grid.points, coarse_grid.length, coarse_grid.x_min
    )
    coarse_field = HelmholtzOperator(coarse_grid).solve(coarse_state.regular_momentum)
    fine_field = np.asarray(resample(coarse_field, fine.points)).real
    initial_momentum = FourierOperators(fine).helmholtz_apply(fine_field)
    q0 = np.asarray(coarse_state.positions, dtype=np.float64)
    amplitudes = np.asarray(coarse_state.amplitudes, dtype=np.float64)
    initial = np.concatenate((initial_momentum, q0))

    def rhs(_time: float, values: NDArray[np.floating]) -> NDArray[np.floating]:
        momentum = values[: fine.points]
        positions = wrap_positions(fine, values[fine.points :])
        dm, dq = spectral_hybrid_rhs(momentum, positions, amplitudes, fine, parameters)
        return np.concatenate((dm, dq))

    solution = solve_ivp(
        rhs, (0.0, final_time), initial, method="DOP853",
        rtol=rtol, atol=atol, t_eval=[final_time],
    )
    if not solution.success:
        raise RuntimeError(f"P4 reference failed: {solution.message}")
    final = solution.y[:, -1]
    positions = wrap_positions(fine, final[fine.points :])
    separation = _minimum_separation(positions, fine.length)
    if separation <= collision_tolerance:
        raise RuntimeError("P4 reference reached the collision guard")
    state = CMAMEHybridState(
        np.asarray(final[: fine.points]), positions, amplitudes.copy()
    )
    return P21ReferenceTrajectory(fine, state, separation, int(solution.nfev))
