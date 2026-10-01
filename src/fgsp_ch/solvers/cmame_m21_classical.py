"""Non-learned classical comparators for the frozen CMAME-M2 protocol.

The adapters deliberately expose the same terminal-state contract.  No class
in this module imports a checkpoint, a neural feature, or an M1.1 estimator.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fgsp_ch.discretization.adaptive_mesh import dorfler_mark, finite_volume_divergence, periodic_halo, PeriodicRefinementPatch
from fgsp_ch.discretization.difference import first_derivative
from fgsp_ch.discretization.grids import PeriodicGrid
from fgsp_ch.discretization.helmholtz import HelmholtzOperator
from fgsp_ch.discretization.spectral import FourierOperators
from fgsp_ch.equations.modified_ch import ModifiedCHParameters
from fgsp_ch.representations.cmame_p2_hybrid import CMAMEHybridState, hybrid_h1_squared, hybrid_mass
from fgsp_ch.solvers.mch_compatible_invariant import CompatibleInvariantHybridMCHSolver
from fgsp_ch.solvers.mch_compatible_local_amr import (
    CompatibleLocalAMRSolver,
    composite_from_fine_uniform,
    migrate_patch_invariantly,
    uniform_fine_state,
)
from fgsp_ch.solvers.mch_hybrid_amr import HybridMCHState, hybrid_flux_and_source, hybrid_particle_velocity
from fgsp_ch.solvers.mch_multipeakon import ConservativePeriodicMCHMultipeakonSolver, mch_particle_h1


@dataclass(frozen=True, slots=True)
class ClassicalTrajectory:
    method: str
    grid: PeriodicGrid
    state: HybridMCHState
    accepted_steps: int
    active_dof_time: float
    maximum_mass_drift: float
    maximum_h1_drift: float
    patch_changes: int = 0
    time_adjustments: int = 0


def _represented(state: HybridMCHState) -> CMAMEHybridState:
    return CMAMEHybridState(state.field_momentum, state.positions, state.amplitudes)


def compatible_uniform_rollout(
    initial: HybridMCHState,
    grid: PeriodicGrid,
    parameters: ModifiedCHParameters,
    *,
    dt: float,
    final_time: float,
    nonlinear_tolerance: float = 1.0e-11,
    maximum_iterations: int = 80,
) -> ClassicalTrajectory:
    """Uniform compatible conservative finite-difference comparator."""
    if dt <= 0.0 or final_time <= 0.0:
        raise ValueError("dt and final_time must be positive")
    current = initial
    initial_mass = hybrid_mass(_represented(initial), grid)
    initial_h1 = hybrid_h1_squared(_represented(initial), grid)
    time = 0.0
    steps = 0
    mass_drift = h1_drift = 0.0
    while time < final_time - 10.0 * np.finfo(float).eps:
        step_dt = min(dt, final_time - time)
        solver = CompatibleInvariantHybridMCHSolver(
            grid, parameters, step_dt,
            nonlinear_tolerance=nonlinear_tolerance,
            maximum_iterations=maximum_iterations,
        )
        current, diagnostics = solver.step(current)
        time += step_dt
        steps += 1
        mass_drift = max(mass_drift, diagnostics.mass_drift)
        h1_drift = max(h1_drift, diagnostics.h1_drift)
    return ClassicalTrajectory(
        "conservative_finite_difference", grid, current, steps,
        float((grid.points + initial.positions.size) * final_time),
        mass_drift, h1_drift,
    )


def _spectral_mch_rhs(
    momentum: np.ndarray,
    operators: FourierOperators,
    alpha: float,
) -> np.ndarray:
    state = operators.helmholtz_solve(momentum)
    slope = operators.first_derivative(state)
    state_squared = operators.nonlinear_product(state, state, dealias=True)
    slope_squared = operators.nonlinear_product(slope, slope, dealias=True)
    flux = operators.nonlinear_product(
        state_squared - slope_squared, momentum, dealias=True
    )
    return np.asarray(-alpha * operators.first_derivative(flux))


def fourier_ifrk4_rollout(
    initial: HybridMCHState,
    grid: PeriodicGrid,
    parameters: ModifiedCHParameters,
    *,
    dt: float,
    final_time: float,
) -> ClassicalTrajectory:
    """Dealiased Fourier IFRK4 comparator for smooth states.

    Under the registered ``gamma=0`` pure-mCH equation there is no separate
    linear generator, hence the integrating factor is the identity and IFRK4
    is exactly classical RK4 applied to the spectral momentum equation.
    """
    if initial.positions.size:
        raise ValueError("Fourier IFRK4 is registered only for field states")
    if parameters.gamma != 0.0:
        raise ValueError("M2.1 Fourier IFRK4 contract requires gamma=0")
    operators = FourierOperators(grid)
    momentum = np.asarray(initial.field_momentum, dtype=np.float64).copy()
    initial_state = operators.helmholtz_solve(momentum)
    mass0 = float(grid.h * np.sum(initial_state))
    h10 = float(operators.energy(initial_state))
    time = 0.0
    steps = 0
    mass_drift = h1_drift = 0.0
    while time < final_time - 10.0 * np.finfo(float).eps:
        step_dt = min(dt, final_time - time)
        k1 = _spectral_mch_rhs(momentum, operators, parameters.alpha)
        k2 = _spectral_mch_rhs(momentum + 0.5 * step_dt * k1, operators, parameters.alpha)
        k3 = _spectral_mch_rhs(momentum + 0.5 * step_dt * k2, operators, parameters.alpha)
        k4 = _spectral_mch_rhs(momentum + step_dt * k3, operators, parameters.alpha)
        momentum = operators.dealias(
            momentum + step_dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
        )
        state = operators.helmholtz_solve(momentum)
        mass_drift = max(mass_drift, abs(float(grid.h * np.sum(state)) - mass0))
        h1_drift = max(
            h1_drift,
            abs(float(operators.energy(state)) - h10) / max(abs(h10), np.finfo(float).eps),
        )
        time += step_dt
        steps += 1
    final = HybridMCHState(np.empty(0), np.empty(0), momentum)
    return ClassicalTrajectory(
        "fourier_ifrk4", grid, final, steps, float(grid.points * final_time),
        mass_drift, h1_drift,
    )


def particle_dop853_rollout(
    initial: HybridMCHState,
    grid: PeriodicGrid,
    parameters: ModifiedCHParameters,
    *,
    final_time: float,
    rtol: float = 1.0e-9,
    atol: float = 1.0e-11,
    collision_tolerance: float = 1.0e-8,
) -> ClassicalTrajectory:
    if np.linalg.norm(initial.field_momentum) > 100.0 * np.finfo(float).eps:
        raise ValueError("particle DOP853 requires a pure atomic state")
    solver = ConservativePeriodicMCHMultipeakonSolver(
        grid.length, alpha=parameters.alpha, rtol=rtol, atol=atol,
        collision_tolerance=collision_tolerance,
    )
    trajectory = solver.solve(
        initial.positions, initial.amplitudes,
        final_time=final_time, output_dt=final_time,
    )
    h10 = mch_particle_h1(initial.positions, initial.amplitudes, grid.length)
    h1_drift = float(
        np.max(np.abs(trajectory.h1_squared - h10))
        / max(abs(h10), np.finfo(float).eps)
    )
    final = HybridMCHState(
        trajectory.positions[-1].copy(), trajectory.amplitudes[-1].copy(),
        np.zeros(grid.points),
    )
    return ClassicalTrajectory(
        "particle_dop853", grid, final, len(trajectory.times) - 1,
        float(2 * initial.positions.size * final_time), 0.0, h1_drift,
    )


def _residual_patch(
    state: HybridMCHState,
    grid: PeriodicGrid,
    parameters: ModifiedCHParameters,
    *,
    theta: float,
    maximum_fraction: float,
    halo_cells: int,
    particle_halo_cells: int,
) -> PeriodicRefinementPatch:
    flux, source = hybrid_flux_and_source(state, grid, parameters)
    rate = -finite_volume_divergence(flux, grid.h) + source
    curvature = np.roll(state.field_momentum, -1) - 2.0 * state.field_momentum + np.roll(state.field_momentum, 1)
    scale = max(float(np.linalg.norm(state.field_momentum) / np.sqrt(grid.points)), np.finfo(float).eps)
    indicators = np.sqrt((grid.h * rate / scale) ** 2 + (curvature / scale) ** 2)
    patch = dorfler_mark(
        indicators, theta=theta, maximum_fraction=maximum_fraction,
        halo_cells=halo_cells,
    )
    mask = patch.coarse_mask.copy()
    for position in state.positions:
        index = int(np.floor((position - grid.x_min) / grid.h)) % grid.points
        mask[index] = True
    mask = periodic_halo(mask, particle_halo_cells)
    if np.mean(mask) > maximum_fraction:
        # Geometric particle protection has priority over the residual cap.
        # This is recorded by the actual active fraction rather than hidden.
        pass
    energy = indicators**2
    captured = float(energy[mask].sum() / max(energy.sum(), np.finfo(float).eps))
    return PeriodicRefinementPatch(mask, np.repeat(mask, 2), captured, float(np.mean(mask)))


def residual_space_time_amr_rollout(
    fine_initial: HybridMCHState,
    coarse_grid: PeriodicGrid,
    parameters: ModifiedCHParameters,
    *,
    tolerance: float,
    final_time: float,
    minimum_dt: float,
    maximum_dt: float,
    initial_dt: float,
    theta: float = 0.65,
    maximum_patch_fraction: float = 0.75,
    halo_cells: int = 1,
    particle_halo_cells: int = 2,
    temporal_safety: float = 0.8,
    nonlinear_tolerance: float = 1.0e-11,
    maximum_iterations: int = 80,
) -> ClassicalTrajectory:
    """Classical residual/Dörfler local AMR with residual time control.

    The controller sees only the current residual, a CFL bound and the user
    tolerance.  It does not call M1.1, a neural checkpoint or a reference.
    """
    if not 0.0 < minimum_dt <= initial_dt <= maximum_dt:
        raise ValueError("invalid residual-AMR time-step bounds")
    if tolerance <= 0.0 or final_time <= 0.0:
        raise ValueError("tolerance and final_time must be positive")
    fine_grid = PeriodicGrid(2 * coarse_grid.points, coarse_grid.length, coarse_grid.x_min)
    state = composite_from_fine_uniform(fine_initial, coarse_grid)
    initial_rep = _represented(uniform_fine_state(state))
    mass0 = hybrid_mass(initial_rep, fine_grid)
    h10 = hybrid_h1_squared(initial_rep, fine_grid)
    dt = initial_dt
    time = 0.0
    steps = patch_changes = adjustments = 0
    active_dof_time = 0.0
    maximum_mass = maximum_h1 = 0.0
    while time < final_time - 10.0 * np.finfo(float).eps:
        coarse_proxy = HybridMCHState(
            state.positions.copy(), state.amplitudes.copy(), state.coarse_momentum.copy()
        )
        target = _residual_patch(
            coarse_proxy, coarse_grid, parameters,
            theta=theta, maximum_fraction=maximum_patch_fraction,
            halo_cells=halo_cells, particle_halo_cells=particle_halo_cells,
        )
        if not np.array_equal(target.coarse_mask, state.patch.coarse_mask):
            state, _, _, _ = migrate_patch_invariantly(state, fine_grid, target)
            patch_changes += 1

        flux, source = hybrid_flux_and_source(coarse_proxy, coarse_grid, parameters)
        rate = -finite_volume_divergence(flux, coarse_grid.h) + source
        velocity = hybrid_particle_velocity(coarse_proxy, coarse_grid, alpha=parameters.alpha)
        state_scale = max(
            float(np.linalg.norm(coarse_proxy.field_momentum)),
            float(np.linalg.norm(coarse_proxy.positions)), 1.0,
        )
        defect_rate = float(np.sqrt(np.linalg.norm(rate) ** 2 + np.linalg.norm(velocity) ** 2) / state_scale)
        total = HelmholtzOperator(coarse_grid).solve(coarse_proxy.field_momentum)
        slope = first_derivative(total, coarse_grid)
        speed = float(np.max(np.abs(parameters.alpha * (total**2 - slope**2))))
        cfl_dt = maximum_dt if speed <= np.finfo(float).eps else 0.45 * coarse_grid.h / speed
        residual_dt = np.sqrt(tolerance / max(defect_rate, np.finfo(float).eps)) * temporal_safety
        proposed = min(maximum_dt, cfl_dt, residual_dt, final_time - time)
        proposed = max(minimum_dt, proposed) if final_time - time >= minimum_dt else final_time - time
        if not np.isclose(proposed, dt):
            adjustments += 1
        dt = proposed
        solver = CompatibleLocalAMRSolver(
            coarse_grid, parameters, dt,
            nonlinear_tolerance=nonlinear_tolerance,
            maximum_iterations=maximum_iterations,
        )
        state, diagnostics = solver.step(state)
        represented = _represented(uniform_fine_state(state))
        maximum_mass = max(
            maximum_mass,
            abs(hybrid_mass(represented, fine_grid) - mass0) / max(abs(mass0), np.finfo(float).eps),
        )
        maximum_h1 = max(
            maximum_h1,
            abs(hybrid_h1_squared(represented, fine_grid) - h10) / max(abs(h10), np.finfo(float).eps),
        )
        active_dof_time += dt * (
            coarse_grid.points + np.count_nonzero(state.patch.coarse_mask) + state.positions.size
        )
        time += dt
        steps += 1
        if steps > 100000:
            raise RuntimeError("residual AMR exceeded deterministic step budget")
    return ClassicalTrajectory(
        "residual_space_time_amr", fine_grid, uniform_fine_state(state), steps,
        float(active_dof_time), maximum_mass, maximum_h1,
        patch_changes=patch_changes, time_adjustments=adjustments,
    )
