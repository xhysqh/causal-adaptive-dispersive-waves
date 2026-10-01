"""Deterministic conservative space--time controller built on CMAME-M1.0.1.

The accepted time update is always the compatible midpoint method.  Mesh
migration preserves the representation-exact mass and H1 functionals by an
analytic two-parameter transfer; it is not a post-time-step state projection.
M1.2 certifies global h-adaptivity and local Dörfler diagnostics.  It does not
claim that the historical projected P3 patch integrator is compatible AMR.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.signal import resample

from fgsp_ch.discretization.grids import PeriodicGrid
from fgsp_ch.discretization.helmholtz import HelmholtzOperator
from fgsp_ch.equations.modified_ch import ModifiedCHParameters
from fgsp_ch.estimators.mch_hybrid_aposteriori import compatible_rollout, dorfler_mark, hierarchical_estimate
from fgsp_ch.representations.cmame_p2_hybrid import CMAMEHybridState, hybrid_h1_squared, hybrid_mass
from fgsp_ch.solvers.mch_hybrid_amr import HybridMCHState, hybrid_particle_velocity


def _represented(state: HybridMCHState) -> CMAMEHybridState:
    return CMAMEHybridState(state.field_momentum, state.positions, state.amplitudes)


def representation_route(state: HybridMCHState, grid: PeriodicGrid, *, field_floor: float = 1.0e-12) -> str:
    field_norm = float(np.linalg.norm(state.field_momentum))
    if not state.positions.size:
        return "eulerian"
    if field_norm <= field_floor:
        return "particle"
    return "hybrid"


def minimum_periodic_separation(positions: NDArray[np.floating], length: float) -> float:
    q = np.asarray(positions, dtype=np.float64)
    if q.size < 2:
        return float(length)
    delta = np.abs((q[:, None] - q[None, :] + 0.5 * length) % length - 0.5 * length)
    delta[np.eye(q.size, dtype=bool)] = np.inf
    return float(np.min(delta))


def relative_particle_cfl(state: HybridMCHState, grid: PeriodicGrid, alpha: float, dt: float) -> float:
    if state.positions.size < 2:
        return 0.0
    velocity = hybrid_particle_velocity(state, grid, alpha=alpha)
    relative = float(np.max(np.abs(velocity[:, None] - velocity[None, :])))
    return dt * relative / max(minimum_periodic_separation(state.positions, grid.length), np.finfo(float).eps)


def invariant_preserving_transfer(
    state: HybridMCHState, source: PeriodicGrid, target: PeriodicGrid
) -> tuple[HybridMCHState, float, float]:
    """Transfer a hybrid state while matching mass and H1 analytically."""
    before = _represented(state)
    target_mass = hybrid_mass(before, source)
    target_h1 = hybrid_h1_squared(before, source)
    field = HelmholtzOperator(source).solve(state.field_momentum)
    field = np.asarray(resample(field, target.points)).real
    zero_mean = field - np.mean(field)
    atomic_mass = hybrid_mass(
        CMAMEHybridState(np.zeros(target.points), state.positions, state.amplitudes), target
    )
    constant = (target_mass - atomic_mass) / target.length

    def represented_at(scale: float) -> CMAMEHybridState:
        candidate_field = constant + scale * zero_mean
        momentum = HelmholtzOperator(target).apply(candidate_field)
        return CMAMEHybridState(momentum, state.positions.copy(), state.amplitudes.copy())

    e0 = hybrid_h1_squared(represented_at(0.0), target)
    ep = hybrid_h1_squared(represented_at(1.0), target)
    em = hybrid_h1_squared(represented_at(-1.0), target)
    quadratic = 0.5 * (ep + em) - e0
    linear = 0.5 * (ep - em)
    constant_term = e0 - target_h1
    roots: list[float] = []
    if abs(constant_term) <= 1.0e-13 * max(1.0, abs(target_h1)):
        # Pure particle states (and constant regular fields) contain no
        # zero-mean degree of freedom.  Their invariants already agree.
        roots.append(1.0)
    if abs(quadratic) <= 1.0e-15:
        if abs(linear) > 1.0e-15:
            roots.append(-constant_term / linear)
    else:
        discriminant = linear * linear - 4.0 * quadratic * constant_term
        if discriminant >= -1.0e-12 * max(1.0, linear * linear):
            root = np.sqrt(max(discriminant, 0.0))
            roots.extend(((-linear + root) / (2.0 * quadratic), (-linear - root) / (2.0 * quadratic)))
    finite = [value for value in roots if np.isfinite(value)]
    if not finite:
        raise RuntimeError("no real invariant-preserving grid-transfer scale")
    scale = min(finite, key=lambda value: abs(value - 1.0))
    final = represented_at(scale)
    transferred = HybridMCHState(final.positions, final.amplitudes, final.regular_momentum)
    mass_defect = abs(hybrid_mass(final, target) - target_mass) / max(abs(target_mass), np.finfo(float).eps)
    h1_defect = abs(hybrid_h1_squared(final, target) - target_h1) / max(abs(target_h1), np.finfo(float).eps)
    return transferred, mass_defect, h1_defect


@dataclass(frozen=True, slots=True)
class AdaptiveControllerConfig:
    tolerance: float
    time_fraction: float
    spatial_fraction: float
    safety_factor: float
    dorfler_theta: float
    minimum_points: int
    maximum_points: int
    minimum_dt: float
    maximum_dt: float
    time_safety: float = 0.9
    time_growth_limit: float = 2.0
    time_shrink_limit: float = 0.5
    coarsen_ratio: float = 0.2
    coarsen_hysteresis: int = 2
    minimum_cells_per_separation: float = 2.0
    maximum_relative_particle_cfl: float = 0.15
    maximum_rejections: int = 12

    def __post_init__(self) -> None:
        if self.tolerance <= 0.0 or self.minimum_dt <= 0.0 or self.maximum_dt < self.minimum_dt:
            raise ValueError("invalid adaptive tolerance or time-step bounds")
        if self.minimum_points < 16 or self.maximum_points < self.minimum_points:
            raise ValueError("invalid adaptive grid bounds")
        if not np.isclose(self.time_fraction + self.spatial_fraction, 1.0):
            raise ValueError("time and spatial error fractions must sum to one")


@dataclass(frozen=True, slots=True)
class AdaptiveStepDiagnostics:
    accepted_dt: float
    proposed_dt: float
    points_before: int
    points_after: int
    time_estimate: float
    spatial_estimate: float
    total_estimate: float
    marked_fraction: float
    marked_mask: NDArray[np.bool_]
    minimum_separation: float
    separation_cells: float
    relative_particle_cfl: float
    rejections: int
    refined: bool
    coarsened: bool
    geometry_forced_refinement: bool
    mass_transfer_defect: float
    h1_transfer_defect: float
    route: str


@dataclass(slots=True)
class CompatibleAdaptiveHybridSolver:
    parameters: ModifiedCHParameters
    controller: AdaptiveControllerConfig
    nonlinear: dict[str, float | int]
    low_error_streak: int = 0

    def step(
        self, state: HybridMCHState, grid: PeriodicGrid, dt: float, *, remaining_time: float | None = None
    ) -> tuple[HybridMCHState, PeriodicGrid, float, AdaptiveStepDiagnostics]:
        cfg = self.controller
        if remaining_time is not None and remaining_time <= 0.0:
            raise ValueError("remaining_time must be positive")
        step_floor = min(cfg.minimum_dt, remaining_time) if remaining_time is not None else cfg.minimum_dt
        step_ceiling = min(cfg.maximum_dt, remaining_time) if remaining_time is not None else cfg.maximum_dt
        trial_dt = float(np.clip(dt, step_floor, step_ceiling))
        rejections = 0
        while True:
            candidate, estimate = hierarchical_estimate(
                state, grid, self.parameters, trial_dt, trial_dt, self.nonlinear
            )
            eta_t = cfg.safety_factor * estimate.temporal
            eta_x = cfg.safety_factor * (estimate.field + estimate.particle + estimate.interface)
            particle_cfl = relative_particle_cfl(state, grid, self.parameters.alpha, trial_dt)
            reject = eta_t > cfg.time_fraction * cfg.tolerance or particle_cfl > cfg.maximum_relative_particle_cfl
            if not reject or trial_dt <= step_floor * (1.0 + 1.0e-12):
                break
            trial_dt = max(step_floor, cfg.time_shrink_limit * trial_dt)
            rejections += 1
            if rejections > cfg.maximum_rejections:
                raise RuntimeError("M1.2 exceeded its rejection budget")

        marked = dorfler_mark(estimate.local_indicators, cfg.dorfler_theta)
        marked_fraction = float(np.mean(marked))
        separation = minimum_periodic_separation(candidate.positions, grid.length)
        separation_cells = separation / grid.h
        geometry_refine = bool(candidate.positions.size >= 2 and separation_cells < cfg.minimum_cells_per_separation)
        refine = (eta_x > cfg.spatial_fraction * cfg.tolerance or geometry_refine) and grid.points < cfg.maximum_points
        if eta_x < cfg.coarsen_ratio * cfg.spatial_fraction * cfg.tolerance and not geometry_refine:
            self.low_error_streak += 1
        else:
            self.low_error_streak = 0
        coarsen = (
            self.low_error_streak >= cfg.coarsen_hysteresis
            and grid.points > cfg.minimum_points and not refine
        )
        next_grid = grid
        mass_transfer = h1_transfer = 0.0
        if refine:
            next_grid = PeriodicGrid(min(2 * grid.points, cfg.maximum_points), grid.length, grid.x_min)
            migrated, mass_transfer, h1_transfer = invariant_preserving_transfer(state, grid, next_grid)
            candidate = compatible_rollout(
                migrated, next_grid, self.parameters, trial_dt, trial_dt, self.nonlinear
            )
            self.low_error_streak = 0
        elif coarsen:
            next_grid = PeriodicGrid(max(grid.points // 2, cfg.minimum_points), grid.length, grid.x_min)
            migrated, mass_transfer, h1_transfer = invariant_preserving_transfer(state, grid, next_grid)
            candidate = compatible_rollout(
                migrated, next_grid, self.parameters, trial_dt, trial_dt, self.nonlinear
            )
            self.low_error_streak = 0

        time_target = cfg.time_fraction * cfg.tolerance
        if eta_t <= np.finfo(float).eps:
            factor = cfg.time_growth_limit
        else:
            factor = cfg.time_safety * (time_target / eta_t) ** (1.0 / 3.0)
            factor = float(np.clip(factor, cfg.time_shrink_limit, cfg.time_growth_limit))
        next_dt = float(np.clip(trial_dt * factor, cfg.minimum_dt, cfg.maximum_dt))
        diagnostics = AdaptiveStepDiagnostics(
            trial_dt, next_dt, grid.points, next_grid.points, eta_t, eta_x,
            eta_t + eta_x, marked_fraction, marked, separation,
            separation_cells, particle_cfl, rejections, refine, coarsen,
            geometry_refine, mass_transfer, h1_transfer,
            representation_route(candidate, next_grid),
        )
        return candidate, next_grid, next_dt, diagnostics
