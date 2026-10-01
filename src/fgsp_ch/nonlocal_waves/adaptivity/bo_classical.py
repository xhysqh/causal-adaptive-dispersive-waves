"""Classical, reference-free space--time adaptivity for periodic BO."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from fgsp_ch.discretization.grids import PeriodicGrid
from fgsp_ch.nonlocal_waves.evaluation.bo_metrics import bo_invariants, relative_spectral_h1
from fgsp_ch.nonlocal_waves.operators.bo_fourier import BOFourierOperator
from fgsp_ch.nonlocal_waves.solvers.bo_reference import BOIFRK4Solver


@dataclass(frozen=True, slots=True)
class BOResolutionIndicators:
    tail_ratio: float
    aliasing_defect: float
    nonlinear_dispersion_ratio: float
    maximum_linear_phase: float


@dataclass(frozen=True, slots=True)
class BOAdaptiveConfig:
    tolerance: float
    minimum_points: int = 32
    maximum_points: int = 256
    initial_dt: float = 1.0e-2
    minimum_dt: float = 2.5e-5
    maximum_dt: float = 2.0e-2
    temporal_budget_fraction: float = 2.0e-2
    richardson_safety: float = 1.25
    controller_safety: float = 0.85
    minimum_step_factor: float = 0.25
    maximum_step_factor: float = 1.8
    refine_tail: float = 2.0e-3
    coarsen_tail: float = 2.0e-7
    refine_aliasing: float = 2.0e-4
    coarsen_aliasing: float = 2.0e-8
    coarsen_patience: int = 2
    maximum_migration_defect: float = 0.15
    maximum_mass_drift: float = 1.0e-10
    maximum_l2_drift: float = 2.0e-5
    maximum_hamiltonian_drift: float = 2.0e-5
    maximum_attempts: int = 20000

    def __post_init__(self) -> None:
        if self.tolerance <= 0.0:
            raise ValueError("BO adaptive tolerance must be positive")
        if self.minimum_points < 16 or self.maximum_points < self.minimum_points:
            raise ValueError("invalid BO adaptive point bounds")
        if self.minimum_points & (self.minimum_points - 1):
            raise ValueError("minimum_points must be a power of two")
        if self.maximum_points & (self.maximum_points - 1):
            raise ValueError("maximum_points must be a power of two")
        if not 0.0 < self.minimum_step_factor <= 1.0 <= self.maximum_step_factor:
            raise ValueError("invalid BO adaptive step factors")


@dataclass(frozen=True, slots=True)
class BOAdaptiveResult:
    final_state: NDArray[np.floating]
    final_grid: PeriodicGrid
    final_time: float
    rows: tuple[dict[str, float | int | str | bool], ...]
    accepted_steps: int
    rejected_steps: int
    refinements: int
    coarsenings: int
    coarsening_vetoes: int
    time_growth_events: int
    time_shrink_events: int
    active_dof_time: float
    maximum_mass_drift: float
    maximum_l2_drift: float
    maximum_hamiltonian_drift: float
    maximum_migration_defect: float


def project_periodic_spectrum(
    values: NDArray[np.floating], source: PeriodicGrid, target_points: int
) -> tuple[NDArray[np.floating], PeriodicGrid]:
    """Migrate a real periodic state by Fourier truncation/zero-padding."""
    if target_points < 16 or target_points % 2:
        raise ValueError("target Fourier grid must be even and contain at least 16 points")
    data = np.asarray(values, dtype=np.float64)
    source.validate_state(data)
    target = PeriodicGrid(int(target_points), source.length)
    source_hat = BOFourierOperator(source).spectrum(data)
    target_hat = np.zeros(target.points // 2 + 1, dtype=np.complex128)
    copied = min(source.points // 2, target.points // 2)
    target_hat[:copied] = (target.points / source.points) * source_hat[:copied]
    target_hat[0] = (target.points / source.points) * source_hat[0]
    target_hat[-1] = 0.0
    migrated = BOFourierOperator(target).physical(target_hat)
    return migrated, target


def bo_resolution_indicators(
    values: NDArray[np.floating], grid: PeriodicGrid, dt: float,
    *, tail_fraction: float = 0.65,
) -> BOResolutionIndicators:
    operator = BOFourierOperator(grid)
    spectrum = operator.spectrum(values)
    weights = 1.0 + operator.wave_numbers**2
    cutoff = max(1, int(tail_fraction * (spectrum.size - 1)))
    total = float(np.sum(weights * np.abs(spectrum) ** 2))
    tail = float(np.sum(weights[cutoff:] * np.abs(spectrum[cutoff:]) ** 2))
    nonlinear = operator.nonlinear_spectrum(spectrum)
    dispersive = operator.linear_symbol * spectrum
    nonlinear_ratio = float(
        np.linalg.norm(nonlinear) /
        max(np.linalg.norm(dispersive), np.finfo(float).eps)
    )
    return BOResolutionIndicators(
        tail_ratio=float(np.sqrt(tail / max(total, np.finfo(float).eps))),
        aliasing_defect=operator.aliasing_defect(values),
        nonlinear_dispersion_ratio=nonlinear_ratio,
        maximum_linear_phase=float(abs(dt) * np.max(np.abs(operator.linear_symbol))),
    )


def _relative_invariant_drift(current, initial) -> dict[str, float]:
    result: dict[str, float] = {}
    for key, reference in initial.as_dict().items():
        scale = max(abs(reference), 1.0e-12)
        result[key] = abs(current.as_dict()[key] - reference) / scale
    return result


class BOClassicalAdaptiveSolver:
    """Bidirectional spectral-order and IFRK4 step-doubling controller."""

    def __init__(self, config: BOAdaptiveConfig):
        self.config = config

    def solve(
        self,
        initial: NDArray[np.floating],
        grid: PeriodicGrid,
        *,
        final_time: float,
    ) -> BOAdaptiveResult:
        cfg = self.config
        if final_time <= 0.0 or not np.isfinite(final_time):
            raise ValueError("adaptive BO final_time must be positive and finite")
        values = np.asarray(initial, dtype=np.float64).copy()
        grid.validate_state(values)
        if not cfg.minimum_points <= grid.points <= cfg.maximum_points:
            raise ValueError("initial BO grid is outside adaptive bounds")
        initial_invariants = bo_invariants(values, grid)
        dt = float(np.clip(cfg.initial_dt, cfg.minimum_dt, cfg.maximum_dt))
        time = 0.0
        quiet_steps = 0
        rows: list[dict[str, float | int | str | bool]] = []
        accepted = rejected = refinements = coarsenings = vetoes = 0
        growth = shrink = 0
        active_dof_time = 0.0
        maximum_migration = 0.0
        maximum_drifts = {"mass": 0.0, "half_l2_squared": 0.0, "hamiltonian": 0.0}

        for attempt in range(cfg.maximum_attempts):
            if time >= final_time - 10.0 * np.finfo(float).eps:
                break
            indicators = bo_resolution_indicators(values, grid, dt)
            needs_refine = (
                indicators.tail_ratio > cfg.refine_tail
                or indicators.aliasing_defect > cfg.refine_aliasing
            )
            if needs_refine and grid.points < cfg.maximum_points:
                old_points = grid.points
                values, grid = project_periodic_spectrum(values, grid, 2 * grid.points)
                refinements += 1
                quiet_steps = 0
                rows.append({
                    "attempt": attempt, "time": time, "action": "refine",
                    "points_before": old_points, "points_after": grid.points,
                    "dt": dt, "tail_ratio": indicators.tail_ratio,
                    "aliasing_defect": indicators.aliasing_defect,
                    "accepted": True,
                })
                continue

            quiet = (
                indicators.tail_ratio < cfg.coarsen_tail
                and indicators.aliasing_defect < cfg.coarsen_aliasing
            )
            quiet_steps = quiet_steps + 1 if quiet else 0
            if quiet_steps >= cfg.coarsen_patience and grid.points > cfg.minimum_points:
                coarse, coarse_grid = project_periodic_spectrum(values, grid, grid.points // 2)
                reconstructed, _ = project_periodic_spectrum(coarse, coarse_grid, grid.points)
                migration = relative_spectral_h1(reconstructed, values, grid)
                maximum_migration = max(maximum_migration, migration)
                if migration <= cfg.maximum_migration_defect * cfg.tolerance:
                    old_points = grid.points
                    values, grid = coarse, coarse_grid
                    coarsenings += 1
                    rows.append({
                        "attempt": attempt, "time": time, "action": "coarsen",
                        "points_before": old_points, "points_after": grid.points,
                        "dt": dt, "migration_defect": migration,
                        "accepted": True,
                    })
                else:
                    vetoes += 1
                    rows.append({
                        "attempt": attempt, "time": time, "action": "coarsen_veto",
                        "points_before": grid.points, "points_after": grid.points,
                        "dt": dt, "migration_defect": migration,
                        "accepted": False,
                    })
                quiet_steps = 0
                continue

            trial_dt = min(dt, final_time - time)
            solver = BOIFRK4Solver(grid)
            full = solver.step(values, trial_dt)
            half = solver.step(values, 0.5 * trial_dt)
            half = solver.step(half, 0.5 * trial_dt)
            richardson = relative_spectral_h1(full, half, grid) / 15.0
            estimated_error = cfg.richardson_safety * richardson
            temporal_limit = cfg.temporal_budget_fraction * cfg.tolerance
            candidate_invariants = bo_invariants(half, grid)
            drift = _relative_invariant_drift(candidate_invariants, initial_invariants)
            for key in maximum_drifts:
                maximum_drifts[key] = max(maximum_drifts[key], drift[key])
            temporal_ratio = estimated_error / max(temporal_limit, np.finfo(float).eps)
            invariant_ratio = max(
                drift["mass"] / cfg.maximum_mass_drift,
                drift["half_l2_squared"] / cfg.maximum_l2_drift,
                drift["hamiltonian"] / cfg.maximum_hamiltonian_drift,
            )
            risk = max(temporal_ratio, invariant_ratio)
            accepted_step = risk <= 1.0 or trial_dt <= 1.01 * cfg.minimum_dt
            rows.append({
                "attempt": attempt, "time": time, "action": "step",
                "points_before": grid.points, "points_after": grid.points,
                "dt": trial_dt, "tail_ratio": indicators.tail_ratio,
                "aliasing_defect": indicators.aliasing_defect,
                "temporal_error": estimated_error,
                "temporal_ratio": temporal_ratio, "invariant_ratio": invariant_ratio,
                "accepted": accepted_step,
            })
            exponent = -0.2
            factor = cfg.maximum_step_factor if risk <= np.finfo(float).eps else (
                cfg.controller_safety * risk**exponent
            )
            factor = float(np.clip(factor, cfg.minimum_step_factor, cfg.maximum_step_factor))
            proposed_dt = float(np.clip(trial_dt * factor, cfg.minimum_dt, cfg.maximum_dt))
            if accepted_step:
                values = half
                time += trial_dt
                active_dof_time += grid.points * trial_dt
                accepted += 1
                if proposed_dt > 1.05 * trial_dt:
                    growth += 1
                elif proposed_dt < 0.95 * trial_dt:
                    shrink += 1
                dt = proposed_dt
            else:
                rejected += 1
                shrink += 1
                dt = min(proposed_dt, 0.8 * trial_dt)
        else:
            raise RuntimeError("BO adaptive controller exceeded maximum attempts")

        if time < final_time - 1.0e-12:
            raise RuntimeError("BO adaptive controller did not reach final time")
        return BOAdaptiveResult(
            final_state=values,
            final_grid=grid,
            final_time=time,
            rows=tuple(rows),
            accepted_steps=accepted,
            rejected_steps=rejected,
            refinements=refinements,
            coarsenings=coarsenings,
            coarsening_vetoes=vetoes,
            time_growth_events=growth,
            time_shrink_events=shrink,
            active_dof_time=active_dof_time,
            maximum_mass_drift=maximum_drifts["mass"],
            maximum_l2_drift=maximum_drifts["half_l2_squared"],
            maximum_hamiltonian_drift=maximum_drifts["hamiltonian"],
            maximum_migration_defect=maximum_migration,
        )
