from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.signal import resample

from fgsp_ch.discretization.grids import PeriodicGrid
from fgsp_ch.nonlocal_waves.evaluation.bo_metrics import relative_spectral_h1
from fgsp_ch.nonlocal_waves.initial_conditions.bo_families import bo_initial_condition
from fgsp_ch.nonlocal_waves.operators.bo_fourier import BOFourierOperator
from fgsp_ch.nonlocal_waves.solvers.bo_reference import BOIFRK4Solver, solve_bo_dop853


@dataclass(frozen=True, slots=True)
class BOConvergenceRow:
    family: str
    amplitude: float
    base_points: int
    finest_points: int
    base_dt: float
    finest_dt: float
    spatial_coarse_defect: float
    spatial_fine_defect: float
    temporal_coarse_defect: float
    temporal_fine_defect: float
    observed_time_order: float
    ifrk4_dop853_defect: float
    reversibility_defect: float
    maximum_mass_drift: float
    maximum_l2_drift: float
    maximum_hamiltonian_drift: float
    aliasing_defect: float

    def as_dict(self) -> dict[str, float | int | str]:
        return asdict(self)


def periodic_project(values: np.ndarray, points: int) -> np.ndarray:
    data = np.asarray(values, dtype=np.float64)
    if data.ndim != 1 or points < 8:
        raise ValueError("periodic projection requires a vector and at least eight points")
    return np.asarray(resample(data, points), dtype=np.float64)


def _compare(high: np.ndarray, low: np.ndarray, low_grid: PeriodicGrid) -> float:
    return relative_spectral_h1(low, periodic_project(high, low_grid.points), low_grid)


def audit_bo_family(
    family: str,
    *,
    amplitude: float,
    base_points: int,
    base_dt: float,
    final_time: float,
    length: float,
    x_min: float = 0.0,
    dop853_rtol: float = 1.0e-11,
    dop853_atol: float = 1.0e-13,
) -> tuple[BOConvergenceRow, np.ndarray, np.ndarray]:
    point_levels = (base_points, 2 * base_points, 4 * base_points)
    dt_levels = (base_dt, 0.5 * base_dt, 0.25 * base_dt)
    spatial_finals: list[np.ndarray] = []
    spatial_trajectories = []
    for points in point_levels:
        grid = PeriodicGrid(points, length, x_min)
        initial = bo_initial_condition(family, grid, amplitude=amplitude)
        trajectory = BOIFRK4Solver(grid).solve(
            initial, final_time=final_time, dt=dt_levels[-1]
        )
        spatial_finals.append(trajectory.final_state)
        spatial_trajectories.append(trajectory)

    coarse_grid = PeriodicGrid(point_levels[0], length, x_min)
    middle_grid = PeriodicGrid(point_levels[1], length, x_min)
    spatial_coarse = _compare(spatial_finals[1], spatial_finals[0], coarse_grid)
    spatial_fine = _compare(spatial_finals[2], spatial_finals[1], middle_grid)

    finest_grid = PeriodicGrid(point_levels[-1], length, x_min)
    finest_initial = bo_initial_condition(family, finest_grid, amplitude=amplitude)
    time_trajectories = [
        BOIFRK4Solver(finest_grid).solve(
            finest_initial, final_time=final_time, dt=dt
        )
        for dt in dt_levels
    ]
    time_coarse = relative_spectral_h1(
        time_trajectories[0].final_state, time_trajectories[1].final_state, finest_grid
    )
    time_fine = relative_spectral_h1(
        time_trajectories[1].final_state, time_trajectories[2].final_state, finest_grid
    )
    if time_fine <= 10.0 * np.finfo(float).eps:
        observed_order = 4.0
    else:
        observed_order = float(np.log2(max(time_coarse, np.finfo(float).eps) / time_fine))

    dop853 = solve_bo_dop853(
        finest_initial,
        finest_grid,
        final_time=final_time,
        rtol=dop853_rtol,
        atol=dop853_atol,
        maximum_step=base_dt,
    )
    parity = relative_spectral_h1(
        time_trajectories[-1].final_state, dop853.final_state, finest_grid
    )
    reverse = BOIFRK4Solver(finest_grid).solve(
        time_trajectories[-1].final_state,
        initial_time=final_time,
        final_time=0.0,
        dt=dt_levels[-1],
    )
    # The even-grid Nyquist coefficient is outside the signed real Fourier
    # reference space and is removed once, when the initial state is accepted.
    # Reversibility must compare against that accepted state rather than
    # charging the time integrator for the initial representation projection.
    accepted_initial = time_trajectories[-1].states[0]
    reversibility = relative_spectral_h1(
        reverse.final_state, accepted_initial, finest_grid
    )
    drift = time_trajectories[-1].maximum_relative_invariant_drift()
    aliasing = BOFourierOperator(finest_grid).aliasing_defect(finest_initial)
    row = BOConvergenceRow(
        family,
        amplitude,
        base_points,
        point_levels[-1],
        base_dt,
        dt_levels[-1],
        spatial_coarse,
        spatial_fine,
        time_coarse,
        time_fine,
        observed_order,
        parity,
        reversibility,
        drift["mass"],
        drift["half_l2_squared"],
        drift["hamiltonian"],
        aliasing,
    )
    return row, time_trajectories[-1].times, time_trajectories[-1].states
