from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from fgsp_ch.geometry.metric import discrete_energy
from fgsp_ch.solvers.semi_implicit import SemiImplicitCHSolver


@dataclass(frozen=True, slots=True)
class Trajectory:
    times: NDArray[np.floating]
    states: NDArray[np.floating]
    momenta: NDArray[np.floating]
    energies: NDArray[np.floating]
    projection_scales: NDArray[np.floating]


def solve_trajectory(
    solver: SemiImplicitCHSolver,
    initial_state: NDArray[np.floating],
    *,
    steps: int,
) -> Trajectory:
    """Run the baseline and retain every state and structural diagnostic."""
    if steps < 0:
        raise ValueError("Number of steps must be non-negative.")
    grid = solver.equation.grid
    grid.validate_state(initial_state)
    states = np.empty((steps + 1, grid.points), dtype=initial_state.dtype)
    momenta = np.empty_like(states)
    energies = np.empty(steps + 1, dtype=np.float64)
    scales = np.ones(steps + 1, dtype=np.float64)
    states[0] = initial_state
    momenta[0] = solver.equation.helmholtz.apply(initial_state)
    energies[0] = discrete_energy(initial_state, solver.equation.helmholtz)

    previous: NDArray[np.floating] | None = None
    current = initial_state.copy()
    for index in range(steps):
        result = solver.step(current, previous)
        previous, current = current, result.state
        states[index + 1] = current
        momenta[index + 1] = solver.equation.helmholtz.apply(current)
        energies[index + 1] = result.energy_after
        scales[index + 1] = result.projection_scale
    if not (
        np.all(np.isfinite(states))
        and np.all(np.isfinite(momenta))
        and np.all(np.isfinite(energies))
    ):
        raise FloatingPointError("Trajectory contains NaN or Inf.")
    return Trajectory(
        times=np.arange(steps + 1, dtype=np.float64) * solver.dt,
        states=states,
        momenta=momenta,
        energies=energies,
        projection_scales=scales,
    )
