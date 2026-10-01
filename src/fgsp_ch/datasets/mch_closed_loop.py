from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from numpy.typing import NDArray

from fgsp_ch.equations.modified_ch import ModifiedCH
from fgsp_ch.geometry.metric import a_norm
from fgsp_ch.solvers.mch_energy_projected import MassH1ProjectedMCHSolver


State = NDArray[np.floating]
LearnedStepper = Callable[[State, State, float], State]
OracleStepper = Callable[[State, float], State]


@dataclass(frozen=True, slots=True)
class MCHClosedLoopSample:
    """Oracle label anchored at a state actually visited by the learned model."""

    trajectory: str
    step: int
    previous_state: State
    visited_state: State
    baseline_state: State
    predicted_state: State
    oracle_state: State
    embedded_direction: State
    relative_h1_error: float


def projected_substep_oracle(
    equation: ModifiedCH, *, substeps: int = 8
) -> OracleStepper:
    """Independent-from-the-coarse-step local oracle for data aggregation."""
    if substeps < 2:
        raise ValueError("The closed-loop oracle needs at least two substeps.")

    def advance(state: State, dt: float) -> State:
        result = np.asarray(state).copy()
        solver = MassH1ProjectedMCHSolver(equation, dt / substeps)
        for _ in range(substeps):
            result = solver.step(result).state
        return result

    return advance


def collect_mch_closed_loop_samples(
    equation: ModifiedCH,
    initial: State,
    learned_stepper: LearnedStepper,
    *,
    dt: float,
    steps: int,
    trajectory: str,
    oracle_stepper: OracleStepper | None = None,
) -> tuple[MCHClosedLoopSample, ...]:
    """DAgger-style collection without teacher-forcing the next model state.

    The oracle labels every model-visited state, but the next input remains the
    model prediction.  Consequently exposure errors are represented rather
    than silently reset to the reference trajectory after each step.
    """
    equation.grid.validate_state(initial)
    if dt <= 0.0 or steps < 1 or not trajectory:
        raise ValueError("dt, steps, and trajectory must be valid.")
    oracle = oracle_stepper or projected_substep_oracle(equation)
    previous = np.asarray(initial).copy()
    current = previous.copy()
    records: list[MCHClosedLoopSample] = []
    for index in range(steps):
        baseline = MassH1ProjectedMCHSolver(equation, dt).step(current).state
        half = MassH1ProjectedMCHSolver(equation, 0.5 * dt).step(current).state
        fine = MassH1ProjectedMCHSolver(equation, 0.5 * dt).step(half).state
        prediction = np.asarray(learned_stepper(previous, current, dt))
        equation.grid.validate_state(prediction)
        target = np.asarray(oracle(current, dt))
        equation.grid.validate_state(target)
        error = a_norm(prediction - target, equation.helmholtz) / max(
            a_norm(target, equation.helmholtz), np.finfo(float).eps
        )
        records.append(
            MCHClosedLoopSample(
                trajectory, index, previous.copy(), current.copy(), baseline.copy(),
                prediction.copy(), target.copy(), (fine - baseline).copy(), float(error),
            )
        )
        # This is the defining closed-loop step: do not replace with target.
        previous, current = current, prediction
    return tuple(records)
