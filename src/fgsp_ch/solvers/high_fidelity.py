from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp

from fgsp_ch.equations.generalized_ch import GeneralizedCH
from fgsp_ch.geometry.metric import discrete_energy
from fgsp_ch.simulation import Trajectory


@dataclass(slots=True)
class HighFidelitySolver:
    """Independent adaptive Runge-Kutta reference integrator.

    This solver uses SciPy's high-order DOP853 integrator on the semidiscrete
    generalized CH system. It does not reuse the linearly implicit
    Crank-Nicolson update or the energy-shell projection, so it provides an
    independent temporal reference format.
    """

    equation: GeneralizedCH
    method: str = "DOP853"
    rtol: float = 1.0e-10
    atol: float = 1.0e-12

    def __post_init__(self) -> None:
        if self.rtol <= 0.0 or self.atol <= 0.0:
            raise ValueError("Reference tolerances must be positive.")

    def solve(
        self,
        initial_state: NDArray[np.floating],
        *,
        final_time: float,
        output_dt: float,
    ) -> Trajectory:
        """Integrate to ``final_time`` and return uniformly sampled states."""
        self.equation.grid.validate_state(initial_state)
        if final_time <= 0.0 or output_dt <= 0.0:
            raise ValueError("final_time and output_dt must be positive.")
        steps = round(final_time / output_dt)
        if not np.isclose(steps * output_dt, final_time, rtol=0.0, atol=1e-12):
            raise ValueError("final_time must be an integer multiple of output_dt.")
        times = np.linspace(0.0, final_time, steps + 1)
        solution = solve_ivp(
            lambda _time, state: self.equation.state_rhs(state),
            (0.0, final_time),
            np.asarray(initial_state, dtype=np.float64),
            method=self.method,
            t_eval=times,
            rtol=self.rtol,
            atol=self.atol,
        )
        if not solution.success:
            raise RuntimeError(
                f"High-fidelity integration failed: {solution.message}"
            )
        states = np.asarray(solution.y.T, dtype=initial_state.dtype)
        momenta = np.stack(
            [self.equation.helmholtz.apply(state) for state in states]
        )
        energies = np.asarray(
            [
                discrete_energy(state, self.equation.helmholtz)
                for state in states
            ],
            dtype=np.float64,
        )
        if not (
            np.all(np.isfinite(states))
            and np.all(np.isfinite(momenta))
            and np.all(np.isfinite(energies))
        ):
            raise FloatingPointError("High-fidelity trajectory contains NaN or Inf.")
        return Trajectory(
            times=times,
            states=states,
            momenta=momenta,
            energies=energies,
            projection_scales=np.ones(steps + 1, dtype=np.float64),
        )
