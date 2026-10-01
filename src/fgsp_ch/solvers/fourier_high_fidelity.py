from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp

from fgsp_ch.equations.spectral_generalized_ch import SpectralGeneralizedCH
from fgsp_ch.simulation import Trajectory


@dataclass(slots=True)
class FourierHighFidelitySolver:
    """DOP853 integration of the Fourier pseudospectral momentum equation."""

    equation: SpectralGeneralizedCH
    method: str = "DOP853"
    rtol: float = 1.0e-10
    atol: float = 1.0e-12

    def solve(
        self,
        initial_state: NDArray[np.floating],
        *,
        final_time: float,
        output_dt: float,
    ) -> Trajectory:
        self.equation.grid.validate_state(initial_state)
        steps = round(final_time / output_dt)
        if steps < 1:
            raise ValueError("At least one output step is required.")
        output_dt = final_time / steps
        times = np.linspace(0.0, final_time, steps + 1)
        initial_momentum = self.equation.operators.helmholtz_apply(initial_state)
        solution = solve_ivp(
            lambda _time, momentum: self.equation.momentum_rhs(momentum),
            (0.0, final_time),
            np.asarray(initial_momentum, dtype=np.float64),
            method=self.method,
            t_eval=times,
            rtol=self.rtol,
            atol=self.atol,
        )
        if not solution.success:
            raise RuntimeError(
                f"Fourier reference integration failed: {solution.message}"
            )
        momenta = np.asarray(solution.y.T, dtype=initial_state.dtype)
        states = np.stack(
            [
                self.equation.operators.helmholtz_solve(momentum)
                for momentum in momenta
            ]
        )
        energies = np.asarray(
            [self.equation.operators.energy(state) for state in states]
        )
        if not np.all(np.isfinite(states)):
            raise FloatingPointError("Fourier trajectory contains NaN or Inf.")
        return Trajectory(
            times=times,
            states=states,
            momenta=momenta,
            energies=energies,
            projection_scales=np.ones(steps + 1),
        )

