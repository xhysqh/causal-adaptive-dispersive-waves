"""Interaction-picture solver and independent audit for power-law waves."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp

from fgsp_ch.nonlocal_waves.evaluation.power_law_metrics import power_law_invariants
from fgsp_ch.nonlocal_waves.operators.power_law_fourier import PowerLawFourierOperator


@dataclass(frozen=True, slots=True)
class PowerLawTrajectory:
    times: np.ndarray
    states: np.ndarray
    invariants: tuple
    method: str


@dataclass(frozen=True, slots=True)
class PowerLawIFRK4Solver:
    grid: object
    traits: object
    padding_factor: float = 1.5

    @property
    def operator(self):
        return PowerLawFourierOperator(self.grid, self.traits, self.padding_factor)

    def _interaction_rhs(self, stage, transformed, dt):
        linear = self.operator.linear_symbol
        phase = np.exp(linear * stage * dt)
        return np.exp(-linear * stage * dt) * self.operator.nonlinear_spectrum(
            phase * transformed
        )

    def step_embedded_spectrum(self, coefficients, dt):
        initial = np.asarray(coefficients, dtype=np.complex128)
        f = lambda c, y: self._interaction_rhs(c, y, dt)
        k1 = f(0., initial)
        k2 = f(1/5, initial + dt*k1/5)
        k3 = f(3/10, initial + dt*(3*k1/40 + 9*k2/40))
        k4 = f(4/5, initial + dt*(44*k1/45 - 56*k2/15 + 32*k3/9))
        k5 = f(8/9, initial + dt*(19372*k1/6561 - 25360*k2/2187
                                  + 64448*k3/6561 - 212*k4/729))
        k6 = f(1., initial + dt*(9017*k1/3168 - 355*k2/33
                                 + 46732*k3/5247 + 49*k4/176
                                 - 5103*k5/18656))
        fifth_t = initial + dt*(35*k1/384 + 500*k3/1113 + 125*k4/192
                                - 2187*k5/6784 + 11*k6/84)
        k7 = f(1., fifth_t)
        fourth_t = initial + dt*(5179*k1/57600 + 7571*k3/16695
                                 + 393*k4/640 - 92097*k5/339200
                                 + 187*k6/2100 + k7/40)
        phase = np.exp(self.operator.linear_symbol * dt)
        fifth, fourth = phase*fifth_t, phase*fourth_t
        for result in (fifth, fourth):
            result[0], result[-1] = initial[0], 0.0
            if not np.all(np.isfinite(result)):
                raise FloatingPointError("power-law embedded step is non-finite")
        return fifth, fourth

    def step_embedded(self, values, dt):
        high, low = self.step_embedded_spectrum(self.operator.spectrum(values), dt)
        return self.operator.physical(high), self.operator.physical(low)

    def step(self, values, dt):
        return self.step_embedded(values, dt)[0]


def solve_power_law_dop853(initial, grid, traits, *, evaluation_times,
                           rtol=1e-11, atol=1e-13, maximum_step=None):
    times = np.asarray(evaluation_times, dtype=np.float64)
    if times.ndim != 1 or len(times) < 2 or times[0] != 0 or np.any(np.diff(times) <= 0):
        raise ValueError("power-law reference times must increase from zero")
    operator = PowerLawFourierOperator(grid, traits)
    initial_hat, linear = operator.spectrum(initial), operator.linear_symbol

    def rhs(time, transformed):
        phase = np.exp(linear * time)
        return np.exp(-linear*time) * operator.nonlinear_spectrum(phase*transformed)

    solution = solve_ivp(
        rhs, (0., float(times[-1])), initial_hat, method="DOP853",
        rtol=rtol, atol=atol, t_eval=times,
        max_step=np.inf if maximum_step is None else float(maximum_step),
    )
    if not solution.success:
        raise RuntimeError(solution.message)
    states = []
    for index, time in enumerate(solution.t):
        spectrum = np.exp(linear*time) * solution.y[:, index]
        spectrum[0], spectrum[-1] = initial_hat[0], 0.0
        states.append(operator.physical(spectrum))
    states = np.asarray(states)
    return PowerLawTrajectory(
        np.asarray(solution.t), states,
        tuple(power_law_invariants(row, grid, traits) for row in states),
        "interaction-picture-dop853",
    )
