"""High-accuracy fixed-resolution reference integrators for periodic BO."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp

from fgsp_ch.discretization.grids import PeriodicGrid
from fgsp_ch.nonlocal_waves.evaluation.bo_metrics import BOInvariants, bo_invariants
from fgsp_ch.nonlocal_waves.operators.bo_fourier import BOFourierOperator


@dataclass(frozen=True, slots=True)
class BOReferenceTrajectory:
    times: NDArray[np.floating]
    states: NDArray[np.floating]
    invariants: tuple[BOInvariants, ...]
    method: str

    @property
    def final_state(self) -> NDArray[np.floating]:
        return np.asarray(self.states[-1], dtype=np.float64)

    def maximum_relative_invariant_drift(self) -> dict[str, float]:
        initial = self.invariants[0].as_dict()
        result: dict[str, float] = {}
        for key, value in initial.items():
            scale = max(abs(value), np.finfo(float).eps)
            result[key] = max(
                abs(row.as_dict()[key] - value) / scale for row in self.invariants
            )
        return result


@dataclass(frozen=True, slots=True)
class BOIFRK4Solver:
    grid: PeriodicGrid
    padding_factor: float = 1.5

    @property
    def operator(self) -> BOFourierOperator:
        return BOFourierOperator(self.grid, self.padding_factor)

    def step_spectrum(
        self, coefficients: NDArray[np.complexfloating], dt: float
    ) -> NDArray[np.complexfloating]:
        if not np.isfinite(dt) or dt == 0.0:
            raise ValueError("BO IFRK4 step requires a finite nonzero dt")
        operator = self.operator
        initial = np.asarray(coefficients, dtype=np.complex128)
        if initial.shape != (self.grid.points // 2 + 1,):
            raise ValueError("BO spectrum has an incompatible shape")
        linear = operator.linear_symbol

        def interaction_rhs(stage: float, transformed: NDArray[np.complexfloating]):
            phase = np.exp(linear * stage)
            physical_spectrum = phase * transformed
            return np.exp(-linear * stage) * operator.nonlinear_spectrum(physical_spectrum)

        k1 = interaction_rhs(0.0, initial)
        k2 = interaction_rhs(0.5 * dt, initial + 0.5 * dt * k1)
        k3 = interaction_rhs(0.5 * dt, initial + 0.5 * dt * k2)
        k4 = interaction_rhs(dt, initial + dt * k3)
        transformed = initial + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        result = np.exp(linear * dt) * transformed
        result[0] = initial[0]
        result[-1] = 0.0
        if not np.all(np.isfinite(result)):
            raise FloatingPointError("BO IFRK4 step produced a non-finite spectrum")
        return np.asarray(result, dtype=np.complex128)

    def step(self, values: NDArray[np.floating], dt: float) -> NDArray[np.floating]:
        coefficients = self.operator.spectrum(values)
        return self.operator.physical(self.step_spectrum(coefficients, dt))

    def step_embedded_spectrum(
        self, coefficients: NDArray[np.complexfloating], dt: float
    ) -> tuple[NDArray[np.complexfloating], NDArray[np.complexfloating]]:
        """Dormand--Prince 5(4) pair in the BO interaction picture.

        Both solutions share seven nonlinear evaluations.  The fifth-order
        state is committed; their difference is a causal local-error estimate.
        """
        if not np.isfinite(dt) or dt == 0.0:
            raise ValueError("embedded BO step requires a finite nonzero dt")
        operator = self.operator
        initial = np.asarray(coefficients, dtype=np.complex128)
        if initial.shape != (self.grid.points // 2 + 1,):
            raise ValueError("BO spectrum has an incompatible shape")
        linear = operator.linear_symbol

        def rhs(stage: float, transformed: NDArray[np.complexfloating]):
            phase = np.exp(linear * (stage * dt))
            return np.exp(-linear * (stage * dt)) * operator.nonlinear_spectrum(
                phase * transformed
            )

        k1 = rhs(0.0, initial)
        k2 = rhs(1.0 / 5.0, initial + dt * (k1 / 5.0))
        k3 = rhs(3.0 / 10.0, initial + dt * (3.0 * k1 / 40.0 + 9.0 * k2 / 40.0))
        k4 = rhs(4.0 / 5.0, initial + dt * (
            44.0 * k1 / 45.0 - 56.0 * k2 / 15.0 + 32.0 * k3 / 9.0
        ))
        k5 = rhs(8.0 / 9.0, initial + dt * (
            19372.0 * k1 / 6561.0 - 25360.0 * k2 / 2187.0
            + 64448.0 * k3 / 6561.0 - 212.0 * k4 / 729.0
        ))
        k6 = rhs(1.0, initial + dt * (
            9017.0 * k1 / 3168.0 - 355.0 * k2 / 33.0
            + 46732.0 * k3 / 5247.0 + 49.0 * k4 / 176.0
            - 5103.0 * k5 / 18656.0
        ))
        fifth_transformed = initial + dt * (
            35.0 * k1 / 384.0 + 500.0 * k3 / 1113.0
            + 125.0 * k4 / 192.0 - 2187.0 * k5 / 6784.0
            + 11.0 * k6 / 84.0
        )
        k7 = rhs(1.0, fifth_transformed)
        fourth_transformed = initial + dt * (
            5179.0 * k1 / 57600.0 + 7571.0 * k3 / 16695.0
            + 393.0 * k4 / 640.0 - 92097.0 * k5 / 339200.0
            + 187.0 * k6 / 2100.0 + k7 / 40.0
        )
        phase = np.exp(linear * dt)
        fifth = phase * fifth_transformed
        fourth = phase * fourth_transformed
        for result in (fifth, fourth):
            result[0] = initial[0]
            result[-1] = 0.0
            if not np.all(np.isfinite(result)):
                raise FloatingPointError("embedded BO step produced a non-finite spectrum")
        return (
            np.asarray(fifth, dtype=np.complex128),
            np.asarray(fourth, dtype=np.complex128),
        )

    def step_embedded(
        self, values: NDArray[np.floating], dt: float
    ) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        coefficients = self.operator.spectrum(values)
        fifth, fourth = self.step_embedded_spectrum(coefficients, dt)
        return self.operator.physical(fifth), self.operator.physical(fourth)

    def solve(
        self,
        initial: NDArray[np.floating],
        *,
        final_time: float,
        dt: float,
        initial_time: float = 0.0,
    ) -> BOReferenceTrajectory:
        interval = float(final_time - initial_time)
        if interval == 0.0 or not np.isfinite(interval):
            raise ValueError("BO solve interval must be finite and nonzero")
        if dt <= 0.0 or not np.isfinite(dt):
            raise ValueError("BO solve dt is an absolute positive step size")
        steps = int(round(abs(interval) / dt))
        if steps < 1 or not np.isclose(steps * dt, abs(interval), atol=1.0e-13):
            raise ValueError("BO interval magnitude must be an integer multiple of dt")
        signed_dt = np.copysign(dt, interval)
        coefficients = self.operator.spectrum(initial)
        states = [self.operator.physical(coefficients)]
        times = [float(initial_time)]
        for index in range(steps):
            coefficients = self.step_spectrum(coefficients, signed_dt)
            states.append(self.operator.physical(coefficients))
            times.append(float(initial_time + (index + 1) * signed_dt))
        state_array = np.asarray(states, dtype=np.float64)
        invariants = tuple(bo_invariants(row, self.grid) for row in state_array)
        return BOReferenceTrajectory(
            np.asarray(times, dtype=np.float64), state_array, invariants, "ifrk4"
        )


def solve_bo_dop853(
    initial: NDArray[np.floating],
    grid: PeriodicGrid,
    *,
    final_time: float,
    rtol: float = 1.0e-11,
    atol: float = 1.0e-13,
    maximum_step: float | None = None,
) -> BOReferenceTrajectory:
    if final_time <= 0.0:
        raise ValueError("DOP853 BO reference final_time must be positive")
    operator = BOFourierOperator(grid)
    initial_spectrum = operator.spectrum(initial)
    linear = operator.linear_symbol

    def rhs(time: float, transformed: NDArray[np.complexfloating]):
        phase = np.exp(linear * time)
        return np.exp(-linear * time) * operator.nonlinear_spectrum(phase * transformed)

    solution = solve_ivp(
        rhs,
        (0.0, float(final_time)),
        initial_spectrum,
        method="DOP853",
        rtol=rtol,
        atol=atol,
        max_step=np.inf if maximum_step is None else maximum_step,
        t_eval=np.asarray([0.0, float(final_time)]),
    )
    if not solution.success:
        raise RuntimeError(f"BO DOP853 reference failed: {solution.message}")
    states = []
    for index, time in enumerate(solution.t):
        physical_spectrum = np.exp(linear * time) * solution.y[:, index]
        physical_spectrum[0] = initial_spectrum[0]
        physical_spectrum[-1] = 0.0
        states.append(operator.physical(physical_spectrum))
    states = np.asarray(states, dtype=np.float64)
    invariants = tuple(bo_invariants(row, grid) for row in states)
    return BOReferenceTrajectory(
        np.asarray(solution.t, dtype=np.float64), states, invariants, "dop853"
    )
