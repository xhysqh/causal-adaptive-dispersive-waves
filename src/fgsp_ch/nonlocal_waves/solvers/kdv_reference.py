"""Interaction-picture KdV integrators and an independent DOP853 audit."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp

from fgsp_ch.discretization.grids import PeriodicGrid
from fgsp_ch.nonlocal_waves.evaluation.kdv_metrics import KdVInvariants, kdv_invariants
from fgsp_ch.nonlocal_waves.operators.kdv_fourier import KdVFourierOperator


@dataclass(frozen=True, slots=True)
class KdVReferenceTrajectory:
    times: NDArray[np.floating]
    states: NDArray[np.floating]
    invariants: tuple[KdVInvariants, ...]
    method: str

    @property
    def final_state(self) -> NDArray[np.floating]:
        return np.asarray(self.states[-1], dtype=np.float64)


@dataclass(frozen=True, slots=True)
class KdVIFRK4Solver:
    grid: PeriodicGrid
    padding_factor: float = 1.5

    @property
    def operator(self) -> KdVFourierOperator:
        return KdVFourierOperator(self.grid, self.padding_factor)

    def step_spectrum(
        self, coefficients: NDArray[np.complexfloating], dt: float,
    ) -> NDArray[np.complexfloating]:
        if not np.isfinite(dt) or dt == 0.0:
            raise ValueError("KdV IFRK4 step requires a finite nonzero dt")
        initial = np.asarray(coefficients, dtype=np.complex128)
        operator, linear = self.operator, self.operator.linear_symbol
        if initial.shape != linear.shape:
            raise ValueError("KdV spectrum has an incompatible shape")

        def rhs(stage, transformed):
            phase = np.exp(linear * stage)
            return np.exp(-linear * stage) * operator.nonlinear_spectrum(phase * transformed)

        k1 = rhs(0.0, initial)
        k2 = rhs(0.5 * dt, initial + 0.5 * dt * k1)
        k3 = rhs(0.5 * dt, initial + 0.5 * dt * k2)
        k4 = rhs(dt, initial + dt * k3)
        transformed = initial + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
        result = np.exp(linear * dt) * transformed
        result[0], result[-1] = initial[0], 0.0
        if not np.all(np.isfinite(result)):
            raise FloatingPointError("KdV IFRK4 step produced a non-finite spectrum")
        return np.asarray(result, dtype=np.complex128)

    def step(self, values: NDArray[np.floating], dt: float) -> NDArray[np.floating]:
        coefficients = self.operator.spectrum(values)
        return self.operator.physical(self.step_spectrum(coefficients, dt))

    def step_embedded_spectrum(
        self, coefficients: NDArray[np.complexfloating], dt: float,
    ) -> tuple[NDArray[np.complexfloating], NDArray[np.complexfloating]]:
        """Dormand--Prince 5(4) pair in the KdV interaction picture."""
        if not np.isfinite(dt) or dt == 0.0:
            raise ValueError("embedded KdV step requires a finite nonzero dt")
        initial = np.asarray(coefficients, dtype=np.complex128)
        operator, linear = self.operator, self.operator.linear_symbol
        if initial.shape != linear.shape:
            raise ValueError("KdV spectrum has an incompatible shape")

        def rhs(stage, transformed):
            phase = np.exp(linear * stage * dt)
            return np.exp(-linear * stage * dt) * operator.nonlinear_spectrum(
                phase * transformed
            )

        k1 = rhs(0.0, initial)
        k2 = rhs(1 / 5, initial + dt * k1 / 5)
        k3 = rhs(3 / 10, initial + dt * (3 * k1 / 40 + 9 * k2 / 40))
        k4 = rhs(4 / 5, initial + dt * (44 * k1 / 45 - 56 * k2 / 15 + 32 * k3 / 9))
        k5 = rhs(8 / 9, initial + dt * (
            19372 * k1 / 6561 - 25360 * k2 / 2187 + 64448 * k3 / 6561 - 212 * k4 / 729
        ))
        k6 = rhs(1.0, initial + dt * (
            9017 * k1 / 3168 - 355 * k2 / 33 + 46732 * k3 / 5247
            + 49 * k4 / 176 - 5103 * k5 / 18656
        ))
        fifth_t = initial + dt * (
            35 * k1 / 384 + 500 * k3 / 1113 + 125 * k4 / 192
            - 2187 * k5 / 6784 + 11 * k6 / 84
        )
        k7 = rhs(1.0, fifth_t)
        fourth_t = initial + dt * (
            5179 * k1 / 57600 + 7571 * k3 / 16695 + 393 * k4 / 640
            - 92097 * k5 / 339200 + 187 * k6 / 2100 + k7 / 40
        )
        phase = np.exp(linear * dt)
        fifth, fourth = phase * fifth_t, phase * fourth_t
        for result in (fifth, fourth):
            result[0], result[-1] = initial[0], 0.0
            if not np.all(np.isfinite(result)):
                raise FloatingPointError("embedded KdV step produced a non-finite spectrum")
        return np.asarray(fifth), np.asarray(fourth)

    def step_embedded(
        self, values: NDArray[np.floating], dt: float,
    ) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        coefficients = self.operator.spectrum(values)
        high, low = self.step_embedded_spectrum(coefficients, dt)
        return self.operator.physical(high), self.operator.physical(low)


def solve_kdv_dop853(
    initial: NDArray[np.floating], grid: PeriodicGrid, *, evaluation_times,
    rtol: float = 1e-11, atol: float = 1e-13, maximum_step: float | None = None,
) -> KdVReferenceTrajectory:
    """Independent high-accuracy interaction-picture trajectory.

    ``evaluation_times`` must include zero.  It is supplied only after the
    complete adaptive action sequence has been committed.
    """
    times = np.asarray(evaluation_times, dtype=np.float64)
    if times.ndim != 1 or len(times) < 2 or times[0] != 0 or np.any(np.diff(times) <= 0):
        raise ValueError("KdV reference times must be strictly increasing from zero")
    operator = KdVFourierOperator(grid)
    initial_spectrum = operator.spectrum(initial)
    linear = operator.linear_symbol

    def rhs(time, transformed):
        phase = np.exp(linear * time)
        return np.exp(-linear * time) * operator.nonlinear_spectrum(phase * transformed)

    solution = solve_ivp(
        rhs, (0.0, float(times[-1])), initial_spectrum, method="DOP853",
        rtol=rtol, atol=atol, t_eval=times,
        max_step=np.inf if maximum_step is None else float(maximum_step),
    )
    if not solution.success:
        raise RuntimeError(f"KdV DOP853 reference failed: {solution.message}")
    states = []
    for index, time in enumerate(solution.t):
        spectrum = np.exp(linear * time) * solution.y[:, index]
        spectrum[0], spectrum[-1] = initial_spectrum[0], 0.0
        states.append(operator.physical(spectrum))
    state_array = np.asarray(states, dtype=np.float64)
    invariants = tuple(kdv_invariants(row, grid) for row in state_array)
    return KdVReferenceTrajectory(
        np.asarray(solution.t), state_array, invariants, "interaction-picture-dop853"
    )
