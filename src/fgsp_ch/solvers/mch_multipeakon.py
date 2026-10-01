from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp

from fgsp_ch.solvers.multipeakon import (
    periodic_peakon_kernel,
    periodic_peakon_kernel_derivative,
)


def conservative_periodic_mch_velocity(
    positions: NDArray[np.floating],
    amplitudes: NDArray[np.floating],
    length: float,
    *,
    alpha: float = 1.0,
) -> NDArray[np.floating]:
    """Velocity from ``u(q)^2-<u_x^2>(q)`` under the M0 weak contract."""
    q = np.asarray(positions, dtype=np.float64)
    a = np.asarray(amplitudes, dtype=np.float64)
    if q.ndim != 1 or q.shape != a.shape or q.size == 0:
        raise ValueError("Positions and amplitudes must be equal nonempty 1-D arrays.")
    if not np.all(np.isfinite(q)) or not np.all(np.isfinite(a)):
        raise FloatingPointError("Particle data contain NaN or Inf.")
    if length <= 0.0 or not np.isfinite(length) or not np.isfinite(alpha):
        raise ValueError("length must be positive and alpha must be finite.")
    delta = q[:, None] - q[None, :]
    u_at_q = periodic_peakon_kernel(delta, length) @ a
    regular_ux = periodic_peakon_kernel_derivative(delta, length) @ a
    self_slope = a * np.tanh(0.5 * length)
    velocity = alpha * (u_at_q**2 - regular_ux**2 - self_slope**2)
    if not np.all(np.isfinite(velocity)):
        raise FloatingPointError("mCH particle velocity contains NaN or Inf.")
    return velocity


def mch_particle_h1(
    positions: NDArray[np.floating],
    amplitudes: NDArray[np.floating],
    length: float,
) -> float:
    """Exact periodic ``H1^2`` for the project's unit-peak Green kernel."""
    q = np.asarray(positions, dtype=np.float64)
    a = np.asarray(amplitudes, dtype=np.float64)
    if q.ndim != 1 or q.shape != a.shape or q.size == 0:
        raise ValueError("Positions and amplitudes must be equal nonempty 1-D arrays.")
    if not np.all(np.isfinite(q)) or not np.all(np.isfinite(a)):
        raise FloatingPointError("Particle data contain NaN or Inf.")
    if length <= 0.0 or not np.isfinite(length):
        raise ValueError("Periodic length must be positive and finite.")
    kernel = periodic_peakon_kernel(q[:, None] - q[None, :], length)
    return float(2.0 * np.tanh(0.5 * length) * (a @ kernel @ a))


@dataclass(frozen=True, slots=True)
class MCHMultipeakonTrajectory:
    times: NDArray[np.floating]
    positions: NDArray[np.floating]
    amplitudes: NDArray[np.floating]
    h1_squared: NDArray[np.floating]
    minimum_separation: float


@dataclass(slots=True)
class ConservativePeriodicMCHMultipeakonSolver:
    """Independent conservative periodic cubic-mCH particle reference."""

    length: float
    alpha: float = 1.0
    method: str = "DOP853"
    rtol: float = 1.0e-11
    atol: float = 1.0e-13
    collision_tolerance: float = 1.0e-8

    def __post_init__(self) -> None:
        if self.length <= 0.0 or not np.isfinite(self.length):
            raise ValueError("Periodic length must be positive and finite.")
        if not np.isfinite(self.alpha) or self.alpha == 0.0:
            raise ValueError("alpha must be finite and nonzero.")
        if self.rtol <= 0.0 or self.atol <= 0.0 or self.collision_tolerance <= 0.0:
            raise ValueError("Solver tolerances must be positive.")

    def rhs(
        self, _time: float, positions: NDArray[np.floating], amplitudes: NDArray[np.floating]
    ) -> NDArray[np.floating]:
        return conservative_periodic_mch_velocity(
            positions, amplitudes, self.length, alpha=self.alpha
        )

    def solve(
        self,
        positions: NDArray[np.floating],
        amplitudes: NDArray[np.floating],
        *,
        final_time: float,
        output_dt: float,
    ) -> MCHMultipeakonTrajectory:
        q0 = np.asarray(positions, dtype=np.float64)
        a = np.asarray(amplitudes, dtype=np.float64)
        if q0.ndim != 1 or q0.shape != a.shape or q0.size == 0:
            raise ValueError("Positions and amplitudes must be equal nonempty arrays.")
        if np.any(a <= 0.0):
            raise ValueError("M1 reference is restricted to positive peakons.")
        if not np.all(np.isfinite(q0)) or not np.all(np.isfinite(a)):
            raise FloatingPointError("Initial particle data contain NaN or Inf.")
        if final_time <= 0.0 or output_dt <= 0.0:
            raise ValueError("final_time and output_dt must be positive.")
        steps = round(final_time / output_dt)
        if steps < 1 or not np.isclose(steps * output_dt, final_time, atol=1e-12):
            raise ValueError("final_time must be a positive multiple of output_dt.")
        times = np.linspace(0.0, final_time, steps + 1)
        def separation_event(_time: float, q: NDArray[np.floating]) -> float:
            if q.size == 1:
                return self.length
            distance = np.abs(
                (q[:, None] - q[None, :] + self.length / 2) % self.length
                - self.length / 2
            )
            distance[np.eye(q.size, dtype=bool)] = np.inf
            return float(np.min(distance) - self.collision_tolerance)

        separation_event.terminal = True  # type: ignore[attr-defined]
        separation_event.direction = -1.0  # type: ignore[attr-defined]
        solution = solve_ivp(
            lambda t, q: self.rhs(t, q, a),
            (0.0, final_time),
            q0,
            method=self.method,
            t_eval=times,
            rtol=self.rtol,
            atol=self.atol,
            events=separation_event,
        )
        if not solution.success:
            raise RuntimeError(f"mCH particle integration failed: {solution.message}")
        if solution.t_events and solution.t_events[0].size:
            raise RuntimeError("mCH particle integration stopped before collision.")
        q_history = np.asarray(solution.y.T)
        a_history = np.broadcast_to(a, q_history.shape).copy()
        h1 = np.asarray([mch_particle_h1(q, a, self.length) for q in q_history])
        minimum = self.length
        if q0.size > 1:
            for q in q_history:
                distance = np.abs((q[:, None] - q[None, :] + self.length / 2) % self.length - self.length / 2)
                distance[np.eye(q.size, dtype=bool)] = np.inf
                minimum = min(minimum, float(np.min(distance)))
        if minimum <= self.collision_tolerance:
            raise RuntimeError("mCH particle separation reached collision tolerance.")
        if not np.all(np.isfinite(q_history)) or not np.all(np.isfinite(h1)):
            raise FloatingPointError("mCH particle trajectory contains NaN or Inf.")
        return MCHMultipeakonTrajectory(times, q_history, a_history, h1, float(minimum))
