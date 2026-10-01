from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp

from fgsp_ch.discretization.grids import PeriodicGrid


def periodic_peakon_kernel(
    displacement: NDArray[np.floating] | float, length: float
) -> NDArray[np.floating]:
    """Normalized periodic Green profile used by the project.

    ``K_L(z) = cosh(L/2 - |wrap_L(z)|) / cosh(L/2)`` and ``K_L(0)=1``.
    """
    if length <= 0.0 or not np.isfinite(length):
        raise ValueError("Periodic length must be positive and finite.")
    values = np.asarray(displacement, dtype=np.float64)
    wrapped = (values + 0.5 * length) % length - 0.5 * length
    return np.cosh(0.5 * length - np.abs(wrapped)) / np.cosh(0.5 * length)


def periodic_peakon_kernel_derivative(
    displacement: NDArray[np.floating] | float, length: float
) -> NDArray[np.floating]:
    """Symmetric derivative of ``periodic_peakon_kernel`` with ``K'(0)=0``."""
    values = np.asarray(displacement, dtype=np.float64)
    wrapped = (values + 0.5 * length) % length - 0.5 * length
    derivative = (
        -np.sign(wrapped)
        * np.sinh(0.5 * length - np.abs(wrapped))
        / np.cosh(0.5 * length)
    )
    return np.where(wrapped == 0.0, 0.0, derivative)


def particle_hamiltonian(
    positions: NDArray[np.floating],
    amplitudes: NDArray[np.floating],
    length: float,
) -> float:
    """Return ``1/2 sum_ij p_i p_j K_L(q_i-q_j)``."""
    positions = np.asarray(positions, dtype=np.float64)
    amplitudes = np.asarray(amplitudes, dtype=np.float64)
    if positions.ndim != 1 or positions.shape != amplitudes.shape:
        raise ValueError("Particle positions and amplitudes must be equal 1-D arrays.")
    kernel = periodic_peakon_kernel(
        positions[:, None] - positions[None, :], length
    )
    return float(0.5 * amplitudes @ kernel @ amplitudes)


def reconstruct_multipeakon(
    grid: PeriodicGrid,
    positions: NDArray[np.floating],
    amplitudes: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Reconstruct a periodic multipeakon on ``grid`` from particle data."""
    positions = np.asarray(positions, dtype=np.float64)
    amplitudes = np.asarray(amplitudes, dtype=np.float64)
    if positions.ndim != 1 or positions.shape != amplitudes.shape:
        raise ValueError("Particle positions and amplitudes must be equal 1-D arrays.")
    values = periodic_peakon_kernel(
        grid.x[:, None] - positions[None, :], grid.length
    ) @ amplitudes
    result = np.asarray(values, dtype=grid.dtype)
    grid.validate_state(result)
    return result


@dataclass(frozen=True, slots=True)
class MultipeakonTrajectory:
    times: NDArray[np.floating]
    positions: NDArray[np.floating]
    amplitudes: NDArray[np.floating]
    momenta: NDArray[np.floating]
    hamiltonians: NDArray[np.floating]
    minimum_separation: float

    def reconstruct(self, grid: PeriodicGrid) -> NDArray[np.floating]:
        return np.stack(
            [
                reconstruct_multipeakon(grid, positions, amplitudes)
                for positions, amplitudes in zip(
                    self.positions, self.amplitudes, strict=True
                )
            ]
        )


@dataclass(slots=True)
class PeriodicMultipeakonSolver:
    """Independent particle reference for positive standard-CH peakons.

    With the normalized periodic kernel, the weak-solution particle system is
    ``qdot_i = sum_j p_j K(q_i-q_j)`` and
    ``pdot_i = -p_i sum_j p_j K'(q_i-q_j)``.
    """

    length: float
    method: str = "DOP853"
    rtol: float = 1.0e-11
    atol: float = 1.0e-13
    collision_tolerance: float = 1.0e-10

    def __post_init__(self) -> None:
        if self.length <= 0.0 or not np.isfinite(self.length):
            raise ValueError("Periodic length must be positive and finite.")
        if self.rtol <= 0.0 or self.atol <= 0.0:
            raise ValueError("Particle tolerances must be positive.")
        if self.collision_tolerance <= 0.0:
            raise ValueError("Collision tolerance must be positive.")

    def rhs(self, state: NDArray[np.floating]) -> NDArray[np.floating]:
        if state.ndim != 1 or state.size < 2 or state.size % 2:
            raise ValueError("Particle state must contain equally many q and p values.")
        particles = state.size // 2
        positions = state[:particles]
        amplitudes = state[particles:]
        difference = positions[:, None] - positions[None, :]
        kernel = periodic_peakon_kernel(difference, self.length)
        derivative = periodic_peakon_kernel_derivative(difference, self.length)
        qdot = kernel @ amplitudes
        pdot = -amplitudes * (derivative @ amplitudes)
        result = np.concatenate([qdot, pdot])
        if not np.all(np.isfinite(result)):
            raise FloatingPointError("Multipeakon RHS contains NaN or Inf.")
        return result

    def solve(
        self,
        positions: NDArray[np.floating],
        amplitudes: NDArray[np.floating],
        *,
        final_time: float,
        output_dt: float,
    ) -> MultipeakonTrajectory:
        positions = np.asarray(positions, dtype=np.float64)
        amplitudes = np.asarray(amplitudes, dtype=np.float64)
        if positions.ndim != 1 or positions.shape != amplitudes.shape or not len(positions):
            raise ValueError("Particle positions and amplitudes must be equal nonempty arrays.")
        if not np.all(np.isfinite(positions)) or not np.all(np.isfinite(amplitudes)):
            raise FloatingPointError("Initial particle data contain NaN or Inf.")
        if np.any(amplitudes <= 0.0):
            raise ValueError("B0.3 reference permits positive peakons only.")
        steps = round(final_time / output_dt)
        if steps < 1 or not np.isclose(steps * output_dt, final_time, atol=1.0e-12):
            raise ValueError("final_time must be a positive multiple of output_dt.")
        times = np.linspace(0.0, final_time, steps + 1)
        initial = np.concatenate([positions, amplitudes])
        solution = solve_ivp(
            lambda _time, values: self.rhs(values),
            (0.0, final_time),
            initial,
            method=self.method,
            t_eval=times,
            rtol=self.rtol,
            atol=self.atol,
        )
        if not solution.success:
            raise RuntimeError(f"Multipeakon integration failed: {solution.message}")
        states = np.asarray(solution.y.T, dtype=np.float64)
        particle_positions = states[:, : len(positions)]
        particle_amplitudes = states[:, len(positions) :]
        momenta = np.sum(particle_amplitudes, axis=1)
        hamiltonians = np.asarray(
            [
                particle_hamiltonian(q, p, self.length)
                for q, p in zip(
                    particle_positions, particle_amplitudes, strict=True
                )
            ]
        )
        minimum = self.length
        if len(positions) > 1:
            for row in particle_positions:
                pairwise = np.abs(
                    (row[:, None] - row[None, :] + 0.5 * self.length)
                    % self.length
                    - 0.5 * self.length
                )
                pairwise[np.eye(len(row), dtype=bool)] = np.inf
                minimum = min(minimum, float(np.min(pairwise)))
        if minimum <= self.collision_tolerance:
            raise RuntimeError(
                "Particle separation reached the configured collision tolerance."
            )
        if not (
            np.all(np.isfinite(states))
            and np.all(np.isfinite(momenta))
            and np.all(np.isfinite(hamiltonians))
        ):
            raise FloatingPointError("Multipeakon trajectory contains NaN or Inf.")
        return MultipeakonTrajectory(
            times=times,
            positions=particle_positions,
            amplitudes=particle_amplitudes,
            momenta=momenta,
            hamiltonians=hamiltonians,
            minimum_separation=float(minimum),
        )

