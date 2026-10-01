from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from fgsp_ch.discretization.grids import PeriodicGrid
from fgsp_ch.discretization.spectral import FourierOperators
from fgsp_ch.solvers.multipeakon import reconstruct_multipeakon


def _validate_atomic_data(
    grid: PeriodicGrid,
    positions: NDArray[np.floating],
    weights: NDArray[np.floating],
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    if grid.points % 2 == 0:
        raise ValueError("Odd grid size is required to avoid a Nyquist atom ambiguity.")
    q = np.asarray(positions, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    if q.ndim != 1 or q.shape != w.shape or q.size == 0:
        raise ValueError("Positions and weights must be equal nonempty 1-D arrays.")
    if not np.all(np.isfinite(q)) or not np.all(np.isfinite(w)):
        raise FloatingPointError("Atomic measure data contain NaN or Inf.")
    return q, w


def spectral_atomic_measure(
    grid: PeriodicGrid,
    positions: NDArray[np.floating],
    weights: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Embed an atomic periodic measure in the grid's Fourier space."""
    q, w = _validate_atomic_data(grid, positions, weights)
    kappa = 2.0 * np.pi * np.fft.fftfreq(grid.points, d=grid.h)
    phase = q - grid.x_min
    coefficients = np.exp(-1j * kappa[:, None] * phase[None, :]) @ w / grid.h
    density = np.fft.ifft(coefficients).real
    return np.asarray(density, dtype=grid.dtype)


def spectral_measure_weak_residual(
    grid: PeriodicGrid,
    positions: NDArray[np.floating],
    weights: NDArray[np.floating],
    velocities: NDArray[np.floating],
    test_function: NDArray[np.floating],
) -> float:
    """Eulerian Fourier weak residual of the transported atomic measure."""
    q, w = _validate_atomic_data(grid, positions, weights)
    velocity = np.asarray(velocities, dtype=np.float64)
    if velocity.shape != q.shape or not np.all(np.isfinite(velocity)):
        raise ValueError("Velocities must be finite and match the particle positions.")
    grid.validate_state(test_function)
    flux = spectral_atomic_measure(grid, q, w * velocity)
    derivative = FourierOperators(grid).first_derivative
    momentum_t = -derivative(flux)
    return float(
        grid.h
        * np.sum(momentum_t * test_function - flux * derivative(test_function))
    )


@dataclass(frozen=True, slots=True)
class CrossRepresentationAudit:
    mass_error: float
    test_action_error: float
    weak_residual: float
    field_l2_error: float


def audit_particle_eulerian_embedding(
    grid: PeriodicGrid,
    positions: NDArray[np.floating],
    amplitudes: NDArray[np.floating],
    velocities: NDArray[np.floating],
    test_function: NDArray[np.floating],
    exact_test_at_particles: NDArray[np.floating],
) -> CrossRepresentationAudit:
    """Compare particle actions with their contract-aware Eulerian embedding."""
    q = np.asarray(positions, dtype=np.float64)
    a = np.asarray(amplitudes, dtype=np.float64)
    phi_q = np.asarray(exact_test_at_particles, dtype=np.float64)
    if phi_q.shape != q.shape:
        raise ValueError("Particle test values must match positions.")
    weights = 2.0 * np.tanh(0.5 * grid.length) * a
    momentum = spectral_atomic_measure(grid, q, weights)
    operators = FourierOperators(grid)
    embedded_state = operators.helmholtz_solve(momentum)
    exact_state = reconstruct_multipeakon(grid, q, a)
    return CrossRepresentationAudit(
        mass_error=abs(float(grid.h * np.sum(momentum) - np.sum(weights))),
        test_action_error=abs(
            float(grid.h * np.dot(momentum, test_function) - np.dot(weights, phi_q))
        ),
        weak_residual=abs(
            spectral_measure_weak_residual(
                grid, q, weights, velocities, test_function
            )
        ),
        field_l2_error=float(
            np.sqrt(grid.h * np.sum((embedded_state - exact_state) ** 2))
        ),
    )
