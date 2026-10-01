from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from fgsp_ch.discretization.difference import first_derivative
from fgsp_ch.equations.modified_ch import ModifiedCH


def smooth_instantaneous_weak_residual(
    equation: ModifiedCH,
    state: NDArray[np.floating],
    test_function: NDArray[np.floating],
) -> float:
    """Evaluate the semidiscrete periodic weak residual at one instant."""
    equation.grid.validate_state(state)
    equation.grid.validate_state(test_function)
    mt = equation.momentum_rhs(state)
    test_x = first_derivative(test_function, equation.grid)
    ux = first_derivative(state, equation.grid)
    integrand = (
        mt * test_function
        - equation.flux(state) * test_x
        + equation.parameters.gamma * ux * test_function
    )
    return float(equation.grid.h * np.sum(integrand))


def particle_distribution_weak_residual(
    qdot: NDArray[np.floating],
    weight_rates: NDArray[np.floating],
    velocity: NDArray[np.floating],
    weights: NDArray[np.floating],
    test_values: NDArray[np.floating],
    test_derivatives: NDArray[np.floating],
) -> float:
    """Coefficient residual of ``m=sum_i weights_i delta(x-q_i)``.

    ``weights`` are atomic momentum weights, not peak heights. For the unit-peak
    periodic Green kernel they equal ``2*tanh(L/2)*amplitudes``.
    """
    arrays = [
        np.asarray(value, dtype=np.float64)
        for value in (
            qdot,
            weight_rates,
            velocity,
            weights,
            test_values,
            test_derivatives,
        )
    ]
    shape = arrays[0].shape
    if len(shape) != 1 or any(value.shape != shape for value in arrays):
        raise ValueError("All particle weak-residual inputs must be equal 1-D arrays.")
    if not all(np.all(np.isfinite(value)) for value in arrays):
        raise FloatingPointError("Particle weak-residual inputs contain NaN or Inf.")
    qd, wd, vel, weight, phi, phi_x = arrays
    return float(np.sum(wd * phi + weight * (qd - vel) * phi_x))
