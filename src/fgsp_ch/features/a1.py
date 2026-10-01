from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from fgsp_ch.discretization.difference import (
    first_derivative,
    second_derivative,
)
from fgsp_ch.equations.generalized_ch import GeneralizedCH


A1_CHANNEL_NAMES = (
    "u_previous",
    "u_current",
    "momentum_current",
    "d1_current",
    "d2_current",
    "baseline_next",
    "baseline_momentum_next",
    "baseline_residual",
)


def baseline_residual(
    equation: GeneralizedCH,
    current: NDArray[np.floating],
    baseline_next: NDArray[np.floating],
    dt: float,
) -> NDArray[np.floating]:
    """Causal residual ``A(U_B-U_n)/dt - F_m((U_B+U_n)/2)``."""
    midpoint = 0.5 * (current + baseline_next)
    return (
        equation.helmholtz.apply(baseline_next - current) / dt
        - equation.momentum_rhs(midpoint)
    )


def extract_a1_features(
    equation: GeneralizedCH,
    previous: NDArray[np.floating],
    current: NDArray[np.floating],
    baseline_next: NDArray[np.floating],
    dt: float,
) -> NDArray[np.floating]:
    """Construct the eight causal A1 field channels."""
    grid = equation.grid
    channels = (
        previous,
        current,
        equation.helmholtz.apply(current),
        first_derivative(current, grid),
        second_derivative(current, grid),
        baseline_next,
        equation.helmholtz.apply(baseline_next),
        baseline_residual(equation, current, baseline_next, dt),
    )
    result = np.stack(channels)
    if not np.all(np.isfinite(result)):
        raise FloatingPointError("A1 features contain NaN or Inf.")
    return result


def parameter_vector(
    equation: GeneralizedCH, dt: float
) -> NDArray[np.floating]:
    p = equation.parameters
    grid = equation.grid
    return np.asarray(
        [p.k1, p.k2, p.k, grid.h, dt, grid.length, grid.points],
        dtype=np.float64,
    )

