from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from fgsp_ch.discretization.grids import PeriodicGrid
from fgsp_ch.discretization.helmholtz import HelmholtzOperator


def discrete_inner(
    left: NDArray[np.floating],
    right: NDArray[np.floating],
    grid: PeriodicGrid,
) -> float:
    """Discrete periodic inner product ``h left^T right``."""
    grid.validate_state(left)
    grid.validate_state(right)
    return float(grid.h * np.dot(left, right))


def a_inner(
    left: NDArray[np.floating],
    right: NDArray[np.floating],
    helmholtz: HelmholtzOperator,
) -> float:
    """Helmholtz metric ``<left, right>_A = <left, A_h right>_h``."""
    return discrete_inner(left, helmholtz.apply(right), helmholtz.grid)


def a_norm(
    values: NDArray[np.floating], helmholtz: HelmholtzOperator
) -> float:
    """Discrete ``H^1`` norm induced by ``A_h``."""
    squared = a_inner(values, values, helmholtz)
    if squared < -100.0 * np.finfo(values.dtype).eps:
        raise FloatingPointError(f"Negative squared A-norm: {squared}.")
    return float(np.sqrt(max(squared, 0.0)))


def discrete_energy(
    values: NDArray[np.floating], helmholtz: HelmholtzOperator
) -> float:
    """Energy ``E_h(U) = 0.5 <U, A_h U>_h``."""
    return 0.5 * a_inner(values, values, helmholtz)

