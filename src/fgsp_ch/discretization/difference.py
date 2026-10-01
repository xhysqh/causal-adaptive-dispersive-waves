from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import sparse

from .grids import PeriodicGrid


def first_derivative(
    values: NDArray[np.floating], grid: PeriodicGrid
) -> NDArray[np.floating]:
    """Apply periodic second-order centered ``D_1``."""
    grid.validate_state(values)
    return (np.roll(values, -1) - np.roll(values, 1)) / (2.0 * grid.h)


def second_derivative(
    values: NDArray[np.floating], grid: PeriodicGrid
) -> NDArray[np.floating]:
    """Apply periodic second-order centered ``D_2``."""
    grid.validate_state(values)
    return (
        np.roll(values, -1) - 2.0 * values + np.roll(values, 1)
    ) / grid.h**2


def first_derivative_matrix(grid: PeriodicGrid) -> sparse.csr_matrix:
    """Return the circulant matrix for periodic ``D_1``."""
    n = grid.points
    matrix = sparse.diags(
        (-np.ones(n), np.ones(n)), offsets=(-1, 1), shape=(n, n), format="lil"
    )
    matrix[0, n - 1] = -1.0
    matrix[n - 1, 0] = 1.0
    return (matrix.tocsr() / (2.0 * grid.h)).astype(grid.dtype)


def second_derivative_matrix(grid: PeriodicGrid) -> sparse.csr_matrix:
    """Return the symmetric negative-semidefinite circulant ``D_2``."""
    n = grid.points
    matrix = sparse.diags(
        (np.ones(n), -2.0 * np.ones(n), np.ones(n)),
        offsets=(-1, 0, 1),
        shape=(n, n),
        format="lil",
    )
    matrix[0, n - 1] = 1.0
    matrix[n - 1, 0] = 1.0
    return (matrix.tocsr() / grid.h**2).astype(grid.dtype)

