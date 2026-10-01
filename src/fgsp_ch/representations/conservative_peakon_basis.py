from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from fgsp_ch.discretization.grids import PeriodicGrid
from fgsp_ch.solvers.multipeakon import (
    periodic_peakon_kernel,
    periodic_peakon_kernel_derivative,
)

from .peakon_tangent_basis import build_peakon_tangent_basis


@dataclass(frozen=True, slots=True)
class ConservativePeakonBasis:
    full_basis: NDArray[np.floating]
    constraint_matrix: NDArray[np.floating]
    nullspace: NDArray[np.floating]
    constrained_basis: NDArray[np.floating]
    constraint_rank: int
    finite: bool

    def physical_coefficients(
        self, reduced_coefficients: NDArray[np.floating]
    ) -> NDArray[np.floating]:
        values = self.nullspace @ np.asarray(reduced_coefficients)
        return np.asarray(values)


def particle_invariant_constraint_matrix(
    positions: NDArray[np.floating],
    amplitudes: NDArray[np.floating],
    length: float,
) -> NDArray[np.floating]:
    """Linearized standard-CH particle momentum and Hamiltonian constraints.

    Coefficients are ordered ``z=(delta p, delta q)``.  The rows enforce
    ``d sum(p)=0`` and ``d[1/2 p^T K(q) p]=0``.
    """
    positions = np.asarray(positions, dtype=np.float64)
    amplitudes = np.asarray(amplitudes, dtype=np.float64)
    if positions.ndim != 1 or positions.shape != amplitudes.shape or not len(positions):
        raise ValueError("Particle positions and amplitudes must be equal nonempty arrays.")
    displacement = positions[:, None] - positions[None, :]
    kernel = periodic_peakon_kernel(displacement, length)
    derivative = periodic_peakon_kernel_derivative(displacement, length)
    gradient_amplitude = kernel @ amplitudes
    gradient_position = amplitudes * (derivative @ amplitudes)
    momentum = np.concatenate(
        [np.ones_like(amplitudes), np.zeros_like(amplitudes)]
    )
    hamiltonian = np.concatenate([gradient_amplitude, gradient_position])
    result = np.stack([momentum, hamiltonian])
    if not np.all(np.isfinite(result)):
        raise FloatingPointError("Particle invariant constraints contain NaN or Inf.")
    return result


def build_conservative_peakon_basis(
    grid: PeriodicGrid,
    positions: NDArray[np.floating],
    amplitudes: NDArray[np.floating],
    *,
    rank_tolerance: float = 1.0e-12,
) -> ConservativePeakonBasis:
    """Restrict the peakon tangent basis to the invariant-tangent nullspace."""
    if not 0.0 < rank_tolerance < 1.0:
        raise ValueError("rank_tolerance must lie in (0, 1).")
    full = build_peakon_tangent_basis(grid, positions, amplitudes)
    constraints = particle_invariant_constraint_matrix(
        positions, amplitudes, grid.length
    )
    _, singular_values, right = np.linalg.svd(constraints, full_matrices=True)
    scale = singular_values[0] if singular_values.size else 0.0
    rank = int(np.sum(singular_values > rank_tolerance * max(scale, 1.0)))
    nullspace = right[rank:].T
    constrained = full @ nullspace
    values = np.concatenate(
        [constraints.ravel(), nullspace.ravel(), constrained.ravel()]
    )
    return ConservativePeakonBasis(
        full_basis=np.asarray(full),
        constraint_matrix=np.asarray(constraints),
        nullspace=np.asarray(nullspace),
        constrained_basis=np.asarray(constrained),
        constraint_rank=rank,
        finite=bool(np.isfinite(values).all()),
    )
