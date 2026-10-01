from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from fgsp_ch.discretization.grids import PeriodicGrid
from fgsp_ch.discretization.helmholtz import HelmholtzOperator
from fgsp_ch.geometry.metric import a_norm
from fgsp_ch.solvers.multipeakon import (
    periodic_peakon_kernel,
    periodic_peakon_kernel_derivative,
)


@dataclass(frozen=True, slots=True)
class SubspaceDecomposition:
    coefficients: NDArray[np.floating]
    component: NDArray[np.floating]
    remainder: NDArray[np.floating]
    gram_matrix: NDArray[np.floating]
    gram_condition_number: float
    effective_rank: int
    explained_fraction: float
    relative_orthogonality_error: float
    finite: bool


@dataclass(frozen=True, slots=True)
class PeakonDefectDecomposition:
    """A-orthogonal decomposition into peak-amplitude/position and field parts."""

    coefficients: NDArray[np.floating]
    amplitude_coefficients: NDArray[np.floating]
    position_coefficients: NDArray[np.floating]
    peakon_component: NDArray[np.floating]
    field_remainder: NDArray[np.floating]
    gram_matrix: NDArray[np.floating]
    gram_condition_number: float
    effective_rank: int
    explained_fraction: float
    relative_orthogonality_error: float
    finite: bool


def build_peakon_tangent_basis(
    grid: PeriodicGrid,
    positions: NDArray[np.floating],
    amplitudes: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Build ``[K_i, -p_i K'_i]`` from causal particle coordinates.

    The first ``P`` columns encode amplitude changes and the last ``P``
    columns encode position changes in the linearization
    ``delta u = sum_i delta p_i K_i - p_i delta q_i K'_i``.
    """
    positions = np.asarray(positions, dtype=np.float64)
    amplitudes = np.asarray(amplitudes, dtype=np.float64)
    if positions.ndim != 1 or positions.shape != amplitudes.shape or not len(positions):
        raise ValueError("Peakon positions and amplitudes must be equal nonempty arrays.")
    if not np.all(np.isfinite(positions)) or not np.all(np.isfinite(amplitudes)):
        raise FloatingPointError("Peakon coordinates contain NaN or Inf.")
    displacement = grid.x[:, None] - positions[None, :]
    amplitude_basis = periodic_peakon_kernel(displacement, grid.length)
    position_basis = -periodic_peakon_kernel_derivative(
        displacement, grid.length
    ) * amplitudes[None, :]
    basis = np.concatenate([amplitude_basis, position_basis], axis=1)
    if not np.all(np.isfinite(basis)):
        raise FloatingPointError("Peakon tangent basis contains NaN or Inf.")
    return np.asarray(basis, dtype=grid.dtype)


def decompose_peakon_defect(
    defect: NDArray[np.floating],
    basis: NDArray[np.floating],
    helmholtz: HelmholtzOperator,
    *,
    rank_tolerance: float = 1.0e-12,
    epsilon: float = 1.0e-15,
) -> PeakonDefectDecomposition:
    """Compute the explicit A-least-squares peakon tangent decomposition.

    Rank deficiency is not hidden: the SVD least-squares solution is returned
    together with ``effective_rank`` and the Gram condition number.  Acceptance
    logic must reject a deficient or ill-conditioned basis.
    """
    if basis.shape[1] < 2 or basis.shape[1] % 2:
        raise ValueError("Peakon basis must contain equal amplitude/position blocks.")
    result = decompose_a_subspace(
        defect,
        basis,
        helmholtz,
        rank_tolerance=rank_tolerance,
        epsilon=epsilon,
    )
    particles = basis.shape[1] // 2
    return PeakonDefectDecomposition(
        coefficients=result.coefficients,
        amplitude_coefficients=np.asarray(result.coefficients[:particles]),
        position_coefficients=np.asarray(result.coefficients[particles:]),
        peakon_component=result.component,
        field_remainder=result.remainder,
        gram_matrix=result.gram_matrix,
        gram_condition_number=result.gram_condition_number,
        effective_rank=result.effective_rank,
        explained_fraction=result.explained_fraction,
        relative_orthogonality_error=result.relative_orthogonality_error,
        finite=result.finite,
    )


def decompose_a_subspace(
    defect: NDArray[np.floating],
    basis: NDArray[np.floating],
    helmholtz: HelmholtzOperator,
    *,
    rank_tolerance: float = 1.0e-12,
    epsilon: float = 1.0e-15,
) -> SubspaceDecomposition:
    """Project ``defect`` onto an arbitrary explicit basis in the A metric."""
    helmholtz.grid.validate_state(defect)
    if basis.ndim != 2 or basis.shape[0] != helmholtz.grid.points:
        raise ValueError("Basis must have shape (grid.points, components).")
    if basis.shape[1] < 1:
        raise ValueError("Subspace basis must contain at least one component.")
    if not 0.0 < rank_tolerance < 1.0:
        raise ValueError("rank_tolerance must lie in (0, 1).")
    applied_basis = np.stack(
        [helmholtz.apply(basis[:, column]) for column in range(basis.shape[1])],
        axis=1,
    )
    gram = helmholtz.grid.h * basis.T @ applied_basis
    rhs = helmholtz.grid.h * basis.T @ helmholtz.apply(defect)
    coefficients, _, rank, singular_values = np.linalg.lstsq(
        gram, rhs, rcond=rank_tolerance
    )
    condition = (
        float(np.inf)
        if singular_values.size == 0 or singular_values[-1] <= epsilon
        else float(singular_values[0] / singular_values[-1])
    )
    peakon = basis @ coefficients
    remainder = defect - peakon
    defect_norm = a_norm(defect, helmholtz)
    remainder_norm = a_norm(remainder, helmholtz)
    explained = 0.0 if defect_norm <= epsilon else 1.0 - (remainder_norm / defect_norm) ** 2
    column_norms = np.asarray(
        [a_norm(basis[:, column], helmholtz) for column in range(basis.shape[1])]
    )
    residual_moments = helmholtz.grid.h * basis.T @ helmholtz.apply(remainder)
    orthogonality = float(
        np.max(
            np.abs(residual_moments)
            / np.maximum(column_norms * max(defect_norm, epsilon), epsilon)
        )
    )
    values = np.concatenate(
        [
            coefficients,
            peakon,
            remainder,
            gram.ravel(),
            np.asarray([condition, explained, orthogonality]),
        ]
    )
    return SubspaceDecomposition(
        coefficients=np.asarray(coefficients),
        component=np.asarray(peakon),
        remainder=np.asarray(remainder),
        gram_matrix=np.asarray(gram),
        gram_condition_number=condition,
        effective_rank=int(rank),
        explained_fraction=float(explained),
        relative_orthogonality_error=orthogonality,
        finite=bool(np.isfinite(values).all()),
    )
