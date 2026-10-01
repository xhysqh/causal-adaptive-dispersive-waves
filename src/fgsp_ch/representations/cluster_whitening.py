from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from fgsp_ch.atlas.mch_atlas import AtlasDecision
from fgsp_ch.discretization.helmholtz import HelmholtzOperator
from fgsp_ch.geometry.metric import a_norm
from fgsp_ch.solvers.multipeakon import periodic_peakon_kernel_derivative


@dataclass(frozen=True, slots=True)
class ClusterWhitening:
    basis: NDArray[np.floating]
    gram: NDArray[np.floating]
    square_root: NDArray[np.floating]
    inverse_square_root: NDArray[np.floating]
    eigenvalues: NDArray[np.floating]
    condition: float
    rank: int
    identifiable: bool


def cluster_position_basis(
    decision: AtlasDecision, helmholtz: HelmholtzOperator
) -> NDArray[np.floating]:
    clusters = decision.clusters
    if clusters.centers.size == 0:
        return np.empty((helmholtz.grid.points, 0), dtype=np.float64)
    derivative = periodic_peakon_kernel_derivative(
        helmholtz.grid.x[:, None] - clusters.centers[None, :],
        helmholtz.grid.length,
    )
    return np.asarray(-derivative * clusters.zeroth[None, :])


def build_cluster_whitening(
    decision: AtlasDecision,
    helmholtz: HelmholtzOperator,
    *,
    relative_eigenvalue_floor: float = 1.0e-10,
    condition_maximum: float = 1.0e8,
) -> ClusterWhitening:
    basis = cluster_position_basis(decision, helmholtz)
    count = basis.shape[1]
    if count == 0:
        empty = np.empty((0, 0), dtype=np.float64)
        return ClusterWhitening(
            basis, empty, empty, empty, np.empty(0), np.inf, 0, False
        )
    applied = np.stack([helmholtz.apply(basis[:, index]) for index in range(count)], axis=1)
    gram = helmholtz.grid.h * basis.T @ applied
    gram = 0.5 * (gram + gram.T)
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    maximum = max(float(np.max(eigenvalues)), np.finfo(float).eps)
    cutoff = relative_eigenvalue_floor * maximum
    active = eigenvalues > cutoff
    rank = int(np.count_nonzero(active))
    safe = np.maximum(eigenvalues, cutoff)
    square_root = (eigenvectors * np.sqrt(safe)) @ eigenvectors.T
    inverse_square_root = (eigenvectors / np.sqrt(safe)) @ eigenvectors.T
    minimum = float(np.min(eigenvalues))
    condition = np.inf if minimum <= 0.0 else maximum / minimum
    identifiable = bool(
        rank == count and np.isfinite(condition) and condition <= condition_maximum
    )
    return ClusterWhitening(
        basis, gram, square_root, inverse_square_root, eigenvalues,
        float(condition), rank, identifiable,
    )


def whiten_cluster_position_defect_rate(
    coefficients: NDArray[np.floating],
    whitening: ClusterWhitening,
    *,
    state_scale: float,
    dt: float,
) -> NDArray[np.floating]:
    values = np.asarray(coefficients, dtype=np.float64)
    if values.shape != (whitening.basis.shape[1],):
        raise ValueError("Cluster coefficient count does not match the whitening basis.")
    if state_scale <= 0.0 or dt <= 0.0:
        raise ValueError("Cluster whitening scales must be positive.")
    return np.asarray(whitening.square_root @ values / (state_scale * dt))


def reconstruct_cluster_position_defect(
    whitened_rate: NDArray[np.floating],
    whitening: ClusterWhitening,
    *,
    state_scale: float,
    dt: float,
) -> NDArray[np.floating]:
    values = np.asarray(whitened_rate, dtype=np.float64)
    if values.shape != (whitening.basis.shape[1],):
        raise ValueError("Whitened cluster coordinate count is inconsistent.")
    coefficients = state_scale * dt * (whitening.inverse_square_root @ values)
    return np.asarray(whitening.basis @ coefficients)


def cluster_position_signal_norm(
    coefficients: NDArray[np.floating], whitening: ClusterWhitening,
    helmholtz: HelmholtzOperator,
) -> float:
    return a_norm(whitening.basis @ np.asarray(coefficients), helmholtz)
