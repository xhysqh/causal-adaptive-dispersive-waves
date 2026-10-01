from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from fgsp_ch.atlas.mch_atlas import AtlasDecision, AtlasKind
from fgsp_ch.discretization.difference import first_derivative, second_derivative
from fgsp_ch.discretization.helmholtz import HelmholtzOperator
from fgsp_ch.geometry.metric import a_norm
from fgsp_ch.representations.peakon_tangent_basis import (
    build_peakon_tangent_basis,
    decompose_a_subspace,
)
from fgsp_ch.solvers.multipeakon import periodic_peakon_kernel_derivative


ATLAS_CODE = {
    AtlasKind.FIELD: 0,
    AtlasKind.SEPARATED: 1,
    AtlasKind.CLUSTER: 2,
}


@dataclass(frozen=True, slots=True)
class MCHAtlasLabel:
    atlas: AtlasKind
    position: NDArray[np.floating]
    amplitude: NDArray[np.floating]
    cluster_position: NDArray[np.floating]
    cluster_moment_change: NDArray[np.floating]
    peak_state: NDArray[np.floating]
    field_state: NDArray[np.floating]
    peak_momentum: NDArray[np.floating]
    field_momentum: NDArray[np.floating]
    effective_rank: int
    expected_rank: int
    gram_condition: float
    relative_reconstruction_error: float
    finite: bool


def _cluster_position_basis(
    current: AtlasDecision, helmholtz: HelmholtzOperator
) -> NDArray[np.floating]:
    clusters = current.clusters
    if clusters.centers.size == 0:
        return np.empty((helmholtz.grid.points, 0), dtype=np.float64)
    derivative = periodic_peakon_kernel_derivative(
        helmholtz.grid.x[:, None] - clusters.centers[None, :],
        helmholtz.grid.length,
    )
    return np.asarray(-derivative * clusters.zeroth[None, :])


def _cluster_moment_change(
    current: AtlasDecision, future: AtlasDecision | None, length: float
) -> NDArray[np.floating]:
    now = current.clusters
    if future is None or now.centers.size == 0:
        return np.zeros((now.centers.size, 4), dtype=np.float64)
    later = future.clusters
    result = np.zeros((now.centers.size, 4), dtype=np.float64)
    used: set[int] = set()
    for index, (center, sign) in enumerate(zip(now.centers, now.signs, strict=True)):
        candidates = np.flatnonzero(later.signs == sign)
        candidates = np.asarray([item for item in candidates if int(item) not in used])
        if candidates.size == 0:
            continue
        distance = np.abs(
            (later.centers[candidates] - center + length / 2) % length - length / 2
        )
        selected = int(candidates[int(np.argmin(distance))])
        used.add(selected)
        result[index, 0] = (
            later.centers[selected] - center + length / 2
        ) % length - length / 2
        result[index, 1] = later.zeroth[selected] - now.zeroth[index]
        result[index, 2] = later.first[selected] - now.first[index]
        result[index, 3] = later.second[selected] - now.second[index]
    return result


def build_mch_atlas_label(
    state_defect: NDArray[np.floating],
    current: AtlasDecision,
    helmholtz: HelmholtzOperator,
    *,
    future: AtlasDecision | None = None,
    rank_tolerance: float = 1.0e-12,
) -> MCHAtlasLabel:
    """Decompose a future defect in a basis determined only by current state."""
    helmholtz.grid.validate_state(state_defect)
    particles = current.decomposition.positions.size
    position = np.zeros(particles, dtype=np.float64)
    amplitude = np.zeros(particles, dtype=np.float64)
    cluster_position = np.zeros(current.clusters.centers.size, dtype=np.float64)
    expected_rank = 0
    effective_rank = 0
    condition = 1.0
    peak = np.zeros_like(state_defect)
    if current.kind == AtlasKind.SEPARATED and particles:
        full = build_peakon_tangent_basis(
            helmholtz.grid,
            current.decomposition.positions,
            current.decomposition.amplitudes,
        )
        basis = full[:, particles:]
        decomposition = decompose_a_subspace(
            state_defect, basis, helmholtz, rank_tolerance=rank_tolerance
        )
        position = np.asarray(decomposition.coefficients)
        peak = np.asarray(decomposition.component)
        expected_rank = particles
        effective_rank = decomposition.effective_rank
        condition = decomposition.gram_condition_number
    elif current.kind == AtlasKind.CLUSTER:
        basis = _cluster_position_basis(current, helmholtz)
        if basis.shape[1]:
            decomposition = decompose_a_subspace(
                state_defect, basis, helmholtz, rank_tolerance=rank_tolerance
            )
            cluster_position = np.asarray(decomposition.coefficients)
            peak = np.asarray(decomposition.component)
            expected_rank = basis.shape[1]
            effective_rank = decomposition.effective_rank
            condition = decomposition.gram_condition_number
    field = np.asarray(state_defect - peak)
    peak_momentum = helmholtz.apply(peak)
    field_momentum = helmholtz.apply(field)
    denominator = max(a_norm(state_defect, helmholtz), np.finfo(float).eps)
    reconstruction = helmholtz.solve(peak_momentum + field_momentum)
    reconstruction_error = a_norm(
        reconstruction - state_defect, helmholtz
    ) / denominator
    moment_change = _cluster_moment_change(
        current, future, helmholtz.grid.length
    )
    values = np.concatenate(
        [
            position,
            amplitude,
            cluster_position,
            moment_change.ravel(),
            peak,
            field,
            peak_momentum,
            field_momentum,
        ]
    )
    return MCHAtlasLabel(
        atlas=current.kind,
        position=position,
        amplitude=amplitude,
        cluster_position=cluster_position,
        cluster_moment_change=moment_change,
        peak_state=peak,
        field_state=field,
        peak_momentum=peak_momentum,
        field_momentum=field_momentum,
        effective_rank=effective_rank,
        expected_rank=expected_rank,
        gram_condition=float(condition),
        relative_reconstruction_error=float(reconstruction_error),
        finite=bool(np.all(np.isfinite(values))),
    )


def causal_feature_channels(
    previous_state: NDArray[np.floating],
    current_state: NDArray[np.floating],
    baseline_state: NDArray[np.floating],
    decision: AtlasDecision,
    helmholtz: HelmholtzOperator,
) -> NDArray[np.floating]:
    """Build field channels without reading a future reference."""
    for state in (previous_state, current_state, baseline_state):
        helmholtz.grid.validate_state(state)
    momentum = helmholtz.apply(current_state)
    ux = first_derivative(current_state, helmholtz.grid)
    uxx = second_derivative(current_state, helmholtz.grid)
    baseline_increment = baseline_state - current_state
    confidence = np.full_like(current_state, decision.decomposition.detection_confidence)
    separation = np.full_like(
        current_state,
        min(decision.identifiability.separation_cells, 1.0e6),
    )
    return np.stack(
        [
            previous_state,
            current_state,
            momentum,
            ux,
            uxx,
            baseline_increment,
            decision.decomposition.atomic_state,
            decision.decomposition.field_state,
            confidence,
            separation,
        ]
    )


def normalize_mch_atlas_label(
    label: MCHAtlasLabel,
    *,
    length: float,
    dt: float,
    amplitude_scale: float,
    state_scale: float,
) -> dict[str, NDArray[np.floating]]:
    scales = np.asarray([length, dt, amplitude_scale, state_scale])
    if np.any(scales <= 0.0) or not np.all(np.isfinite(scales)):
        raise ValueError("Normalization scales must be positive, finite, and causal.")
    cluster = label.cluster_moment_change.copy()
    if cluster.size:
        cluster[:, 0] /= length
        cluster[:, 1] /= amplitude_scale
        cluster[:, 2] /= amplitude_scale * length
        cluster[:, 3] /= amplitude_scale * length**2
        cluster /= dt
    return {
        "position_rate": np.asarray(label.position / (length * dt)),
        "amplitude_rate": np.asarray(label.amplitude / (amplitude_scale * dt)),
        "cluster_position_rate": np.asarray(label.cluster_position / (length * dt)),
        "cluster_moment_rate": cluster,
        "field_momentum_rate": np.asarray(label.field_momentum / (state_scale * dt)),
    }
