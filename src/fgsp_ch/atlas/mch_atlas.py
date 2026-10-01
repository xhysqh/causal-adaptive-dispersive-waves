from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import least_squares

from fgsp_ch.discretization.difference import first_derivative, second_derivative
from fgsp_ch.discretization.grids import PeriodicGrid
from fgsp_ch.discretization.helmholtz import HelmholtzOperator
from fgsp_ch.geometry.metric import a_norm
from fgsp_ch.representations.peakon_tangent_basis import build_peakon_tangent_basis
from fgsp_ch.solvers.multipeakon import periodic_peakon_kernel


class AtlasKind(str, Enum):
    SEPARATED = "separated"
    CLUSTER = "cluster"
    FIELD = "field"


@dataclass(frozen=True, slots=True)
class MCHAtlasConfig:
    maximum_atoms: int = 8
    candidate_exclusion_cells: int = 1
    minimum_cusp_strength: float = 0.08
    momentum_prominence_sigma: float = 8.0
    field_enter_confidence: float = 0.72
    field_exit_confidence: float = 0.58
    separation_enter_cells: float = 6.0
    separation_exit_cells: float = 4.0
    gram_condition_max: float = 1.0e10
    rank_tolerance: float = 1.0e-10
    cluster_radius_cells: float = 6.0
    reference_floor: float = 1.0e-10
    fit_max_nfev: int = 300

    def __post_init__(self) -> None:
        if self.maximum_atoms < 1 or self.candidate_exclusion_cells < 0:
            raise ValueError("Atom-count controls are invalid.")
        if not 0 < self.field_exit_confidence < self.field_enter_confidence < 1:
            raise ValueError("Field confidence thresholds must form a hysteresis band.")
        if not 0 < self.separation_exit_cells < self.separation_enter_cells:
            raise ValueError("Separation thresholds must form a hysteresis band.")
        positive = (
            self.minimum_cusp_strength,
            self.momentum_prominence_sigma,
            self.gram_condition_max,
            self.rank_tolerance,
            self.cluster_radius_cells,
            self.reference_floor,
            self.fit_max_nfev,
        )
        if any(value <= 0 or not np.isfinite(value) for value in positive):
            raise ValueError("Atlas thresholds must be positive and finite.")


@dataclass(frozen=True, slots=True)
class AtomicFieldDecomposition:
    positions: NDArray[np.floating]
    amplitudes: NDArray[np.floating]
    atomic_state: NDArray[np.floating]
    field_state: NDArray[np.floating]
    cusp_strengths: NDArray[np.floating]
    relative_field_a_norm: float
    detection_confidence: float
    fit_success: bool


@dataclass(frozen=True, slots=True)
class IdentifiabilityState:
    minimum_separation: float
    separation_cells: float
    gram_rank: int
    expected_rank: int
    gram_condition: float
    signal_to_floor: float
    finite: bool


@dataclass(frozen=True, slots=True)
class SignedClusterMoments:
    cluster_ids: NDArray[np.integer]
    centers: NDArray[np.floating]
    signs: NDArray[np.integer]
    zeroth: NDArray[np.floating]
    first: NDArray[np.floating]
    second: NDArray[np.floating]


@dataclass(frozen=True, slots=True)
class AtlasDecision:
    kind: AtlasKind
    decomposition: AtomicFieldDecomposition
    identifiability: IdentifiabilityState
    clusters: SignedClusterMoments
    reason: str
    switched: bool


def periodic_separation(positions: NDArray[np.floating], length: float) -> float:
    q = np.asarray(positions, dtype=np.float64)
    if q.size < 2:
        return float(length)
    distance = np.abs((q[:, None] - q[None, :] + length / 2) % length - length / 2)
    distance[np.eye(q.size, dtype=bool)] = np.inf
    return float(np.min(distance))


def _candidate_indices(
    state: NDArray[np.floating],
    momentum: NDArray[np.floating],
    grid: PeriodicGrid,
    config: MCHAtlasConfig,
) -> tuple[list[int], NDArray[np.floating]]:
    left = (state - np.roll(state, 1)) / grid.h
    right = (np.roll(state, -1) - state) / grid.h
    cusp = np.abs(right - left) / max(float(np.max(np.abs(state))), np.finfo(float).eps)
    magnitude = np.abs(momentum)
    median = float(np.median(magnitude))
    mad = float(np.median(np.abs(magnitude - median)))
    threshold = median + config.momentum_prominence_sigma * max(
        1.4826 * mad, np.finfo(float).eps
    )
    selected: list[int] = []
    for index in np.argsort(magnitude)[::-1]:
        if magnitude[index] < threshold or cusp[index] < config.minimum_cusp_strength:
            continue
        circular_cells = [
            min((int(index) - existing) % grid.points, (existing - int(index)) % grid.points)
            for existing in selected
        ]
        if circular_cells and min(circular_cells) <= config.candidate_exclusion_cells:
            continue
        selected.append(int(index))
        if len(selected) == config.maximum_atoms:
            break
    selected.sort(key=lambda index: grid.x[index])
    return selected, cusp


def decompose_atomic_field(
    state: NDArray[np.floating],
    grid: PeriodicGrid,
    config: MCHAtlasConfig | None = None,
) -> AtomicFieldDecomposition:
    """Fit a sparse periodic peakon component using current Eulerian state only."""
    cfg = config or MCHAtlasConfig()
    grid.validate_state(state)
    helmholtz = HelmholtzOperator(grid)
    momentum = helmholtz.apply(state)
    indices, cusp = _candidate_indices(state, momentum, grid, cfg)
    if not indices:
        zeros = np.empty(0, dtype=np.float64)
        return AtomicFieldDecomposition(
            zeros,
            zeros,
            np.zeros_like(state),
            state.copy(),
            zeros,
            1.0,
            0.0,
            True,
        )
    initial_positions = grid.x[np.asarray(indices)]

    def variable_projection(offsets: NDArray[np.floating]) -> NDArray[np.floating]:
        positions = (initial_positions + offsets - grid.x_min) % grid.length + grid.x_min
        basis = periodic_peakon_kernel(
            grid.x[:, None] - positions[None, :], grid.length
        )
        amplitudes, *_ = np.linalg.lstsq(basis, state, rcond=cfg.rank_tolerance)
        return np.asarray(basis @ amplitudes - state, dtype=np.float64)

    fit = least_squares(
        variable_projection,
        np.zeros(len(indices)),
        bounds=(-2.0 * grid.h, 2.0 * grid.h),
        xtol=1.0e-12,
        ftol=1.0e-12,
        gtol=1.0e-12,
        max_nfev=cfg.fit_max_nfev,
    )
    positions = (initial_positions + fit.x - grid.x_min) % grid.length + grid.x_min
    order = np.argsort(positions)
    positions = positions[order]
    basis = periodic_peakon_kernel(grid.x[:, None] - positions[None, :], grid.length)
    amplitudes, *_ = np.linalg.lstsq(basis, state, rcond=cfg.rank_tolerance)
    basis = periodic_peakon_kernel(grid.x[:, None] - positions[None, :], grid.length)
    atomic = np.asarray(basis @ amplitudes, dtype=state.dtype)
    field = np.asarray(state - atomic, dtype=state.dtype)
    state_norm = max(a_norm(state, helmholtz), np.finfo(float).eps)
    relative_field = a_norm(field, helmholtz) / state_norm
    cusp_strengths = np.asarray(
        [cusp[int(np.argmin(np.abs(grid.periodic_distance(grid.x, q))))] for q in positions]
    )
    cusp_score = float(np.mean(np.clip(cusp_strengths / 0.5, 0.0, 1.0)))
    fit_score = float(np.clip(1.0 - relative_field, 0.0, 1.0))
    confidence = 0.5 * cusp_score + 0.5 * fit_score
    finite = all(
        np.all(np.isfinite(value))
        for value in (positions, amplitudes, atomic, field, cusp_strengths)
    )
    return AtomicFieldDecomposition(
        positions=np.asarray(positions),
        amplitudes=np.asarray(amplitudes),
        atomic_state=atomic,
        field_state=field,
        cusp_strengths=cusp_strengths,
        relative_field_a_norm=float(relative_field),
        detection_confidence=float(confidence),
        fit_success=bool(fit.success and finite),
    )


def identifiability_state(
    decomposition: AtomicFieldDecomposition,
    grid: PeriodicGrid,
    config: MCHAtlasConfig | None = None,
) -> IdentifiabilityState:
    cfg = config or MCHAtlasConfig()
    particles = decomposition.positions.size
    if particles == 0:
        return IdentifiabilityState(grid.length, np.inf, 0, 0, np.inf, 0.0, True)
    helmholtz = HelmholtzOperator(grid)
    basis = build_peakon_tangent_basis(
        grid, decomposition.positions, decomposition.amplitudes
    )
    gram = grid.h * basis.T @ np.stack([helmholtz.apply(column) for column in basis.T]).T
    singular = np.linalg.svd(gram, compute_uv=False)
    scale = max(float(singular[0]), np.finfo(float).eps)
    rank = int(np.sum(singular > cfg.rank_tolerance * scale))
    condition = float(np.inf if singular[-1] <= 0 else singular[0] / singular[-1])
    separation = periodic_separation(decomposition.positions, grid.length)
    signal = a_norm(decomposition.atomic_state, helmholtz) / cfg.reference_floor
    values = np.asarray([separation, rank, condition, signal])
    return IdentifiabilityState(
        minimum_separation=separation,
        separation_cells=separation / grid.h,
        gram_rank=rank,
        expected_rank=basis.shape[1],
        gram_condition=condition,
        signal_to_floor=float(signal),
        finite=bool(np.all(np.isfinite(values)) or np.isinf(condition)),
    )


def _cluster_components(
    positions: NDArray[np.floating], length: float, radius: float
) -> NDArray[np.integer]:
    count = positions.size
    labels = -np.ones(count, dtype=np.int64)
    current = 0
    for start in range(count):
        if labels[start] >= 0:
            continue
        labels[start] = current
        stack = [start]
        while stack:
            left = stack.pop()
            distance = np.abs(
                (positions - positions[left] + length / 2) % length - length / 2
            )
            for right in np.flatnonzero((distance < radius) & (labels < 0)):
                labels[right] = current
                stack.append(int(right))
        current += 1
    return labels


def signed_cluster_moments(
    positions: NDArray[np.floating],
    amplitudes: NDArray[np.floating],
    length: float,
    radius: float,
) -> SignedClusterMoments:
    q = np.asarray(positions, dtype=np.float64)
    a = np.asarray(amplitudes, dtype=np.float64)
    if q.shape != a.shape or q.ndim != 1:
        raise ValueError("Cluster inputs must be equal 1-D arrays.")
    labels = _cluster_components(q, length, radius) if q.size else np.empty(0, dtype=int)
    cluster_ids: list[int] = []
    centers: list[float] = []
    signs: list[int] = []
    zeroth: list[float] = []
    first: list[float] = []
    second: list[float] = []
    for cluster in np.unique(labels):
        for sign in (1, -1):
            selected = (labels == cluster) & (np.sign(a) == sign)
            if not np.any(selected):
                continue
            weights = np.abs(a[selected])
            angles = 2 * np.pi * (q[selected] / length)
            vector = np.sum(weights * np.exp(1j * angles))
            center = float((np.angle(vector) % (2 * np.pi)) * length / (2 * np.pi))
            displacement = (q[selected] - center + length / 2) % length - length / 2
            cluster_ids.append(int(cluster))
            centers.append(center)
            signs.append(sign)
            zeroth.append(float(sign * np.sum(weights)))
            first.append(float(sign * np.sum(weights * displacement)))
            second.append(float(sign * np.sum(weights * displacement**2)))
    return SignedClusterMoments(
        np.asarray(cluster_ids, dtype=np.int64),
        np.asarray(centers),
        np.asarray(signs, dtype=np.int64),
        np.asarray(zeroth),
        np.asarray(first),
        np.asarray(second),
    )


def select_atlas(
    state: NDArray[np.floating],
    grid: PeriodicGrid,
    config: MCHAtlasConfig | None = None,
    *,
    previous: AtlasKind | None = None,
) -> AtlasDecision:
    cfg = config or MCHAtlasConfig()
    decomposition = decompose_atomic_field(state, grid, cfg)
    identity = identifiability_state(decomposition, grid, cfg)
    confidence_threshold = (
        cfg.field_exit_confidence if previous in (AtlasKind.SEPARATED, AtlasKind.CLUSTER)
        else cfg.field_enter_confidence
    )
    if (
        not decomposition.fit_success
        or decomposition.positions.size == 0
        or decomposition.detection_confidence < confidence_threshold
    ):
        kind = AtlasKind.FIELD
        reason = "low_atomic_confidence"
    else:
        separation_threshold = (
            cfg.separation_exit_cells
            if previous == AtlasKind.SEPARATED
            else cfg.separation_enter_cells
        )
        well_conditioned = (
            identity.gram_rank == identity.expected_rank
            and identity.gram_condition <= cfg.gram_condition_max
        )
        if identity.separation_cells >= separation_threshold and well_conditioned:
            kind = AtlasKind.SEPARATED
            reason = "resolved_full_rank_atoms"
        else:
            kind = AtlasKind.CLUSTER
            reason = "near_or_ill_conditioned_atoms"
    clusters = signed_cluster_moments(
        decomposition.positions,
        decomposition.amplitudes,
        grid.length,
        cfg.cluster_radius_cells * grid.h,
    )
    return AtlasDecision(
        kind=kind,
        decomposition=decomposition,
        identifiability=identity,
        clusters=clusters,
        reason=reason,
        switched=bool(previous is not None and previous != kind),
    )
