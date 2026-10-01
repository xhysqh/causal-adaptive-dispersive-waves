from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from fgsp_ch.atlas.mch_atlas import MCHAtlasConfig, decompose_atomic_field
from fgsp_ch.discretization.grids import PeriodicGrid
from fgsp_ch.solvers.multipeakon import reconstruct_multipeakon


@dataclass(frozen=True, slots=True)
class StructuredMCHState:
    """Canonical mCH state: atoms plus a disjoint Eulerian remainder.

    Cluster coordinates are auxiliary conditioning coordinates.  They never
    contribute a second copy of the atomic measure to reconstruction.
    """

    positions: NDArray[np.floating]
    amplitudes: NDArray[np.floating]
    field_state: NDArray[np.floating]
    cluster_ids: NDArray[np.integer]
    cluster_features: NDArray[np.floating]
    detection_confidence: float


def _cluster_labels(
    positions: NDArray[np.floating], length: float, radius: float
) -> NDArray[np.integer]:
    count = positions.size
    labels = -np.ones(count, dtype=np.int64)
    label = 0
    for start in range(count):
        if labels[start] >= 0:
            continue
        component = [start]
        labels[start] = label
        while component:
            left = component.pop()
            distance = np.abs(
                (positions - positions[left] + 0.5 * length) % length
                - 0.5 * length
            )
            for right in np.flatnonzero((distance < radius) & (labels < 0)):
                labels[right] = label
                component.append(int(right))
        label += 1
    return labels


def _cluster_features(
    positions: NDArray[np.floating],
    amplitudes: NDArray[np.floating],
    labels: NDArray[np.integer],
    grid: PeriodicGrid,
) -> NDArray[np.floating]:
    """Return periodic center, mass, width, and four normalized moments."""
    rows: list[list[float]] = []
    for label in np.unique(labels):
        selected = labels == label
        q = positions[selected]
        p = amplitudes[selected]
        weights = np.abs(p)
        angle = 2.0 * np.pi * (q - grid.x_min) / grid.length
        vector = np.sum(weights * np.exp(1j * angle))
        center = grid.x_min + (np.angle(vector) % (2.0 * np.pi)) * grid.length / (
            2.0 * np.pi
        )
        displacement = (q - center + 0.5 * grid.length) % grid.length - 0.5 * grid.length
        width = max(float(np.sqrt(np.average(displacement**2, weights=weights))), grid.h)
        signed_moments = [float(np.sum(p * (displacement / width) ** order)) for order in range(1, 5)]
        rows.append(
            [
                np.sin(2.0 * np.pi * (center - grid.x_min) / grid.length),
                np.cos(2.0 * np.pi * (center - grid.x_min) / grid.length),
                float(np.sum(p)),
                float(np.log1p(width / grid.h)),
                float(np.count_nonzero(selected)),
                *signed_moments,
            ]
        )
    return np.asarray(rows, dtype=np.float64).reshape(-1, 9)


def encode_mch_state(
    state: NDArray[np.floating],
    grid: PeriodicGrid,
    config: MCHAtlasConfig | None = None,
) -> StructuredMCHState:
    """Causally encode an Eulerian state without losing its field remainder."""
    cfg = config or MCHAtlasConfig()
    decomposition = decompose_atomic_field(state, grid, cfg)
    labels = (
        _cluster_labels(
            decomposition.positions,
            grid.length,
            cfg.cluster_radius_cells * grid.h,
        )
        if decomposition.positions.size
        else np.empty(0, dtype=np.int64)
    )
    features = _cluster_features(
        decomposition.positions, decomposition.amplitudes, labels, grid
    ) if labels.size else np.empty((0, 9), dtype=np.float64)
    return StructuredMCHState(
        positions=np.asarray(decomposition.positions),
        amplitudes=np.asarray(decomposition.amplitudes),
        field_state=np.asarray(decomposition.field_state),
        cluster_ids=labels,
        cluster_features=features,
        detection_confidence=decomposition.detection_confidence,
    )


def reconstruct_mch_state(encoded: StructuredMCHState, grid: PeriodicGrid) -> NDArray[np.floating]:
    """Exactly reconstruct the encoded grid state (up to roundoff)."""
    grid.validate_state(encoded.field_state)
    atomic = reconstruct_multipeakon(grid, encoded.positions, encoded.amplitudes)
    reconstructed = np.asarray(atomic + encoded.field_state, dtype=grid.dtype)
    grid.validate_state(reconstructed)
    return reconstructed


def padded_peak_tokens(
    encoded: StructuredMCHState,
    grid: PeriodicGrid,
    *,
    maximum_atoms: int,
) -> tuple[NDArray[np.floating], NDArray[np.bool_]]:
    """Build the canonical eight-channel periodic peak token tensor."""
    if maximum_atoms < encoded.positions.size:
        raise ValueError("maximum_atoms cannot truncate detected atoms.")
    tokens = np.zeros((maximum_atoms, 8), dtype=np.float64)
    mask = np.zeros(maximum_atoms, dtype=bool)
    count = encoded.positions.size
    if count == 0:
        return tokens, mask
    q = encoded.positions
    p = encoded.amplitudes
    scale = max(float(np.max(np.abs(p))), np.finfo(float).eps)
    difference = (q[:, None] - q[None, :] + 0.5 * grid.length) % grid.length - 0.5 * grid.length
    distance = np.abs(difference)
    distance[np.eye(count, dtype=bool)] = np.inf
    nearest = np.min(distance, axis=1) if count > 1 else np.full(count, grid.length)
    phase = 2.0 * np.pi * (q - grid.x_min) / grid.length
    tokens[:count, 0] = np.sin(phase)
    tokens[:count, 1] = np.cos(phase)
    tokens[:count, 2] = p / scale
    tokens[:count, 3] = np.sign(p) * np.log1p(np.abs(p) / scale)
    tokens[:count, 4] = nearest / grid.h
    tokens[:count, 5] = nearest / grid.h
    # Exact Green field and its symmetric slope are added by the graph layer.
    tokens[:count, 6:] = 0.0
    mask[:count] = True
    return tokens, mask
