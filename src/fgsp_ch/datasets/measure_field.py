from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from fgsp_ch.discretization.helmholtz import HelmholtzOperator
from fgsp_ch.representations.conservative_peakon_basis import (
    build_conservative_peakon_basis,
)
from fgsp_ch.representations.peakon_tangent_basis import (
    build_peakon_tangent_basis,
    decompose_a_subspace,
)


@dataclass(frozen=True, slots=True)
class MeasureFieldLabel:
    """Selected peak-coordinate label plus its exact field remainder.

    Coefficients are physical increments ordered by particle.  When the
    conditional amplitude expert is disabled, ``amplitude`` is exactly zero
    and ``position`` is the A-projection on the position tangent basis.  When
    enabled, both are the physical coefficients of the invariant-nullspace
    projection.  In both cases ``defect = peak_component + field``.
    """

    amplitude: NDArray[np.floating]
    position: NDArray[np.floating]
    peak_component: NDArray[np.floating]
    field: NDArray[np.floating]
    amplitude_gate: bool
    representation: str
    effective_rank: int
    expected_rank: int
    gram_condition_number: float
    relative_orthogonality_error: float
    conservative_constraint_residual: float
    finite: bool


def build_measure_field_label(
    defect: NDArray[np.floating],
    positions: NDArray[np.floating],
    amplitudes: NDArray[np.floating],
    helmholtz: HelmholtzOperator,
    *,
    amplitude_gate: bool,
    rank_tolerance: float = 1.0e-12,
) -> MeasureFieldLabel:
    """Decompose a future defect using a basis built only from current state.

    The future reference enters only through ``defect`` (the supervised
    target).  Particle coordinates used to construct either basis must be the
    coordinates at the current time and are therefore causal model inputs.
    """
    positions = np.asarray(positions, dtype=np.float64)
    amplitudes = np.asarray(amplitudes, dtype=np.float64)
    particles = len(positions)
    if particles < 1 or positions.shape != amplitudes.shape:
        raise ValueError("Particle coordinates must be equal nonempty vectors.")
    full_basis = build_peakon_tangent_basis(
        helmholtz.grid, positions, amplitudes
    )
    constraint_residual = 0.0
    if amplitude_gate:
        if particles < 2:
            raise ValueError("The conditional amplitude expert requires >=2 particles.")
        conservative = build_conservative_peakon_basis(
            helmholtz.grid,
            positions,
            amplitudes,
            rank_tolerance=rank_tolerance,
        )
        decomposition = decompose_a_subspace(
            defect,
            conservative.constrained_basis,
            helmholtz,
            rank_tolerance=rank_tolerance,
        )
        physical = conservative.physical_coefficients(
            decomposition.coefficients
        )
        amplitude = np.asarray(physical[:particles])
        position = np.asarray(physical[particles:])
        denominator = max(
            float(np.linalg.norm(conservative.constraint_matrix))
            * float(np.linalg.norm(physical)),
            np.finfo(float).eps,
        )
        constraint_residual = float(
            np.linalg.norm(conservative.constraint_matrix @ physical)
            / denominator
        )
        expected_rank = conservative.constrained_basis.shape[1]
        representation = "conservative_amplitude_position_plus_field"
    else:
        position_basis = full_basis[:, particles:]
        decomposition = decompose_a_subspace(
            defect,
            position_basis,
            helmholtz,
            rank_tolerance=rank_tolerance,
        )
        amplitude = np.zeros(particles, dtype=np.float64)
        position = np.asarray(decomposition.coefficients)
        expected_rank = particles
        representation = "position_plus_field"
    values = np.concatenate(
        [amplitude, position, decomposition.component, decomposition.remainder]
    )
    return MeasureFieldLabel(
        amplitude=amplitude,
        position=position,
        peak_component=np.asarray(decomposition.component),
        field=np.asarray(decomposition.remainder),
        amplitude_gate=bool(amplitude_gate),
        representation=representation,
        effective_rank=decomposition.effective_rank,
        expected_rank=expected_rank,
        gram_condition_number=decomposition.gram_condition_number,
        relative_orthogonality_error=decomposition.relative_orthogonality_error,
        conservative_constraint_residual=constraint_residual,
        finite=bool(decomposition.finite and np.isfinite(values).all()),
    )


def normalized_measure_field_targets(
    label: MeasureFieldLabel,
    *,
    length: float,
    dt: float,
    current_amplitude_scale: float,
    current_state_scale: float,
) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
    """Return dimensionless coefficient rates and causally scaled field rate."""
    scales = np.asarray(
        [length, dt, current_amplitude_scale, current_state_scale],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(scales)) or np.any(scales <= 0.0):
        raise ValueError("All normalization scales must be positive and finite.")
    position = label.position / (length * dt)
    amplitude = label.amplitude / (current_amplitude_scale * dt)
    field = label.field / (current_state_scale * dt)
    return np.asarray(position), np.asarray(amplitude), np.asarray(field)
