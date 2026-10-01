from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fgsp_ch.discretization.grids import PeriodicGrid
from fgsp_ch.discretization.helmholtz import HelmholtzOperator
from fgsp_ch.discretization.spectral import fourier_restrict
from fgsp_ch.evaluation.m63433521_scale_audit import (
    ScaleErrorBudget,
    scale_consistent_shadow_label,
)
from fgsp_ch.geometry.metric import a_norm
from fgsp_ch.solvers.mch_hybrid_amr import (
    HierarchicalPeriodicGreenSolver,
    HybridMCHState,
    UniformHybridMCHReferenceSolver,
    fourier_prolong,
    hybrid_total_state,
)
from fgsp_ch.solvers.multipeakon import reconstruct_multipeakon


@dataclass(frozen=True)
class ScaleReferenceMetrics:
    absolute_error: float
    reference_uncertainty: float
    state_norm: float
    scale_label: float
    direction: float
    temporal_gap: float
    spatial_gap: float


def physical_scale_risk_components(diagnostic, metrics: ScaleReferenceMetrics, label: dict):
    direction_risk = max(
        0.0,
        (1.0 - metrics.direction) / (1.0 - float(label["minimum_direction_cosine"])),
    )
    values = np.asarray((
        metrics.scale_label,
        direction_risk,
        float(diagnostic.mass_drift) / float(label["maximum_mass_drift"]),
        float(diagnostic.relative_energy_drift) / float(label["maximum_relative_energy_drift"]),
        float(diagnostic.maximum_cfl) / float(label["maximum_cfl"]),
        (
            float(diagnostic.patch_migration_to_step_increment)
            / float(label["maximum_patch_migration_to_step_increment"])
            if diagnostic.patch_changed else 0.0
        ),
    ))
    return np.where(np.isfinite(values), values, np.inf)


def _advance(state, grid, parameters, dt: float, substeps: int):
    solver = UniformHybridMCHReferenceSolver(
        grid, parameters, dt / substeps, project_invariants=True,
    )
    current = state
    for _ in range(substeps):
        current = solver.step(current)
    return current


def paired_scale_reference_metrics(
    state,
    candidate,
    grid: PeriodicGrid,
    parameters,
    dt: float,
    budget: ScaleErrorBudget,
    *,
    spatial_order: float = 2.0,
    temporal_order: float = 2.0,
) -> ScaleReferenceMetrics:
    """Independent uniform paired reference at a common physical endpoint."""
    green = HierarchicalPeriodicGreenSolver(grid)
    common_grid = green.fine_grid
    field = green.solve(state.composite_field)
    atomic = (
        reconstruct_multipeakon(common_grid, state.positions, state.amplitudes)
        if state.positions.size else np.zeros(common_grid.points)
    )
    initial_total = field + atomic
    base_state = HybridMCHState(
        state.positions.copy(), state.amplitudes.copy(),
        HelmholtzOperator(common_grid).apply(field),
    )
    base_full = hybrid_total_state(_advance(
        base_state, common_grid, parameters, dt, 1,
    ), common_grid)
    base_half = hybrid_total_state(_advance(
        base_state, common_grid, parameters, dt, 2,
    ), common_grid)

    fine_grid = PeriodicGrid(2 * common_grid.points, common_grid.length)
    fine_field = fourier_prolong(field, fine_grid.points)
    fine_state = HybridMCHState(
        state.positions.copy(), state.amplitudes.copy(),
        HelmholtzOperator(fine_grid).apply(fine_field),
    )
    fine_initial = hybrid_total_state(fine_state, fine_grid)
    fine_full = hybrid_total_state(_advance(
        fine_state, fine_grid, parameters, dt, 1,
    ), fine_grid)
    fine_half = hybrid_total_state(_advance(
        fine_state, fine_grid, parameters, dt, 2,
    ), fine_grid)
    fine_full_aligned = initial_total + fourier_restrict(
        fine_full - fine_initial, common_grid.points,
    )
    fine_half_aligned = initial_total + fourier_restrict(
        fine_half - fine_initial, common_grid.points,
    )
    temporal_correction = (fine_half_aligned - fine_full_aligned) / (
        2.0**temporal_order - 1.0
    )
    spatial_correction = (fine_half_aligned - base_half) / (
        2.0**spatial_order - 1.0
    )
    reference = fine_half_aligned + temporal_correction + spatial_correction
    candidate_total = green.solve(candidate.composite_field)
    if candidate.positions.size:
        candidate_total += reconstruct_multipeakon(
            common_grid, candidate.positions, candidate.amplitudes,
        )
    operator = HelmholtzOperator(common_grid)
    absolute_error = a_norm(candidate_total - reference, operator)
    reference_uncertainty = (
        a_norm(temporal_correction, operator) + a_norm(spatial_correction, operator)
    )
    state_norm = a_norm(initial_total, operator)
    label = float(scale_consistent_shadow_label(
        np.asarray([absolute_error]), np.asarray([state_norm]),
        np.asarray([reference_uncertainty]), np.asarray([grid.h]),
        np.asarray([dt]), budget,
    )[0])
    candidate_increment = candidate_total - initial_total
    reference_increment = reference - initial_total
    denominator = a_norm(candidate_increment, operator) * a_norm(reference_increment, operator)
    direction = float(
        common_grid.h * np.sum(candidate_increment * operator.apply(reference_increment))
        / max(denominator, np.finfo(float).eps)
    )
    return ScaleReferenceMetrics(
        float(absolute_error), float(reference_uncertainty), float(state_norm),
        label, direction,
        float(a_norm(fine_half_aligned - fine_full_aligned, operator)),
        float(a_norm(fine_half_aligned - base_half, operator)),
    )
