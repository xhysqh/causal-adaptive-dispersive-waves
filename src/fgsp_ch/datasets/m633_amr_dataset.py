from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from fgsp_ch.discretization.adaptive_mesh import dorfler_mark, reflux_two_to_one
from fgsp_ch.discretization.difference import first_derivative
from fgsp_ch.discretization.grids import PeriodicGrid
from fgsp_ch.discretization.helmholtz import HelmholtzOperator
from fgsp_ch.discretization.spectral import fourier_restrict
from fgsp_ch.equations.modified_ch import ModifiedCHParameters
from fgsp_ch.solvers.mch_hybrid_amr import (
    CompositeMomentumState,
    HierarchicalPeriodicGreenSolver,
    HybridAMRState,
    HybridMCHState,
    HybridMCHAMRSolver,
    UniformHybridMCHReferenceSolver,
    hybrid_flux_and_source,
    hybrid_total_state,
)
from fgsp_ch.solvers.multipeakon import reconstruct_multipeakon


@dataclass(frozen=True, slots=True)
class M633AMRSample:
    group: str
    split: str
    family: str
    time_index: int
    features: NDArray[np.floating]
    parameters: NDArray[np.floating]
    refinement_score: NDArray[np.floating]
    refinement_mask: NDArray[np.bool_]
    reflux_correction: NDArray[np.floating]
    reflux_support: NDArray[np.bool_]
    reflux_face_target: NDArray[np.floating]
    boundary_face_mask: NDArray[np.bool_]
    fine_detail_target: NDArray[np.floating]
    refinement_reference_floor: NDArray[np.floating]
    reflux_reference_floor: NDArray[np.floating]
    reflux_face_reference_floor: NDArray[np.floating]
    marked_energy_fraction: float


def _periodic_distance(x: NDArray[np.floating], center: float, length: float):
    return (x - center + 0.5 * length) % length - 0.5 * length


def geometry_indicator(
    grid: PeriodicGrid,
    field: NDArray[np.floating],
    positions: NDArray[np.floating],
    amplitudes: NDArray[np.floating],
) -> NDArray[np.floating]:
    slope = first_derivative(field, grid)
    curvature = first_derivative(slope, grid)
    scale = max(float(np.sqrt(np.mean(curvature**2))), np.finfo(float).eps)
    indicator = np.abs(curvature) / scale
    width = max(3.0 * grid.h, 0.25)
    for position, amplitude in zip(positions, amplitudes, strict=True):
        distance = _periodic_distance(grid.x, float(position), grid.length)
        indicator += 2.0 * abs(float(amplitude)) * np.exp(-(distance / width) ** 2)
    return np.asarray(indicator)


def _fine_advance_with_average_flux(
    state: HybridMCHState,
    grid: PeriodicGrid,
    parameters: ModifiedCHParameters,
    dt: float,
    substeps: int,
) -> tuple[HybridMCHState, NDArray[np.floating]]:
    if substeps < 1:
        raise ValueError("substeps must be positive.")
    current = state
    fluxes = []
    solver = UniformHybridMCHReferenceSolver(grid, parameters, dt / substeps)
    for _ in range(substeps):
        flux, _ = hybrid_flux_and_source(current, grid, parameters)
        fluxes.append(flux)
        current = solver.step(current)
    return current, np.mean(np.stack(fluxes), axis=0)


def _label_maps(
    state: HybridAMRState,
    grid: PeriodicGrid,
    parameters: ModifiedCHParameters,
    dt: float,
    *,
    reference_substeps: int,
    theta: float,
    maximum_fraction: float,
    halo_cells: int,
) -> tuple[
    NDArray[np.floating], NDArray[np.bool_], NDArray[np.floating],
    NDArray[np.bool_], NDArray[np.floating], NDArray[np.bool_],
    NDArray[np.floating], float,
]:
    green = HierarchicalPeriodicGreenSolver(grid)
    fine_grid = green.fine_grid
    coarse_state = HybridMCHState(
        state.positions, state.amplitudes, state.composite_field.coarse_momentum
    )
    fine_state = HybridMCHState(
        state.positions, state.amplitudes,
        green.reconstruct_fine_momentum(state.composite_field),
    )
    coarse_next = UniformHybridMCHReferenceSolver(grid, parameters, dt).step(coarse_state)
    fine_next, fine_average_flux = _fine_advance_with_average_flux(
        fine_state, fine_grid, parameters, dt, reference_substeps
    )
    coarse_total = hybrid_total_state(coarse_next, grid)
    fine_total = fourier_restrict(hybrid_total_state(fine_next, fine_grid), grid.points)
    defect = fine_total - coarse_total
    defect_slope = first_derivative(defect, grid)
    state_scale = max(
        float(np.sqrt(np.mean(fine_total**2 + first_derivative(fine_total, grid) ** 2))),
        np.finfo(float).eps,
    )
    refinement_score = np.sqrt(defect**2 + defect_slope**2) / state_scale
    patch = dorfler_mark(
        refinement_score, theta=theta,
        maximum_fraction=maximum_fraction, halo_cells=halo_cells,
    )

    coarse_flux, _ = hybrid_flux_and_source(coarse_state, grid, parameters)
    fine_field = fine_next.field_momentum
    overwrite_only = coarse_next.field_momentum.copy()
    overwrite_only[patch.coarse_mask] = 0.5 * (
        fine_field[0::2][patch.coarse_mask]
        + fine_field[1::2][patch.coarse_mask]
    )
    synchronized = reflux_two_to_one(
        coarse_next.field_momentum, fine_field, coarse_flux, fine_average_flux,
        patch.coarse_mask, dt=dt, coarse_h=grid.h,
    )
    reflux = synchronized - overwrite_only
    support = np.zeros(grid.points, dtype=bool)
    boundary_faces = np.zeros(grid.points, dtype=bool)
    for face in range(grid.points):
        left, right = face, (face + 1) % grid.points
        if patch.coarse_mask[left] == patch.coarse_mask[right]:
            continue
        boundary_faces[face] = True
        support[left if not patch.coarse_mask[left] else right] = True

    base = green.base_fine_momentum(coarse_next.field_momentum)
    detail = np.where(patch.fine_mask, fine_next.field_momentum - base, 0.0)
    return (
        np.asarray(refinement_score), patch.coarse_mask, np.asarray(reflux),
        support, np.asarray(fine_average_flux[1::2] - coarse_flux),
        boundary_faces, np.asarray(detail), patch.marked_energy_fraction,
    )


def causal_features(
    state: HybridAMRState,
    grid: PeriodicGrid,
    parameters: ModifiedCHParameters,
) -> NDArray[np.floating]:
    green = HierarchicalPeriodicGreenSolver(grid)
    field = HelmholtzOperator(grid).solve(state.composite_field.coarse_momentum)
    atomic = (
        reconstruct_multipeakon(grid, state.positions, state.amplitudes)
        if state.positions.size else np.zeros(grid.points)
    )
    total = field + atomic
    slope = first_derivative(total, grid)
    curvature = first_derivative(slope, grid)
    particle_geometry = np.zeros(grid.points)
    width = max(3.0 * grid.h, 0.25)
    for position, amplitude in zip(state.positions, state.amplitudes, strict=True):
        distance = _periodic_distance(grid.x, float(position), grid.length)
        particle_geometry += abs(float(amplitude)) * np.exp(-(distance / width) ** 2)
    fine_detail = np.abs(state.composite_field.fine_detail_momentum)
    detail_envelope = np.sqrt(fine_detail[0::2] ** 2 + fine_detail[1::2] ** 2)
    fine_state = HybridMCHState(
        state.positions, state.amplitudes,
        green.reconstruct_fine_momentum(state.composite_field),
    )
    coarse_state = HybridMCHState(
        state.positions, state.amplitudes, state.composite_field.coarse_momentum
    )
    coarse_flux, _ = hybrid_flux_and_source(coarse_state, grid, parameters)
    fine_flux, _ = hybrid_flux_and_source(fine_state, green.fine_grid, parameters)
    face_mismatch = fine_flux[1::2] - coarse_flux
    flux_envelope = np.abs(face_mismatch) + np.abs(np.roll(face_mismatch, 1))
    return np.stack((
        state.composite_field.coarse_momentum,
        field,
        atomic,
        total,
        slope,
        curvature,
        particle_geometry,
        detail_envelope,
        flux_envelope,
        face_mismatch,
        np.roll(face_mismatch, 1),
    ))


def _initial_amr_state(
    rng: np.random.Generator,
    grid: PeriodicGrid,
    family: str,
    refinement: dict,
) -> HybridAMRState:
    phase = 2.0 * np.pi * grid.x / grid.length
    field = rng.uniform(0.08, 0.2) + rng.uniform(-0.06, 0.06) * np.sin(phase)
    field += rng.uniform(-0.04, 0.04) * np.cos(2.0 * phase)
    field += rng.uniform(-0.02, 0.02) * np.sin(3.0 * phase)
    if family == "smooth":
        positions, amplitudes = np.empty(0), np.empty(0)
    elif family == "separated":
        center = float(rng.uniform(0.0, grid.length))
        positions = np.remainder(center + np.asarray([0.0, 0.42 * grid.length]), grid.length)
        amplitudes = rng.uniform(0.3, 0.8, 2)
    elif family == "clustered":
        center = float(rng.uniform(0.0, grid.length))
        separation = float(rng.uniform(0.35, 0.55))
        positions = np.remainder(
            center + np.asarray([-0.5 * separation, 0.5 * separation, 0.35 * grid.length]),
            grid.length,
        )
        amplitudes = rng.uniform(0.25, 0.65, 3)
    else:
        raise ValueError(f"Unknown family {family!r}.")
    indicator = geometry_indicator(grid, field, positions, amplitudes)
    patch = dorfler_mark(
        indicator, theta=float(refinement["theta"]),
        maximum_fraction=float(refinement["maximum_fraction"]),
        halo_cells=int(refinement["halo_cells"]),
    )
    coarse_momentum = HelmholtzOperator(grid).apply(field)
    fine_grid = HierarchicalPeriodicGreenSolver(grid).fine_grid
    detail = np.zeros(fine_grid.points)
    detail[patch.fine_mask] = rng.uniform(5.0e-4, 2.0e-3) * np.cos(
        int(rng.integers(2, 6)) * 2.0 * np.pi * fine_grid.x[patch.fine_mask] / grid.length
        + rng.uniform(0.0, 2.0 * np.pi)
    )
    return HybridAMRState(
        positions, amplitudes,
        CompositeMomentumState(coarse_momentum, detail, patch),
    )


def build_m633_initial_state(
    *, seed: int, points: int, length: float, family: str, refinement: dict,
) -> tuple[HybridAMRState, ModifiedCHParameters]:
    """Public deterministic scenario constructor for post-training rollouts."""
    rng = np.random.default_rng(seed)
    grid = PeriodicGrid(points, length)
    parameters = ModifiedCHParameters(
        alpha=float(rng.uniform(0.7, 1.25)),
        gamma=float(rng.uniform(0.0, 0.08)),
    )
    return _initial_amr_state(rng, grid, family, refinement), parameters


def build_m633_amr_samples(
    *, seed: int, points: int, length: float, dt: float,
    groups: dict[str, int], snapshots_per_group: int,
    refinement: dict,
) -> list[M633AMRSample]:
    rows: list[M633AMRSample] = []
    family_cycle = ("smooth", "separated", "clustered")
    for split_index, (split, count) in enumerate(groups.items()):
        for group_index in range(int(count)):
            local_seed = seed + 10000 * split_index + group_index
            rng = np.random.default_rng(local_seed)
            family = family_cycle[group_index % len(family_cycle)]
            grid = PeriodicGrid(points, length)
            parameters = ModifiedCHParameters(
                alpha=float(rng.uniform(0.7, 1.25)),
                gamma=float(rng.uniform(0.0, 0.08)),
            )
            state = _initial_amr_state(rng, grid, family, refinement)
            group = f"{split}_{family}_{group_index}"
            for time_index in range(snapshots_per_group):
                primary = _label_maps(
                    state, grid, parameters, dt, reference_substeps=2,
                    theta=float(refinement["theta"]),
                    maximum_fraction=float(refinement["maximum_fraction"]),
                    halo_cells=int(refinement["halo_cells"]),
                )
                refined = _label_maps(
                    state, grid, parameters, dt, reference_substeps=4,
                    theta=float(refinement["theta"]),
                    maximum_fraction=float(refinement["maximum_fraction"]),
                    halo_cells=int(refinement["halo_cells"]),
                )
                rows.append(M633AMRSample(
                    group, split, family, time_index,
                    causal_features(state, grid, parameters),
                    np.asarray([
                        parameters.alpha, parameters.gamma, length, dt, grid.h,
                        float(state.positions.size),
                    ]),
                    primary[0], primary[1], primary[2], primary[3],
                    primary[4], primary[5], primary[6],
                    np.abs(primary[0] - refined[0]),
                    np.abs(primary[2] - refined[2]),
                    np.abs(primary[4] - refined[4]),
                    primary[7],
                ))
                state, _ = HybridMCHAMRSolver(
                    grid, parameters, dt,
                    maximum_cfl=float(refinement["maximum_cfl"]),
                ).step(state)
    return rows
