from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import NDArray

from fgsp_ch.datasets.m633_amr_dataset import build_m633_initial_state, causal_features
from fgsp_ch.discretization.adaptive_mesh import finite_volume_divergence
from fgsp_ch.discretization.difference import first_derivative
from fgsp_ch.discretization.grids import PeriodicGrid
from fgsp_ch.discretization.helmholtz import HelmholtzOperator
from fgsp_ch.equations.modified_ch import ModifiedCHParameters
from fgsp_ch.features.m63431_scale_consistent import (
    proposal_residual_features,
    select_residual_features,
)
from fgsp_ch.geometry.metric import a_norm, discrete_energy
from fgsp_ch.models.m633_local_amr_operator import ConservativeLocalAMROperator
from fgsp_ch.solvers.m633_safe_learned_amr import CertifiedLocalAMRSolver, M633SafeConfig
from fgsp_ch.solvers.mch_hybrid_amr import (
    HierarchicalPeriodicGreenSolver,
    HybridAMRState,
    HybridMCHAMRSolver,
    HybridMCHState,
    hybrid_flux_and_source,
    hybrid_particle_velocity,
)
from fgsp_ch.solvers.multipeakon import reconstruct_multipeakon
from fgsp_ch.training.m633_risk_calibration import M633RiskCalibration


@dataclass(frozen=True, slots=True)
class M6340ShadowSample:
    group: str
    split: str
    family: str
    points: int
    time_index: int
    seed: int
    feature_summary: NDArray[np.floating]
    parameters: NDArray[np.floating]
    residual_features: NDArray[np.floating]
    shadow_defect_ratio: float
    direction_cosine: float
    maximum_safe_weight: float
    raw_mass_drift: float
    raw_relative_energy_drift: float
    shadow_increment_h1: float
    raw_residual_features: NDArray[np.floating] | None = None


def _fine_total(state: HybridAMRState, green: HierarchicalPeriodicGreenSolver) -> np.ndarray:
    total = green.solve(state.composite_field)
    if state.positions.size:
        total += reconstruct_multipeakon(
            green.fine_grid, state.positions, state.amplitudes
        )
    return np.asarray(total)


def _periodic_displacement(after: np.ndarray, before: np.ndarray, length: float) -> np.ndarray:
    return (after - before + 0.5 * length) % length - 0.5 * length


def _summarize_feature_map(features: np.ndarray) -> np.ndarray:
    return np.concatenate((
        np.mean(features, axis=1),
        np.sqrt(np.mean(features**2, axis=1)),
        np.max(np.abs(features), axis=1),
    ))


@torch.no_grad()
def _ensemble_features(
    models: list[ConservativeLocalAMROperator],
    features: np.ndarray,
    parameters: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    x = torch.as_tensor(features[None], dtype=torch.float32, device=device)
    p = torch.as_tensor(parameters[None], dtype=torch.float32, device=device)
    outputs = [model(x, p) for model in models]
    result = []
    for name in ("refinement_score", "fine_detail", "face_flux_mismatch"):
        values = torch.stack([getattr(row, name)[0] for row in outputs])
        mean = values.mean(dim=0)
        standard_deviation = values.var(dim=0, unbiased=False).sqrt()
        result.extend((
            float(mean.square().mean().sqrt().cpu()),
            float(standard_deviation.square().mean().sqrt().cpu()),
            float((
                standard_deviation.square().mean().sqrt()
                / mean.square().mean().sqrt().clamp_min(1.0e-12)
            ).cpu()),
        ))
    return np.asarray(result, dtype=np.float64)


def _proposal_residuals(
    initial: HybridAMRState,
    proposal: HybridAMRState,
    grid: PeriodicGrid,
    parameters: ModifiedCHParameters,
    dt: float,
    *,
    feature_version: str = "raw_v1",
) -> np.ndarray:
    green = HierarchicalPeriodicGreenSolver(grid)
    fine = green.fine_grid
    before = HybridMCHState(
        initial.positions, initial.amplitudes,
        green.reconstruct_fine_momentum(initial.composite_field),
    )
    after = HybridMCHState(
        proposal.positions, proposal.amplitudes,
        green.reconstruct_fine_momentum(proposal.composite_field),
    )
    flux0, source0 = hybrid_flux_and_source(before, fine, parameters)
    flux1, source1 = hybrid_flux_and_source(after, fine, parameters)
    rhs = -finite_volume_divergence(0.5 * (flux0 + flux1), fine.h)
    rhs += 0.5 * (source0 + source1)
    eulerian = (after.field_momentum - before.field_momentum) / dt - rhs
    helmholtz = HelmholtzOperator(fine)

    if before.positions.size:
        velocity0 = hybrid_particle_velocity(before, fine, alpha=parameters.alpha)
        velocity1 = hybrid_particle_velocity(after, fine, alpha=parameters.alpha)
        particle = _periodic_displacement(
            after.positions, before.positions, grid.length
        ) / dt - 0.5 * (velocity0 + velocity1)
        displacement = _periodic_displacement(
            after.positions, before.positions, grid.length
        )
        mid_velocity = 0.5 * (velocity0 + velocity1)
    else:
        particle = displacement = mid_velocity = np.zeros(0)

    coarse = HybridMCHState(
        proposal.positions, proposal.amplitudes,
        proposal.composite_field.coarse_momentum,
    )
    coarse_flux, _ = hybrid_flux_and_source(coarse, grid, parameters)
    fine_flux, _ = hybrid_flux_and_source(after, fine, parameters)
    mismatch = fine_flux[1::2] - coarse_flux
    mask = proposal.composite_field.patch.coarse_mask
    boundary = mask != np.roll(mask, -1)
    interface = mismatch[boundary] if np.any(boundary) else np.zeros(0)
    maximum_cfl = float(
        dt * np.max(np.abs(parameters.alpha * (
            _fine_total(initial, green) ** 2
            - first_derivative(_fine_total(initial, green), fine) ** 2
        ))) / fine.h
    )
    features = proposal_residual_features(
        eulerian_residual=eulerian,
        field_increment=after.field_momentum - before.field_momentum,
        eulerian_rhs=rhs,
        operator=helmholtz,
        dt=dt,
        particle_residual=particle,
        particle_displacement=displacement,
        particle_mid_velocity=mid_velocity,
        amplitude_increment=after.amplitudes - before.amplitudes,
        amplitude_reference=before.amplitudes,
        interface_mismatch=interface,
        maximum_cfl=maximum_cfl,
        active_fraction=proposal.composite_field.patch.active_fraction,
    )
    return select_residual_features(features, feature_version)


def _solver_config(refinement: dict, controller: dict, *, permissive: bool) -> M633SafeConfig:
    return M633SafeConfig(
        theta=float(refinement["theta"]),
        maximum_fraction=float(refinement["maximum_fraction"]),
        halo_cells=int(refinement["halo_cells"]),
        refinement_tolerance=1.0e12,
        detail_tolerance=1.0e12,
        face_tolerance=1.0e12,
        minimum_dt=float(controller["minimum_dt"]),
        shrink_factor=float(controller["shrink_factor"]),
        maximum_cfl=float(refinement["maximum_cfl"]),
        collision_tolerance=1.0e-12,
        cluster_guard_cells=1.0e-6,
        maximum_shadow_increment_ratio=(
            1.0e12 if permissive else float(controller["maximum_shadow_increment_ratio"])
        ),
        maximum_composite_energy_drift=(
            1.0e12 if permissive else float(controller["maximum_composite_energy_drift"])
        ),
        minimum_direction_cosine=(
            -1.0 if permissive else float(controller["minimum_direction_cosine"])
        ),
        minimum_shadow_blend=float(controller["minimum_shadow_blend"]),
        enable_time_consistent_detail=bool(
            controller.get("enable_scale_consistent_proposal", False)
        ),
        proposal_reference_dt=float(controller.get("proposal_reference_dt", 2.0e-4)),
    )


def build_shadow_defect_samples(
    models: list[ConservativeLocalAMROperator],
    calibration: M633RiskCalibration,
    *,
    device: torch.device,
    seed: int,
    length: float,
    dt: float,
    dts: tuple[float, ...] | None = None,
    points: tuple[int, ...],
    groups: dict[str, int],
    snapshots_per_group: int,
    refinement: dict,
    controller: dict,
    feature_version: str = "raw_v1",
) -> list[M6340ShadowSample]:
    rows: list[M6340ShadowSample] = []
    if bool(controller.get("enable_scale_consistent_proposal", False)):
        for model in models:
            model.bounded_degenerate_parameter_conditioning = True
    families = ("smooth", "separated", "clustered")
    for split_index, (split, group_count) in enumerate(groups.items()):
        for group_index in range(int(group_count)):
            family = families[group_index % len(families)]
            local_seed = seed + 10000 * split_index + group_index
            group = f"{split}_{family}_{group_index}"
            for grid_points in points:
              for local_dt in (dts if dts is not None else (dt,)):
                grid = PeriodicGrid(int(grid_points), length)
                state, parameters = build_m633_initial_state(
                    seed=local_seed, points=grid.points, length=length,
                    family=family, refinement=refinement,
                )
                permissive_solver = CertifiedLocalAMRSolver(
                    models, calibration, grid, parameters, device=device,
                    config=_solver_config(refinement, controller, permissive=True),
                )
                safe_solver = CertifiedLocalAMRSolver(
                    models, calibration, grid, parameters, device=device,
                    config=_solver_config(refinement, controller, permissive=False),
                )
                for time_index in range(snapshots_per_group):
                    feature_map = causal_features(state, grid, parameters)
                    parameter_vector = np.asarray((
                        parameters.alpha, parameters.gamma, length, local_dt,
                        grid.h, float(state.positions.size),
                    ))
                    ensemble = _ensemble_features(
                        models, feature_map, parameter_vector, device
                    )
                    raw, raw_diagnostic = permissive_solver.step(
                        state, dt=local_dt, enable_calibration=False
                    )
                    if not raw_diagnostic.learned:
                        raise RuntimeError(
                            f"Permissive proposal unexpectedly fell back: {raw_diagnostic.fallback_reason}."
                        )
                    certified, certified_diagnostic = safe_solver.step(
                        state, dt=local_dt, enable_calibration=False
                    )
                    shadow, _ = HybridMCHAMRSolver(
                        grid, parameters, local_dt,
                        maximum_cfl=float(refinement["maximum_cfl"]),
                        collision_tolerance=1.0e-12,
                    ).step(state)
                    green = HierarchicalPeriodicGreenSolver(grid)
                    operator = HelmholtzOperator(green.fine_grid)
                    initial_fine = _fine_total(state, green)
                    raw_fine = _fine_total(raw, green)
                    shadow_fine = _fine_total(shadow, green)
                    shadow_increment = shadow_fine - initial_fine
                    raw_increment = raw_fine - initial_fine
                    shadow_norm = a_norm(shadow_increment, operator)
                    defect_ratio = a_norm(raw_fine - shadow_fine, operator) / max(
                        shadow_norm, np.finfo(float).eps
                    )
                    inner = float(green.fine_grid.h * np.sum(
                        raw_increment * operator.apply(shadow_increment)
                    ))
                    direction = inner / max(
                        a_norm(raw_increment, operator) * shadow_norm,
                        np.finfo(float).eps,
                    )
                    initial_mass = float(green.fine_grid.h * np.sum(initial_fine))
                    initial_energy = discrete_energy(initial_fine, operator)
                    raw_residual = _proposal_residuals(
                        state, raw, grid, parameters, local_dt,
                        feature_version="raw_v1",
                    )
                    residual = _proposal_residuals(
                        state, raw, grid, parameters, local_dt,
                        feature_version=feature_version,
                    )
                    rows.append(M6340ShadowSample(
                        group, split, family, grid.points, time_index, local_seed,
                        np.concatenate((_summarize_feature_map(feature_map), ensemble)),
                        parameter_vector, residual,
                        float(defect_ratio), float(direction),
                        float(certified_diagnostic.learned_weight),
                        abs(float(green.fine_grid.h * np.sum(raw_fine)) - initial_mass),
                        abs(discrete_energy(raw_fine, operator) - initial_energy)
                        / max(initial_energy, np.finfo(float).eps),
                        float(shadow_norm),
                        raw_residual,
                    ))
                    # Dataset trajectories are generated by the independent
                    # classical path; neither learned nor certified outputs
                    # are teacher-forced into the next feature vector.
                    state = shadow
                    del certified
    return rows
