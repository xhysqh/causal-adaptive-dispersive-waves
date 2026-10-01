from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
import torch

from fgsp_ch.datasets.m633_amr_dataset import causal_features
from fgsp_ch.datasets.m6340_shadow_defect import _fine_total, _summarize_feature_map
from fgsp_ch.discretization.adaptive_mesh import (
    PeriodicRefinementPatch,
    finite_volume_divergence,
    reflux_two_to_one,
)
from fgsp_ch.discretization.helmholtz import HelmholtzOperator
from fgsp_ch.discretization.difference import first_derivative
from fgsp_ch.equations.modified_ch import ModifiedCH, ModifiedCHParameters
from fgsp_ch.features.m63431_scale_consistent import (
    proposal_residual_features,
    select_residual_features,
)
from fgsp_ch.geometry.metric import a_norm, discrete_energy
from fgsp_ch.models.m633_local_amr_operator import ConservativeLocalAMROperator
from fgsp_ch.models.m6341_shadow_risk import MonotoneShadowDefectRiskOperator
from fgsp_ch.models.m63432_deployment_safety import (
    DeploymentCandidateSafetyOperator,
)
from fgsp_ch.safety.m63432_selective_certificate import DeploymentSupportCertificate
from fgsp_ch.solvers.m633_safe_learned_amr import (
    CertifiedLocalAMRSolver,
    M633EnsembleCache,
    M633SafeConfig,
)
from fgsp_ch.solvers.mch_energy_projected import project_to_mass_h1
from fgsp_ch.solvers.mch_hybrid_amr import (
    CompositeMomentumState,
    HierarchicalPeriodicGreenSolver,
    HybridAMRState,
    HybridMCHAMRSolver,
    HybridMCHState,
    UniformHybridMCHReferenceSolver,
    hybrid_flux_and_source,
    hybrid_particle_velocity,
    hybrid_total_state,
)
from fgsp_ch.solvers.multipeakon import reconstruct_multipeakon
from fgsp_ch.training.m633_risk_calibration import M633RiskCalibration


@dataclass(frozen=True, slots=True)
class M6342Config:
    selected_weight: float = 0.0625
    safe_selector_threshold: float = 0.0
    direction_correction: float = 0.0
    minimum_direction_cosine: float = -0.05
    maximum_relative_energy_drift: float = 0.02
    maximum_mass_drift: float = 1.0e-10
    maximum_cfl: float = 0.45
    minimum_dt: float = 5.0e-5
    shrink_factor: float = 0.5
    enable_direction_gate: bool = False
    enable_patch_commit: bool = False
    maximum_patch_migration_relative_h1: float = 0.02
    maximum_patch_migration_to_step_increment: float = float("inf")
    enable_time_growth: bool = False
    maximum_dt: float = 2.0e-4
    growth_factor: float = 1.25
    growth_patience: int = 2
    growth_risk_margin: float = 0.1
    growth_cfl_fraction: float = 0.6
    risk_feature_version: str = "raw_v1"
    hold_dt_on_proposal_risk: bool = False
    enable_scale_consistent_proposal: bool = False
    proposal_reference_dt: float = 2.0e-4
    selector_score_mode: str = "defect_upper"
    selector_family_thresholds: tuple[float, ...] = ()
    enable_deployment_selector: bool = False
    deployment_selector_threshold: float = 0.0
    deployment_selector_thresholds: tuple[float, ...] = ()
    deployment_registered_points: tuple[int, ...] = (32, 64, 128)
    deployment_ensemble_sigma: float = 1.0
    deployment_defect_correction: float = 0.0
    deployment_direction_correction: float = 0.0
    collect_deployment_features: bool = False

    def __post_init__(self) -> None:
        if not 0.0 < self.selected_weight <= 1.0:
            raise ValueError("selected_weight must lie in (0, 1].")
        if self.maximum_mass_drift <= 0.0 or self.maximum_relative_energy_drift <= 0.0:
            raise ValueError("Invariant tolerances must be positive.")
        if self.maximum_patch_migration_relative_h1 <= 0.0:
            raise ValueError("Patch migration tolerance must be positive.")
        if self.maximum_patch_migration_to_step_increment <= 0.0:
            raise ValueError("Patch migration increment tolerance must be positive.")
        if not np.isfinite(self.maximum_dt) or self.maximum_dt < self.minimum_dt:
            raise ValueError("maximum_dt must not be smaller than minimum_dt.")
        if self.growth_factor < 1.0 or self.growth_patience < 1:
            raise ValueError("Time growth requires factor >= 1 and positive patience.")
        if not 0.0 < self.growth_cfl_fraction < 1.0:
            raise ValueError("growth_cfl_fraction must lie in (0, 1).")
        if self.growth_risk_margin < 0.0 or not np.isfinite(self.growth_risk_margin):
            raise ValueError("growth_risk_margin must be finite and nonnegative.")
        if self.risk_feature_version not in {"raw_v1", "scale_consistent_v2"}:
            raise ValueError("Unknown risk_feature_version.")
        if self.proposal_reference_dt <= 0.0:
            raise ValueError("proposal_reference_dt must be positive.")
        if self.selector_score_mode not in {"defect_upper", "family_hybrid_v2"}:
            raise ValueError("Unknown selector_score_mode.")
        if (
            self.selector_score_mode == "family_hybrid_v2"
            and len(self.selector_family_thresholds) != 3
        ):
            raise ValueError("family_hybrid_v2 requires three family thresholds.")
        if self.deployment_ensemble_sigma < 0.0:
            raise ValueError("deployment_ensemble_sigma must be nonnegative.")
        if self.enable_deployment_selector and (
            len(self.deployment_selector_thresholds)
            != 3 * len(self.deployment_registered_points)
        ):
            raise ValueError("A threshold is required for every family/grid cell.")


@dataclass(frozen=True, slots=True)
class M6342StepDiagnostics:
    dt: float
    next_dt: float
    learned: bool
    fallback: bool
    fallback_reason: str | None
    learned_weight: float
    safe_score: float
    selector_threshold: float
    direction_lower_bound: float
    mass_drift: float
    relative_energy_drift: float
    maximum_cfl: float
    proposal_shadow_calls: int
    classical_fallback_calls: int
    phase_seconds: dict[str, float] | None = None
    patch_changed: bool = False
    patch_entered_fraction: float = 0.0
    patch_exited_fraction: float = 0.0
    patch_migration_relative_h1: float = 0.0
    patch_migration_to_step_increment: float = 0.0
    safe_streak: int = 0
    time_action: str = "hold"
    raw_residual_features: tuple[float, ...] = ()
    risk_residual_features: tuple[float, ...] = ()
    ensemble_upper_log_defect: tuple[float, ...] = ()
    ensemble_direction: tuple[float, ...] = ()
    risk_margin: float = 0.0
    risk_feature_version: str = "raw_v1"
    ensemble_unsafe_logit: tuple[float, ...] = ()
    defect_safe_score: float = 0.0
    selector_component: float = 0.0
    selector_family_index: int = 0
    deployment_features: tuple[float, ...] = ()
    deployment_unsafe_members: tuple[float, ...] = ()
    deployment_unsafe_upper: float = float("nan")
    deployment_defect_upper: float = float("nan")
    deployment_direction_lower: float = float("nan")
    deployment_support_ratio: float = float("inf")


@dataclass(frozen=True, slots=True)
class M6342ProposalCache:
    """All expensive proposal-time quantities, evaluated exactly once."""

    ensemble: M633EnsembleCache
    before_fine_state: HybridMCHState
    after_fine_state: HybridMCHState
    initial_fine_total: np.ndarray
    flux0: np.ndarray
    source0: np.ndarray
    flux1: np.ndarray
    source1: np.ndarray
    velocity0: np.ndarray
    velocity1: np.ndarray
    residual_features: np.ndarray
    raw_residual_features: np.ndarray
    scale_consistent_residual_features: np.ndarray


class ShadowFreeSafeAMRSolver:
    """Risk-gated cheap-backbone AMR controller with no accepted-step shadow."""

    def __init__(
        self,
        proposal_models: list[ConservativeLocalAMROperator],
        risk_models: list[MonotoneShadowDefectRiskOperator],
        proposal_calibration: M633RiskCalibration,
        grid,
        parameters: ModifiedCHParameters,
        *,
        device: torch.device,
        refinement: dict,
        config: M6342Config,
        deployment_models: list[DeploymentCandidateSafetyOperator] | None = None,
        deployment_certificate: DeploymentSupportCertificate | None = None,
    ) -> None:
        if len(proposal_models) < 3 or len(risk_models) < 3:
            raise ValueError("Shadow-free controller requires two three-model ensembles.")
        self.proposal_models = [model.eval() for model in proposal_models]
        if config.enable_scale_consistent_proposal:
            for model in self.proposal_models:
                model.bounded_degenerate_parameter_conditioning = True
        self.risk_models = [model.eval() for model in risk_models]
        self.proposal_calibration = proposal_calibration
        self.grid = grid
        self.parameters = parameters
        self.device = device
        self.refinement = refinement
        self.config = config
        self.deployment_models = [
            model.eval() for model in (deployment_models or [])
        ]
        self.deployment_certificate = deployment_certificate
        if config.enable_deployment_selector and len(self.deployment_models) < 3:
            raise ValueError(
                "Deployment selection requires a three-model safety ensemble."
            )
        if config.enable_deployment_selector and deployment_certificate is None:
            raise ValueError("Deployment selection requires a support certificate.")
        self._safe_streak = 0
        self._proposal_solver = CertifiedLocalAMRSolver(
            self.proposal_models, proposal_calibration, grid, parameters,
            device=device,
            config=M633SafeConfig(
                theta=float(refinement["theta"]),
                maximum_fraction=float(refinement["maximum_fraction"]),
                halo_cells=int(refinement["halo_cells"]),
                refinement_tolerance=1.0e12,
                detail_tolerance=1.0e12,
                face_tolerance=1.0e12,
                minimum_dt=config.minimum_dt,
                shrink_factor=config.shrink_factor,
                maximum_cfl=config.maximum_cfl,
                collision_tolerance=1.0e-12,
                cluster_guard_cells=1.0e-6,
                maximum_shadow_increment_ratio=1.0e12,
                maximum_composite_energy_drift=1.0e12,
                minimum_direction_cosine=-1.0,
                enable_time_consistent_detail=config.enable_scale_consistent_proposal,
                proposal_reference_dt=config.proposal_reference_dt,
            ),
        )

    def _fallback(
        self, state: HybridAMRState, dt: float, reason: str,
        *, safe_score: float, direction_lower: float, cfl: float,
        phase_seconds: dict[str, float] | None = None,
        raw_residual: np.ndarray | None = None,
        risk_residual: np.ndarray | None = None,
        upper: np.ndarray | None = None,
        direction: np.ndarray | None = None,
        unsafe: np.ndarray | None = None,
        shrink_dt: bool = True,
        defect_safe_score: float | None = None,
        selector_component: float | None = None,
        selector_family_index: int = 0,
        deployment_features: np.ndarray | None = None,
        deployment_unsafe_members: np.ndarray | None = None,
        deployment_unsafe_upper: float = float("nan"),
        deployment_defect_upper: float = float("nan"),
        deployment_direction_lower: float = float("nan"),
        deployment_support_ratio: float = float("inf"),
    ) -> tuple[HybridAMRState, M6342StepDiagnostics]:
        self._safe_streak = 0
        started = perf_counter()
        classical, diagnostic = HybridMCHAMRSolver(
            self.grid, self.parameters, dt,
            maximum_cfl=self.config.maximum_cfl,
            collision_tolerance=1.0e-12,
        ).step(state)
        phases = dict(phase_seconds or {})
        phases["fallback_classical"] = perf_counter() - started
        next_dt = (
            min(dt, max(self.config.minimum_dt, self.config.shrink_factor * dt))
            if shrink_dt else dt
        )
        return classical, M6342StepDiagnostics(
            dt, next_dt,
            False, True, reason, 0.0, safe_score,
            self.config.safe_selector_threshold, direction_lower,
            diagnostic.composite_mass_drift,
            diagnostic.composite_relative_energy_drift,
            cfl, 0, 1, phases,
            False, 0.0, 0.0, 0.0, 0.0, 0, "shrink" if shrink_dt else "hold",
            tuple(np.asarray(raw_residual if raw_residual is not None else (), dtype=float)),
            tuple(np.asarray(risk_residual if risk_residual is not None else (), dtype=float)),
            tuple(np.asarray(upper if upper is not None else (), dtype=float)),
            tuple(np.asarray(direction if direction is not None else (), dtype=float)),
            safe_score - self.config.safe_selector_threshold,
            self.config.risk_feature_version,
            tuple(np.asarray(unsafe if unsafe is not None else (), dtype=float)),
            safe_score if defect_safe_score is None else defect_safe_score,
            safe_score if selector_component is None else selector_component,
            selector_family_index,
            tuple(np.asarray(
                deployment_features if deployment_features is not None else (),
                dtype=float,
            )),
            tuple(np.asarray(
                deployment_unsafe_members
                if deployment_unsafe_members is not None else (), dtype=float,
            )),
            deployment_unsafe_upper,
            deployment_defect_upper,
            deployment_direction_lower,
            deployment_support_ratio,
        )

    def _cheap_backbone(
        self, state: HybridAMRState, proposal: HybridAMRState, dt: float,
        cache: M6342ProposalCache,
    ) -> HybridAMRState:
        green = HierarchicalPeriodicGreenSolver(self.grid)
        initial_fine = cache.initial_fine_total
        initial_mass = float(green.fine_grid.h * np.sum(initial_fine))
        coarse_state = HybridMCHState(
            state.positions, state.amplitudes,
            state.composite_field.coarse_momentum,
        )
        coarse_total = hybrid_total_state(coarse_state, self.grid)
        target_coarse_mass = float(self.grid.h * np.sum(coarse_total))
        target_coarse_energy = discrete_energy(
            coarse_total, HelmholtzOperator(self.grid)
        )
        coarse_flux, coarse_source = hybrid_flux_and_source(
            coarse_state, self.grid, self.parameters
        )
        coarse_provisional = (
            state.composite_field.coarse_momentum
            - dt * finite_volume_divergence(coarse_flux, self.grid.h)
            + dt * coarse_source
        )
        fine_state = cache.before_fine_state
        fine_flux, fine_source = cache.flux0, cache.source0
        proposal_fine_state = cache.after_fine_state
        proposal_flux, proposal_source = cache.flux1, cache.source1
        fine_rhs = (
            -finite_volume_divergence(
                0.5 * (fine_flux + proposal_flux), green.fine_grid.h
            )
            + 0.5 * (fine_source + proposal_source)
        )
        # One-step-lagged causal support: the current certified patch is used
        # for this advance. The proposal patch is a next-step signal and must
        # not create an instantaneous representation switch.
        patch = state.composite_field.patch
        fine_momentum = np.where(
            patch.fine_mask,
            fine_state.field_momentum + dt * fine_rhs,
            fine_state.field_momentum,
        )
        velocity, proposal_velocity = cache.velocity0, cache.velocity1
        positions = np.remainder(
            state.positions + 0.5 * dt * (velocity + proposal_velocity),
            self.grid.length,
        )
        synchronized = reflux_two_to_one(
            coarse_provisional, fine_momentum, coarse_flux,
            0.5 * (fine_flux + proposal_flux),
            patch.coarse_mask, dt=dt, coarse_h=self.grid.h,
        )
        total = hybrid_total_state(
            HybridMCHState(positions, state.amplitudes, synchronized),
            self.grid,
        )
        projected = project_to_mass_h1(
            total, target_coarse_mass, target_coarse_energy,
            ModifiedCH(self.grid, self.parameters),
        ).state
        atomic = (
            reconstruct_multipeakon(
                self.grid, positions, state.amplitudes
            ) if positions.size else np.zeros(self.grid.points)
        )
        coarse_momentum = HelmholtzOperator(self.grid).apply(projected - atomic)
        detail = np.where(
            patch.fine_mask,
            fine_momentum - green.base_fine_momentum(coarse_momentum),
            0.0,
        )
        backbone = HybridAMRState(
            positions, state.amplitudes.copy(),
            CompositeMomentumState(
                coarse_momentum, detail, patch,
            ),
        )
        mask = patch.fine_mask
        if np.any(mask):
            mass_defect = initial_mass - float(
                green.fine_grid.h * np.sum(_fine_total(backbone, green))
            )
            detail[mask] += mass_defect / (
                green.fine_grid.h * np.count_nonzero(mask)
            )
            backbone = HybridAMRState(
                backbone.positions, backbone.amplitudes,
                CompositeMomentumState(
                    coarse_momentum, detail, patch,
                ),
            )
        return backbone

    def _commit_next_patch(
        self,
        candidate: HybridAMRState,
        proposal: HybridAMRState,
        initial: HybridAMRState | None = None,
    ) -> tuple[HybridAMRState, bool, float, float, float, float]:
        """Causally migrate the completed state onto the next-step patch.

        Newly entered cells start with zero hierarchical detail and are
        evolved on the fine level from the following step.  Leaving-cell
        child means are first synchronized into the coarse momentum.  The
        migration is rejected if its represented-field H1 defect is too large.
        """
        old_patch = candidate.composite_field.patch
        proposed_patch = proposal.composite_field.patch
        if np.array_equal(old_patch.coarse_mask, proposed_patch.coarse_mask):
            return candidate, False, 0.0, 0.0, 0.0, 0.0
        old = old_patch.coarse_mask
        new = proposed_patch.coarse_mask
        entered = new & ~old
        exited = old & ~new
        target_patch = PeriodicRefinementPatch(
            new.copy(), np.repeat(new, 2),
            proposed_patch.marked_energy_fraction,
            float(np.mean(new)),
        )
        green = HierarchicalPeriodicGreenSolver(self.grid)
        before_total = _fine_total(candidate, green)
        represented_momentum = green.reconstruct_fine_momentum(
            candidate.composite_field
        )
        coarse = candidate.composite_field.coarse_momentum.copy()
        if np.any(exited):
            child_mean = 0.5 * (
                represented_momentum[0::2] + represented_momentum[1::2]
            )
            coarse[exited] = child_mean[exited]
        initial_coarse = HybridMCHState(
            candidate.positions, candidate.amplitudes,
            candidate.composite_field.coarse_momentum,
        )
        target_mass = float(self.grid.h * np.sum(
            hybrid_total_state(initial_coarse, self.grid)
        ))
        target_energy = discrete_energy(
            hybrid_total_state(initial_coarse, self.grid),
            HelmholtzOperator(self.grid),
        )
        provisional_total = hybrid_total_state(
            HybridMCHState(candidate.positions, candidate.amplitudes, coarse),
            self.grid,
        )
        projected = project_to_mass_h1(
            provisional_total, target_mass, target_energy,
            ModifiedCH(self.grid, self.parameters),
        ).state
        atomic = (
            reconstruct_multipeakon(
                self.grid, candidate.positions, candidate.amplitudes
            ) if candidate.positions.size else np.zeros(self.grid.points)
        )
        projected_momentum = HelmholtzOperator(self.grid).apply(projected - atomic)
        new_base = green.base_fine_momentum(projected_momentum)
        overlap = np.repeat(old & new, 2)
        detail = np.where(overlap, represented_momentum - new_base, 0.0)
        migrated = HybridAMRState(
            candidate.positions, candidate.amplitudes.copy(),
            CompositeMomentumState(projected_momentum, detail, target_patch),
        )
        fine_mask = target_patch.fine_mask
        if np.any(fine_mask):
            mass_defect = float(green.fine_grid.h * np.sum(
                before_total - _fine_total(migrated, green)
            ))
            detail = detail.copy()
            detail[fine_mask] += mass_defect / (
                green.fine_grid.h * np.count_nonzero(fine_mask)
            )
            migrated = HybridAMRState(
                migrated.positions, migrated.amplitudes,
                CompositeMomentumState(projected_momentum, detail, target_patch),
            )
        after_total = _fine_total(migrated, green)
        operator = HelmholtzOperator(green.fine_grid)
        migration = a_norm(after_total - before_total, operator) / max(
            a_norm(before_total, operator), np.finfo(float).eps
        )
        migration_to_increment = 0.0
        if initial is not None:
            initial_total = _fine_total(initial, green)
            migration_to_increment = a_norm(
                after_total - before_total, operator
            ) / max(
                a_norm(before_total - initial_total, operator),
                np.sqrt(np.finfo(float).eps),
            )
        if (
            not np.all(np.isfinite(after_total))
            or migration > self.config.maximum_patch_migration_relative_h1
            or migration_to_increment
            > self.config.maximum_patch_migration_to_step_increment
        ):
            return (
                candidate, False, 0.0, 0.0, float(migration),
                float(migration_to_increment),
            )
        return (
            migrated, True, float(np.mean(entered)),
            float(np.mean(exited)), float(migration),
            float(migration_to_increment),
        )

    def _next_dt_after_accept(
        self, dt: float, *, safe_score: float, cfl: float, energy_drift: float,
        deployment_margin: float | None = None,
    ) -> tuple[float, str]:
        risk_comfortable = (
            deployment_margin >= self.config.growth_risk_margin
            if deployment_margin is not None else
            safe_score >= self.config.safe_selector_threshold
            + self.config.growth_risk_margin
        )
        comfortably_safe = (
            risk_comfortable
            and cfl <= self.config.growth_cfl_fraction * self.config.maximum_cfl
            and energy_drift <= 0.5 * self.config.maximum_relative_energy_drift
        )
        self._safe_streak = self._safe_streak + 1 if comfortably_safe else 0
        if (
            not self.config.enable_time_growth
            or self._safe_streak < self.config.growth_patience
            or dt >= self.config.maximum_dt
        ):
            return dt, "hold"
        cfl_limit = (
            self.config.maximum_dt if cfl <= np.finfo(float).eps else
            dt * self.config.growth_cfl_fraction * self.config.maximum_cfl / cfl
        )
        next_dt = min(
            self.config.maximum_dt,
            self.config.growth_factor * dt,
            cfl_limit,
        )
        if next_dt <= dt * (1.0 + 1.0e-12):
            return dt, "hold"
        self._safe_streak = 0
        return next_dt, "grow"

    def _build_proposal_cache(
        self,
        state: HybridAMRState,
        proposal: HybridAMRState,
        dt: float,
        ensemble: M633EnsembleCache,
    ) -> M6342ProposalCache:
        green = HierarchicalPeriodicGreenSolver(self.grid)
        fine = green.fine_grid
        before = HybridMCHState(
            state.positions, state.amplitudes,
            green.reconstruct_fine_momentum(state.composite_field),
        )
        after = HybridMCHState(
            proposal.positions, proposal.amplitudes,
            green.reconstruct_fine_momentum(proposal.composite_field),
        )
        flux0, source0 = hybrid_flux_and_source(before, fine, self.parameters)
        flux1, source1 = hybrid_flux_and_source(after, fine, self.parameters)
        rhs = -finite_volume_divergence(0.5 * (flux0 + flux1), fine.h)
        rhs += 0.5 * (source0 + source1)
        eulerian = (after.field_momentum - before.field_momentum) / dt - rhs
        helmholtz = HelmholtzOperator(fine)
        if before.positions.size:
            velocity0 = hybrid_particle_velocity(
                before, fine, alpha=self.parameters.alpha
            )
            velocity1 = hybrid_particle_velocity(
                after, fine, alpha=self.parameters.alpha
            )
            displacement = (
                after.positions - before.positions + 0.5 * self.grid.length
            ) % self.grid.length - 0.5 * self.grid.length
            mid_velocity = 0.5 * (velocity0 + velocity1)
            particle_residual = displacement / dt - mid_velocity
        else:
            velocity0 = velocity1 = np.zeros(0)
            displacement = mid_velocity = particle_residual = np.zeros(0)
        coarse = HybridMCHState(
            proposal.positions, proposal.amplitudes,
            proposal.composite_field.coarse_momentum,
        )
        coarse_flux, _ = hybrid_flux_and_source(coarse, self.grid, self.parameters)
        mismatch = flux1[1::2] - coarse_flux
        mask = proposal.composite_field.patch.coarse_mask
        boundary = mask != np.roll(mask, -1)
        interface = mismatch[boundary] if np.any(boundary) else np.zeros(0)
        initial_total = _fine_total(state, green)
        slope = first_derivative(initial_total, fine)
        cfl = float(dt * np.max(np.abs(
            self.parameters.alpha * (initial_total**2 - slope**2)
        )) / fine.h)
        feature_pair = proposal_residual_features(
            eulerian_residual=eulerian,
            field_increment=after.field_momentum - before.field_momentum,
            eulerian_rhs=rhs,
            operator=helmholtz,
            dt=dt,
            particle_residual=particle_residual,
            particle_displacement=displacement,
            particle_mid_velocity=mid_velocity,
            amplitude_increment=after.amplitudes - before.amplitudes,
            amplitude_reference=before.amplitudes,
            interface_mismatch=interface,
            maximum_cfl=cfl,
            active_fraction=proposal.composite_field.patch.active_fraction,
        )
        residual = select_residual_features(
            feature_pair, self.config.risk_feature_version
        )
        return M6342ProposalCache(
            ensemble, before, after, initial_total,
            flux0, source0, flux1, source1,
            velocity0, velocity1, residual,
            feature_pair.raw, feature_pair.scale_consistent,
        )

    def _blend_and_project(
        self, initial: HybridAMRState, backbone: HybridAMRState,
        proposal: HybridAMRState, weight: float,
    ) -> HybridAMRState:
        green = HierarchicalPeriodicGreenSolver(self.grid)
        length = self.grid.length
        displacement = (
            proposal.positions - backbone.positions + 0.5 * length
        ) % length - 0.5 * length
        positions = (backbone.positions + weight * displacement) % length
        amplitudes = (
            (1.0 - weight) * backbone.amplitudes + weight * proposal.amplitudes
        )
        coarse = (
            (1.0 - weight) * backbone.composite_field.coarse_momentum
            + weight * proposal.composite_field.coarse_momentum
        )
        detail = (
            (1.0 - weight) * backbone.composite_field.fine_detail_momentum
            + weight * proposal.composite_field.fine_detail_momentum
        )
        patch = backbone.composite_field.patch
        mask = patch.fine_mask
        detail = np.where(mask, detail, 0.0)
        represented = green.base_fine_momentum(coarse) + detail
        initial_coarse = HybridMCHState(
            initial.positions, initial.amplitudes,
            initial.composite_field.coarse_momentum,
        )
        initial_total = hybrid_total_state(initial_coarse, self.grid)
        target_mass = float(self.grid.h * np.sum(initial_total))
        target_energy = discrete_energy(
            initial_total, HelmholtzOperator(self.grid)
        )
        total = hybrid_total_state(
            HybridMCHState(positions, amplitudes, coarse), self.grid
        )
        projected = project_to_mass_h1(
            total, target_mass, target_energy,
            ModifiedCH(self.grid, self.parameters),
        ).state
        atomic = (
            reconstruct_multipeakon(self.grid, positions, amplitudes)
            if positions.size else np.zeros(self.grid.points)
        )
        projected_momentum = HelmholtzOperator(self.grid).apply(projected - atomic)
        detail = np.where(
            mask,
            represented - green.base_fine_momentum(projected_momentum),
            0.0,
        )
        candidate = HybridAMRState(
            positions, amplitudes,
            CompositeMomentumState(
                projected_momentum, detail, patch
            ),
        )
        initial_fine = _fine_total(initial, green)
        mass_defect = float(
            green.fine_grid.h * np.sum(initial_fine - _fine_total(candidate, green))
        )
        if np.any(mask):
            detail = detail.copy()
            detail[mask] += mass_defect / (
                green.fine_grid.h * np.count_nonzero(mask)
            )
            candidate = HybridAMRState(
                positions, amplitudes,
                CompositeMomentumState(
                    projected_momentum, detail,
                    patch,
                ),
            )
        return candidate

    def _deployment_candidate_features(
        self,
        state: HybridAMRState,
        raw: HybridAMRState,
        backbone: HybridAMRState,
        candidate: HybridAMRState,
        cache: M6342ProposalCache,
        ensemble_cache: M633EnsembleCache,
        *,
        dt: float,
        upper: np.ndarray,
        direction: np.ndarray,
        unsafe: np.ndarray,
        patch_entered: float,
        patch_exited: float,
        migration_defect: float,
        mass_drift: float,
        energy_drift: float,
    ) -> np.ndarray:
        """Causal features of the state that would actually be committed."""
        green = HierarchicalPeriodicGreenSolver(self.grid)
        operator = HelmholtzOperator(green.fine_grid)
        initial_fine = cache.initial_fine_total
        raw_fine = _fine_total(raw, green)
        backbone_fine = _fine_total(backbone, green)
        candidate_fine = _fine_total(candidate, green)
        increment_scale = max(
            a_norm(backbone_fine - initial_fine, operator),
            np.sqrt(np.finfo(float).eps),
        )
        distances = np.asarray((
            a_norm(raw_fine - backbone_fine, operator) / increment_scale,
            a_norm(candidate_fine - backbone_fine, operator) / increment_scale,
            a_norm(candidate_fine - raw_fine, operator) / increment_scale,
        ))
        candidate_cache = self._build_proposal_cache(
            state, candidate, dt, ensemble_cache
        )
        summary = np.asarray((
            float(np.mean(upper)), float(np.std(upper)),
            float(np.mean(direction)), float(np.std(direction)),
            float(np.mean(unsafe)), float(np.std(unsafe)),
        ))
        features = np.concatenate((
            cache.scale_consistent_residual_features,
            candidate_cache.scale_consistent_residual_features,
            summary,
            distances,
            np.asarray((patch_entered, patch_exited, migration_defect)),
            np.asarray((mass_drift, energy_drift, cache.residual_features[-2])),
            np.asarray((
                np.log1p(dt / self.grid.h),
                np.log1p(1.0 / self.grid.h),
                float(state.positions.size),
            )),
        )).astype(np.float64, copy=False)
        if features.shape != (32,) or not np.all(np.isfinite(features)):
            raise FloatingPointError(
                "Deployment candidate features must be 32 finite coordinates."
            )
        return features

    @torch.no_grad()
    def step(
        self, state: HybridAMRState, *, dt: float,
    ) -> tuple[HybridAMRState, M6342StepDiagnostics]:
        phase_start = perf_counter()
        ensemble_cache = self._proposal_solver._ensemble(state, dt)
        raw, proposal_diagnostic = self._proposal_solver.step(
            state, dt=dt, enable_calibration=False,
            enable_shadow_audit=False, enable_fallback=False,
            ensemble_cache=ensemble_cache,
        )
        proposal_seconds = perf_counter() - phase_start
        phase_start = perf_counter()
        feature_map = ensemble_cache.features
        parameter_vector = ensemble_cache.parameters
        ensemble = ensemble_cache.summary()
        context = np.concatenate((
            _summarize_feature_map(feature_map), ensemble, parameter_vector
        ))
        cache = self._build_proposal_cache(state, raw, dt, ensemble_cache)
        residual = cache.residual_features
        context_tensor = torch.as_tensor(
            context[None], dtype=torch.float32, device=self.device
        )
        residual_tensor = torch.as_tensor(
            residual[None], dtype=torch.float32, device=self.device
        )
        risk_outputs = [
            model(context_tensor, residual_tensor) for model in self.risk_models
        ]
        upper = np.asarray([
            float(row.upper_log_defect[0].cpu()) for row in risk_outputs
        ])
        direction = np.asarray([
            float(row.direction[0].cpu()) for row in risk_outputs
        ])
        unsafe = np.asarray([
            float(row.unsafe_logit[0].cpu()) for row in risk_outputs
        ])
        defect_safe_score = -float(np.mean(upper))
        direction_lower = float(np.mean(direction) - self.config.direction_correction)
        family_index = (
            0 if state.positions.size == 0 else
            1 if state.positions.size <= 2 else 2
        )
        selector_component = defect_safe_score
        selector_threshold = self.config.safe_selector_threshold
        if self.config.selector_score_mode == "family_hybrid_v2":
            if family_index == 0:
                selector_component = -float(np.mean(unsafe))
            elif family_index == 1:
                selector_component = direction_lower
            else:
                selector_component = defect_safe_score + 0.5 * direction_lower
            selector_threshold = self.config.selector_family_thresholds[family_index]
            safe_score = selector_component - selector_threshold
            selector_threshold = 0.0
        else:
            safe_score = defect_safe_score
        cfl = float(residual[-2])
        risk_seconds = perf_counter() - phase_start
        pre_gate_phases = {
            "proposal": proposal_seconds,
            "risk_and_residual": risk_seconds,
        }
        if (
            not self.config.enable_deployment_selector
            and safe_score < selector_threshold
        ):
            return self._fallback(
                state, dt, "risk_selector", safe_score=safe_score,
                direction_lower=direction_lower, cfl=cfl,
                phase_seconds=pre_gate_phases,
                raw_residual=cache.raw_residual_features,
                risk_residual=residual, upper=upper, direction=direction,
                unsafe=unsafe,
                shrink_dt=not self.config.hold_dt_on_proposal_risk,
                defect_safe_score=defect_safe_score,
                selector_component=selector_component,
                selector_family_index=family_index,
            )
        if (
            not self.config.enable_deployment_selector
            and self.config.enable_direction_gate
            and direction_lower < self.config.minimum_direction_cosine
        ):
            return self._fallback(
                state, dt, "direction", safe_score=safe_score,
                direction_lower=direction_lower, cfl=cfl,
                phase_seconds=pre_gate_phases,
                raw_residual=cache.raw_residual_features,
                risk_residual=residual, upper=upper, direction=direction,
                unsafe=unsafe,
                shrink_dt=not self.config.hold_dt_on_proposal_risk,
                defect_safe_score=defect_safe_score,
                selector_component=selector_component,
                selector_family_index=family_index,
            )
        if cfl > self.config.maximum_cfl:
            return self._fallback(
                state, dt, "cfl", safe_score=safe_score,
                direction_lower=direction_lower, cfl=cfl,
                phase_seconds=pre_gate_phases,
                raw_residual=cache.raw_residual_features,
                risk_residual=residual, upper=upper, direction=direction,
                unsafe=unsafe,
                shrink_dt=True,
                defect_safe_score=defect_safe_score,
                selector_component=selector_component,
                selector_family_index=family_index,
            )
        phase_start = perf_counter()
        backbone = self._cheap_backbone(state, raw, dt, cache)
        candidate = self._blend_and_project(
            state, backbone, raw, self.config.selected_weight
        )
        patch_changed = False
        entered_fraction = exited_fraction = migration_defect = 0.0
        migration_to_increment = 0.0
        if self.config.enable_patch_commit:
            (
                candidate, patch_changed, entered_fraction,
                exited_fraction, migration_defect, migration_to_increment,
            ) = self._commit_next_patch(candidate, raw, state)
        backbone_seconds = perf_counter() - phase_start
        green = HierarchicalPeriodicGreenSolver(self.grid)
        initial_fine, final_fine = _fine_total(state, green), _fine_total(candidate, green)
        operator = HelmholtzOperator(green.fine_grid)
        mass_drift = abs(float(
            green.fine_grid.h * np.sum(final_fine - initial_fine)
        ))
        initial_energy = discrete_energy(initial_fine, operator)
        energy_drift = abs(
            discrete_energy(final_fine, operator) - initial_energy
        ) / max(initial_energy, np.finfo(float).eps)
        reason = None
        if not np.all(np.isfinite(final_fine)):
            reason = "nonfinite"
        elif mass_drift > self.config.maximum_mass_drift:
            reason = "mass"
        elif energy_drift > self.config.maximum_relative_energy_drift:
            reason = "energy"
        if reason is not None:
            return self._fallback(
                state, dt, reason, safe_score=safe_score,
                direction_lower=direction_lower, cfl=cfl,
                phase_seconds={
                    **pre_gate_phases,
                    "backbone_and_projection": backbone_seconds,
                },
                raw_residual=cache.raw_residual_features,
                risk_residual=residual, upper=upper, direction=direction,
                unsafe=unsafe,
                shrink_dt=reason in {"nonfinite", "mass", "energy"},
                defect_safe_score=defect_safe_score,
                selector_component=selector_component,
                selector_family_index=family_index,
            )
        deployment_features = np.empty(0, dtype=np.float64)
        deployment_members = np.empty(0, dtype=np.float64)
        deployment_upper = float("nan")
        deployment_defect_upper = float("nan")
        deployment_direction_lower = float("nan")
        deployment_support_ratio = float("inf")
        deployment_threshold = self.config.deployment_selector_threshold
        if (
            self.config.collect_deployment_features
            or self.config.enable_deployment_selector
        ):
            deployment_features = self._deployment_candidate_features(
                state, raw, backbone, candidate, cache, ensemble_cache,
                dt=dt, upper=upper, direction=direction, unsafe=unsafe,
                patch_entered=entered_fraction, patch_exited=exited_fraction,
                migration_defect=max(migration_defect, migration_to_increment),
                mass_drift=mass_drift, energy_drift=energy_drift,
            )
        if self.config.enable_deployment_selector:
            deployment_tensor = torch.as_tensor(
                deployment_features[None], dtype=torch.float32,
                device=self.device,
            )
            deployment_outputs = [
                model(deployment_tensor) for model in self.deployment_models
            ]
            deployment_members = np.asarray([
                float(torch.sigmoid(row.unsafe_logit[0]).cpu())
                for row in deployment_outputs
            ])
            probability_upper = float(
                np.mean(deployment_members)
                + self.config.deployment_ensemble_sigma
                * np.std(deployment_members)
            )
            defect_members = np.asarray([
                float(row.log_defect[0].cpu()) for row in deployment_outputs
            ])
            direction_members = np.asarray([
                float(row.direction[0].cpu()) for row in deployment_outputs
            ])
            deployment_defect_upper = float(
                np.mean(defect_members)
                + self.config.deployment_ensemble_sigma * np.std(defect_members)
                + self.config.deployment_defect_correction
            )
            deployment_direction_lower = float(
                np.mean(direction_members)
                - self.config.deployment_ensemble_sigma * np.std(direction_members)
                - self.config.deployment_direction_correction
            )
            defect_vote = float(1.0 / (1.0 + np.exp(-8.0 * (
                deployment_defect_upper - np.log1p(0.1)
            ))))
            direction_vote = float(1.0 / (1.0 + np.exp(8.0 * (
                deployment_direction_lower + 0.05
            ))))
            deployment_upper = max(probability_upper, defect_vote, direction_vote)
            family_name = ("smooth", "separated", "clustered")[family_index]
            try:
                grid_index = self.config.deployment_registered_points.index(
                    int(self.grid.points)
                )
            except ValueError:
                grid_index = -1
            if grid_index >= 0:
                threshold_index = (
                    family_index * len(self.config.deployment_registered_points)
                    + grid_index
                )
                deployment_threshold = self.config.deployment_selector_thresholds[
                    threshold_index
                ]
                deployment_support_ratio = float(
                    self.deployment_certificate.support_ratio(
                        np.asarray([[
                            probability_upper, deployment_defect_upper,
                            deployment_direction_lower,
                        ]]), np.asarray([family_name]),
                        np.asarray([self.grid.points]),
                    )[0]
                )
            selector_reason = (
                "deployment_unregistered_grid" if grid_index < 0 else
                "deployment_support" if deployment_support_ratio > 1.0 else
                "deployment_selector" if deployment_upper > deployment_threshold else
                None
            )
            if selector_reason is not None:
                return self._fallback(
                    state, dt, selector_reason,
                    safe_score=safe_score,
                    direction_lower=direction_lower, cfl=cfl,
                    phase_seconds={
                        **pre_gate_phases,
                        "backbone_and_projection": backbone_seconds,
                    },
                    raw_residual=cache.raw_residual_features,
                    risk_residual=residual, upper=upper,
                    direction=direction, unsafe=unsafe,
                    shrink_dt=False,
                    defect_safe_score=defect_safe_score,
                    selector_component=selector_component,
                    selector_family_index=family_index,
                    deployment_features=deployment_features,
                    deployment_unsafe_members=deployment_members,
                    deployment_unsafe_upper=deployment_upper,
                    deployment_defect_upper=deployment_defect_upper,
                    deployment_direction_lower=deployment_direction_lower,
                    deployment_support_ratio=deployment_support_ratio,
                )
        next_dt, time_action = self._next_dt_after_accept(
            dt, safe_score=safe_score, cfl=cfl, energy_drift=energy_drift,
            deployment_margin=(
                deployment_threshold - deployment_upper
                if self.config.enable_deployment_selector else None
            ),
        )
        return candidate, M6342StepDiagnostics(
            dt, next_dt, True, False, None, self.config.selected_weight,
            safe_score, self.config.safe_selector_threshold,
            direction_lower, mass_drift, energy_drift, cfl, 0, 0,
            {
                "proposal": proposal_seconds,
                "risk_and_residual": risk_seconds,
                "backbone_and_projection": backbone_seconds,
            },
            patch_changed, entered_fraction, exited_fraction,
            migration_defect, migration_to_increment,
            self._safe_streak, time_action,
            tuple(cache.raw_residual_features), tuple(residual),
            tuple(upper), tuple(direction),
            safe_score - self.config.safe_selector_threshold,
            self.config.risk_feature_version,
            tuple(unsafe),
            defect_safe_score, selector_component, family_index,
            tuple(deployment_features), tuple(deployment_members),
            deployment_upper,
            deployment_defect_upper,
            deployment_direction_lower,
            deployment_support_ratio,
        )
