from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from fgsp_ch.datasets.m633_amr_dataset import causal_features, geometry_indicator
from fgsp_ch.discretization.adaptive_mesh import PeriodicRefinementPatch, dorfler_mark
from fgsp_ch.discretization.difference import first_derivative
from fgsp_ch.discretization.grids import PeriodicGrid
from fgsp_ch.discretization.helmholtz import HelmholtzOperator
from fgsp_ch.equations.modified_ch import ModifiedCH, ModifiedCHParameters
from fgsp_ch.geometry.metric import a_norm, discrete_energy
from fgsp_ch.models.m633_local_amr_operator import ConservativeLocalAMROperator
from fgsp_ch.solvers.mch_energy_projected import project_to_mass_h1
from fgsp_ch.solvers.mch_hybrid_amr import (
    CompositeMomentumState,
    HierarchicalPeriodicGreenSolver,
    HybridAMRState,
    HybridMCHAMRSolver,
    HybridMCHState,
    UniformHybridMCHReferenceSolver,
    hybrid_total_state,
)
from fgsp_ch.solvers.multipeakon import reconstruct_multipeakon
from fgsp_ch.training.m633_risk_calibration import M633RiskCalibration


@dataclass(frozen=True, slots=True)
class M633SafeConfig:
    theta: float = 0.6
    maximum_fraction: float = 0.4
    halo_cells: int = 1
    refinement_tolerance: float = 1.0
    detail_tolerance: float = 1.0
    face_tolerance: float = 1.0
    minimum_dt: float = 5.0e-5
    shrink_factor: float = 0.5
    maximum_cfl: float = 0.45
    collision_tolerance: float = 1.0e-8
    cluster_guard_cells: float = 4.0
    maximum_shadow_increment_ratio: float = 0.1
    maximum_composite_energy_drift: float = 0.01
    minimum_direction_cosine: float = -0.05
    minimum_shadow_blend: float = 0.0625
    enable_time_consistent_detail: bool = False
    proposal_reference_dt: float = 2.0e-4

    def __post_init__(self) -> None:
        if not 0.0 < self.theta <= 1.0 or not 0.0 < self.maximum_fraction <= 1.0:
            raise ValueError("Dörfler parameters must lie in (0, 1].")
        if self.halo_cells < 0 or self.minimum_dt <= 0.0:
            raise ValueError("halo_cells must be nonnegative and minimum_dt positive.")
        if not 0.0 < self.shrink_factor <= 1.0:
            raise ValueError("shrink_factor must lie in (0, 1].")
        if min(
            self.refinement_tolerance, self.detail_tolerance, self.face_tolerance,
            self.maximum_cfl, self.maximum_shadow_increment_ratio,
            self.maximum_composite_energy_drift,
            self.minimum_shadow_blend,
        ) <= 0.0:
            raise ValueError("Safety tolerances must be positive.")
        if self.minimum_shadow_blend > 0.5:
            raise ValueError("minimum_shadow_blend must not exceed 0.5.")
        if self.proposal_reference_dt <= 0.0:
            raise ValueError("proposal_reference_dt must be positive.")


@dataclass(frozen=True, slots=True)
class M633SafeStepDiagnostics:
    dt: float
    next_dt: float
    family: str
    learned: bool
    fallback: bool
    fallback_reason: str | None
    refinement_used: bool
    detail_used: bool
    face_used: bool
    budget_saturated: bool
    active_fraction: float
    risk_upper_refinement: float
    risk_upper_detail: float
    risk_upper_face: float
    maximum_cfl: float
    minimum_separation: float
    composite_mass_drift: float
    composite_relative_energy_drift: float
    shadow_increment_ratio: float
    direction_cosine: float
    learned_weight: float


@dataclass(frozen=True, slots=True)
class M633EnsembleCache:
    """Single-step neural outputs shared by proposal and risk evaluation."""

    features: np.ndarray
    parameters: np.ndarray
    refinement: torch.Tensor
    detail: torch.Tensor
    face: torch.Tensor
    state_identity: int

    def summary(self) -> np.ndarray:
        result: list[float] = []
        for values in (self.refinement, self.detail, self.face):
            mean = values.mean(dim=0)
            deviation = values.var(dim=0, unbiased=False).sqrt()
            mean_rms = mean.square().mean().sqrt()
            deviation_rms = deviation.square().mean().sqrt()
            result.extend((
                float(mean_rms.cpu()),
                float(deviation_rms.cpu()),
                float((deviation_rms / mean_rms.clamp_min(1.0e-12)).cpu()),
            ))
        return np.asarray(result, dtype=np.float64)


def _minimum_separation(q: np.ndarray, length: float) -> float:
    if q.size < 2:
        return length
    distance = np.abs((q[:, None] - q[None, :] + 0.5 * length) % length - 0.5 * length)
    distance[np.eye(q.size, dtype=bool)] = np.inf
    return float(np.min(distance))


def _family(state: HybridAMRState, grid: PeriodicGrid, cluster_cells: float) -> str:
    if state.positions.size == 0:
        return "smooth"
    if _minimum_separation(state.positions, grid.length) <= cluster_cells * grid.h:
        return "clustered"
    return "separated"


def _fine_total(state: HybridAMRState, green: HierarchicalPeriodicGreenSolver) -> np.ndarray:
    total = green.solve(state.composite_field)
    if state.positions.size:
        total = total + reconstruct_multipeakon(
            green.fine_grid, state.positions, state.amplitudes
        )
    return np.asarray(total)


class CertifiedLocalAMRSolver:
    def __init__(
        self,
        models: list[ConservativeLocalAMROperator],
        calibration: M633RiskCalibration,
        grid: PeriodicGrid,
        parameters: ModifiedCHParameters,
        *,
        device: torch.device,
        config: M633SafeConfig | None = None,
    ) -> None:
        if len(models) < 3:
            raise ValueError("Certified rollout requires at least three ensemble members.")
        self.models = [model.eval() for model in models]
        self.calibration = calibration
        self.grid = grid
        self.parameters = parameters
        self.device = device
        self.config = config or M633SafeConfig()

    @torch.no_grad()
    def _ensemble(self, state: HybridAMRState, dt: float) -> M633EnsembleCache:
        features = causal_features(state, self.grid, self.parameters)
        parameters = np.asarray([
            self.parameters.alpha, self.parameters.gamma, self.grid.length,
            dt, self.grid.h, float(state.positions.size),
        ])
        x = torch.as_tensor(features[None], dtype=torch.float32, device=self.device)
        p = torch.as_tensor(parameters[None], dtype=torch.float32, device=self.device)
        outputs = [model(x, p) for model in self.models]
        refinement = torch.stack([row.refinement_score[0] for row in outputs])
        detail = torch.stack([row.fine_detail[0] for row in outputs])
        face = torch.stack([row.face_flux_mismatch[0] for row in outputs])
        return M633EnsembleCache(
            features, parameters, refinement, detail, face, id(state)
        )

    @staticmethod
    def _raw_risk(values: torch.Tensor, mask: torch.Tensor | None = None) -> float:
        mean = values.mean(dim=0)
        variance = values.var(dim=0, unbiased=False)
        if mask is not None and torch.any(mask):
            mean, variance = mean[mask], variance[mask]
        numerator = variance.mean().sqrt()
        denominator = mean.square().mean().sqrt().clamp_min(1.0e-12)
        return float((numerator / denominator).cpu())

    def _fallback(
        self, state: HybridAMRState, dt: float, reason: str,
        family: str, risks: tuple[float, float, float], cfl: float,
    ) -> tuple[HybridAMRState, M633SafeStepDiagnostics]:
        classical, diagnostic = HybridMCHAMRSolver(
            self.grid, self.parameters, dt,
            maximum_cfl=self.config.maximum_cfl,
            collision_tolerance=self.config.collision_tolerance,
        ).step(state)
        return classical, M633SafeStepDiagnostics(
            dt, min(dt, max(self.config.minimum_dt, self.config.shrink_factor * dt)),
            family, False, True, reason, False, False, False,
            state.composite_field.patch.marked_energy_fraction < self.config.theta,
            state.composite_field.patch.active_fraction,
            *risks, cfl, diagnostic.minimum_particle_separation,
            diagnostic.composite_mass_drift,
            diagnostic.composite_relative_energy_drift,
            0.0, 1.0, 0.0,
        )

    def _shadow_blend(
        self,
        learned: HybridAMRState,
        shadow: HybridAMRState,
        *,
        weight: float,
        initial_mass: float,
        target_coarse_mass: float,
        target_coarse_energy: float,
        green: HierarchicalPeriodicGreenSolver,
    ) -> HybridAMRState:
        """Blend only the learned residual, then restore the hard invariants.

        The periodic particle displacement is blended on the shortest arc.  A
        union patch is used when it respects the registered AMR budget;
        otherwise the certified shadow patch is retained.  Projection and
        composite mass closure are deliberately repeated for every trial
        weight: convex interpolation alone does not preserve H1 energy.
        """
        if not 0.0 < weight < 1.0:
            raise ValueError("shadow blend weight must lie strictly between zero and one.")
        union = learned.composite_field.patch.coarse_mask | shadow.composite_field.patch.coarse_mask
        if float(np.mean(union)) > self.config.maximum_fraction:
            union = shadow.composite_field.patch.coarse_mask.copy()
        fine_mask = np.repeat(union, 2)
        patch = PeriodicRefinementPatch(
            union, fine_mask,
            min(
                learned.composite_field.patch.marked_energy_fraction,
                shadow.composite_field.patch.marked_energy_fraction,
            ),
            float(np.mean(fine_mask)),
        )

        length = self.grid.length
        displacement = (
            learned.positions - shadow.positions + 0.5 * length
        ) % length - 0.5 * length
        positions = (shadow.positions + weight * displacement) % length
        amplitudes = (
            (1.0 - weight) * shadow.amplitudes + weight * learned.amplitudes
        )
        coarse = (
            (1.0 - weight) * shadow.composite_field.coarse_momentum
            + weight * learned.composite_field.coarse_momentum
        )
        detail = np.where(
            fine_mask,
            (1.0 - weight) * shadow.composite_field.fine_detail_momentum
            + weight * learned.composite_field.fine_detail_momentum,
            0.0,
        )
        # Preserve the represented fine field while re-projecting its coarse
        # component onto the frozen mass/H1 manifold.
        represented_fine_momentum = green.base_fine_momentum(coarse) + detail
        total = hybrid_total_state(
            HybridMCHState(positions, amplitudes, coarse), self.grid
        )
        projected = project_to_mass_h1(
            total, target_coarse_mass, target_coarse_energy,
            ModifiedCH(self.grid, self.parameters),
        ).state
        atomic = (
            reconstruct_multipeakon(self.grid, positions, amplitudes)
            if positions.size else np.zeros(self.grid.points)
        )
        projected_momentum = HelmholtzOperator(self.grid).apply(projected - atomic)
        detail = np.where(
            fine_mask,
            represented_fine_momentum - green.base_fine_momentum(projected_momentum),
            0.0,
        )
        candidate = HybridAMRState(
            positions, amplitudes,
            CompositeMomentumState(projected_momentum, detail, patch),
        )
        if np.any(fine_mask):
            mass_defect = initial_mass - float(
                green.fine_grid.h * np.sum(_fine_total(candidate, green))
            )
            detail = detail.copy()
            detail[fine_mask] += mass_defect / (
                green.fine_grid.h * np.count_nonzero(fine_mask)
            )
            candidate = HybridAMRState(
                positions, amplitudes,
                CompositeMomentumState(projected_momentum, detail, patch),
            )
        return candidate

    @staticmethod
    def _candidate_audit(
        candidate: HybridAMRState,
        shadow: HybridAMRState,
        initial_fine: np.ndarray,
        initial_mass: float,
        initial_energy: float,
        green: HierarchicalPeriodicGreenSolver,
    ) -> tuple[np.ndarray, float, float, float, float]:
        final_fine = _fine_total(candidate, green)
        shadow_fine = _fine_total(shadow, green)
        operator = HelmholtzOperator(green.fine_grid)
        mass_drift = abs(float(green.fine_grid.h * np.sum(final_fine)) - initial_mass)
        energy_drift = abs(discrete_energy(final_fine, operator) - initial_energy) / max(
            initial_energy, np.finfo(float).eps
        )
        shadow_increment = shadow_fine - initial_fine
        learned_increment = final_fine - initial_fine
        shadow_increment_ratio = a_norm(final_fine - shadow_fine, operator) / max(
            a_norm(shadow_increment, operator), np.finfo(float).eps
        )
        inner = float(green.fine_grid.h * np.sum(
            learned_increment * operator.apply(shadow_increment)
        ))
        direction = inner / max(
            a_norm(learned_increment, operator) * a_norm(shadow_increment, operator),
            np.finfo(float).eps,
        )
        return final_fine, mass_drift, energy_drift, shadow_increment_ratio, direction

    def step(
        self, state: HybridAMRState, *, dt: float,
        enable_calibration: bool = True,
        enable_refinement: bool = True,
        enable_detail: bool = True,
        enable_face: bool = True,
        enable_fallback: bool = True,
        enable_geometry_guard: bool = True,
        enable_mass_closure: bool = True,
        enable_shadow_audit: bool = True,
        ensemble_cache: M633EnsembleCache | None = None,
    ) -> tuple[HybridAMRState, M633SafeStepDiagnostics]:
        config = self.config
        green = HierarchicalPeriodicGreenSolver(self.grid)
        family = _family(state, self.grid, config.cluster_guard_cells)
        initial_fine = _fine_total(state, green)
        initial_mass = float(green.fine_grid.h * np.sum(initial_fine))
        initial_energy = discrete_energy(initial_fine, HelmholtzOperator(green.fine_grid))
        slope = first_derivative(initial_fine, green.fine_grid)
        speed = np.max(np.abs(self.parameters.alpha * (initial_fine**2 - slope**2)))
        cfl = float(dt * speed / green.fine_grid.h)
        separation = _minimum_separation(state.positions, self.grid.length)
        cached = ensemble_cache or self._ensemble(state, dt)
        expected_parameters = np.asarray([
            self.parameters.alpha, self.parameters.gamma, self.grid.length,
            dt, self.grid.h, float(state.positions.size),
        ])
        if (
            cached.state_identity != id(state)
            or cached.parameters.shape != (6,)
            or not np.array_equal(cached.parameters, expected_parameters)
            or cached.features.shape[1] != self.grid.points
        ):
            raise ValueError("ensemble_cache is incompatible with this solver step.")
        features = cached.features
        refinement_members = cached.refinement
        detail_members = cached.detail
        face_members = cached.face
        refinement_mean = refinement_members.mean(dim=0).cpu().numpy()
        refinement_std = refinement_members.var(dim=0, unbiased=False).sqrt().cpu().numpy()
        if enable_geometry_guard:
            field = HelmholtzOperator(self.grid).solve(state.composite_field.coarse_momentum)
            geometry = geometry_indicator(
                self.grid, field, state.positions, state.amplitudes
            )
            geometry = geometry / max(float(np.sqrt(np.mean(geometry**2))), np.finfo(float).eps)
            safe_score = refinement_mean + refinement_std + 0.1 * np.mean(refinement_mean) * geometry
        else:
            safe_score = refinement_mean + refinement_std
        patch = dorfler_mark(
            np.maximum(safe_score, 0.0), theta=config.theta,
            maximum_fraction=config.maximum_fraction, halo_cells=config.halo_cells,
        )
        boundary = np.zeros(self.grid.points, dtype=bool)
        for face_index in range(self.grid.points):
            if patch.coarse_mask[face_index] != patch.coarse_mask[(face_index + 1) % self.grid.points]:
                boundary[face_index] = True
        fine_mask_t = torch.as_tensor(patch.fine_mask, device=self.device)
        boundary_t = torch.as_tensor(boundary, device=self.device)
        raw_risks = (
            self._raw_risk(refinement_members),
            self._raw_risk(detail_members, fine_mask_t),
            self._raw_risk(face_members, boundary_t),
        )
        if enable_calibration:
            risks = tuple(
                self.calibration.scale(branch, family)
                * max(raw, self.calibration.raw_floors[branch])
                for branch, raw in zip(("refinement", "detail", "face"), raw_risks, strict=True)
            )
        else:
            risks = raw_risks
        budget_saturated = patch.marked_energy_fraction < config.theta
        refinement_ok = enable_refinement and risks[0] <= config.refinement_tolerance
        detail_ok = enable_detail and risks[1] <= config.detail_tolerance
        face_ok = enable_face and risks[2] <= config.face_tolerance
        guard_reason = None
        if cfl > config.maximum_cfl:
            guard_reason = "cfl"
        elif separation <= max(config.collision_tolerance, config.cluster_guard_cells * self.grid.h):
            guard_reason = "cluster_guard"
        elif budget_saturated and not refinement_ok:
            guard_reason = "budget_saturated"
        elif not refinement_ok:
            guard_reason = "refinement_risk"
        elif not detail_ok:
            guard_reason = "detail_risk"
        elif not face_ok:
            guard_reason = "face_risk"
        if guard_reason is not None:
            if enable_fallback:
                return self._fallback(state, dt, guard_reason, family, risks, cfl)
            raise RuntimeError(f"Unsafe learned AMR proposal without fallback: {guard_reason}.")

        coarse_initial = HybridMCHState(
            state.positions, state.amplitudes, state.composite_field.coarse_momentum
        )
        coarse_next = UniformHybridMCHReferenceSolver(
            self.grid, self.parameters, dt
        ).step(coarse_initial)
        detail_raw = detail_members.mean(dim=0, keepdim=True)
        if config.enable_time_consistent_detail:
            current_detail = torch.as_tensor(
                state.composite_field.fine_detail_momentum[None],
                dtype=detail_raw.dtype, device=detail_raw.device,
            )
            ratio = min(dt / config.proposal_reference_dt, 1.25)
            detail_raw = current_detail + ratio * (detail_raw - current_detail)
        detail = self.models[0].constrain_fine_detail(
            detail_raw,
            torch.as_tensor(patch.coarse_mask[None], device=self.device),
            torch.zeros(1, dtype=torch.float64, device=self.device),
        )[0].cpu().numpy()
        base = green.base_fine_momentum(coarse_next.field_momentum)
        fine_momentum = base + detail
        synchronized = coarse_next.field_momentum.copy()
        synchronized[patch.coarse_mask] = 0.5 * (
            fine_momentum[0::2][patch.coarse_mask]
            + fine_momentum[1::2][patch.coarse_mask]
        )
        face_mean = face_members.mean(dim=0, keepdim=True)
        reflux = self.models[0].conservative_reflux_correction(
            face_mean,
            torch.as_tensor(boundary[None], device=self.device),
            dt_over_h=dt / self.grid.h,
        )[0].cpu().numpy()
        synchronized[~patch.coarse_mask] += reflux[~patch.coarse_mask]

        initial_coarse_total = hybrid_total_state(coarse_initial, self.grid)
        target_coarse_mass = float(self.grid.h * np.sum(initial_coarse_total))
        target_coarse_energy = discrete_energy(initial_coarse_total, HelmholtzOperator(self.grid))
        candidate_coarse = HybridMCHState(
            coarse_next.positions, state.amplitudes, synchronized
        )
        candidate_total = hybrid_total_state(candidate_coarse, self.grid)
        projected = project_to_mass_h1(
            candidate_total, target_coarse_mass, target_coarse_energy,
            ModifiedCH(self.grid, self.parameters),
        ).state
        atomic = (
            reconstruct_multipeakon(self.grid, coarse_next.positions, state.amplitudes)
            if state.positions.size else np.zeros(self.grid.points)
        )
        projected_momentum = HelmholtzOperator(self.grid).apply(projected - atomic)
        new_base = green.base_fine_momentum(projected_momentum)
        detail = np.where(patch.fine_mask, fine_momentum - new_base, 0.0)
        candidate = HybridAMRState(
            coarse_next.positions, state.amplitudes.copy(),
            CompositeMomentumState(projected_momentum, detail, patch),
        )
        if enable_mass_closure and np.any(patch.fine_mask):
            provisional = _fine_total(candidate, green)
            mass_defect = initial_mass - float(green.fine_grid.h * np.sum(provisional))
            detail = candidate.composite_field.fine_detail_momentum.copy()
            detail[patch.fine_mask] += mass_defect / (
                green.fine_grid.h * np.count_nonzero(patch.fine_mask)
            )
            candidate = HybridAMRState(
                candidate.positions, candidate.amplitudes,
                CompositeMomentumState(projected_momentum, detail, patch),
            )
        if not enable_shadow_audit:
            final_fine = _fine_total(candidate, green)
            mass_drift = abs(
                float(green.fine_grid.h * np.sum(final_fine)) - initial_mass
            )
            energy_drift = abs(
                discrete_energy(
                    final_fine, HelmholtzOperator(green.fine_grid)
                ) - initial_energy
            ) / max(initial_energy, np.finfo(float).eps)
            hard_reason = None
            if not np.all(np.isfinite(final_fine)):
                hard_reason = "nonfinite"
            elif mass_drift > 1.0e-10:
                hard_reason = "mass"
            elif energy_drift > config.maximum_composite_energy_drift:
                hard_reason = "energy"
            if hard_reason is not None:
                if enable_fallback:
                    return self._fallback(
                        state, dt, hard_reason, family, risks, cfl
                    )
                raise RuntimeError(
                    f"Learned proposal failed shadow-free hard audit: {hard_reason}."
                )
            return candidate, M633SafeStepDiagnostics(
                dt, dt, family, True, False, None,
                refinement_ok, detail_ok, face_ok, budget_saturated,
                patch.active_fraction, *risks, cfl,
                _minimum_separation(candidate.positions, self.grid.length),
                mass_drift, energy_drift, float("nan"), float("nan"), 1.0,
            )
        # M6.3.3.3 is an audit phase: the classical shadow is intentionally
        # retained for direction and embedding-defect checks. No speedup claim.
        shadow, _ = HybridMCHAMRSolver(
            self.grid, self.parameters, dt,
            maximum_cfl=config.maximum_cfl,
            collision_tolerance=config.collision_tolerance,
        ).step(state)
        learned_weight = 1.0

        def audit(current: HybridAMRState):
            values = self._candidate_audit(
                current, shadow, initial_fine, initial_mass, initial_energy, green
            )
            final, mass, energy, ratio, cosine = values
            reason = None
            if not np.all(np.isfinite(final)):
                reason = "nonfinite"
            elif mass > 1.0e-10:
                reason = "mass"
            elif energy > config.maximum_composite_energy_drift:
                reason = "energy"
            elif ratio > config.maximum_shadow_increment_ratio:
                reason = "shadow_defect"
            elif cosine < config.minimum_direction_cosine:
                reason = "direction"
            return values, reason

        (final_fine, mass_drift, energy_drift, shadow_increment_ratio, direction), hard_reason = audit(candidate)
        if hard_reason is not None:
            full_candidate = candidate
            learned_weight = 0.5
            while learned_weight + np.finfo(float).eps >= config.minimum_shadow_blend:
                trial = self._shadow_blend(
                    full_candidate, shadow, weight=learned_weight,
                    initial_mass=initial_mass,
                    target_coarse_mass=target_coarse_mass,
                    target_coarse_energy=target_coarse_energy,
                    green=green,
                )
                values, trial_reason = audit(trial)
                if trial_reason is None:
                    candidate = trial
                    final_fine, mass_drift, energy_drift, shadow_increment_ratio, direction = values
                    hard_reason = None
                    break
                learned_weight *= 0.5
        if hard_reason is not None:
            if enable_fallback:
                return self._fallback(state, dt, hard_reason, family, risks, cfl)
            raise RuntimeError(f"Learned candidate failed hard audit: {hard_reason}.")
        return candidate, M633SafeStepDiagnostics(
            dt, dt, family, True, False, None,
            refinement_ok, detail_ok, face_ok, budget_saturated,
            patch.active_fraction, *risks, cfl,
            _minimum_separation(candidate.positions, self.grid.length),
            mass_drift, energy_drift, shadow_increment_ratio, direction,
            learned_weight,
        )

    def solve(
        self, initial: HybridAMRState, *, final_time: float, initial_dt: float,
        **step_options,
    ) -> tuple[tuple[HybridAMRState, ...], tuple[M633SafeStepDiagnostics, ...]]:
        states, diagnostics = [initial], []
        time, dt = 0.0, initial_dt
        while time < final_time - 1.0e-15:
            dt = min(dt, final_time - time)
            state, row = self.step(states[-1], dt=dt, **step_options)
            states.append(state)
            diagnostics.append(row)
            time += dt
            dt = min(dt, row.next_dt)
            if len(diagnostics) > 10000:
                raise RuntimeError("Safe AMR rollout exceeded its step budget.")
        return tuple(states), tuple(diagnostics)
