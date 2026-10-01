"""Hierarchical certified adaptivity for the periodic Benjamin--Ono equation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from fgsp_ch.discretization.grids import PeriodicGrid
from fgsp_ch.nonlocal_waves.adaptivity.bo_classical import (
    bo_resolution_indicators,
    project_periodic_spectrum,
)
from fgsp_ch.nonlocal_waves.evaluation.bo_metrics import bo_invariants, relative_spectral_h1
from fgsp_ch.nonlocal_waves.operators.bo_fourier import BOFourierOperator
from fgsp_ch.nonlocal_waves.solvers.bo_reference import BOIFRK4Solver


@dataclass(frozen=True, slots=True)
class BOCertifiedConfig:
    tolerance: float
    minimum_points: int = 32
    maximum_points: int = 256
    initial_dt: float = 1.0e-2
    minimum_dt: float = 2.5e-5
    maximum_dt: float = 2.0e-2
    embedded_safety: float = 1.25
    residual_safety: float = 1.25
    hierarchical_safety: float = 1.25
    causal_safety: float = 0.85
    minimum_local_budget_fraction: float = 1.0e-3
    refine_tail: float = 2.0e-3
    coarsen_tail: float = 2.0e-7
    refine_aliasing: float = 2.0e-4
    coarsen_aliasing: float = 2.0e-8
    coarsen_patience: int = 2
    hierarchical_probe_interval: int = 3
    maximum_migration_fraction: float = 0.15
    maximum_mass_drift: float = 1.0e-10
    maximum_l2_drift: float = 2.0e-5
    maximum_hamiltonian_drift: float = 2.0e-5
    minimum_step_factor: float = 0.3
    maximum_step_factor: float = 1.6
    maximum_attempts: int = 30000

    def __post_init__(self) -> None:
        if self.tolerance <= 0.0:
            raise ValueError("certified BO tolerance must be positive")
        if self.minimum_points & (self.minimum_points - 1):
            raise ValueError("minimum_points must be a power of two")
        if self.maximum_points & (self.maximum_points - 1):
            raise ValueError("maximum_points must be a power of two")
        if self.minimum_points > self.maximum_points:
            raise ValueError("invalid certified BO resolution interval")
        if self.hierarchical_probe_interval < 1:
            raise ValueError("hierarchical_probe_interval must be positive")


@dataclass(frozen=True, slots=True)
class BOCertifiedResult:
    final_state: NDArray[np.floating]
    final_grid: PeriodicGrid
    final_time: float
    rows: tuple[dict[str, float | int | str | bool], ...]
    accepted_steps: int
    rejected_steps: int
    refinements: int
    coarsenings: int
    coarsening_vetoes: int
    hierarchical_probes: int
    time_growth_events: int
    time_shrink_events: int
    active_dof_time: float
    nonlinear_evaluation_work: float
    consumed_error_budget: float
    minimum_remaining_error_budget: float
    maximum_embedded_error: float
    maximum_residual_error: float
    maximum_hierarchical_error: float
    maximum_mass_drift: float
    maximum_l2_drift: float
    maximum_hamiltonian_drift: float
    learned_effectivity_steps: int
    skipped_hierarchical_probes: int
    forced_hierarchical_probes: int
    out_of_distribution_fallbacks: int
    maximum_effectivity_uncertainty: float


def hierarchical_one_step_defect(
    values: NDArray[np.floating], grid: PeriodicGrid, dt: float,
    *, coarse_step: NDArray[np.floating] | None = None,
) -> float:
    """Compare one embedded step at N and 2N from a common band-limited state."""
    coarse = (
        BOIFRK4Solver(grid).step_embedded(values, dt)[0]
        if coarse_step is None else np.asarray(coarse_step, dtype=np.float64)
    )
    grid.validate_state(coarse)
    fine_initial, fine_grid = project_periodic_spectrum(values, grid, 2 * grid.points)
    fine, _ = BOIFRK4Solver(fine_grid).step_embedded(fine_initial, dt)
    restricted, _ = project_periodic_spectrum(fine, fine_grid, grid.points)
    return relative_spectral_h1(coarse, restricted, grid)


def _invariant_drift(current, initial) -> dict[str, float]:
    result = {}
    for key, initial_value in initial.as_dict().items():
        result[key] = abs(current.as_dict()[key] - initial_value) / max(
            abs(initial_value), 1.0e-12
        )
    return result


def _predicted_action(
    *,
    cfg: BOCertifiedConfig,
    points: int,
    dt: float,
    temporal_error: float,
    spatial_error: float,
    next_budget: float,
    coarsen_allowed: bool,
    learned_action_log_risk: tuple[float, ...] | None = None,
) -> tuple[int, float, float]:
    """Lowest FFT-work admissible (N, dt) action under predicted error."""
    candidates: list[tuple[float, float, int, float, float]] = []
    point_options = [points]
    if points < cfg.maximum_points:
        point_options.append(2 * points)
    if coarsen_allowed and points > cfg.minimum_points:
        point_options.append(points // 2)
    for target_points in point_options:
        spatial_scale = (points / target_points) ** 2
        for factor in (0.5, 1.0, 1.5):
            target_dt = float(np.clip(dt * factor, cfg.minimum_dt, cfg.maximum_dt))
            predicted_temporal = temporal_error * (target_dt / max(dt, np.finfo(float).eps)) ** 5
            predicted_spatial = spatial_error * spatial_scale
            predicted = predicted_temporal + predicted_spatial
            if predicted <= next_budget:
                work_rate = 7.0 * target_points * np.log2(target_points) / target_dt
                if learned_action_log_risk is None:
                    learned_score = 0.0
                else:
                    n_factor = float(target_points / points)
                    n_index = (0.5, 1.0, 2.0).index(n_factor)
                    dt_index = (0.5, 1.0, 1.5).index(factor)
                    learned_score = float(learned_action_log_risk[3 * n_index + dt_index])
                candidates.append((learned_score, work_rate, target_points, target_dt, predicted))
    if not candidates:
        return points, max(cfg.minimum_dt, 0.5 * dt), temporal_error + spatial_error
    # The learned score ranks only candidates already admitted by the frozen
    # analytical error constraint; it cannot make an unsafe action admissible.
    _, _, target_points, target_dt, predicted = min(
        candidates, key=lambda row: (row[0], row[1])
    )
    return target_points, target_dt, predicted


class BOCertifiedAdaptiveSolver:
    def __init__(
        self, config: BOCertifiedConfig, *, effectivity_ensemble=None, probe_pruner=None,
    ):
        self.config = config
        # Deliberately duck-typed to keep the deterministic P1.1 backbone free
        # from a hard torch dependency.  The ensemble may only tighten error
        # estimates or request a probe; invariant gates remain authoritative.
        self.effectivity_ensemble = effectivity_ensemble
        self.probe_pruner = probe_pruner
        if effectivity_ensemble is not None and probe_pruner is not None:
            raise ValueError("effectivity ensemble and probe pruner are mutually exclusive")

    def solve(
        self,
        initial: NDArray[np.floating],
        grid: PeriodicGrid,
        *,
        final_time: float,
        shadow_probe_audit: bool = False,
    ) -> BOCertifiedResult:
        cfg = self.config
        if final_time <= 0.0 or not np.isfinite(final_time):
            raise ValueError("certified BO final_time must be positive")
        values = np.asarray(initial, dtype=np.float64).copy()
        grid.validate_state(values)
        if not cfg.minimum_points <= grid.points <= cfg.maximum_points:
            raise ValueError("initial BO grid is outside certified adaptive bounds")
        initial_invariants = bo_invariants(values, grid)
        time = 0.0
        dt = float(np.clip(cfg.initial_dt, cfg.minimum_dt, cfg.maximum_dt))
        consumed = 0.0
        minimum_remaining = cfg.tolerance
        quiet_steps = 0
        rows: list[dict[str, float | int | str | bool]] = []
        accepted = rejected = refinements = coarsenings = vetoes = probes = 0
        growth = shrink = 0
        learned_steps = skipped_probes = forced_probes = ood_fallbacks = 0
        maximum_uncertainty = 0.0
        skipped_since_probe = 0
        active_dof_time = 0.0
        nonlinear_work = 0.0
        maxima = {"embedded": 0.0, "residual": 0.0, "hierarchical": 0.0,
                  "mass": 0.0, "half_l2_squared": 0.0, "hamiltonian": 0.0}

        for attempt in range(cfg.maximum_attempts):
            if time >= final_time - 10.0 * np.finfo(float).eps:
                break
            trial_dt = min(dt, final_time - time)
            indicators = bo_resolution_indicators(values, grid, trial_dt)
            if (
                indicators.tail_ratio > cfg.refine_tail
                or indicators.aliasing_defect > cfg.refine_aliasing
            ) and grid.points < cfg.maximum_points:
                before = grid.points
                values, grid = project_periodic_spectrum(values, grid, 2 * grid.points)
                refinements += 1
                quiet_steps = 0
                rows.append({"attempt": attempt, "time": time, "action": "refine_indicator",
                             "points_before": before, "points_after": grid.points,
                             "dt": trial_dt, "tail_ratio": indicators.tail_ratio,
                             "aliasing_defect": indicators.aliasing_defect, "accepted": True})
                continue

            solver = BOIFRK4Solver(grid)
            high, low = solver.step_embedded(values, trial_dt)
            nonlinear_work += 7.0 * grid.points
            embedded = cfg.embedded_safety * relative_spectral_h1(high, low, grid)
            residual = cfg.residual_safety * BOFourierOperator(grid).unresolved_nonlinear_increment(
                values, trial_dt
            )
            remaining = max(cfg.tolerance - consumed, 0.0)
            causal_share = trial_dt / max(final_time - time, trial_dt)
            local_budget = max(
                cfg.minimum_local_budget_fraction * cfg.tolerance,
                cfg.causal_safety * remaining * causal_share,
            )
            drift = _invariant_drift(bo_invariants(high, grid), initial_invariants)
            learned_temporal = embedded
            learned_spatial = residual
            assessment = None
            pruner_assessment = None
            if self.effectivity_ensemble is not None:
                from fgsp_ch.nonlocal_waves.adaptivity.bo_effectivity import build_effectivity_features
                invariant_fractions = (
                    drift["mass"] / cfg.maximum_mass_drift,
                    drift["half_l2_squared"] / cfg.maximum_l2_drift,
                    drift["hamiltonian"] / cfg.maximum_hamiltonian_drift,
                )
                features = build_effectivity_features(
                    values, grid, indicators, embedded=embedded, residual=residual,
                    dt=trial_dt, tolerance=cfg.tolerance, local_budget=local_budget,
                    remaining_time_fraction=max(final_time - time, 0.0) / final_time,
                    remaining_budget_fraction=remaining / cfg.tolerance,
                    invariant_budget_fractions=invariant_fractions,
                    previous_rejected=bool(rows and not rows[-1].get("accepted", True)),
                )
                assessment = self.effectivity_ensemble.assess(
                    features, embedded=embedded, residual=residual
                )
                learned_steps += 1
                learned_temporal = max(embedded, assessment.temporal_error_upper)
                learned_spatial = max(residual, assessment.spatial_error_upper)
                maximum_uncertainty = max(maximum_uncertainty, assessment.uncertainty_width)
                ood_fallbacks += int(assessment.out_of_distribution)
            hard_probe = indicators.tail_ratio > 0.5 * cfg.refine_tail
            baseline_periodic_probe = accepted % cfg.hierarchical_probe_interval == 0
            baseline_due_probe = grid.points < cfg.maximum_points and (
                hard_probe or baseline_periodic_probe
            )
            if self.probe_pruner is not None:
                # P2.1 is evaluated only on a probe already scheduled by P1.1.
                # It cannot add probes, alter embedded/residual estimates, or
                # rank the deterministic action candidates.
                due_probe = baseline_due_probe
                if baseline_due_probe and not hard_probe:
                    from fgsp_ch.nonlocal_waves.adaptivity.bo_probe_pruner import (
                        build_probe_features,
                    )
                    probe_features = build_probe_features(
                        values, grid, indicators, embedded=embedded, residual=residual,
                        dt=trial_dt,
                        previous_rejected=bool(rows and not rows[-1].get("accepted", True)),
                    )
                    pruner_assessment = self.probe_pruner.assess(
                        probe_features,
                        spatial_budget=max(local_budget - embedded, np.finfo(float).eps),
                        baseline_due=True,
                        hard_trigger=False,
                    )
                    learned_steps += 1
                    maximum_uncertainty = max(
                        maximum_uncertainty, pruner_assessment.uncertainty_width
                    )
                    ood_fallbacks += int(pruner_assessment.out_of_distribution)
                    due_probe = pruner_assessment.should_probe
                    skipped_probes += int(pruner_assessment.skip_certified)
            else:
                maximum_skip = (
                    int(self.effectivity_ensemble.manifest.get(
                        "maximum_probe_skip_steps", cfg.hierarchical_probe_interval
                    )) if self.effectivity_ensemble is not None else cfg.hierarchical_probe_interval
                )
                periodic_probe = skipped_since_probe >= maximum_skip
                due_probe = grid.points < cfg.maximum_points and (
                    hard_probe or periodic_probe or
                    (assessment.probe_required if assessment is not None else
                     baseline_periodic_probe)
                )
            hierarchical = 0.0
            shadow_hierarchical = 0.0
            shadow_decision_change = False
            if due_probe:
                hierarchical = cfg.hierarchical_safety * hierarchical_one_step_defect(
                    values, grid, trial_dt, coarse_step=high
                )
                # Reuse the N candidate; only the 2N embedded solve is extra.
                nonlinear_work += 14.0 * grid.points
                probes += 1
                forced_probes += int(hard_probe)
                skipped_since_probe = 0
            elif assessment is not None and self.probe_pruner is None:
                skipped_probes += 1
                skipped_since_probe += 1
            if (
                shadow_probe_audit
                and pruner_assessment is not None
                and pruner_assessment.skip_certified
            ):
                # Audit-only truth is never charged to policy work and never
                # enters the online state or decision path.
                shadow_hierarchical = cfg.hierarchical_safety * hierarchical_one_step_defect(
                    values, grid, trial_dt, coarse_step=high
                )
                cheap_accept = embedded + residual <= local_budget
                shadow_accept = embedded + max(residual, shadow_hierarchical) <= local_budget
                shadow_decision_change = cheap_accept != shadow_accept
            # A probe request or abstention returns completely to the frozen
            # P1.1 error path.  Learned bounds are used only when the model is
            # sufficiently certain to skip the expensive probe.
            if self.probe_pruner is not None:
                temporal = embedded
                spatial = max(residual, hierarchical) if due_probe else residual
            elif due_probe:
                temporal = embedded
                spatial = max(residual, hierarchical)
            else:
                temporal = learned_temporal
                spatial = learned_spatial
            invariant_risk = max(
                drift["mass"] / cfg.maximum_mass_drift,
                drift["half_l2_squared"] / cfg.maximum_l2_drift,
                drift["hamiltonian"] / cfg.maximum_hamiltonian_drift,
            )
            estimated = temporal + spatial
            risk = max(estimated / max(local_budget, np.finfo(float).eps), invariant_risk)
            maxima["embedded"] = max(maxima["embedded"], embedded)
            maxima["residual"] = max(maxima["residual"], residual)
            maxima["hierarchical"] = max(maxima["hierarchical"], hierarchical)
            for key in ("mass", "half_l2_squared", "hamiltonian"):
                maxima[key] = max(maxima[key], drift[key])

            if risk > 1.0:
                rejected += 1
                if spatial > embedded and grid.points < cfg.maximum_points:
                    before = grid.points
                    values, grid = project_periodic_spectrum(values, grid, 2 * grid.points)
                    refinements += 1
                    action = "reject_refine"
                    after = grid.points
                else:
                    dt = max(cfg.minimum_dt, trial_dt * max(
                        cfg.minimum_step_factor,
                        cfg.causal_safety * risk ** (-0.2),
                    ))
                    shrink += 1
                    action = "reject_shrink"
                    before = after = grid.points
                rows.append({"attempt": attempt, "time": time, "action": action,
                             "points_before": before, "points_after": after,
                             "dt": trial_dt, "embedded_error": embedded,
                             "learned_temporal_upper": temporal,
                             "learned_spatial_upper": learned_spatial,
                             "residual_error": residual, "hierarchical_error": hierarchical,
                             "shadow_hierarchical_error": shadow_hierarchical,
                             "shadow_decision_change": shadow_decision_change,
                             "local_budget": local_budget, "risk": risk, "accepted": False})
                continue

            # Commit the high-order state before choosing the next action.
            values = high
            time += trial_dt
            accepted += 1
            active_dof_time += grid.points * trial_dt
            consumed = min(cfg.tolerance, consumed + estimated)
            minimum_remaining = min(minimum_remaining, cfg.tolerance - consumed)
            quiet = (
                indicators.tail_ratio < cfg.coarsen_tail
                and indicators.aliasing_defect < cfg.coarsen_aliasing
            )
            quiet_steps = quiet_steps + 1 if quiet else 0
            coarsen_allowed = quiet_steps >= cfg.coarsen_patience
            migration = np.inf
            if coarsen_allowed and grid.points > cfg.minimum_points:
                coarse, coarse_grid = project_periodic_spectrum(values, grid, grid.points // 2)
                restored, _ = project_periodic_spectrum(coarse, coarse_grid, grid.points)
                migration = relative_spectral_h1(restored, values, grid)
                if migration > cfg.maximum_migration_fraction * max(local_budget, cfg.tolerance * 1.0e-3):
                    coarsen_allowed = False
                    vetoes += 1
            next_remaining = max(cfg.tolerance - consumed, cfg.tolerance * cfg.minimum_local_budget_fraction)
            next_dt_scale = min(cfg.maximum_dt, max(cfg.minimum_dt, trial_dt))
            next_budget = max(
                cfg.minimum_local_budget_fraction * cfg.tolerance,
                cfg.causal_safety * next_remaining * next_dt_scale
                / max(final_time - time, next_dt_scale),
            )
            target_points, target_dt, predicted = _predicted_action(
                cfg=cfg, points=grid.points, dt=trial_dt,
                temporal_error=embedded, spatial_error=spatial,
                next_budget=next_budget, coarsen_allowed=coarsen_allowed,
                learned_action_log_risk=(
                    assessment.action_log_risk
                    if assessment is not None and not due_probe and self.probe_pruner is None
                    else None
                ),
            )
            before = grid.points
            if target_points != grid.points:
                values, grid = project_periodic_spectrum(values, grid, target_points)
                if target_points > before:
                    refinements += 1
                    action = "accept_refine"
                else:
                    coarsenings += 1
                    action = "accept_coarsen"
                quiet_steps = 0
            else:
                action = "accept_hold"
            if target_dt > 1.05 * trial_dt:
                growth += 1
            elif target_dt < 0.95 * trial_dt:
                shrink += 1
            dt = target_dt
            rows.append({"attempt": attempt, "time": time, "action": action,
                         "points_before": before, "points_after": grid.points,
                         "dt": trial_dt, "next_dt": dt,
                         "embedded_error": embedded, "residual_error": residual,
                         "learned_temporal_upper": temporal,
                         "learned_spatial_upper": learned_spatial,
                         "probe_probability": assessment.probe_probability if assessment else 1.0,
                         "effectivity_uncertainty": assessment.uncertainty_width if assessment else 0.0,
                         "effectivity_ood": assessment.out_of_distribution if assessment else False,
                         "p11_probe_scheduled": baseline_due_probe,
                         "p21_probe_skipped": bool(
                             pruner_assessment and pruner_assessment.skip_certified
                         ),
                         "p21_hidden_excess_upper": (
                             pruner_assessment.hidden_excess_upper if pruner_assessment else 0.0
                         ),
                         "p21_decision_change_probability": (
                             pruner_assessment.decision_change_probability
                             if pruner_assessment else 1.0
                         ),
                         "p21_ood": bool(
                             pruner_assessment and pruner_assessment.out_of_distribution
                         ),
                         "shadow_hierarchical_error": shadow_hierarchical,
                         "shadow_decision_change": shadow_decision_change,
                         "hierarchical_error": hierarchical, "local_budget": local_budget,
                         "consumed_budget": consumed, "predicted_next_error": predicted,
                         "migration_defect": migration if np.isfinite(migration) else 0.0,
                         "risk": risk, "accepted": True})
        else:
            raise RuntimeError("certified BO controller exceeded maximum attempts")
        if time < final_time - 1.0e-12:
            raise RuntimeError("certified BO controller did not reach final time")
        return BOCertifiedResult(
            final_state=values, final_grid=grid, final_time=time, rows=tuple(rows),
            accepted_steps=accepted, rejected_steps=rejected,
            refinements=refinements, coarsenings=coarsenings,
            coarsening_vetoes=vetoes, hierarchical_probes=probes,
            time_growth_events=growth, time_shrink_events=shrink,
            active_dof_time=active_dof_time, consumed_error_budget=consumed,
            nonlinear_evaluation_work=nonlinear_work,
            minimum_remaining_error_budget=minimum_remaining,
            maximum_embedded_error=maxima["embedded"],
            maximum_residual_error=maxima["residual"],
            maximum_hierarchical_error=maxima["hierarchical"],
            maximum_mass_drift=maxima["mass"],
            maximum_l2_drift=maxima["half_l2_squared"],
            maximum_hamiltonian_drift=maxima["hamiltonian"],
            learned_effectivity_steps=learned_steps,
            skipped_hierarchical_probes=skipped_probes,
            forced_hierarchical_probes=forced_probes,
            out_of_distribution_fallbacks=ood_fallbacks,
            maximum_effectivity_uncertainty=maximum_uncertainty,
        )
