"""Build stages-0--2 data for the V2 finite-horizon risk operator.

Every label is an H-step, post-action zero-order-hold rollout on mCH or BO.
Reference states are used only after candidate and history features are fixed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts.build_cmame_p21_mixed_causal_dataset import build_initial_state  # noqa: E402
from fgsp_ch.discretization.grids import PeriodicGrid  # noqa: E402
from fgsp_ch.equations.modified_ch import ModifiedCHParameters  # noqa: E402
from fgsp_ch.estimators.mch_hybrid_aposteriori import (  # noqa: E402
    component_distance, hierarchical_estimate, hybrid_total_h1_norm,
    transfer_hybrid_state,
)
from fgsp_ch.representations.cmame_p2_hybrid import (  # noqa: E402
    CMAMEHybridState, hybrid_h1_squared, hybrid_mass,
)
from fgsp_ch.solvers.cmame_p21_reference import spectral_hybrid_rollout  # noqa: E402
from fgsp_ch.solvers.mch_hybrid_amr import HybridMCHState  # noqa: E402
from fgsp_ch.nonlocal_waves.adaptivity.bo_certified import hierarchical_one_step_defect  # noqa: E402
from fgsp_ch.nonlocal_waves.adaptivity.bo_classical import (  # noqa: E402
    bo_resolution_indicators, project_periodic_spectrum,
)
from fgsp_ch.nonlocal_waves.adaptivity.mechanism_coordinates import (  # noqa: E402
    CAUSAL_CONTROL_FEATURE_ORDER, CAUSAL_HISTORY_FEATURE_ORDER,
    CausalHistoryState, MECHANISM_FEATURE_ORDER, V2_CAUSAL_FEATURE_ORDER,
    causal_control_coordinates, coordinates_for_equation,
    mechanism_history_coordinates, v2_causal_coordinates,
)
from fgsp_ch.nonlocal_waves.adaptivity.prospective_causal_operator import ACTIONS, prospective_features  # noqa: E402
from fgsp_ch.nonlocal_waves.evaluation.bo_metrics import bo_invariants, relative_spectral_h1  # noqa: E402
from fgsp_ch.nonlocal_waves.initial_conditions.bo_families import bo_initial_condition  # noqa: E402
from fgsp_ch.nonlocal_waves.operators.bo_fourier import BOFourierOperator  # noqa: E402
from fgsp_ch.nonlocal_waves.solvers.bo_reference import BOIFRK4Solver, solve_bo_dop853  # noqa: E402
from fgsp_ch.utils.config import load_yaml  # noqa: E402


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def append(store: dict, **row) -> None:
    for key, value in row.items():
        store.setdefault(key, []).append(value)


def build_group_transaction(store: dict, builder, **kwargs) -> None:
    """Rollback every partial row when one candidate rollout is invalid."""
    lengths = {key: len(values) for key, values in store.items()}
    try:
        builder(store, **kwargs)
    except Exception:
        for key in list(store):
            if key not in lengths:
                del store[key]
            else:
                del store[key][lengths[key]:]
        raise


def represented(state: HybridMCHState) -> CMAMEHybridState:
    return CMAMEHybridState(state.field_momentum, state.positions, state.amplitudes)


def spectral_h1_components(approximation, reference, grid):
    wave_numbers = 2 * np.pi * np.fft.rfftfreq(grid.points, d=grid.h)
    weights = 1.0 + wave_numbers**2
    difference = np.fft.rfft(np.asarray(approximation) - np.asarray(reference))
    reference_hat = np.fft.rfft(reference)
    absolute = float(np.sqrt(np.sum(weights * np.abs(difference)**2)))
    scale = float(np.sqrt(np.sum(weights * np.abs(reference_hat)**2)))
    if not np.isfinite(scale) or scale <= np.finfo(float).eps:
        raise ValueError("BO future-risk reference has no resolvable H1 scale")
    return absolute, scale, absolute / scale


def action_scales(action: str) -> tuple[float, float]:
    return ({"shrink_time": 0.5, "hold": 1.0, "refine_space": 1.0}[action],
            {"shrink_time": 1.0, "hold": 1.0, "refine_space": 2.0}[action])


def rollout_budget(tolerance: float, prefixes: int, floor: float) -> float:
    """Frozen uniform allocation, independent of any future online decision."""
    return max(float(tolerance) / max(int(prefixes), 1), float(floor))


def mch_rollout(target_state, target_grid, params, step, horizon, cfg):
    states, errors, etas = [], [], []
    current = target_state
    for index in range(1, horizon + 1):
        current, eta = hierarchical_estimate(
            current, target_grid, params, step, step, cfg["nonlinear_solver"],
        )
        reference = spectral_hybrid_rollout(
            represented(target_state), target_grid, params, index * step,
            refinement=int(cfg["_settings"]["reference_refinement"]),
            rtol=float(cfg["reference"]["rtol"]), atol=float(cfg["reference"]["atol"]),
            collision_tolerance=float(cfg["reference"]["collision_tolerance"]),
        )
        truth = HybridMCHState(
            reference.state.positions, reference.state.amplitudes,
            reference.state.regular_momentum,
        )
        distance = component_distance(current, target_grid, truth, reference.grid, reference.grid)
        scale = hybrid_total_h1_norm(truth, reference.grid, reference.grid)
        errors.append(float(distance["total"]) / scale)
        states.append(current); etas.append(eta)
    return states, etas, errors


def bo_rollout(target_values, target_grid, step, horizon, cfg):
    states, embedded_errors, errors = [], [], []
    current = np.asarray(target_values, dtype=np.float64)
    fine, fine_grid = project_periodic_spectrum(
        target_values, target_grid,
        int(cfg["_settings"]["reference_refinement"]) * target_grid.points,
    )
    solver = BOIFRK4Solver(target_grid)
    for index in range(1, horizon + 1):
        high, low = solver.step_embedded(current, step)
        current = high
        reference = solve_bo_dop853(
            fine, fine_grid, final_time=index * step,
            rtol=float(cfg["reference"]["rtol"]), atol=float(cfg["reference"]["atol"]),
            maximum_step=step / 4.0,
        ).final_state
        projected, _ = project_periodic_spectrum(high, target_grid, fine_grid.points)
        _, _, relative = spectral_h1_components(projected, reference, fine_grid)
        states.append(high); embedded_errors.append(relative_spectral_h1(high, low, target_grid))
        errors.append(relative)
    return states, embedded_errors, errors


def append_v2_row(store, *, base, history, control, risk, errors, action, group,
                  split, family, seed, prefix, tolerance, equation, work,
                  deterministic_error, target_probe, horizon, local_budget):
    features = v2_causal_coordinates(base, history, control)
    append(
        store,
        features=features.astype(np.float32),
        mechanism_features=base.astype(np.float32),
        history_physical=history.physical.astype(np.float32),
        history_mean=history.exponential_mean.astype(np.float32),
        history_deviation=history.deviation.astype(np.float32),
        history_velocity=history.velocity.astype(np.float32),
        history_acceleration=history.acceleration.astype(np.float32),
        control_features=control.astype(np.float32),
        target_log_future_risk=np.log(max(float(risk), 1.0e-16)),
        future_risk=float(risk),
        rollout_relative_errors=np.asarray(errors, dtype=np.float64),
        future_horizon=int(horizon),
        rollout_policy="zero_order_hold_post_action_discretization",
        local_rollout_budget=float(local_budget),
        target_log_cost=np.log(max(float(work), 1.0e-16)),
        target_probe=float(target_probe),
        deterministic_error=float(deterministic_error),
        equation=equation, family=family, split=split, group_id=group,
        seed=int(seed), prefix=int(prefix), action=action,
        tolerance=float(tolerance), reference_in_features=False,
        history_source_prefix=int(prefix),
    )


def build_mch_group(store, cfg, *, split, family, seed, group):
    settings = cfg["_settings"]
    grid = PeriodicGrid(int(settings["points"]), float(cfg["domain"]["length"]))
    state, alpha = build_initial_state(seed, family, grid, (0.5, 1.5))
    state = HybridMCHState(state.positions, state.amplitudes, state.regular_momentum)
    params = ModifiedCHParameters(alpha=alpha, gamma=0.0)
    tolerance, horizon = float(settings["tolerance"]), int(cfg["protocol"]["horizon_steps"])
    base_dt = float(settings["dt"])
    history_state = CausalHistoryState(float(cfg["protocol"]["history_rho"]))
    initial_mass = hybrid_mass(represented(state), grid)
    initial_h1 = hybrid_h1_squared(represented(state), grid)
    for prefix in range(int(settings["prefixes"])):
        remaining_fraction = max(1.0 - prefix / max(int(settings["prefixes"]), 1), float(cfg["protocol"]["reserve_fraction"]))
        candidate_rows = []
        for action in ACTIONS:
            dt_factor, resolution_factor = action_scales(action)
            target_grid, target_state = grid, state
            if resolution_factor == 2.0:
                target_grid = PeriodicGrid(2 * grid.points, grid.length, grid.x_min)
                target_state = transfer_hybrid_state(state, grid, target_grid)
            step = base_dt * dt_factor
            states, etas, errors = mch_rollout(target_state, target_grid, params, step, horizon, cfg)
            candidate, eta = states[0], etas[0]
            mass = abs(hybrid_mass(represented(candidate), target_grid) - initial_mass) / max(abs(initial_mass), 1.0e-12)
            h1 = abs(hybrid_h1_squared(represented(candidate), target_grid) - initial_h1) / max(abs(initial_h1), 1.0e-12)
            drift = max(mass, h1)
            sharp = float(np.max(eta.local_indicators) / max(np.linalg.norm(eta.local_indicators), 1.0e-16))
            local_budget = rollout_budget(tolerance, int(settings["prefixes"]), float(cfg["protocol"]["local_budget_floor"]))
            work = float(target_grid.points * (3 + 2 * resolution_factor) / dt_factor)
            legacy = prospective_features(
                temporal=eta.temporal, spatial=eta.field + eta.particle,
                representation=eta.interface, migration=0.0,
                invariant_drift=drift, invariant_limit=1.0e-10,
                stiffness=sharp, local_budget=local_budget,
                remaining_budget_fraction=remaining_fraction,
                remaining_time_fraction=remaining_fraction,
                relative_work=work / (grid.points * 5), previous_rejected=False,
                localized_sharpness=sharp, nonlocal_tail=0.0,
                dt_factor=dt_factor, resolution_factor=resolution_factor,
                probe_requested=resolution_factor > 1.0,
                representation_change=resolution_factor > 1.0,
            )
            base = coordinates_for_equation(legacy, "modified_camassa_holm")
            candidate_rows.append((action, candidate, eta, errors, base, work, local_budget, resolution_factor))
        hold = next(row for row in candidate_rows if row[0] == "hold")
        history = history_state.observe(mechanism_history_coordinates(hold[4]))
        control = causal_control_coordinates(
            remaining_budget_fraction=remaining_fraction,
            remaining_time_fraction=remaining_fraction,
            resolution_fraction=grid.points / (2.0 * int(settings["points"])),
            current_dt=base_dt,
            cooldown_fraction=float(cfg["protocol"]["cooldown_fraction"]),
        )
        for action, candidate, eta, errors, base, work, local_budget, resolution_factor in candidate_rows:
            risk = max(max(errors) / local_budget, 1.0e-16)
            append_v2_row(
                store, base=base, history=history, control=control, risk=risk,
                errors=errors, action=action, group=group, split=split,
                family=family, seed=seed, prefix=prefix, tolerance=tolerance,
                equation="modified_camassa_holm", work=work,
                deterministic_error=max(eta.temporal + eta.hierarchical_total, 1.0e-14),
                target_probe=resolution_factor > 1.0 and eta.hierarchical_total > 0.25 * local_budget,
                horizon=horizon, local_budget=local_budget,
            )
        state = hold[1]


def build_bo_group(store, cfg, *, split, family, seed, group):
    settings = cfg["_settings"]
    grid = PeriodicGrid(int(settings["points"]), float(cfg["domain"]["length"]))
    rng = np.random.default_rng(seed)
    values = bo_initial_condition(family, grid, amplitude=float(rng.uniform(0.55, 1.45)), phase_shift=float(rng.uniform(0, grid.length)))
    tolerance, horizon = float(settings["tolerance"]), int(cfg["protocol"]["horizon_steps"])
    base_dt = float(settings["dt"]) * 8.0
    history_state = CausalHistoryState(float(cfg["protocol"]["history_rho"]))
    for prefix in range(int(settings["prefixes"])):
        remaining_fraction = max(1.0 - prefix / max(int(settings["prefixes"]), 1), float(cfg["protocol"]["reserve_fraction"]))
        initial_invariants = bo_invariants(values, grid)
        candidate_rows = []
        for action in ACTIONS:
            dt_factor, resolution_factor = action_scales(action)
            target_grid, target_values = grid, values
            if resolution_factor == 2.0:
                target_values, target_grid = project_periodic_spectrum(values, grid, 2 * grid.points)
            step = base_dt * dt_factor
            states, embedded_errors, errors = bo_rollout(target_values, target_grid, step, horizon, cfg)
            candidate, embedded = states[0], embedded_errors[0]
            residual = BOFourierOperator(target_grid).unresolved_nonlinear_increment(target_values, step)
            hierarchy = hierarchical_one_step_defect(target_values, target_grid, step, coarse_step=candidate)
            invariants = bo_invariants(candidate, target_grid)
            drift = max(
                abs(invariants.mass - initial_invariants.mass) / max(abs(initial_invariants.mass), 1.0e-12),
                abs(invariants.half_l2_squared - initial_invariants.half_l2_squared) / max(abs(initial_invariants.half_l2_squared), 1.0e-12),
                abs(invariants.hamiltonian - initial_invariants.hamiltonian) / max(abs(initial_invariants.hamiltonian), 1.0e-12),
            )
            indicator = bo_resolution_indicators(target_values, target_grid, step)
            local_budget = rollout_budget(tolerance, int(settings["prefixes"]), float(cfg["protocol"]["local_budget_floor"]))
            work = float(7 * target_grid.points * np.log2(target_grid.points) / dt_factor)
            legacy = prospective_features(
                temporal=embedded, spatial=max(residual, hierarchy), representation=indicator.aliasing_defect,
                migration=0.0, invariant_drift=drift, invariant_limit=2.0e-5,
                stiffness=indicator.tail_ratio / max(indicator.aliasing_defect, 1.0e-12),
                local_budget=local_budget, remaining_budget_fraction=remaining_fraction,
                remaining_time_fraction=remaining_fraction,
                relative_work=work / (7 * grid.points * np.log2(grid.points)), previous_rejected=False,
                localized_sharpness=indicator.aliasing_defect, nonlocal_tail=indicator.tail_ratio,
                dt_factor=dt_factor, resolution_factor=resolution_factor,
                probe_requested=resolution_factor > 1.0, representation_change=resolution_factor > 1.0,
            )
            base = coordinates_for_equation(legacy, "benjamin_ono")
            candidate_rows.append((action, candidate, embedded, hierarchy, errors, base, work, local_budget, resolution_factor))
        hold = next(row for row in candidate_rows if row[0] == "hold")
        history = history_state.observe(mechanism_history_coordinates(hold[5]))
        control = causal_control_coordinates(
            remaining_budget_fraction=remaining_fraction,
            remaining_time_fraction=remaining_fraction,
            resolution_fraction=grid.points / (2.0 * int(settings["points"])),
            current_dt=base_dt,
            cooldown_fraction=float(cfg["protocol"]["cooldown_fraction"]),
        )
        for action, candidate, embedded, hierarchy, errors, base, work, local_budget, resolution_factor in candidate_rows:
            risk = max(max(errors) / local_budget, 1.0e-16)
            append_v2_row(
                store, base=base, history=history, control=control, risk=risk,
                errors=errors, action=action, group=group, split=split,
                family=family, seed=seed, prefix=prefix, tolerance=tolerance,
                equation="benjamin_ono", work=work,
                deterministic_error=max(embedded, hierarchy, 1.0e-14),
                target_probe=resolution_factor > 1.0 and hierarchy > 0.25 * local_budget,
                horizon=horizon, local_budget=local_budget,
            )
        values = hold[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment/cmame_v2_predictive_risk.yaml")
    parser.add_argument("--mode", choices=("development", "full"), default="development")
    args = parser.parse_args()
    config_path = ROOT / args.config
    cfg = load_yaml(config_path)
    settings = cfg[args.mode]
    cfg = {**cfg, "_settings": settings}
    output = ROOT / cfg["output_dir"] / args.mode
    output.mkdir(parents=True, exist_ok=True)
    store, group_sets, counts, failures = {}, {}, {}, []
    for split, specification in cfg["splits"].items():
        per_family = int(specification["groups_per_family"])
        if args.mode == "development":
            per_family = min(per_family, int(cfg["development"]["groups_per_family_cap"]))
        groups = set()
        for equation in ("modified_camassa_holm", "benjamin_ono"):
            for family_index, family in enumerate(cfg["families"][equation]):
                successful, attempt = 0, 0
                while successful < per_family:
                    if attempt >= int(cfg["source_rollout"]["maximum_replacement_attempts"]):
                        raise RuntimeError(
                            f"V2 source rollout exhausted reserved seeds for {split}:{equation}:{family}"
                        )
                    seed = int(specification["seed"]) + (0 if equation == "modified_camassa_holm" else 500000) + family_index * 10000 + attempt
                    group = f"{split}:{equation}:{family}:{seed}"
                    print(f"V2 future-risk data {group}", flush=True)
                    try:
                        build_group_transaction(
                            store,
                            build_mch_group if equation == "modified_camassa_holm" else build_bo_group,
                            cfg=cfg, split=split, family=family, seed=seed, group=group,
                        )
                    except (FloatingPointError, RuntimeError, ValueError) as error:
                        failures.append({
                            "group_id": group, "split": split, "equation": equation,
                            "family": family, "seed": seed, "reason": str(error),
                        })
                        attempt += 1
                        continue
                    groups.add(group)
                    counts[(equation, split)] = counts.get((equation, split), 0) + 1
                    successful += 1
                    attempt += 1
        group_sets[split] = groups
    arrays = {key: np.asarray(value) for key, value in store.items()}
    dataset_path = output / "v2_predictive_risk_dataset.npz"
    np.savez_compressed(dataset_path, **arrays)
    failures_path = output / "source_rollout_failures.json"
    failures_path.write_text(json.dumps(failures, indent=2), encoding="utf-8")
    expected_rows = sum(len(cfg["families"][equation]) for equation in cfg["families"]) * sum(
        min(int(spec["groups_per_family"]), int(cfg["development"]["groups_per_family_cap"])) if args.mode == "development" else int(spec["groups_per_family"])
        for spec in cfg["splits"].values()
    ) * int(settings["prefixes"]) * len(ACTIONS)
    cold_start, first_velocity, action_consistent_history = True, True, True
    for group in sorted(set(arrays["group_id"].astype(str))):
        indices = np.flatnonzero(arrays["group_id"].astype(str) == group)
        prefixes = arrays["prefix"][indices].astype(int)
        for prefix in sorted(set(prefixes)):
            local = indices[prefixes == prefix]
            values = arrays["history_deviation"][local]
            action_consistent_history &= bool(np.allclose(values, values[0]))
            action_consistent_history &= bool(np.allclose(
                arrays["history_velocity"][local], arrays["history_velocity"][local][0]
            ))
            action_consistent_history &= bool(np.allclose(
                arrays["history_acceleration"][local], arrays["history_acceleration"][local][0]
            ))
            if prefix == 0:
                cold_start &= bool(np.allclose(arrays["history_deviation"][local], 0.0))
                cold_start &= bool(np.allclose(arrays["history_velocity"][local], 0.0))
                cold_start &= bool(np.allclose(arrays["history_acceleration"][local], 0.0))
            if prefix == 1:
                first_velocity &= bool(np.allclose(arrays["history_acceleration"][local], 0.0))
    disjoint = all(group_sets[left].isdisjoint(group_sets[right]) for i, left in enumerate(group_sets) for right in list(group_sets)[i + 1:])
    checks = {
        "protocol_horizon_is_two": int(cfg["protocol"]["horizon_steps"]) == 2,
        "zero_order_hold_rollout_registered": set(arrays["rollout_policy"].astype(str)) == {"zero_order_hold_post_action_discretization"},
        "future_steps_complete": bool(np.all(arrays["future_horizon"] >= int(cfg["acceptance"]["minimum_future_steps_per_row"]))),
        "split_groups_disjoint": disjoint,
        "mch_bo_trajectory_balanced": all(counts[("modified_camassa_holm", split)] == counts[("benjamin_ono", split)] for split in cfg["splits"]),
        "all_actions_exercised": set(arrays["action"].astype(str)) == set(ACTIONS),
        "future_risk_finite_positive": bool(np.all(np.isfinite(arrays["future_risk"])) and np.all(arrays["future_risk"] > 0.0)),
        "features_finite": bool(np.all(np.isfinite(arrays["features"]))),
        "history_is_predecision": bool(np.array_equal(arrays["history_source_prefix"], arrays["prefix"])),
        "cold_start_has_zero_deformation": cold_start,
        "first_history_velocity_has_zero_acceleration": first_velocity,
        "history_shared_by_actions_at_each_prefix": action_consistent_history,
        "reference_excluded_from_features": bool(not np.any(arrays["reference_in_features"])),
        "feature_width_matches_contract": arrays["features"].shape[1] == len(V2_CAUSAL_FEATURE_ORDER),
        "source_group_count_sufficient": len(set(arrays["group_id"].astype(str))) >= int(cfg["acceptance"]["minimum_source_group_count"]),
        "row_count_matches_protocol": len(arrays["features"]) == expected_rows,
        "replacement_failures_logged": failures_path.exists(),
    }
    blocking = [name for name, value in checks.items() if not value]
    report = {
        "status": "PASS" if not blocking else "STOP", "checks": checks,
        "blocking_failures": blocking, "mode": args.mode,
        "rows": int(len(arrays["features"])),
        "trajectory_groups": {split: len(groups) for split, groups in group_sets.items()},
        "equation_rows": {equation: int(np.sum(arrays["equation"].astype(str) == equation)) for equation in ("modified_camassa_holm", "benjamin_ono")},
        "mechanism_feature_order": MECHANISM_FEATURE_ORDER,
        "history_feature_order": CAUSAL_HISTORY_FEATURE_ORDER,
        "control_feature_order": CAUSAL_CONTROL_FEATURE_ORDER,
        "v2_feature_order": V2_CAUSAL_FEATURE_ORDER,
        "future_risk_quantiles": {str(q): float(np.quantile(arrays["future_risk"], q)) for q in (0.0, 0.5, 0.95, 1.0)},
        "dataset_sha256": sha256(dataset_path), "config_sha256": sha256(config_path),
        "source_rollout_failures": len(failures),
        "source_rollout_failures_sha256": sha256(failures_path),
        "scope": cfg["scope"], "next_phase": cfg["next_phase_on_pass"] if not blocking else None,
    }
    (output / "dataset_acceptance.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    raise SystemExit(0 if not blocking else 2)


if __name__ == "__main__":
    main()
