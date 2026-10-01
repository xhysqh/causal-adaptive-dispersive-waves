"""Run the frozen V2 risk operator in an honest mCH/BO closed loop.

References are deliberately constructed only after the action trace has been
written.  Thus no posthoc error can influence a controller feature, gate, or
action in this script.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from dataclasses import dataclass
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
    component_distance, hierarchical_estimate, hybrid_total_h1_norm, transfer_hybrid_state,
)
from fgsp_ch.nonlocal_waves.adaptivity.bo_certified import hierarchical_one_step_defect  # noqa: E402
from fgsp_ch.nonlocal_waves.adaptivity.bo_classical import bo_resolution_indicators, project_periodic_spectrum  # noqa: E402
from fgsp_ch.nonlocal_waves.adaptivity.mechanism_coordinates import (  # noqa: E402
    CausalHistoryState, causal_control_coordinates, coordinates_for_equation,
    mechanism_history_coordinates, v2_causal_coordinates,
)
from fgsp_ch.nonlocal_waves.adaptivity.prospective_causal_operator import prospective_features  # noqa: E402
from fgsp_ch.nonlocal_waves.adaptivity.trajectory_error_ledger import TrajectoryErrorLedger  # noqa: E402
from fgsp_ch.nonlocal_waves.adaptivity.v2_safety_shielded_controller import (  # noqa: E402
    FrozenV2PredictiveRiskEnsemble, V2Candidate, V2SafetyShieldedController,
    v2_controller_config,
)
from fgsp_ch.nonlocal_waves.evaluation.bo_metrics import bo_invariants, relative_spectral_h1  # noqa: E402
from fgsp_ch.nonlocal_waves.initial_conditions.bo_families import bo_initial_condition  # noqa: E402
from fgsp_ch.nonlocal_waves.operators.bo_fourier import BOFourierOperator  # noqa: E402
from fgsp_ch.representations.cmame_p2_hybrid import (  # noqa: E402
    CMAMEHybridState, hybrid_h1_squared, hybrid_mass, total_state,
)
from fgsp_ch.nonlocal_waves.solvers.bo_reference import BOIFRK4Solver, solve_bo_dop853  # noqa: E402
from fgsp_ch.solvers.cmame_p21_reference import spectral_hybrid_rollout  # noqa: E402
from fgsp_ch.solvers.mch_hybrid_amr import HybridMCHState  # noqa: E402
from fgsp_ch.utils.config import load_yaml  # noqa: E402


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def represented(state: HybridMCHState) -> CMAMEHybridState:
    return CMAMEHybridState(state.field_momentum, state.positions, state.amplitudes)


def relative_drift(current: dict[str, float], initial: dict[str, float]) -> float:
    return max(abs(current[key] - initial[key]) / max(abs(initial[key]), 1.0e-12) for key in initial)


def action_scales(action: str) -> tuple[float, float]:
    return {"shrink_time": (0.5, 1.0), "hold": (1.0, 1.0), "refine_space": (1.0, 2.0),
            "coarsen_space": (1.0, 0.5)}[action]


def periodic_resample(source_x, values, target_x, length):
    """Periodic linear resampling used only for publication visualization."""
    x = np.asarray(source_x, dtype=np.float64)
    y = np.asarray(values, dtype=np.float64)
    target = np.asarray(target_x, dtype=np.float64)
    wrapped = (target - x[0]) % float(length) + x[0]
    return np.interp(wrapped, np.r_[x, x[0] + length], np.r_[y, y[0]])


@dataclass(slots=True)
class LiveCandidate:
    shield: V2Candidate
    next_state: object
    next_grid: PeriodicGrid
    dt: float
    action: str
    invariant_drift: float
    projection_defect: float


def mch_candidate(state, grid, params, *, action, dt, local_budget, remaining_budget, remaining_fraction,
                  time_fraction, initial_invariants, stage2_cfg, minimum_points, maximum_points, history, control):
    dt_factor, resolution_factor = action_scales(action)
    target_grid, target_state = grid, state
    projection_defect = 0.0
    allowed = True
    if resolution_factor > 1.0:
        allowed = grid.points < maximum_points
        if allowed:
            target_grid = PeriodicGrid(2 * grid.points, grid.length, grid.x_min)
            target_state = transfer_hybrid_state(state, grid, target_grid)
            returned = transfer_hybrid_state(target_state, target_grid, grid)
            projection_defect = component_distance(state, grid, returned, grid, grid)["total"] / hybrid_total_h1_norm(state, grid)
    elif resolution_factor < 1.0:
        allowed = grid.points > minimum_points
        if allowed:
            target_grid = PeriodicGrid(grid.points // 2, grid.length, grid.x_min)
            target_state = transfer_hybrid_state(state, grid, target_grid)
            returned = transfer_hybrid_state(target_state, target_grid, grid)
            projection_defect = component_distance(state, grid, returned, grid, grid)["total"] / hybrid_total_h1_norm(state, grid)
    target_initial_invariants = {
        "mass": hybrid_mass(represented(target_state), target_grid),
        "half_h1_squared": 0.5 * hybrid_h1_squared(represented(target_state), target_grid),
    }
    step = dt * dt_factor
    for _ in range(8):
        candidate, eta = hierarchical_estimate(target_state, target_grid, params, step, step, stage2_cfg["nonlinear_solver"])
        inv = {
            "mass": hybrid_mass(represented(candidate), target_grid),
            "half_h1_squared": 0.5 * hybrid_h1_squared(represented(candidate), target_grid),
        }
        drift = relative_drift(inv, target_initial_invariants)
        if action != "shrink_time" or (drift <= 1.0e-10 and eta.temporal + eta.hierarchical_total <= 0.9 * local_budget):
            break
        step *= 0.5
    dt_factor = step / dt
    sharpness = float(np.max(eta.local_indicators) / max(np.linalg.norm(eta.local_indicators), 1.0e-16))
    work = float(target_grid.points * (3.0 + 2.0 * resolution_factor) / dt_factor)
    legacy = prospective_features(
        temporal=eta.temporal, spatial=eta.field + eta.particle, representation=eta.interface,
        migration=projection_defect, invariant_drift=drift, invariant_limit=1.0e-10,
        stiffness=sharpness, local_budget=local_budget, remaining_budget_fraction=remaining_fraction,
        remaining_time_fraction=time_fraction, relative_work=work / (grid.points * 5.0),
        previous_rejected=False, localized_sharpness=sharpness, nonlocal_tail=0.0,
        dt_factor=dt_factor, resolution_factor=resolution_factor,
        probe_requested=resolution_factor > 1.0, representation_change=resolution_factor > 1.0,
    )
    base = coordinates_for_equation(legacy, "modified_camassa_holm")
    features = v2_causal_coordinates(base, history, control)
    return LiveCandidate(
        V2Candidate(action, features, max(eta.temporal + eta.hierarchical_total, 1e-14),
                    remaining_budget, work, bool(allowed and drift <= 1e-10),
                    bool(allowed and projection_defect <= 0.15),
                    local_budget=local_budget, spectral_tail=0.0,
                    localized_sharpness=sharpness,
                    history_velocity_norm=float(np.linalg.norm(history.velocity)),
                    history_acceleration_norm=float(np.linalg.norm(history.acceleration)),
                    external_verification_passed=bool(action == "coarsen_space" and allowed and drift <= 1e-10 and projection_defect <= 0.15),
                    verified_future_risk=max(eta.temporal + eta.hierarchical_total, 1e-14) / local_budget if action == "coarsen_space" else None),
        candidate, target_grid, step, action, drift, projection_defect,
    ), base


def bo_candidate(values, grid, *, action, dt, local_budget, remaining_budget, remaining_fraction,
                 time_fraction, initial_invariants, minimum_points, maximum_points, history, control):
    dt_factor, resolution_factor = action_scales(action)
    target_grid, target_values = grid, values
    projection_defect, allowed = 0.0, True
    if resolution_factor > 1.0:
        allowed = grid.points < maximum_points
        if allowed:
            target_values, target_grid = project_periodic_spectrum(values, grid, 2 * grid.points)
            returned, _ = project_periodic_spectrum(target_values, target_grid, grid.points)
            projection_defect = relative_spectral_h1(returned, values, grid)
    elif resolution_factor < 1.0:
        allowed = grid.points > minimum_points
        if allowed:
            target_values, target_grid = project_periodic_spectrum(values, grid, grid.points // 2)
            returned, _ = project_periodic_spectrum(target_values, target_grid, grid.points)
            projection_defect = relative_spectral_h1(returned, values, grid)
    step = 8.0 * dt * dt_factor
    target_initial_invariants = bo_invariants(target_values, target_grid).as_dict()
    for _ in range(10):
        high, low = BOIFRK4Solver(target_grid).step_embedded(target_values, step)
        embedded = relative_spectral_h1(high, low, target_grid)
        operator = BOFourierOperator(target_grid)
        residual = operator.unresolved_nonlinear_increment(target_values, step)
        hierarchy = hierarchical_one_step_defect(target_values, target_grid, step, coarse_step=high)
        drift = relative_drift(bo_invariants(high, target_grid).as_dict(), target_initial_invariants)
        if action != "shrink_time" or (drift <= 2.0e-5 and max(embedded, residual, hierarchy) <= 0.9 * local_budget):
            break
        step *= 0.5
    dt_factor = step / (8.0 * dt)
    indicator = bo_resolution_indicators(target_values, target_grid, step)
    work = float(7.0 * target_grid.points * np.log2(target_grid.points) / dt_factor)
    legacy = prospective_features(
        temporal=embedded, spatial=max(residual, hierarchy), representation=indicator.aliasing_defect,
        migration=projection_defect, invariant_drift=drift, invariant_limit=2.0e-5,
        stiffness=indicator.tail_ratio / max(indicator.aliasing_defect, 1.0e-12),
        local_budget=local_budget, remaining_budget_fraction=remaining_fraction,
        remaining_time_fraction=time_fraction,
        relative_work=work / (7.0 * grid.points * np.log2(grid.points)), previous_rejected=False,
        localized_sharpness=indicator.aliasing_defect, nonlocal_tail=indicator.tail_ratio,
        dt_factor=dt_factor, resolution_factor=resolution_factor,
        probe_requested=resolution_factor > 1.0, representation_change=resolution_factor > 1.0,
    )
    base = coordinates_for_equation(legacy, "benjamin_ono")
    features = v2_causal_coordinates(base, history, control)
    return LiveCandidate(
        V2Candidate(action, features, max(embedded, residual, hierarchy, 1e-14), remaining_budget, work,
                    bool(allowed and drift <= 2.0e-5),
                    bool(allowed and projection_defect <= 0.15),
                    local_budget=local_budget, spectral_tail=indicator.tail_ratio,
                    localized_sharpness=indicator.aliasing_defect,
                    history_velocity_norm=float(np.linalg.norm(history.velocity)),
                    history_acceleration_norm=float(np.linalg.norm(history.acceleration)),
                    external_verification_passed=bool(action == "coarsen_space" and allowed and drift <= 2.0e-5 and projection_defect <= 0.15),
                    verified_future_risk=max(embedded, residual, hierarchy, 1e-14) / local_budget if action == "coarsen_space" else None),
        high, target_grid, step, action, drift, projection_defect,
    ), base


def run_equation(name, cfg, stage2_cfg, controller, settings):
    grid = PeriodicGrid(int(settings["points"]), float(cfg["domain"]["length"]))
    spec = cfg["showcase"][name]
    tolerance, decisions = float(settings["tolerance"]), int(settings["decisions"])
    ledger = TrajectoryErrorLedger(tolerance, decisions * float(settings["dt"]), reserve_fraction=0.10)
    history_state = CausalHistoryState(float(stage2_cfg["protocol"]["history_rho"]))
    rows, online_work = [], 0.0
    if name == "modified_camassa_holm":
        raw, alpha = build_initial_state(int(spec["seed"]), spec["family"], grid, (0.5, 1.5))
        state = HybridMCHState(raw.positions, raw.amplitudes, raw.regular_momentum)
        initial_state, params = state, ModifiedCHParameters(alpha=alpha, gamma=0.0)
        initial_invariants = {"mass": hybrid_mass(represented(state), grid), "half_h1_squared": 0.5 * hybrid_h1_squared(represented(state), grid)}
    else:
        rng = np.random.default_rng(int(spec["seed"]))
        state = bo_initial_condition(spec["family"], grid, amplitude=float(rng.uniform(.55, 1.45)), phase_shift=float(rng.uniform(0, grid.length)))
        initial_state, params = state.copy(), None
        initial_invariants = bo_invariants(state, grid).as_dict()
    initial_grid = grid
    visualization_points = int(settings.get("visualization_points", settings["reference_refinement"] * settings["points"]))
    visualization_grid = PeriodicGrid(visualization_points, grid.length, grid.x_min)
    elapsed = 0.0
    times = [0.0]
    actions = []
    active_points = [grid.points]
    if name == "modified_camassa_holm":
        fields = [periodic_resample(
            grid.x, total_state(represented(state), grid), visualization_grid.x, grid.length,
        )]
        feature_paths = [state.positions.copy()]
    else:
        fields = [periodic_resample(grid.x, state, visualization_grid.x, grid.length)]
        feature_paths = None
    for prefix in range(decisions):
        local_budget = tolerance / decisions
        remaining_fraction = ledger.remaining / tolerance
        time_fraction = max(1.0 - prefix / decisions, 0.0)
        cache: dict[str, LiveCandidate] = {}
        control = causal_control_coordinates(
            remaining_budget_fraction=np.clip(remaining_fraction, 0.0, 1.0),
            remaining_time_fraction=time_fraction,
            resolution_fraction=grid.points / float(settings["maximum_points"]),
            current_dt=float(settings["dt"]), cooldown_fraction=controller.state.cooldown_remaining / max(controller.config.cooldown_prefixes, 1),
        )
        # Build the physical hold candidate once to establish q_n, then expose a lazy cache to the controller.
        def physical(action, history):
            if name == "modified_camassa_holm":
                return mch_candidate(state, grid, params, action=action, dt=float(settings["dt"]), local_budget=local_budget, remaining_budget=ledger.remaining, remaining_fraction=remaining_fraction, time_fraction=time_fraction, initial_invariants=initial_invariants, stage2_cfg=stage2_cfg, minimum_points=int(settings["minimum_points"]), maximum_points=int(settings["maximum_points"]), history=history, control=control)
            return bo_candidate(state, grid, action=action, dt=float(settings["dt"]), local_budget=local_budget, remaining_budget=ledger.remaining, remaining_fraction=remaining_fraction, time_fraction=time_fraction, initial_invariants=initial_invariants, minimum_points=int(settings["minimum_points"]), maximum_points=int(settings["maximum_points"]), history=history, control=control)
        # Get a provisional hold base without advancing any accepted state. Its history is then frozen for this prefix.
        provisional, hold_base = physical("hold", CausalHistoryState(float(stage2_cfg["protocol"]["history_rho"])).observe(np.zeros(6)))
        history = history_state.observe(mechanism_history_coordinates(hold_base))
        cache["hold"], _ = physical("hold", history)
        def build(actions):
            result = []
            for action in actions:
                if action == "coarsen_space" and not bool(cfg.get("enable_verified_coarsening", False)):
                    # Stage-5 does not yet register an independently verified
                    # live coarsening candidate, so the shield receives none.
                    continue
                if bool(cfg.get("fixed_spatial_grid", False)) and action in {"refine_space", "coarsen_space"}:
                    continue
                if action not in cache:
                    cache[action], _ = physical(action, history)
                result.append(cache[action].shield)
            return tuple(result)
        decision = controller.choose(build)
        if decision.action_id is None:
            detail = ", ".join(
                f"{action}:eta={row.shield.deterministic_error:.3e},"
                f"drift={row.invariant_drift:.3e},proj={row.projection_defect:.3e}"
                for action, row in cache.items()
            )
            raise RuntimeError(f"stage 5 has no deterministic candidate for {name} prefix {prefix}: {detail}")
        selected = cache[decision.action_id]
        before = ledger.remaining
        ledger.commit(selected.shield.deterministic_error)
        state, grid = selected.next_state, selected.next_grid
        elapsed += selected.dt
        times.append(elapsed)
        actions.append(decision.action_id)
        active_points.append(grid.points)
        if name == "modified_camassa_holm":
            fields.append(periodic_resample(
                grid.x, total_state(represented(state), grid), visualization_grid.x, grid.length,
            ))
            feature_paths.append(state.positions.copy())
        else:
            fields.append(periodic_resample(grid.x, state, visualization_grid.x, grid.length))
        online_work += sum(cache[action].shield.work for action in decision.considered_actions if action in cache)
        rows.append({"equation": name, "prefix": prefix, "time": elapsed, "action": decision.action_id,
                     "controller_state": decision.state, "learned_authority": decision.learned_authority,
                     "probe_required": decision.probe_required, "upper_risk": decision.upper_risk,
                     "support_distance": decision.support_distance, "points": grid.points, "dt": selected.dt,
                     "deterministic_error": selected.shield.deterministic_error, "invariant_drift": selected.invariant_drift,
                     "projection_defect": selected.projection_defect, "remaining_before": before,
                     "remaining_after": ledger.remaining, "cooldown_remaining": decision.cooldown_remaining,
                     "stable_hold_prefixes": decision.stable_hold_prefixes,
                     "spectral_tail": selected.shield.spectral_tail,
                     "localized_sharpness": selected.shield.localized_sharpness,
                     "history_velocity_norm": selected.shield.history_velocity_norm,
                     "history_acceleration_norm": selected.shield.history_acceleration_norm})
    # Posthoc reference begins only after every row above is complete.
    if name == "modified_camassa_holm":
        reference = spectral_hybrid_rollout(represented(initial_state), initial_grid, params, elapsed,
                                            refinement=int(settings["reference_refinement"]),
                                            rtol=float(stage2_cfg["reference"]["rtol"]), atol=float(stage2_cfg["reference"]["atol"]),
                                            collision_tolerance=float(stage2_cfg["reference"]["collision_tolerance"]))
        ref_state = HybridMCHState(reference.state.positions, reference.state.amplitudes, reference.state.regular_momentum)
        eval_grid = PeriodicGrid(max(grid.points, reference.grid.points), grid.length)
        terminal = component_distance(state, grid, ref_state, reference.grid, eval_grid)["total"] / hybrid_total_h1_norm(ref_state, reference.grid, eval_grid)
        reference_fields = [fields[0]]
        for time in times[1:]:
            snapshot = spectral_hybrid_rollout(
                represented(initial_state), initial_grid, params, float(time),
                refinement=int(settings["reference_refinement"]),
                rtol=float(stage2_cfg["reference"]["rtol"]),
                atol=float(stage2_cfg["reference"]["atol"]),
                collision_tolerance=float(stage2_cfg["reference"]["collision_tolerance"]),
            )
            reference_fields.append(periodic_resample(
                snapshot.grid.x, total_state(snapshot.state, snapshot.grid),
                visualization_grid.x, initial_grid.length,
            ))
    else:
        fine_initial, fine_grid = project_periodic_spectrum(initial_state, initial_grid, int(settings["reference_refinement"]) * initial_grid.points)
        reference = solve_bo_dop853(fine_initial, fine_grid, final_time=elapsed,
                                    rtol=float(stage2_cfg["reference"]["rtol"]), atol=float(stage2_cfg["reference"]["atol"]), maximum_step=max(elapsed / 8.0, 1e-8))
        projected, _ = project_periodic_spectrum(state, grid, fine_grid.points)
        terminal = relative_spectral_h1(projected, reference.final_state, fine_grid)
        reference_fields = [fields[0]]
        for time in times[1:]:
            snapshot = solve_bo_dop853(
                fine_initial, fine_grid, final_time=float(time),
                rtol=float(stage2_cfg["reference"]["rtol"]),
                atol=float(stage2_cfg["reference"]["atol"]),
                maximum_step=max(float(time) / 8.0, 1e-8),
            )
            reference_fields.append(periodic_resample(
                fine_grid.x, snapshot.final_state, visualization_grid.x, initial_grid.length,
            ))
    showcase = {
        "x": visualization_grid.x,
        "time": np.asarray(times),
        "reference": np.asarray(reference_fields),
        "adaptive": np.asarray(fields),
        "actions": np.asarray(actions),
        "active_points": np.asarray(active_points, dtype=np.int64),
    }
    if feature_paths is not None:
        maximum_paths = max(len(row) for row in feature_paths)
        path_matrix = np.full((len(feature_paths), maximum_paths), np.nan)
        for index, values in enumerate(feature_paths):
            path_matrix[index, :len(values)] = values
        showcase["feature_paths"] = path_matrix
    return rows, float(terminal), online_work, showcase


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment/cmame_v2_stage5_closed_loop.yaml")
    parser.add_argument("--mode", choices=("development", "full"), default="development")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    cfg = load_yaml(ROOT / args.config)
    stage2_cfg = load_yaml(ROOT / cfg["stage2_config"])
    stage4_cfg = load_yaml(ROOT / cfg["stage4_config"])
    root = ROOT / Path(cfg["source_manifest"]).parent
    manifest_path = ROOT / cfg["source_manifest"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_before = sha256(manifest_path)
    ensemble = FrozenV2PredictiveRiskEnsemble.load(root, manifest, device=args.device)
    out = ROOT / cfg["output_dir"] / args.mode
    out.mkdir(parents=True, exist_ok=True)
    all_rows, terminal, work, showcases = [], {}, {}, {}
    controller_mapping = dict(stage4_cfg["controller"])
    controller_mapping.update(cfg.get("controller_overrides", {}))
    for equation in ("modified_camassa_holm", "benjamin_ono"):
        controller = V2SafetyShieldedController(ensemble, v2_controller_config(controller_mapping))
        rows, error, online, showcase = run_equation(equation, cfg, stage2_cfg, controller, cfg[args.mode])
        all_rows.extend(rows); terminal[equation] = error; work[equation] = online
        showcases[equation] = showcase
    trace_path = out / "online_decisions.csv"
    with trace_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0]))
        writer.writeheader(); writer.writerows(all_rows)
    np.savez_compressed(
        out / "v2_closed_loop_showcase.npz",
        x=showcases["modified_camassa_holm"]["x"],
        mch_time=showcases["modified_camassa_holm"]["time"],
        mch_reference=showcases["modified_camassa_holm"]["reference"],
        mch_field=showcases["modified_camassa_holm"]["adaptive"],
        mch_actions=showcases["modified_camassa_holm"]["actions"],
        mch_active_points=showcases["modified_camassa_holm"]["active_points"],
        mch_positions=showcases["modified_camassa_holm"]["feature_paths"],
        bo_time=showcases["benjamin_ono"]["time"],
        bo_reference=showcases["benjamin_ono"]["reference"],
        bo_field=showcases["benjamin_ono"]["adaptive"],
        bo_actions=showcases["benjamin_ono"]["actions"],
        bo_active_points=showcases["benjamin_ono"]["active_points"],
        selection_rule=np.asarray("registered stage-7 showcase; frozen V2 controller"),
    )
    checks = {
        "frozen_manifest_unchanged": sha256(manifest_path) == manifest_before,
        "all_actions_precede_posthoc_reference": True,
        "all_commits_deterministically_admissible": all(row["remaining_after"] >= -1e-14 for row in all_rows),
        "terminal_error_within_registered_limit": max(terminal.values()) <= float(cfg["acceptance"]["maximum_terminal_relative_h1_error"]),
        "coarsening_not_fabricated": (not bool(cfg.get("enable_verified_coarsening", False)))
            or all(row["action"] != "coarsen_space" or row["controller_state"] == "externally_verified_coarsening" for row in all_rows),
        "learned_authority_exercised": bool(any(row["learned_authority"] for row in all_rows)) if args.mode == "full" else True,
        "fallback_decisions_logged": all(row["controller_state"] for row in all_rows),
    }
    blocking = [key for key, value in checks.items() if not value]
    report = {"status": "PASS" if not blocking else "STOP", "checks": checks, "blocking_failures": blocking,
              "terminal_relative_h1_error": terminal, "online_candidate_work": work,
              "learned_authority_decisions": int(sum(row["learned_authority"] for row in all_rows)),
              "deterministic_fallback_decisions": int(sum(row["controller_state"] == "deterministic_fallback" for row in all_rows)),
              "externally_verified_coarsening_decisions": int(sum(row["controller_state"] == "externally_verified_coarsening" for row in all_rows)),
              "space_refinement_decisions": int(sum(row["action"] == "refine_space" for row in all_rows)),
              "decisions": len(all_rows), "manifest_sha256": sha256(manifest_path),
              "trace": trace_path.name, "scope": cfg["scope"]}
    (out / "stage5_acceptance.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    raise SystemExit(0 if not blocking else 2)


if __name__ == "__main__":
    main()
