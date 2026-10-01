"""Generate frozen-V2 KdV and ILW representative trajectories for Figures 2 and 4."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts.run_cmame_kdv_z0_zero_shot_audit import _candidates as kdv_candidates  # noqa: E402
from scripts.run_cmame_u21_ilw_blind_audit import build_candidates as ilw_candidates  # noqa: E402
from fgsp_ch.discretization.grids import PeriodicGrid  # noqa: E402
from fgsp_ch.nonlocal_waves.adaptivity.ilw_unknown import ilw_indicators  # noqa: E402
from fgsp_ch.nonlocal_waves.adaptivity.kdv_zero_shot import (  # noqa: E402
    kdv_resolution_indicators, project_kdv_spectrum,
)
from fgsp_ch.nonlocal_waves.adaptivity.mechanism_coordinates import (  # noqa: E402
    CausalHistoryState, causal_control_coordinates, coordinates_for_equation,
    mechanism_coordinates, mechanism_history_coordinates, v2_causal_coordinates,
)
from fgsp_ch.nonlocal_waves.adaptivity.trajectory_error_ledger import TrajectoryErrorLedger  # noqa: E402
from fgsp_ch.nonlocal_waves.adaptivity.v2_safety_shielded_controller import (  # noqa: E402
    FrozenV2PredictiveRiskEnsemble, V2Candidate, V2SafetyShieldedController,
    v2_controller_config,
)
from fgsp_ch.nonlocal_waves.evaluation.ilw_metrics import ilw_invariants, relative_ilw_h1  # noqa: E402
from fgsp_ch.nonlocal_waves.evaluation.kdv_metrics import kdv_invariants, relative_kdv_h1  # noqa: E402
from fgsp_ch.nonlocal_waves.initial_conditions.kdv_families import kdv_initial_condition  # noqa: E402
from fgsp_ch.nonlocal_waves.operators.ilw_fourier import project_ilw_spectrum  # noqa: E402
from fgsp_ch.nonlocal_waves.solvers.ilw_reference import solve_ilw_dop853  # noqa: E402
from fgsp_ch.nonlocal_waves.solvers.kdv_reference import solve_kdv_dop853  # noqa: E402
from fgsp_ch.utils.config import load_yaml  # noqa: E402


def _shield(candidate, base, history, control, local_budget, indicators):
    features = v2_causal_coordinates(base, history, control)
    envelope = candidate.envelope
    return V2Candidate(
        candidate.action, features, envelope.deterministic_error,
        envelope.remaining_budget, envelope.work,
        bool(envelope.analytically_admissible and envelope.invariant_drift <= envelope.invariant_limit),
        bool(getattr(candidate, "migration_error", 0.0) <= 0.15),
        local_budget=local_budget,
        spectral_tail=float(indicators.tail_ratio),
        localized_sharpness=float(indicators.localized_sharpness),
        history_velocity_norm=float(np.linalg.norm(history.velocity)),
        history_acceleration_norm=float(np.linalg.norm(history.acceleration)),
    )


def _run_kdv(cfg, ensemble, stage2_cfg, controller_cfg):
    source_cfg = load_yaml(ROOT / cfg["kdv_config"])
    settings = dict(source_cfg["full"])
    tolerance = float(cfg["tolerance"])
    grid = PeriodicGrid(int(settings["points"]), **source_cfg["domain"])
    seed = int(source_cfg["seeds"]["external_base"])
    family = source_cfg["figure_representative"]["family"]
    initial = kdv_initial_condition(family, grid, seed=seed)
    state, active = initial.copy(), grid
    initial_invariants = kdv_invariants(state, active)
    decisions, macro_dt = int(settings["decisions"]), float(settings["macro_dt"])
    ledger = TrajectoryErrorLedger(tolerance, decisions * macro_dt, reserve_fraction=0.10)
    history_state = CausalHistoryState(float(stage2_cfg["protocol"]["history_rho"]))
    controller = V2SafetyShieldedController(ensemble, v2_controller_config(controller_cfg))
    states, grids, times, actions, telemetry = [state.copy()], [active], [0.0], [], []
    hold_features = []
    authority = 0
    for prefix in range(decisions):
        local_budget = tolerance / decisions
        control = causal_control_coordinates(
            remaining_budget_fraction=np.clip(ledger.remaining / tolerance, 0.0, 1.0),
            remaining_time_fraction=max(0.0, 1.0 - prefix / decisions),
            resolution_fraction=active.points / float(settings["maximum_points"]),
            current_dt=macro_dt,
            cooldown_fraction=controller.state.cooldown_remaining / max(controller.config.cooldown_prefixes, 1),
        )
        hold_row = kdv_candidates(
            state, active, macro_dt=macro_dt, tolerance=tolerance,
            remaining_budget=ledger.remaining, prefix=prefix, decisions=decisions,
            settings=settings, initial_invariants=initial_invariants, actions=("hold",),
        )[0]
        hold_base = coordinates_for_equation(hold_row.features, "kdv")
        history = history_state.observe(mechanism_history_coordinates(hold_base))
        cache = {}

        def construct(action):
            if action in cache:
                return cache[action]
            row = hold_row if action == "hold" else kdv_candidates(
                state, active, macro_dt=macro_dt, tolerance=tolerance,
                remaining_budget=ledger.remaining, prefix=prefix, decisions=decisions,
                settings=settings, initial_invariants=initial_invariants, actions=(action,),
            )[0]
            base = hold_base if action == "hold" else coordinates_for_equation(row.features, "kdv")
            indicators = kdv_resolution_indicators(
                row.state, row.grid, macro_dt / row.substeps,
                tail_fraction=float(settings["tail_fraction"]),
            )
            cache[action] = (row, _shield(
                row, base, history, control, local_budget, indicators,
            ))
            return cache[action]
        # Record the predictive risk of maintaining the current discretization,
        # independently of the action eventually selected by the controller.
        hold_upper_risk = float(controller.ensemble.assess(
            np.stack([construct("hold")[1].features])
        )[0].upper_risk)
        hold_features.append(construct("hold")[1].features.copy())

        def builder(requested):
            return tuple(construct(action)[1] for action in requested if action != "coarsen_space")

        decision = controller.choose(builder)
        if decision.action_id is None:
            raise RuntimeError(f"no admissible V2 KdV action at prefix {prefix}")
        selected, selected_shield = construct(decision.action_id)
        ledger.commit(selected_shield.deterministic_error)
        state, active = selected.state, selected.grid
        states.append(state.copy()); grids.append(active)
        # ``shrink_time`` subcycles the complete macro interval; it does not
        # shorten the physical horizon represented by this committed prefix.
        times.append(times[-1] + macro_dt)
        actions.append(selected.action); authority += int(decision.learned_authority)
        telemetry.append({
            "equation": "kdv", "prefix": prefix, "time": times[-1],
            "action": selected.action, "points": active.points,
            "upper_risk": float(decision.upper_risk),
            "hold_upper_risk": hold_upper_risk,
            "probe_required": bool(decision.probe_required),
            "learned_authority": bool(decision.learned_authority),
            "deterministic_error": float(selected_shield.deterministic_error),
            "candidate_work": float(sum(item[1].work for item in cache.values())),
            "verification_work": 0.0,
            "remaining_budget": float(ledger.remaining),
            "spectral_tail": float(selected_shield.spectral_tail),
            "localized_sharpness": float(selected_shield.localized_sharpness),
        })
    fine_grid = PeriodicGrid(int(settings["reference_points"]), **source_cfg["domain"])
    fine_initial, _ = project_kdv_spectrum(initial, grid, fine_grid.points)
    reference = solve_kdv_dop853(
        fine_initial, fine_grid, evaluation_times=np.asarray(times),
        rtol=float(source_cfg["reference"]["rtol"]),
        atol=float(source_cfg["reference"]["atol"]), maximum_step=macro_dt / 4,
    )
    adaptive = np.stack([
        project_kdv_spectrum(values, source, fine_grid.points)[0]
        for values, source in zip(states, grids, strict=True)
    ])
    # Strict H=2 fixed-HOLD labels, used only by the posthoc reliability audit.
    realized_hold_risk = np.full(len(states), np.nan)
    local_budget = tolerance / decisions
    for index in range(len(states) - 2):
        held_state, held_grid = states[index].copy(), grids[index]
        risks = []
        for horizon in (1, 2):
            held_options = kdv_candidates(
                held_state, held_grid, macro_dt=macro_dt, tolerance=tolerance,
                remaining_budget=tolerance, prefix=index + horizon - 1,
                decisions=decisions, settings=settings,
                initial_invariants=initial_invariants, actions=("hold",),
            )
            held = next(row for row in held_options if row.action == "hold")
            held_state, held_grid = held.state, held.grid
            projected, _ = project_kdv_spectrum(held_state, held_grid, fine_grid.points)
            risks.append(relative_kdv_h1(projected, reference.states[index + horizon], fine_grid) / local_budget)
        realized_hold_risk[index] = max(risks)
    return {
        "equation": "kdv", "family": family, "seed": seed, "x": fine_grid.x,
        "times": np.asarray(times), "reference": reference.states,
        "adaptive": adaptive,
        "actions": np.concatenate((np.asarray(["initial"]), np.asarray(actions))),
        "active_points": np.asarray([row.points for row in grids]),
        "terminal_error": float(relative_kdv_h1(adaptive[-1], reference.states[-1], fine_grid)),
        "authority": authority, "telemetry": telemetry,
        "hold_features": np.stack(hold_features),
        "realized_hold_risk": realized_hold_risk,
    }


def _run_ilw(cfg, ensemble, stage2_cfg, controller_cfg):
    source_cfg = load_yaml(ROOT / cfg["ilw_config"])
    settings = dict(source_cfg["full"])
    tolerance, depth = float(cfg["tolerance"]), float(cfg["ilw_depth"])
    beta = float(source_cfg["nonlinear_transport"])
    grid = PeriodicGrid(int(settings["points"]), **source_cfg["domain"])
    seed = int(source_cfg["seeds"]["external_base"] + list(map(float, settings["depths"])).index(depth) * 100000)
    family = source_cfg["families"][0]
    initial = kdv_initial_condition(family, grid, seed=seed)
    initial, _ = project_ilw_spectrum(initial, grid, grid.points, depth, beta)
    state, active = initial.copy(), grid
    initial_invariants = ilw_invariants(state, active, depth, beta)
    decisions, macro_dt = int(settings["decisions"]), float(settings["macro_dt"])
    ledger = TrajectoryErrorLedger(tolerance, decisions * macro_dt, reserve_fraction=0.10)
    history_state = CausalHistoryState(float(stage2_cfg["protocol"]["history_rho"]))
    controller = V2SafetyShieldedController(ensemble, v2_controller_config(controller_cfg))
    states, grids, times, actions, telemetry = [state.copy()], [active], [0.0], [], []
    hold_features = []
    authority = 0
    for prefix in range(decisions):
        local_budget = tolerance / decisions
        control = causal_control_coordinates(
            remaining_budget_fraction=np.clip(ledger.remaining / tolerance, 0.0, 1.0),
            remaining_time_fraction=max(0.0, 1.0 - prefix / decisions),
            resolution_fraction=active.points / float(settings["maximum_points"]),
            current_dt=macro_dt,
            cooldown_fraction=controller.state.cooldown_remaining / max(controller.config.cooldown_prefixes, 1),
        )
        # The registered ILW builder already emits equation-neutral mechanism
        # coordinates, whereas the KdV builder emits the legacy row.
        hold_row = ilw_candidates(
            state, active, depth, beta, macro_dt=macro_dt, tolerance=tolerance,
            remaining=ledger.remaining, local_budget=local_budget,
            prefix=prefix, decisions=decisions, settings=settings,
            config=source_cfg, initial_invariants=initial_invariants, actions=("hold",),
        )[0]
        hold_base = np.asarray(hold_row.features, dtype=float)
        history = history_state.observe(mechanism_history_coordinates(hold_base))
        cache = {}

        def construct(action):
            if action in cache:
                return cache[action]
            row = hold_row if action == "hold" else ilw_candidates(
                state, active, depth, beta, macro_dt=macro_dt, tolerance=tolerance,
                remaining=ledger.remaining, local_budget=local_budget,
                prefix=prefix, decisions=decisions, settings=settings,
                config=source_cfg, initial_invariants=initial_invariants, actions=(action,),
            )[0]
            base = hold_base if action == "hold" else np.asarray(row.features, dtype=float)
            cache[action] = (row, _shield(
                row, base, history, control, local_budget, row.indicators,
            ))
            return cache[action]
        hold_upper_risk = float(controller.ensemble.assess(
            np.stack([construct("hold")[1].features])
        )[0].upper_risk)
        hold_features.append(construct("hold")[1].features.copy())

        def builder(requested):
            return tuple(construct(action)[1] for action in requested if action != "coarsen_space")

        decision = controller.choose(builder)
        if decision.action_id is None:
            raise RuntimeError(f"no admissible V2 ILW action at prefix {prefix}")
        selected, selected_shield = construct(decision.action_id)
        ledger.commit(selected_shield.deterministic_error)
        state, active = selected.state, selected.grid
        states.append(state.copy()); grids.append(active)
        # Two half steps still advance one complete physical macro interval.
        times.append(times[-1] + macro_dt)
        actions.append(selected.action); authority += int(decision.learned_authority)
        telemetry.append({
            "equation": "ilw", "prefix": prefix, "time": times[-1],
            "action": selected.action, "points": active.points,
            "upper_risk": float(decision.upper_risk),
            "hold_upper_risk": hold_upper_risk,
            "probe_required": bool(decision.probe_required),
            "learned_authority": bool(decision.learned_authority),
            "deterministic_error": float(selected_shield.deterministic_error),
            "candidate_work": float(sum(item[1].work for item in cache.values())),
            "verification_work": 0.0,
            "remaining_budget": float(ledger.remaining),
            "spectral_tail": float(selected_shield.spectral_tail),
            "localized_sharpness": float(selected_shield.localized_sharpness),
        })
    fine_grid = PeriodicGrid(int(settings["reference_points"]), **source_cfg["domain"])
    fine_initial, _ = project_ilw_spectrum(initial, grid, fine_grid.points, depth, beta)
    reference = solve_ilw_dop853(
        fine_initial, fine_grid, depth, evaluation_times=np.asarray(times),
        nonlinear_transport=beta, rtol=float(source_cfg["reference"]["rtol"]),
        atol=float(source_cfg["reference"]["atol"]), maximum_step=macro_dt / 4,
    )
    adaptive = np.stack([
        project_ilw_spectrum(values, source, fine_grid.points, depth, beta)[0]
        for values, source in zip(states, grids, strict=True)
    ])
    # Posthoc counterfactual labels: from each committed state, keep the
    # current discretization unchanged for H=2 macro intervals.  These labels
    # are never available to the online controller.
    realized_hold_risk = np.full(len(states), np.nan)
    local_budget = tolerance / decisions
    for index in range(len(states) - 2):
        held_state, held_grid = states[index].copy(), grids[index]
        risks = []
        for horizon in (1, 2):
            options = ilw_candidates(
                held_state, held_grid, depth, beta, macro_dt=macro_dt,
                tolerance=tolerance, remaining=tolerance, local_budget=local_budget,
                prefix=index + horizon - 1, decisions=decisions, settings=settings,
                config=source_cfg, initial_invariants=initial_invariants,
            )
            hold = next(item for item in options if item.action == "hold")
            held_state, held_grid = hold.state, hold.grid
            projected, _ = project_ilw_spectrum(
                held_state, held_grid, fine_grid.points, depth, beta,
            )
            risks.append(relative_ilw_h1(projected, reference.states[index + horizon], fine_grid) / local_budget)
        realized_hold_risk[index] = max(risks)
    return {
        "equation": "ilw", "family": family, "seed": seed, "depth": depth,
        "x": fine_grid.x, "times": np.asarray(times), "reference": reference.states,
        "adaptive": adaptive,
        "actions": np.concatenate((np.asarray(["initial"]), np.asarray(actions))),
        "active_points": np.asarray([row.points for row in grids]),
        "terminal_error": float(relative_ilw_h1(adaptive[-1], reference.states[-1], fine_grid)),
        "authority": authority, "telemetry": telemetry,
        "hold_features": np.stack(hold_features),
        "realized_hold_risk": realized_hold_risk,
    }


def _save(path, run, selection_rule):
    payload = {
        "equation": np.asarray(run["equation"]), "family": np.asarray(run["family"]),
        "seed": np.asarray(run["seed"]), "x": run["x"], "times": run["times"],
        "reference": run["reference"], "adaptive": run["adaptive"],
        "absolute_error": np.abs(run["adaptive"] - run["reference"]),
        "actions": run["actions"], "active_points": run["active_points"],
        "selection_rule": np.asarray(selection_rule),
        "upper_risk": np.r_[np.nan, [row["upper_risk"] for row in run["telemetry"]]],
        "hold_upper_risk": np.r_[np.nan, [row["hold_upper_risk"] for row in run["telemetry"]]],
        "probe_required": np.r_[False, [row["probe_required"] for row in run["telemetry"]]],
        "learned_authority": np.r_[False, [row["learned_authority"] for row in run["telemetry"]]],
        "deterministic_error": np.r_[0.0, [row["deterministic_error"] for row in run["telemetry"]]],
        "remaining_budget": np.r_[np.nan, [row["remaining_budget"] for row in run["telemetry"]]],
        "spectral_tail": np.r_[np.nan, [row["spectral_tail"] for row in run["telemetry"]]],
        "localized_sharpness": np.r_[np.nan, [row["localized_sharpness"] for row in run["telemetry"]]],
        "hold_features": run["hold_features"],
        "realized_hold_risk": run.get("realized_hold_risk", np.full(len(run["times"]), np.nan)),
    }
    if "depth" in run:
        payload["depth"] = np.asarray(run["depth"])
    np.savez_compressed(path, **payload)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment/cmame_v2_kdv_ilw_showcases.yaml")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--manifest")
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    cfg = load_yaml(ROOT / args.config)
    stage2_cfg = load_yaml(ROOT / cfg["stage2_config"])
    controller_cfg = load_yaml(ROOT / cfg["stage4_config"])["controller"]
    manifest_path = ROOT / (args.manifest or cfg["source_manifest"])
    ensemble = FrozenV2PredictiveRiskEnsemble.load(
        manifest_path.parent, json.loads(manifest_path.read_text(encoding="utf-8")),
        device=args.device,
    )
    output = ROOT / (args.output_dir or str(Path(cfg["output_dir"]) / "full"))
    output.mkdir(parents=True, exist_ok=True)
    kdv = _run_kdv(cfg, ensemble, stage2_cfg, controller_cfg)
    ilw = _run_ilw(cfg, ensemble, stage2_cfg, controller_cfg)
    _save(output / "kdv_representative_trajectory.npz", kdv,
          "registered collision family; minimum external seed; frozen V2")
    _save(output / "ilw_representative_trajectory.npz", ilw,
          "registered depth delta=1; minimum external seed; frozen V2")
    rows = kdv["telemetry"] + ilw["telemetry"]
    with (output / "online_decisions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    terminal = {"kdv": kdv["terminal_error"], "ilw": ilw["terminal_error"]}
    checks = {
        "all_references_posthoc": True,
        "all_outputs_finite": bool(np.all(np.isfinite(kdv["adaptive"])) and np.all(np.isfinite(ilw["adaptive"]))),
        "registered_tolerance_met": max(terminal.values()) <= float(cfg["tolerance"]),
    }
    report = {
        "status": "PASS" if all(checks.values()) else "STOP", "checks": checks,
        "blocking_failures": [key for key, value in checks.items() if not value],
        "terminal_relative_h1_error": terminal,
        "learned_authority_decisions": {"kdv": kdv["authority"], "ilw": ilw["authority"]},
        "scope": "Frozen V2 zero-shot representative trajectories for KdV and ILW; posthoc references only.",
    }
    (output / "acceptance.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    raise SystemExit(0 if report["status"] == "PASS" else 2)


if __name__ == "__main__":
    main()
