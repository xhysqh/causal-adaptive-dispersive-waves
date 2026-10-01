"""Frozen-V2 deployment on the registered power-law dispersion atlas.

This script is intentionally limited to the five precommitted representative
orders used by the manuscript figure.  The frozen mCH/BO V2 ensemble is never
updated.  DOP853 references are constructed only after all online decisions
have been committed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from fgsp_ch.discretization.grids import PeriodicGrid  # noqa: E402
from fgsp_ch.nonlocal_waves.adaptivity.mechanism_coordinates import (  # noqa: E402
    CausalHistoryState, causal_control_coordinates, mechanism_coordinates,
    mechanism_history_coordinates, v2_causal_coordinates,
)
from fgsp_ch.nonlocal_waves.adaptivity.power_law_unknown import (  # noqa: E402
    power_law_features, power_law_indicators,
)
from fgsp_ch.nonlocal_waves.adaptivity.trajectory_error_ledger import TrajectoryErrorLedger  # noqa: E402
from fgsp_ch.nonlocal_waves.adaptivity.v2_safety_shielded_controller import (  # noqa: E402
    FrozenV2PredictiveRiskEnsemble, V2Candidate, V2SafetyShieldedController,
    v2_controller_config,
)
from fgsp_ch.nonlocal_waves.evaluation.power_law_metrics import (  # noqa: E402
    power_law_invariants, relative_power_law_h1,
)
from fgsp_ch.nonlocal_waves.initial_conditions.kdv_families import kdv_initial_condition  # noqa: E402
from fgsp_ch.nonlocal_waves.operators.power_law_fourier import (  # noqa: E402
    project_power_law_spectrum,
)
from fgsp_ch.nonlocal_waves.operators.wave_traits import DispersiveWaveTraits  # noqa: E402
from fgsp_ch.nonlocal_waves.solvers.power_law_reference import (  # noqa: E402
    PowerLawIFRK4Solver, solve_power_law_dop853,
)
from fgsp_ch.utils.config import load_yaml  # noqa: E402


ORDERS = (2.2, 2.4, 2.6, 2.8, 3.2)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def invariant_drift(current, initial) -> float:
    return max(
        abs(current.as_dict()[key] - value) / max(abs(value), 1.0e-12)
        for key, value in initial.as_dict().items()
    )


def advance(values, grid, traits, macro_dt, substeps):
    state = np.asarray(values, dtype=float)
    temporal = 0.0
    for _ in range(substeps):
        high, low = PowerLawIFRK4Solver(grid, traits).step_embedded(
            state, macro_dt / substeps,
        )
        temporal += relative_power_law_h1(high, low, grid)
        state = high
    return state, temporal


def verify_candidate(state, grid, traits, action, candidate, candidate_grid,
                     macro_dt, maximum_points):
    """Independent 2:1 shadow check used when the frozen controller requests it."""
    target, target_grid = np.asarray(state), grid
    if action == "refine_space" and grid.points < maximum_points:
        target, target_grid = project_power_law_spectrum(
            state, grid, min(2 * grid.points, maximum_points), traits,
        )
    fine_points = min(2 * target_grid.points, 2 * maximum_points)
    fine, fine_grid = project_power_law_spectrum(
        target, target_grid, fine_points, traits,
    )
    substeps = 2 if action == "shrink_time" else 1
    shadow, _ = advance(fine, fine_grid, traits, macro_dt, 2 * substeps)
    approximation, _ = project_power_law_spectrum(
        candidate, candidate_grid, fine_grid.points, traits,
    )
    error = relative_power_law_h1(approximation, shadow, fine_grid)
    work = float(14 * fine_grid.points * np.log2(fine_grid.points) * substeps)
    return max(float(error), 1.0e-16), work


def build_candidate(
    state, grid, traits, *, action, macro_dt, tolerance, remaining,
    local_budget, prefix, decisions, maximum_points, history, control,
    u2_cfg,
):
    substeps = 2 if action == "shrink_time" else 1
    target = np.asarray(state)
    target_grid = grid
    migration = 0.0
    allowed = True
    if action == "refine_space":
        allowed = grid.points < maximum_points
        if allowed:
            target, target_grid = project_power_law_spectrum(
                state, grid, min(2 * grid.points, maximum_points), traits,
            )
            restored, _ = project_power_law_spectrum(
                target, target_grid, grid.points, traits,
            )
            migration = relative_power_law_h1(restored, state, grid)
    candidate, temporal = advance(target, target_grid, traits, macro_dt, substeps)
    indicator = power_law_indicators(
        candidate, target_grid, traits, macro_dt / substeps,
        tail_fraction=float(u2_cfg["traits"]["tail_fraction"]),
    )
    spatial = max(indicator.tail_ratio, indicator.unresolved_increment)
    representation = indicator.aliasing_defect
    initial_invariants = power_law_invariants(target, target_grid, traits)
    drift = invariant_drift(
        power_law_invariants(candidate, target_grid, traits), initial_invariants,
    )
    deterministic = max(
        float(u2_cfg["estimator"]["richardson_safety"]) * temporal
        + float(u2_cfg["estimator"]["spatial_safety"]) * spatial
        + float(u2_cfg["estimator"]["representation_safety"]) * representation
        + migration,
        1.0e-16,
    )
    work = float(7 * target_grid.points * np.log2(target_grid.points) * substeps)
    legacy = power_law_features(
        indicator, temporal=temporal, migration=migration,
        invariant_drift=drift, invariant_limit=1.0e-3,
        local_budget=local_budget,
        remaining_budget_fraction=np.clip(remaining / tolerance, 0.0, 1.0),
        remaining_time_fraction=max(0.0, 1.0 - prefix / decisions),
        relative_work=work / max(7 * grid.points * np.log2(grid.points), 1.0),
        dt_factor=1.0 / substeps,
        resolution_factor=2.0 if action == "refine_space" else 1.0,
    )
    base = mechanism_coordinates(
        legacy, localized_available=True, spectral_available=True,
        migration_available=True,
    )
    features = v2_causal_coordinates(base, history, control)
    admissible = bool(
        allowed and np.all(np.isfinite(candidate)) and drift <= 1.0e-3
        and indicator.nonlinear_phase <= float(u2_cfg["traits"]["maximum_nonlinear_phase"])
    )
    shield = V2Candidate(
        action, features, deterministic, remaining, work, admissible,
        bool(allowed and migration <= 0.15), local_budget=local_budget,
        spectral_tail=indicator.tail_ratio,
        localized_sharpness=indicator.localized_sharpness,
        history_velocity_norm=float(np.linalg.norm(history.velocity)),
        history_acceleration_norm=float(np.linalg.norm(history.acceleration)),
    )
    return shield, candidate, target_grid, base, drift, migration


def run_order(order, index, cfg, stage2_cfg, u2_cfg, ensemble, settings):
    length = float(cfg["domain"]["length"])
    x_min = float(cfg["domain"]["x_min"])
    grid = PeriodicGrid(int(settings["points"]), length, x_min)
    transport = float(u2_cfg["traits"]["nonlinear_intercept"]) + float(
        u2_cfg["traits"]["nonlinear_slope"]
    ) * (order - 2.0)
    traits = DispersiveWaveTraits(order, transport, not float(order).is_integer())
    seed = int(cfg["seed_base"]) + index * 100000
    initial = kdv_initial_condition("single_soliton", grid, seed=seed)
    initial, _ = project_power_law_spectrum(initial, grid, grid.points, traits)
    state = initial.copy()
    initial_grid = grid
    decisions = int(settings["decisions"])
    macro_dt = float(settings["macro_dt"])
    tolerance = float(settings["tolerance"])
    ledger = TrajectoryErrorLedger(tolerance, decisions * macro_dt, reserve_fraction=0.10)
    history_state = CausalHistoryState(float(stage2_cfg["protocol"]["history_rho"]))
    controller = V2SafetyShieldedController(
        ensemble, v2_controller_config(load_yaml(ROOT / cfg["stage4_config"])["controller"]),
    )
    states = [state.copy()]
    grids = [grid]
    times = [0.0]
    actions = []
    rows = []
    hold_features = []
    for prefix in range(decisions):
        local_budget = tolerance / decisions
        control = causal_control_coordinates(
            remaining_budget_fraction=np.clip(ledger.remaining / tolerance, 0.0, 1.0),
            remaining_time_fraction=max(0.0, 1.0 - prefix / decisions),
            resolution_fraction=grid.points / float(settings["maximum_points"]),
            current_dt=macro_dt,
            cooldown_fraction=controller.state.cooldown_remaining / max(controller.config.cooldown_prefixes, 1),
        )
        provisional_history = CausalHistoryState(float(stage2_cfg["protocol"]["history_rho"])).observe(np.zeros(6))
        provisional = build_candidate(
            state, grid, traits, action="hold", macro_dt=macro_dt,
            tolerance=tolerance, remaining=ledger.remaining,
            local_budget=local_budget, prefix=prefix, decisions=decisions,
            maximum_points=int(settings["maximum_points"]), history=provisional_history,
            control=control, u2_cfg=u2_cfg,
        )
        history = history_state.observe(mechanism_history_coordinates(provisional[3]))
        cache = {}

        def construct(action):
            if action not in cache:
                cache[action] = build_candidate(
                    state, grid, traits, action=action, macro_dt=macro_dt,
                    tolerance=tolerance, remaining=ledger.remaining,
                    local_budget=local_budget, prefix=prefix, decisions=decisions,
                    maximum_points=int(settings["maximum_points"]), history=history,
                    control=control, u2_cfg=u2_cfg,
                )
            return cache[action]

        def builder(requested):
            return tuple(construct(action)[0] for action in requested if action != "coarsen_space")

        decision = controller.choose(builder)
        if decision.action_id is None:
            raise RuntimeError(f"no admissible V2 power-law action for p={order:g}, prefix={prefix}")
        shield, state, grid, _, drift_value, migration = construct(decision.action_id)
        hold_upper_risk = float(controller.ensemble.assess(
            np.stack([cache["hold"][0].features])
        )[0].upper_risk)
        hold_features.append(cache["hold"][0].features.copy())
        ledger.commit(shield.deterministic_error)
        candidate_work = float(sum(
            cache[action][0].work for action in decision.considered_actions
            if action in cache
        ))
        verification_error, verification_work = (np.nan, 0.0)
        if decision.probe_required:
            verification_error, verification_work = verify_candidate(
                states[-1], grids[-1], traits, decision.action_id, state, grid,
                macro_dt, int(settings["maximum_points"]),
            )
        states.append(state.copy())
        grids.append(grid)
        # ``shrink_time`` is implemented as two half steps spanning the same
        # macro interval, so every committed prefix advances by ``macro_dt``.
        times.append(times[-1] + macro_dt)
        actions.append(decision.action_id)
        rows.append({
            "order": order, "prefix": prefix, "time": times[-1],
            "action": decision.action_id, "points": grid.points,
            "controller_state": decision.state,
            "learned_authority": decision.learned_authority,
            "upper_risk": decision.upper_risk,
            "hold_upper_risk": hold_upper_risk,
            "deterministic_error": shield.deterministic_error,
            "candidate_work": candidate_work,
            "verification_work": verification_work,
            "verification_error": verification_error,
            "probe_required": decision.probe_required,
            "committed_work": shield.work,
            "remaining_budget": ledger.remaining,
            "invariant_drift": drift_value, "projection_defect": migration,
        })
    fine_grid = PeriodicGrid(int(settings["reference_points"]), length, x_min)
    fine_initial, _ = project_power_law_spectrum(initial, initial_grid, fine_grid.points, traits)
    reference = solve_power_law_dop853(
        fine_initial, fine_grid, traits, evaluation_times=np.asarray(times),
        rtol=float(u2_cfg["reference"]["rtol"]),
        atol=float(u2_cfg["reference"]["atol"]), maximum_step=macro_dt / 4,
    )
    adaptive = np.stack([
        project_power_law_spectrum(values, active_grid, fine_grid.points, traits)[0]
        for values, active_grid in zip(states, grids, strict=True)
    ])
    # Posthoc H=2 counterfactual: retain the current grid and original
    # internal substep for two future macro intervals.  It labels the true
    # HOLD risk without entering the frozen online controller.
    realized_hold_risk = np.full(len(states), np.nan)
    local_budget = tolerance / decisions
    for index in range(len(states) - 2):
        held_state, held_grid = states[index].copy(), grids[index]
        risks = []
        for horizon in (1, 2):
            held_state, _ = advance(held_state, held_grid, traits, macro_dt, 1)
            projected, _ = project_power_law_spectrum(
                held_state, held_grid, fine_grid.points, traits,
            )
            risks.append(relative_power_law_h1(projected, reference.states[index + horizon], fine_grid) / local_budget)
        realized_hold_risk[index] = max(risks)
    terminal = relative_power_law_h1(adaptive[-1], reference.states[-1], fine_grid)
    return {
        "order": order, "seed": seed, "traits": traits.mechanism_vector(),
        "x": fine_grid.x, "times": np.asarray(times),
        "reference": reference.states, "adaptive": adaptive,
        "actions": np.asarray(actions),
        "active_points": np.asarray([row.points for row in grids], dtype=np.int64),
        "terminal_error": float(terminal), "rows": rows,
        "hold_features": np.stack(hold_features),
        "realized_hold_risk": realized_hold_risk,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment/cmame_v2_power_law_showcase.yaml")
    parser.add_argument("--mode", choices=("development", "full"), default="full")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--tolerance", type=float)
    parser.add_argument("--output-dir")
    parser.add_argument("--manifest")
    parser.add_argument("--orders", type=float, nargs="+")
    args = parser.parse_args()
    cfg = load_yaml(ROOT / args.config)
    settings = dict(cfg[args.mode])
    if args.tolerance is not None:
        settings["tolerance"] = float(args.tolerance)
    stage2_cfg = load_yaml(ROOT / cfg["stage2_config"])
    u2_cfg = load_yaml(ROOT / cfg["u2_config"])
    manifest_path = ROOT / (args.manifest or cfg["source_manifest"])
    manifest_before = sha256(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    ensemble = FrozenV2PredictiveRiskEnsemble.load(
        manifest_path.parent, manifest, device=args.device,
    )
    output = ROOT / (args.output_dir or cfg["output_dir"]) / args.mode
    output.mkdir(parents=True, exist_ok=True)
    orders = tuple(args.orders) if args.orders else ORDERS
    if any(order not in ORDERS for order in orders):
        raise ValueError(f"orders must be selected from {ORDERS}")
    runs = [
        run_order(order, ORDERS.index(order), cfg, stage2_cfg, u2_cfg, ensemble, settings)
        for order in orders
    ]
    rows = [row for run in runs for row in run["rows"]]
    with (output / "online_decisions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    common_times = np.asarray(runs[0]["times"])
    if not all(np.allclose(run["times"], common_times, rtol=0.0, atol=1.0e-14) for run in runs):
        raise RuntimeError("power-law representative trajectories do not share the registered macro-time grid")
    np.savez_compressed(
        output / "representative_trajectories.npz",
        dispersion_orders=np.asarray(orders),
        trait_vectors=np.stack([run["traits"] for run in runs]),
        seeds=np.asarray([run["seed"] for run in runs]),
        tolerance=np.asarray(float(settings["tolerance"])),
        x=runs[0]["x"], times=common_times,
        reference=np.stack([run["reference"] for run in runs]),
        adaptive=np.stack([run["adaptive"] for run in runs]),
        absolute_error=np.stack([np.abs(run["adaptive"] - run["reference"]) for run in runs]),
        active_points=np.stack([run["active_points"] for run in runs]),
        actions=np.stack([
            np.concatenate((np.asarray(["initial"]), run["actions"]))
            for run in runs
        ]),
        selection_rule=np.asarray("five precommitted orders; frozen mCH/BO V2 controller"),
        upper_risk=np.stack([
            np.r_[np.nan, [row["upper_risk"] for row in run["rows"]]] for run in runs
        ]),
        hold_upper_risk=np.stack([
            np.r_[np.nan, [row["hold_upper_risk"] for row in run["rows"]]] for run in runs
        ]),
        probe_required=np.stack([
            np.r_[False, [row["probe_required"] for row in run["rows"]]] for run in runs
        ]),
        learned_authority=np.stack([
            np.r_[False, [row["learned_authority"] for row in run["rows"]]] for run in runs
        ]),
        deterministic_error=np.stack([
            np.r_[0.0, [row["deterministic_error"] for row in run["rows"]]] for run in runs
        ]),
        remaining_budget=np.stack([
            np.r_[np.nan, [row["remaining_budget"] for row in run["rows"]]] for run in runs
        ]),
        hold_features=np.stack([run["hold_features"] for run in runs]),
        realized_hold_risk=np.stack([run["realized_hold_risk"] for run in runs]),
    )
    terminal = {str(run["order"]): run["terminal_error"] for run in runs}
    checks = {
        "frozen_manifest_unchanged": sha256(manifest_path) == manifest_before,
        "all_references_posthoc": True,
        "all_orders_present": tuple(run["order"] for run in runs) == orders,
        "all_outputs_finite": all(np.all(np.isfinite(run["adaptive"])) for run in runs),
        "registered_tolerance_met": max(terminal.values()) <= float(settings["tolerance"]),
        "macro_time_consistent_across_actions": all(
            np.allclose(np.diff(run["times"]), float(settings["macro_dt"]), rtol=0.0, atol=1.0e-14)
            for run in runs
        ),
    }
    report = {
        "status": "PASS" if all(checks.values()) else "STOP",
        "checks": checks,
        "blocking_failures": [key for key, value in checks.items() if not value],
        "terminal_relative_h1_error": terminal,
        "decisions": len(rows),
        "learned_authority_decisions": int(sum(bool(row["learned_authority"]) for row in rows)),
        "online_candidate_work": {
            str(order): float(sum(
                row["candidate_work"] for row in rows if float(row["order"]) == order
            )) for order in orders
        },
        "online_total_work": {
            str(order): float(sum(
                row["candidate_work"] + row["verification_work"]
                for row in rows if float(row["order"]) == order
            )) for order in orders
        },
        "verification_calls": int(sum(bool(row["probe_required"]) for row in rows)),
        "manifest_sha256": sha256(manifest_path),
        "scope": "Frozen V2 transfer showcase for the five registered power-law orders; no retraining or calibration.",
    }
    (output / "acceptance.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    raise SystemExit(0 if report["status"] == "PASS" else 2)


if __name__ == "__main__":
    main()
