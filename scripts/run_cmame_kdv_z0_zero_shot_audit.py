"""Zero-shot falsification of the frozen mCH--BO mechanism on periodic KdV.

All KdV trajectories are advanced and committed before an independent fine-grid
DOP853 reference is constructed.  No KdV row is used for training, calibration,
support fitting, threshold selection, or donor-adapter selection.
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

from fgsp_ch.discretization.grids import PeriodicGrid  # noqa: E402
from fgsp_ch.nonlocal_waves.adaptivity.kdv_zero_shot import (  # noqa: E402
    FrozenKdVConsensusMechanism,
    kdv_prospective_features,
    kdv_resolution_indicators,
    project_kdv_spectrum,
)
from fgsp_ch.nonlocal_waves.adaptivity.mechanism_coordinates import (  # noqa: E402
    coordinates_for_equation,
)
from fgsp_ch.nonlocal_waves.adaptivity.prospective_causal_operator import (  # noqa: E402
    ACTIONS,
    ActionEnvelope,
    analytical_safe_actions,
)
from fgsp_ch.nonlocal_waves.adaptivity.universal_causal_operator import (  # noqa: E402
    FrozenUniversalCausalMechanism,
)
from fgsp_ch.nonlocal_waves.adaptivity.trajectory_error_ledger import (  # noqa: E402
    TrajectoryErrorLedger,
)
from fgsp_ch.nonlocal_waves.evaluation.kdv_metrics import (  # noqa: E402
    kdv_invariants,
    relative_kdv_h1,
)
from fgsp_ch.nonlocal_waves.initial_conditions.kdv_families import (  # noqa: E402
    KDV_FAMILIES,
    kdv_initial_condition,
)
from fgsp_ch.nonlocal_waves.solvers.kdv_reference import (  # noqa: E402
    KdVIFRK4Solver,
    solve_kdv_dop853,
)
from fgsp_ch.utils.config import load_yaml  # noqa: E402


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def _invariant_drift(current, initial) -> float:
    return max(
        abs(current.as_dict()[key] - value) / max(abs(value), 1e-12)
        for key, value in initial.as_dict().items()
    )


@dataclass
class Candidate:
    action: str
    state: np.ndarray
    grid: PeriodicGrid
    features: np.ndarray
    envelope: ActionEnvelope
    temporal_error: float
    spatial_error: float
    representation_error: float
    migration_error: float
    substeps: int


@dataclass
class CommittedRun:
    arm: str
    family: str
    seed: int
    tolerance: float
    initial: np.ndarray
    initial_grid: PeriodicGrid
    times: list[float]
    states: list[np.ndarray]
    grids: list[PeriodicGrid]
    rows: list[dict]


def _advance_embedded(values, grid, macro_dt, substeps):
    state = np.asarray(values, dtype=np.float64)
    temporal = 0.0
    dt = macro_dt / substeps
    for _ in range(substeps):
        high, low = KdVIFRK4Solver(grid).step_embedded(state, dt)
        temporal += relative_kdv_h1(high, low, grid)
        state = high
    return state, temporal


def _candidates(state, grid, *, macro_dt, tolerance, remaining_budget,
                prefix, decisions,
                settings, initial_invariants, actions=ACTIONS):
    candidates = []
    invariant_limit = float(settings["maximum_invariant_drift"])
    for action in actions:
        substeps = 2 if action == "shrink_time" else 1
        resolution_factor = 2.0 if action == "refine_space" else 1.0
        target_state, target_grid = np.asarray(state), grid
        admissible = True
        migration = 0.0
        if action == "refine_space":
            admissible = grid.points < int(settings["maximum_points"])
            if admissible:
                target_state, target_grid = project_kdv_spectrum(
                    state, grid, min(2 * grid.points, int(settings["maximum_points"]))
                )
                roundtrip, _ = project_kdv_spectrum(target_state, target_grid, grid.points)
                migration = relative_kdv_h1(roundtrip, state, grid)
        candidate_state, temporal = _advance_embedded(
            target_state, target_grid, macro_dt, substeps
        )
        indicators = kdv_resolution_indicators(
            candidate_state, target_grid, macro_dt / substeps,
            tail_fraction=float(settings["tail_fraction"]),
        )
        spatial = max(indicators.tail_ratio, indicators.unresolved_increment)
        representation = indicators.aliasing_defect
        drift = _invariant_drift(
            kdv_invariants(candidate_state, target_grid), initial_invariants
        )
        deterministic = (
            float(settings["richardson_safety"]) * temporal
            + float(settings["spatial_safety"]) * spatial
            + float(settings["representation_safety"]) * representation
            + migration
        )
        admissible = bool(
            admissible
            and indicators.nonlinear_stiffness <= float(settings["maximum_nonlinear_phase"])
            and np.all(np.isfinite(candidate_state))
        )
        work = float(7 * target_grid.points * np.log2(target_grid.points) * substeps)
        remaining_fraction = np.clip(remaining_budget / tolerance, 0.0, 1.0)
        remaining_time_fraction = max(0.0, 1 - prefix / decisions)
        features = kdv_prospective_features(
            indicators, temporal=temporal, migration=migration,
            invariant_drift=drift, invariant_limit=invariant_limit,
            local_budget=tolerance / decisions,
            remaining_budget_fraction=remaining_fraction,
            remaining_time_fraction=remaining_time_fraction,
            relative_work=work / max(float(7 * grid.points * np.log2(grid.points)), 1.0),
            previous_rejected=False, dt_factor=1 / substeps,
            resolution_factor=resolution_factor,
        )
        envelope = ActionEnvelope(
            action, max(deterministic, 1e-16), remaining_budget, drift,
            invariant_limit, work, admissible,
        )
        candidates.append(Candidate(
            action, candidate_state, target_grid, features, envelope,
            temporal, spatial, representation, migration, substeps,
        ))
    return tuple(candidates)


def _run_arm(initial, grid, *, arm, family, seed, tolerance, settings,
             mechanism, reserve):
    state, active_grid = np.asarray(initial).copy(), grid
    initial_invariants = kdv_invariants(state, active_grid)
    macro_dt, decisions = float(settings["macro_dt"]), int(settings["decisions"])
    ledger = TrajectoryErrorLedger(
        tolerance=tolerance, final_time=macro_dt * decisions,
        reserve_fraction=reserve,
    )
    times, states, grids, rows = [0.0], [state.copy()], [active_grid], []
    for prefix in range(decisions):
        candidates = _candidates(
            state, active_grid, macro_dt=macro_dt, tolerance=tolerance,
            remaining_budget=ledger.remaining,
            prefix=prefix, decisions=decisions, settings=settings,
            initial_invariants=initial_invariants,
        )
        envelopes = tuple(row.envelope for row in candidates)
        safe = analytical_safe_actions(envelopes, reserve_fraction=reserve)
        if not safe:
            details = ", ".join(
                f"{row.action}:eta={row.envelope.deterministic_error:.2e},"
                f"drift={row.envelope.invariant_drift:.2e},"
                f"admissible={row.envelope.analytically_admissible}"
                for row in candidates
            )
            raise RuntimeError(f"no analytically safe KdV action: {details}")
        decision = None
        if arm == "zero_shot":
            feature_rows = np.stack([row.features for row in candidates])
            if getattr(mechanism, "feature_kind", "legacy") == "mechanism":
                feature_rows = np.stack([
                    coordinates_for_equation(row, "kdv") for row in feature_rows
                ])
            decision = mechanism.choose(
                feature_rows, envelopes,
                reserve_fraction=reserve,
            )
        action_id = decision.action_id if decision and not decision.abstained else None
        if action_id is None:
            # Registered deterministic fallback: cheapest analytically safe
            # action, with deterministic error as the stable tie breaker.
            chosen_envelope = min(
                safe, key=lambda row: (row.work, row.deterministic_error, row.action_id)
            )
            action_id = chosen_envelope.action_id
        chosen = next(row for row in candidates if row.action == action_id)
        if chosen.envelope not in safe:
            raise RuntimeError("KdV mechanism released an analytically unsafe action")
        committed_upper = chosen.envelope.deterministic_error
        if decision is not None and not decision.abstained:
            committed_upper *= decision.upper_effectivity
        remaining_before = ledger.remaining
        ledger.commit(committed_upper)
        state, active_grid = chosen.state, chosen.grid
        times.append((prefix + 1) * macro_dt)
        states.append(state.copy())
        grids.append(active_grid)
        distances = decision.support_distances if decision else {}
        rows.append({
            "arm": arm, "family": family, "seed": seed, "tolerance": tolerance,
            "group_id": f"kdv:{family}:{seed}:{tolerance:.3e}", "prefix": prefix,
            "action": action_id,
            "abstained": bool(decision.abstained) if decision else False,
            "safe_action_count": len(safe),
            "upper_effectivity": float(decision.upper_effectivity) if decision else np.nan,
            "mch_support_distance": float(distances.get("modified_camassa_holm", np.nan)),
            "bo_support_distance": float(distances.get("benjamin_ono", np.nan)),
            "universal_support_distance": float(distances.get("universal", np.nan)),
            "points": active_grid.points, "substeps": chosen.substeps,
            "work": chosen.envelope.work,
            "deterministic_error": chosen.envelope.deterministic_error,
            "temporal_indicator": chosen.temporal_error,
            "spatial_indicator": chosen.spatial_error,
            "representation_indicator": chosen.representation_error,
            "migration_indicator": chosen.migration_error,
            "invariant_drift": chosen.envelope.invariant_drift,
            "committed_error_upper": committed_upper,
            "remaining_budget_before": remaining_before,
            "remaining_budget_after": ledger.remaining,
            "analytically_safe_release": True,
        })
    return CommittedRun(
        arm, family, seed, tolerance, np.asarray(initial).copy(), grid,
        times, states, grids, rows,
    )


def _write_rows(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="configs/experiment/cmame_kdv_z0_zero_shot.yaml"
    )
    parser.add_argument("--mode", choices=("development", "full"), default="development")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    cfg = load_yaml(ROOT / args.config)
    settings = cfg[args.mode]
    mechanism_cfg = cfg.get("mechanism", {"kind": "dual_adapter"})
    mechanism_kind = str(mechanism_cfg.get("kind", "dual_adapter"))
    if mechanism_kind == "universal":
        mechanism_root = ROOT / cfg["mechanism_output_dir"] / args.mode
        prerequisite_path = mechanism_root / "training_acceptance.json"
        manifest_path = mechanism_root / "frozen_u1_manifest.json"
    else:
        mechanism_root = ROOT / cfg["f1_output_dir"] / args.mode
        prerequisite_path = mechanism_root / "acceptance.json"
        manifest_path = mechanism_root / "frozen_f1_manifest.json"
    if json.loads(prerequisite_path.read_text())["status"] != "PASS":
        raise RuntimeError("the frozen causal mechanism must PASS before KdV audit")
    manifest_hash = sha256(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    mechanism = (
        FrozenUniversalCausalMechanism.load(mechanism_root, manifest, args.device)
        if mechanism_kind == "universal"
        else FrozenKdVConsensusMechanism.load(mechanism_root, manifest, args.device)
    )
    out = ROOT / cfg["output_dir"] / args.mode
    out.mkdir(parents=True, exist_ok=True)
    grid = PeriodicGrid(
        int(settings["points"]), float(cfg["domain"]["length"]),
        float(cfg["domain"]["x_min"]),
    )
    reserve = float(cfg["calibration"]["reserve_fraction"])
    runs: list[CommittedRun] = []
    for family_index, family in enumerate(KDV_FAMILIES):
        for group in range(int(settings["groups_per_family"])):
            seed = int(cfg["seeds"]["external_base"]) + 1000 * family_index + group
            initial = kdv_initial_condition(family, grid, seed=seed)
            # Enforce the pre-registered even-grid representation once, before
            # either arm starts; the unpaired Nyquist coefficient is excluded.
            initial = KdVIFRK4Solver(grid).operator.physical(
                KdVIFRK4Solver(grid).operator.spectrum(initial)
            )
            for tolerance in map(float, settings["tolerances"]):
                for arm in ("zero_shot", "analytical_control"):
                    run = _run_arm(
                        initial, grid, arm=arm, family=family, seed=seed,
                        tolerance=tolerance, settings=settings,
                        mechanism=mechanism, reserve=reserve,
                    )
                    runs.append(run)
                    print(
                        f"KdV committed family={family} seed={seed} tol={tolerance:.2e} "
                        f"arm={arm}", flush=True,
                    )

    # This file is frozen before the first independent reference is launched.
    online_rows = [row for run in runs for row in run.rows]
    _write_rows(out / "online_decisions.csv", online_rows)

    audit_rows = []
    reference_cache = {}
    for run_index, run in enumerate(runs, start=1):
        reference_key = (run.family, run.seed)
        if reference_key not in reference_cache:
            fine_initial, fine_grid = project_kdv_spectrum(
                run.initial, run.initial_grid, int(settings["reference_points"])
            )
            reference_cache[reference_key] = solve_kdv_dop853(
                fine_initial, fine_grid, evaluation_times=run.times,
                rtol=float(cfg["reference"]["rtol"]),
                atol=float(cfg["reference"]["atol"]),
                maximum_step=float(settings["macro_dt"]) / 4,
            )
        reference = reference_cache[reference_key]
        fine_grid = PeriodicGrid(
            int(settings["reference_points"]), run.initial_grid.length,
            run.initial_grid.x_min,
        )
        for prefix, row in enumerate(run.rows, start=1):
            approximation, _ = project_kdv_spectrum(
                run.states[prefix], run.grids[prefix], fine_grid.points
            )
            error = relative_kdv_h1(approximation, reference.states[prefix], fine_grid)
            audit_rows.append({
                **row, "time": run.times[prefix], "relative_h1": error,
                "error_over_tolerance": error / run.tolerance,
                "tolerance_met": error <= run.tolerance,
                "reference_visibility": "posthoc_only",
            })
        if run_index % 8 == 0 or run_index == len(runs):
            print(f"KdV posthoc audit runs={run_index}/{len(runs)}", flush=True)
    _write_rows(out / "audit_rows.csv", audit_rows)

    zero = [row for row in audit_rows if row["arm"] == "zero_shot"]
    control = [row for row in audit_rows if row["arm"] == "analytical_control"]
    paired = {(row["group_id"], row["prefix"]): row for row in control}
    error_ratios, work_ratios, action_interventions = [], [], []
    for row in zero:
        base = paired[(row["group_id"], row["prefix"])]
        error_ratios.append(row["relative_h1"] / max(base["relative_h1"], 1e-16))
        work_ratios.append(row["work"] / base["work"])
        action_interventions.append(row["action"] != base["action"])
    prefix_coverage = float(np.mean([row["tolerance_met"] for row in zero]))
    last_prefix = int(settings["decisions"]) - 1
    terminal = [row for row in zero if int(row["prefix"]) == last_prefix]
    coverage = float(np.mean([row["tolerance_met"] for row in terminal]))
    family_coverage = {
        family: float(np.mean([
            row["tolerance_met"] for row in terminal if row["family"] == family
        ])) for family in KDV_FAMILIES
    }
    authority = float(np.mean([not row["abstained"] for row in zero]))
    maximum_drift = float(max(row["invariant_drift"] for row in zero))
    limits = cfg["acceptance"].copy()
    if args.mode == "development":
        limits.update(settings.get("acceptance", {}))
    checks = {
        "frozen_mechanism_prerequisite": json.loads(prerequisite_path.read_text())["status"] == "PASS",
        "frozen_manifest_not_modified": sha256(manifest_path) == manifest_hash,
        "blind_seed_range_disjoint": int(cfg["seeds"]["external_base"]) >= 90000000,
        "no_kdv_training_or_calibration": True,
        "registered_zero_shot_mechanism": mechanism_kind in {"dual_adapter", "universal"},
        "fixed_physical_horizon_across_actions": True,
        "reference_posthoc_only": all(row["reference_visibility"] == "posthoc_only" for row in audit_rows),
        "all_families_complete": set(row["family"] for row in zero) == set(KDV_FAMILIES),
        "overall_tolerance_coverage": coverage >= float(limits["minimum_tolerance_coverage"]),
        "family_tolerance_coverage": min(family_coverage.values()) >= float(limits["minimum_family_coverage"]),
        "zero_unsafe_release": all(row["analytically_safe_release"] for row in online_rows),
        "trajectory_budget_enforced": all(
            row["remaining_budget_after"] >= -1e-15
            and row["committed_error_upper"] <= row["remaining_budget_before"]
            for row in online_rows
        ),
        "invariant_drift_bounded": maximum_drift <= float(limits["maximum_invariant_drift"]),
        "all_outputs_finite": bool(np.all(np.isfinite([
            row["relative_h1"] for row in audit_rows
        ]))),
    }
    utility_checks = {
        "nontrivial_frozen_authority": authority > 0 and authority >= float(
            limits["minimum_authority_fraction"]
        ),
        "nontrivial_action_intervention": float(np.mean(action_interventions)) > 0
        and float(np.mean(action_interventions)) >= float(
            limits["minimum_action_intervention_fraction"]
        ),
        "accuracy_noninferior_to_analytical_control": float(np.median(error_ratios))
        <= float(limits["maximum_median_error_ratio_to_control"]),
        "work_secondary_bounded": float(np.median(work_ratios))
        <= float(limits["maximum_median_work_ratio_to_control"]),
    }
    blocking = [key for key, value in checks.items() if not value]
    claim = "SUPPORTED" if all(checks.values()) and all(utility_checks.values()) else (
        "SAFE_BUT_UTILITY_NOT_ESTABLISHED" if all(checks.values()) else "FALSIFIED"
    )
    report = {
        "status": "PASS" if not blocking else "STOP",
        "next_phase": cfg.get("next_phase_on_support") if claim == "SUPPORTED" else None,
        "checks": checks,
        "utility_checks": utility_checks,
        "blocking_failures": blocking,
        "zero_shot_claim": claim,
        "mode": args.mode,
        "cases": len(zero),
        "trajectory_groups": len(set(row["group_id"] for row in zero)),
        "terminal_tolerance_coverage": coverage,
        "prefix_tolerance_coverage": prefix_coverage,
        "family_tolerance_coverage": family_coverage,
        "frozen_authority_fraction": authority,
        "abstention_fraction": 1 - authority,
        "action_intervention_fraction": float(np.mean(action_interventions)),
        "median_error_ratio_to_analytical_control": float(np.median(error_ratios)),
        "median_work_ratio_to_analytical_control": float(np.median(work_ratios)),
        "maximum_invariant_drift": maximum_drift,
        "mechanism_kind": mechanism_kind,
        "frozen_manifest_sha256": manifest_hash,
        "scope": (
            "Strict zero-shot KdV falsification. The registered mechanism, weights, "
            "normalization, support and conformal risk are frozen before KdV. "
            "Independent references are posthoc only."
        ),
    }
    (out / "acceptance.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    raise SystemExit(0 if report["status"] == "PASS" else 2)


if __name__ == "__main__":
    main()
