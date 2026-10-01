"""Frozen U2.1 blind audit on the non-power-law ILW bridge."""

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
from fgsp_ch.nonlocal_waves.adaptivity.ilw_unknown import ilw_features, ilw_indicators  # noqa: E402
from fgsp_ch.nonlocal_waves.adaptivity.mechanism_coordinates import mechanism_coordinates  # noqa: E402
from fgsp_ch.nonlocal_waves.adaptivity.prospective_causal_operator import (  # noqa: E402
    ACTIONS, ActionEnvelope, analytical_safe_actions,
)
from fgsp_ch.nonlocal_waves.adaptivity.self_certifying_operator import (  # noqa: E402
    ReferenceFreeCertificate, SelfCertificationConfig,
)
from fgsp_ch.nonlocal_waves.adaptivity.trajectory_error_ledger import TrajectoryErrorLedger  # noqa: E402
from fgsp_ch.nonlocal_waves.adaptivity.universal_causal_operator import FrozenUniversalCausalMechanism  # noqa: E402
from fgsp_ch.nonlocal_waves.evaluation.ilw_metrics import ilw_invariants, relative_ilw_h1  # noqa: E402
from fgsp_ch.nonlocal_waves.evaluation.grouped_statistics import (  # noqa: E402
    grouped_median_log_ratio_interval,
)
from fgsp_ch.nonlocal_waves.initial_conditions.kdv_families import kdv_initial_condition  # noqa: E402
from fgsp_ch.nonlocal_waves.operators.ilw_fourier import ILWFourierOperator, project_ilw_spectrum  # noqa: E402
from fgsp_ch.nonlocal_waves.solvers.ilw_reference import ILWIFRK54Solver, solve_ilw_dop853  # noqa: E402
from fgsp_ch.utils.config import load_yaml  # noqa: E402


@dataclass(frozen=True, slots=True)
class Candidate:
    action: str
    state: np.ndarray
    grid: PeriodicGrid
    features: np.ndarray
    envelope: ActionEnvelope
    substeps: int
    symbol_vector: np.ndarray
    indicators: object


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest().upper()


def invariant_drift(current, initial):
    return max(
        abs(current.as_dict()[key] - value) / max(abs(value), 1e-12)
        for key, value in initial.as_dict().items()
    )


def advance(values, grid, depth, beta, macro_dt, substeps):
    state, temporal = np.asarray(values), 0.0
    solver = ILWIFRK54Solver(grid, depth, beta)
    for _ in range(substeps):
        high, low = solver.step_embedded(state, macro_dt / substeps)
        temporal += relative_ilw_h1(high, low, grid)
        state = high
    return state, temporal


def build_candidates(state, grid, depth, beta, *, macro_dt, tolerance,
                     remaining, local_budget, prefix, decisions, settings, config,
                     initial_invariants, actions=ACTIONS):
    candidates = []
    for action in actions:
        substeps = 2 if action == "shrink_time" else 1
        target, target_grid, migration = np.asarray(state), grid, 0.0
        admissible = True
        if action == "refine_space":
            admissible = grid.points < int(settings["maximum_points"])
            if admissible:
                target, target_grid = project_ilw_spectrum(
                    state, grid, min(2 * grid.points, int(settings["maximum_points"])),
                    depth, beta,
                )
                restored, _ = project_ilw_spectrum(
                    target, target_grid, grid.points, depth, beta
                )
                migration = relative_ilw_h1(restored, state, grid)
        candidate, temporal = advance(
            target, target_grid, depth, beta, macro_dt, substeps
        )
        indicators = ilw_indicators(
            candidate, target_grid, depth, macro_dt / substeps,
            nonlinear_transport=beta,
            tail_fraction=float(config["traits"]["tail_fraction"]),
        )
        spatial = max(indicators.tail_ratio, indicators.unresolved_increment)
        representation = indicators.aliasing_defect
        drift = invariant_drift(
            ilw_invariants(candidate, target_grid, depth, beta), initial_invariants
        )
        deterministic = max(
            float(config["estimator"]["richardson_safety"]) * temporal
            + float(config["estimator"]["spatial_safety"]) * spatial
            + float(config["estimator"]["representation_safety"]) * representation
            + migration,
            1e-16,
        )
        admissible = bool(
            admissible and np.all(np.isfinite(candidate))
            and indicators.nonlinear_phase <= float(config["traits"]["maximum_nonlinear_phase"])
        )
        work = float(7 * target_grid.points * np.log2(target_grid.points) * substeps)
        legacy = ilw_features(
            indicators,
            temporal=temporal,
            migration=migration,
            invariant_drift=drift,
            invariant_limit=float(settings["maximum_invariant_drift"]),
            local_budget=local_budget,
            remaining_budget_fraction=np.clip(remaining / tolerance, 0.0, 1.0),
            remaining_time_fraction=max(0.0, 1.0 - prefix / decisions),
            relative_work=work / max(7 * grid.points * np.log2(grid.points), 1.0),
            dt_factor=1.0 / substeps,
            resolution_factor=2.0 if action == "refine_space" else 1.0,
        )
        features = mechanism_coordinates(
            legacy,
            localized_available=True,
            spectral_available=True,
            migration_available=True,
        )
        candidates.append(Candidate(
            action, candidate, target_grid, features,
            ActionEnvelope(
                action, deterministic, remaining, drift,
                float(settings["maximum_invariant_drift"]), work, admissible,
            ),
            substeps,
            indicators.symbol.mechanism_vector(),
            indicators,
        ))
    return tuple(candidates)


def hierarchical_probe(state, grid, depth, beta, candidate, macro_dt, settings):
    target, target_grid = np.asarray(state), grid
    if candidate.action == "refine_space":
        target, target_grid = project_ilw_spectrum(
            state, grid, min(2 * grid.points, int(settings["maximum_points"])), depth, beta
        )
    points = min(2 * target_grid.points, int(settings["probe_maximum_points"]))
    fine, fine_grid = project_ilw_spectrum(target, target_grid, points, depth, beta)
    shadow, _ = advance(
        fine, fine_grid, depth, beta, macro_dt, 2 * candidate.substeps
    )
    approximation, _ = project_ilw_spectrum(
        candidate.state, candidate.grid, fine_grid.points, depth, beta
    )
    error = relative_ilw_h1(approximation, shadow, fine_grid)
    work = 14 * fine_grid.points * np.log2(fine_grid.points) * candidate.substeps
    return max(error, 1e-16), float(work)


def run_trajectory(initial, grid, depth, beta, *, arm, family, seed, tolerance,
                   settings, config, mechanism, certificate_config, reserve):
    state, active = np.asarray(initial).copy(), grid
    initial_inv = ilw_invariants(state, active, depth, beta)
    decisions, macro_dt = int(settings["decisions"]), float(settings["macro_dt"])
    ledger = TrajectoryErrorLedger(tolerance, decisions * macro_dt, reserve)
    certificate = ReferenceFreeCertificate(certificate_config)
    states, grids, times, rows = [state.copy()], [active], [0.0], []
    for prefix in range(decisions):
        local_budget = ledger.local_budget(prefix * macro_dt, macro_dt)
        options = build_candidates(
            state, active, depth, beta,
            macro_dt=macro_dt, tolerance=tolerance, remaining=ledger.remaining,
            local_budget=local_budget,
            prefix=prefix, decisions=decisions, settings=settings, config=config,
            initial_invariants=initial_inv,
        )
        envelopes = tuple(option.envelope for option in options)
        safe = analytical_safe_actions(envelopes, reserve_fraction=reserve)
        if not safe:
            raise RuntimeError("U2.1 has no analytically safe ILW action")
        safe_options = [option for option in options if option.envelope in safe]
        candidate_work = float(sum(option.envelope.work for option in options))
        probe_count, probe_work = 0, 0.0
        authority, revoked = False, False
        status, certified, raw = "analytical_fallback", np.nan, np.nan

        if arm == "self_certifying":
            proposal = mechanism.raw_proposal(
                np.stack([option.features for option in options]), envelopes,
                reserve_fraction=reserve,
            )
            proposed = next((option for option in options if option.action == proposal.action_id), None)
            raw = proposal.upper_effectivity
            measured = None
            if proposed is not None:
                preliminary = certificate.decide(
                    action_id=proposed.action, raw_effectivity=raw,
                    deterministic_error=proposed.envelope.deterministic_error,
                    remaining_budget=ledger.remaining,
                    support_distance=proposal.support_distances.get("universal", np.nan),
                    reserve_fraction=reserve,
                )
                decision = preliminary
                if preliminary.probe_required:
                    measured, probe_work = hierarchical_probe(
                        state, active, depth, beta, proposed, macro_dt, settings
                    )
                    probe_count = 1
                    decision = certificate.decide(
                        action_id=proposed.action, raw_effectivity=raw,
                        deterministic_error=proposed.envelope.deterministic_error,
                        remaining_budget=ledger.remaining,
                        support_distance=proposal.support_distances.get("universal", np.nan),
                        probe_error=measured, reserve_fraction=reserve,
                    )
                authority, revoked = decision.authority, decision.revoked
                certified, status = decision.certified_upper, decision.state
                if authority:
                    chosen, committed = proposed, certified
                elif measured is not None and certificate_config.probe_safety * measured <= (1.0 - reserve) * ledger.remaining:
                    chosen, committed = proposed, certificate_config.probe_safety * measured
                    status = "probed_commit"
                else:
                    chosen = min(safe_options, key=lambda row: (row.envelope.work, row.envelope.deterministic_error, row.action))
                    committed = chosen.envelope.deterministic_error
            else:
                chosen = min(safe_options, key=lambda row: row.envelope.work)
                committed = chosen.envelope.deterministic_error
        elif arm == "always_probe":
            measured_options = []
            for option in safe_options:
                error, work = hierarchical_probe(
                    state, active, depth, beta, option, macro_dt, settings
                )
                measured_options.append((option, error))
                probe_count += 1
                probe_work += work
            admitted = [pair for pair in measured_options if certificate_config.probe_safety * pair[1] <= (1.0 - reserve) * ledger.remaining]
            admitted_ids = {id(pair[0]) for pair in admitted}
            chosen, error = min(
                admitted or measured_options,
                key=lambda pair: (0 if id(pair[0]) in admitted_ids else 1, pair[0].envelope.work, pair[1]),
            )
            committed = certificate_config.probe_safety * error if id(chosen) in admitted_ids else chosen.envelope.deterministic_error
            status = "always_probe"
        else:
            chosen = min(safe_options, key=lambda row: (row.envelope.work, row.envelope.deterministic_error, row.action))
            committed = chosen.envelope.deterministic_error

        if not ledger.admissible(committed):
            chosen = min(safe_options, key=lambda row: row.envelope.deterministic_error)
            committed, authority, status = chosen.envelope.deterministic_error, False, "budget_fallback"
        before = ledger.remaining
        ledger.commit(committed)
        state, active = chosen.state, chosen.grid
        states.append(state.copy())
        grids.append(active)
        times.append((prefix + 1) * macro_dt)
        rows.append({
            "arm": arm, "depth": depth,
            "symbol_vector": json.dumps(chosen.symbol_vector.tolist()),
            "family": family, "seed": seed, "tolerance": tolerance,
            "group_id": f"d{depth:.4g}:{family}:{seed}:{tolerance:.2e}",
            "prefix": prefix, "action": chosen.action,
            "certificate_state": status, "neural_authority": authority,
            "revoked": revoked, "probe_count": probe_count,
            "probe_work": probe_work, "raw_effectivity": raw,
            "certified_upper": certified,
            # All candidate states were advanced before ranking.  The committed
            # state reuses one of them and must not be counted a second time.
            "candidate_solver_work": candidate_work,
            "committed_solver_work": chosen.envelope.work,
            "committed_state_reused": True,
            "pde_online_work": candidate_work + probe_work,
            "model_candidate_evaluations": len(options) if arm == "self_certifying" else 0,
            "solver_work": chosen.envelope.work,  # legacy diagnostic only
            "committed_error_upper": committed,
            "local_budget": local_budget,
            "remaining_budget_before": before,
            "remaining_budget_after": ledger.remaining,
            "invariant_drift": chosen.envelope.invariant_drift,
            "analytically_safe_release": chosen.envelope in safe,
        })
    return {
        "depth": depth, "arm": arm, "family": family, "seed": seed,
        "tolerance": tolerance, "initial": np.asarray(initial), "grid": grid,
        "states": states, "grids": grids, "times": times, "rows": rows,
    }


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def physical_group(row):
    return f"d{float(row['depth']):.12g}:{row['family']}:{int(row['seed'])}"


def export_representative_trajectories(output, runs, references, settings, config, beta):
    """Persist precommitted examples needed by paper figures without rerunning."""
    selected = []
    minimum_tolerance = min(map(float, settings["tolerances"]))
    first_family = config["families"][0]
    for depth in map(float, settings["depths"]):
        choices = [run for run in runs if (
            run["arm"] == "self_certifying"
            and float(run["depth"]) == depth
            and run["family"] == first_family
            and float(run["tolerance"]) == minimum_tolerance
        )]
        selected.append(min(choices, key=lambda run: int(run["seed"])))
    fine_grid = PeriodicGrid(int(settings["reference_points"]), **config["domain"])
    adaptive, reference, active_points, actions, symbol_vectors = [], [], [], [], []
    for run in selected:
        adaptive.append(np.stack([
            project_ilw_spectrum(state, grid, fine_grid.points, run["depth"], beta)[0]
            for state, grid in zip(run["states"], run["grids"])
        ]))
        reference.append(references[(run["depth"], run["family"], run["seed"])].states)
        active_points.append([grid.points for grid in run["grids"]])
        actions.append(["initial"] + [row["action"] for row in run["rows"]])
        symbol_vectors.append(np.vstack([
            np.full(5, np.nan),
            *[np.asarray(json.loads(row["symbol_vector"])) for row in run["rows"]],
        ]))
    adaptive_array = np.stack(adaptive)
    reference_array = np.stack(reference)
    np.savez_compressed(
        output / "representative_trajectories.npz",
        depths=np.asarray([run["depth"] for run in selected]),
        families=np.asarray([run["family"] for run in selected]),
        seeds=np.asarray([run["seed"] for run in selected]),
        tolerance=np.asarray(minimum_tolerance),
        x=fine_grid.x,
        times=np.asarray(selected[0]["times"]),
        reference=reference_array,
        adaptive=adaptive_array,
        absolute_error=np.abs(adaptive_array - reference_array),
        active_points=np.asarray(active_points, dtype=np.int64),
        actions=np.asarray(actions),
        symbol_geometry=np.stack(symbol_vectors),
        selection_rule=np.asarray("first registered family and seed; strictest tolerance"),
    )


def reference_floor_audit(output, runs, references, settings, config, beta):
    spec = config.get("reference_floor_audit", {})
    if not bool(spec.get("enabled", False)):
        return [], 0.0
    rows = []
    high_points = int(settings["reference_points"]) * int(spec["points_factor"])
    for depth in map(float, settings["depths"]):
        run = min(
            (row for row in runs if row["arm"] == "self_certifying" and float(row["depth"]) == depth),
            key=lambda row: (config["families"].index(row["family"]), int(row["seed"]), float(row["tolerance"])),
        )
        high_initial, high_grid = project_ilw_spectrum(
            run["initial"], run["grid"], high_points, depth, beta,
        )
        high = solve_ilw_dop853(
            high_initial, high_grid, depth, nonlinear_transport=beta,
            evaluation_times=run["times"],
            rtol=float(config["reference"]["rtol"]) / 10.0,
            atol=float(config["reference"]["atol"]) / 10.0,
            maximum_step=float(settings["macro_dt"]) / 8.0,
        )
        registered = references[(depth, run["family"], run["seed"])].states[-1]
        registered_high, _ = project_ilw_spectrum(
            registered,
            PeriodicGrid(int(settings["reference_points"]), **config["domain"]),
            high_points, depth, beta,
        )
        defect = relative_ilw_h1(registered_high, high.states[-1], high_grid)
        rows.append({
            "depth": depth, "family": run["family"], "seed": run["seed"],
            "registered_points": int(settings["reference_points"]),
            "audit_points": high_points, "final_time": run["times"][-1],
            "relative_h1_reference_floor": defect,
        })
    write_csv(output / "reference_floor_audit.csv", rows)
    return rows, max(float(row["relative_h1_reference_floor"]) for row in rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment/cmame_u21_ilw_blind_bridge.yaml")
    parser.add_argument("--mode", choices=("development", "full"), default="development")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    config_path = ROOT / args.config
    config = load_yaml(config_path)
    settings = config[args.mode]
    mechanism_root = ROOT / config["mechanism_output_dir"] / args.mode
    training_path = mechanism_root / "training_acceptance.json"
    dataset_path = mechanism_root / "universal_mechanism_dataset.npz"
    manifest_path = mechanism_root / "frozen_u1_manifest.json"
    p0_path = ROOT / config["p0_output_dir"] / args.mode / "acceptance.json"
    training = json.loads(training_path.read_text())
    training_equations = set(
        np.load(dataset_path, allow_pickle=False)["equation"].astype(str).tolist()
    )
    p0 = json.loads(p0_path.read_text())
    if training["status"] != "PASS" or p0["status"] != "PASS":
        raise RuntimeError("U2.1 frozen U1 or ILW-P0 prerequisite failed")
    manifest_hash = sha(manifest_path)
    mechanism = FrozenUniversalCausalMechanism.load(
        mechanism_root, json.loads(manifest_path.read_text()), args.device
    )
    certificate_config = SelfCertificationConfig(**{
        key: value for key, value in config["certification"].items()
        if key != "reserve_fraction"
    })
    reserve = float(config["certification"]["reserve_fraction"])
    output = ROOT / config["output_dir"] / args.mode
    output.mkdir(parents=True, exist_ok=True)
    grid = PeriodicGrid(int(settings["points"]), **config["domain"])
    beta = float(config["nonlinear_transport"])
    base = int(config["seeds"]["development_base"] if args.mode == "development" else config["seeds"]["external_base"])

    runs = []
    for di, depth in enumerate(map(float, settings["depths"])):
        for fi, family in enumerate(config["families"]):
            for group in range(int(settings["groups_per_family"])):
                seed = base + di * 100000 + fi * 1000 + group
                initial = kdv_initial_condition(family, grid, seed=seed)
                operator = ILWFourierOperator(grid, depth, beta)
                initial = operator.physical(operator.spectrum(initial))
                for tolerance in map(float, settings["tolerances"]):
                    for arm in ("self_certifying", "always_probe", "analytical_control"):
                        runs.append(run_trajectory(
                            initial, grid, depth, beta,
                            arm=arm, family=family, seed=seed, tolerance=tolerance,
                            settings=settings, config=config, mechanism=mechanism,
                            certificate_config=certificate_config, reserve=reserve,
                        ))
                        print(f"U2.1 depth={depth:g} {family} {seed} {tolerance:.1e} {arm}", flush=True)

    online = [row for run in runs for row in run["rows"]]
    write_csv(output / "online_decisions.csv", online)

    references, audit = {}, []
    for index, run in enumerate(runs, 1):
        key = (run["depth"], run["family"], run["seed"])
        if key not in references:
            fine, fine_grid = project_ilw_spectrum(
                run["initial"], run["grid"], int(settings["reference_points"]),
                run["depth"], beta,
            )
            references[key] = solve_ilw_dop853(
                fine, fine_grid, run["depth"], nonlinear_transport=beta,
                evaluation_times=run["times"],
                rtol=float(config["reference"]["rtol"]),
                atol=float(config["reference"]["atol"]),
                maximum_step=float(settings["macro_dt"]) / 4.0,
            )
        reference = references[key]
        fine_grid = PeriodicGrid(int(settings["reference_points"]), **config["domain"])
        for prefix, row in enumerate(run["rows"], 1):
            approximation, _ = project_ilw_spectrum(
                run["states"][prefix], run["grids"][prefix], fine_grid.points,
                run["depth"], beta,
            )
            error = relative_ilw_h1(approximation, reference.states[prefix], fine_grid)
            audit.append({
                **row, "time": run["times"][prefix], "relative_h1": error,
                "error_over_tolerance": error / run["tolerance"],
                "tolerance_met": error <= run["tolerance"],
                "reference_visibility": "posthoc_only",
            })
        if index % 24 == 0 or index == len(runs):
            print(f"U2.1 posthoc {index}/{len(runs)}", flush=True)
    write_csv(output / "audit_rows.csv", audit)
    export_representative_trajectories(
        output, runs, references, settings, config, beta,
    )
    reference_floor_rows, maximum_reference_floor = reference_floor_audit(
        output, runs, references, settings, config, beta,
    )

    proposed = [row for row in audit if row["arm"] == "self_certifying"]
    always_rows = [row for row in audit if row["arm"] == "always_probe"]
    always = {(row["group_id"], row["prefix"]): row for row in always_rows}
    analytical = {(row["group_id"], row["prefix"]): row for row in audit if row["arm"] == "analytical_control"}
    terminal_prefix = int(settings["decisions"]) - 1
    terminal = [row for row in proposed if int(row["prefix"]) == terminal_prefix]
    coverage = float(np.mean([row["tolerance_met"] for row in terminal]))
    depth_coverage = {
        str(depth): float(np.mean([
            row["tolerance_met"] for row in terminal if float(row["depth"]) == float(depth)
        ])) for depth in settings["depths"]
    }
    authority = float(np.mean([row["neural_authority"] for row in proposed]))
    intervention = float(np.mean([
        row["action"] != analytical[(row["group_id"], row["prefix"])]["action"]
        for row in proposed
    ]))
    self_probes = sum(int(row["probe_count"]) for row in proposed)
    always_probes = sum(int(row["probe_count"]) for row in always_rows)
    probe_reduction = 1.0 - self_probes / max(always_probes, 1)
    self_work = sum(float(row["pde_online_work"]) for row in proposed)
    always_work = sum(float(row["pde_online_work"]) for row in always_rows)
    work_ratio = self_work / max(always_work, 1.0)
    error_ratio = float(np.median([
        row["relative_h1"] / max(always[(row["group_id"], row["prefix"])]["relative_h1"], 1e-16)
        for row in proposed
    ]))
    statistics = grouped_median_log_ratio_interval(
        [
            max(row["relative_h1"], 1e-16)
            / max(always[(row["group_id"], row["prefix"])]["relative_h1"], 1e-16)
            for row in proposed
        ],
        [physical_group(row) for row in proposed],
        samples=int(config["statistics"]["bootstrap_samples"]),
        confidence=float(config["statistics"]["confidence"]),
        seed=int(config["statistics"]["seed"]),
    )
    maximum_drift = max(float(row["invariant_drift"]) for row in proposed)
    limits = dict(config["acceptance"])
    if args.mode == "development":
        limits.update(settings.get("acceptance", {}))
    training_checks = training.get("checks", {})
    checks = {
        "ilw_p0_prerequisite": p0["status"] == "PASS",
        "u1_frozen_prerequisite": training["status"] == "PASS",
        "manifest_immutable": sha(manifest_path) == manifest_hash,
        "no_equation_identifier_or_ilw_training": bool(
            training_checks.get("no_equation_identifier_or_adapter", False)
            and training_equations == {"modified_camassa_holm", "benjamin_ono"}
        ),
        "reference_posthoc_only": all(row["reference_visibility"] == "posthoc_only" for row in audit),
        "long_horizon_reference_floor_resolved": bool(
            reference_floor_rows and maximum_reference_floor
            <= float(config["reference_floor_audit"]["maximum_defect_fraction_of_minimum_tolerance"])
            * min(map(float, settings["tolerances"]))
        ),
        "overall_tolerance_coverage": coverage >= float(limits["minimum_tolerance_coverage"]),
        "depth_tolerance_coverage": min(depth_coverage.values()) >= float(limits["minimum_depth_coverage"]),
        "zero_unsafe_release": all(row["analytically_safe_release"] for row in online),
        "nontrivial_certified_authority": authority >= float(limits["minimum_authority_fraction"]),
        "probes_reduced": probe_reduction >= float(limits["minimum_probe_reduction_fraction"]),
        "adaptive_action_intervention": intervention >= float(limits["minimum_action_intervention_fraction"]),
        "total_work_reduced": work_ratio <= float(limits["maximum_total_work_ratio_to_always_probe"]),
        "accuracy_noninferior": error_ratio <= float(limits["maximum_median_error_ratio_to_always_probe"]),
        "invariants_bounded": maximum_drift <= float(limits["maximum_invariant_drift"]),
        "all_outputs_finite": bool(np.all(np.isfinite([row["relative_h1"] for row in audit]))),
    }
    blocking = [key for key, value in checks.items() if not value]
    report = {
        "status": "PASS" if not blocking else "STOP",
        "next_phase": config["next_phase_on_pass"] if not blocking else None,
        "checks": checks, "blocking_failures": blocking, "mode": args.mode,
        "cases": len({
            (float(row["depth"]), row["family"], int(row["seed"])) for row in proposed
        }),
        "physical_trajectories": len({
            (float(row["depth"]), row["family"], int(row["seed"])) for row in proposed
        }),
        "tolerance_trajectory_groups": len(set(row["group_id"] for row in proposed)),
        "causal_decision_rows": len(proposed),
        "depths": settings["depths"],
        "terminal_tolerance_coverage": coverage,
        "depth_tolerance_coverage": depth_coverage,
        "certified_authority_fraction": authority,
        "action_intervention_fraction": intervention,
        "probe_reduction_fraction": probe_reduction,
        "total_pde_online_work_ratio_to_always_probe": work_ratio,
        "total_work_ratio_to_always_probe": work_ratio,
        "work_accounting": {
            "candidate_advances_included": True,
            "committed_candidate_reused": True,
            "probe_advances_included": True,
            "neural_inference_reported_separately": True,
            "unit": "FFT-equivalent PDE work",
        },
        "training_equations": sorted(training_equations),
        "median_error_ratio_to_always_probe": error_ratio,
        "paired_trajectory_group_statistics": statistics,
        "maximum_long_horizon_reference_floor": maximum_reference_floor,
        "learned_input_contract": {
            "legacy_mechanism_coordinates": True,
            "compressed_symbol_stiffness": True,
            "full_symbol_geometry_is_posthoc_only": True,
            "support_distance_controls_release": False,
            "support_distance_role": "logged diagnostic; online probes certify release",
        },
        "revocations": sum(bool(row["revoked"]) for row in proposed),
        "maximum_invariant_drift": maximum_drift,
        "provenance": {
            "config_sha256": sha(config_path),
            "ilw_p0_acceptance_sha256": sha(p0_path),
            "u1_manifest_sha256": manifest_hash,
            "u1_training_acceptance_sha256": sha(training_path),
            "online_decisions_sha256": sha(output / "online_decisions.csv"),
        },
        "claim": (
            "Frozen property-coordinate transfer across finite-depth ILW without "
            "equation identity; full symbol geometry is posthoc explanatory evidence."
        ),
        "scope": config["scope"],
    }
    (output / "acceptance.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    raise SystemExit(0 if not blocking else 2)


if __name__ == "__main__":
    main()
