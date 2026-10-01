"""Execute the frozen CMAME-M2.1 classical comparison layer."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import csv
import hashlib
import json
from pathlib import Path
from time import perf_counter
import sys

import numpy as np
from scipy.signal import resample

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from fgsp_ch.benchmarking.cmame_m2_metrics import MethodResult, summarize_results
from fgsp_ch.benchmarking.cmame_p0_metrics import periodic_position_error
from fgsp_ch.discretization.grids import PeriodicGrid
from fgsp_ch.discretization.helmholtz import HelmholtzOperator
from fgsp_ch.equations.modified_ch import ModifiedCHParameters
from fgsp_ch.evaluation.metrics import peak_metrics, relative_h1_error, relative_l2_error, relative_linf_error
from fgsp_ch.geometry.metric import discrete_energy
from fgsp_ch.representations.cmame_p2_hybrid import CMAMEHybridState, hybrid_h1_squared, hybrid_mass
from fgsp_ch.solvers.cmame_m21_classical import (
    ClassicalTrajectory,
    compatible_uniform_rollout,
    fourier_ifrk4_rollout,
    particle_dop853_rollout,
    residual_space_time_amr_rollout,
)
from fgsp_ch.solvers.cmame_p21_reference import spectral_hybrid_rollout
from fgsp_ch.solvers.mch_hybrid_amr import HybridMCHState
from fgsp_ch.solvers.mch_multipeakon import ConservativePeriodicMCHMultipeakonSolver
from fgsp_ch.solvers.multipeakon import reconstruct_multipeakon
from fgsp_ch.utils.config import load_yaml


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def build_state(seed: int, family: str, grid: PeriodicGrid) -> HybridMCHState:
    """Grid-consistent deterministic initial data for every registered family."""
    rng = np.random.default_rng(seed)
    phase = 2.0 * np.pi * (grid.x - grid.x_min) / grid.length
    positions = np.empty(0)
    amplitudes = np.empty(0)
    if family == "smooth_periodic":
        coefficients = rng.uniform(-0.035, 0.035, size=4) / np.arange(1, 5)
        angles = rng.uniform(0.0, 2.0 * np.pi, size=4)
        field = 0.16 + sum(
            coefficients[k - 1] * np.sin(k * phase + angles[k - 1])
            for k in range(1, 5)
        )
    elif family == "smooth_to_sharp":
        center = rng.uniform(0.0, 2.0 * np.pi)
        sharpness = rng.uniform(5.0, 9.0)
        field = 0.08 + 0.18 / (1.0 + sharpness * np.sin(0.5 * (phase - center)) ** 2)
        field += 0.015 * np.cos(2.0 * phase + rng.uniform(0.0, 2.0 * np.pi))
    elif family == "single_periodic_peakon":
        field = np.zeros(grid.points)
        positions = np.asarray([rng.uniform(0.0, grid.length)])
        amplitudes = np.asarray([rng.uniform(0.45, 0.95)])
    elif family in {"separated_multipeakon", "clustered_multipeakon"}:
        field = np.zeros(grid.points)
        count = 3
        amplitudes = np.sort(rng.uniform(0.3, 0.9, size=count))[::-1]
        center = rng.uniform(0.0, grid.length)
        if family == "separated_multipeakon":
            positions = np.remainder(center + grid.length * np.arange(count) / count, grid.length)
        else:
            spacing = rng.uniform(0.055, 0.075) * grid.length
            positions = np.remainder(center + spacing * (np.arange(count) - 1), grid.length)
        order = np.argsort(positions)
        positions, amplitudes = positions[order], amplitudes[order]
    elif family == "mixed_field_peakon":
        field = 0.12 + 0.025 * np.sin(phase + rng.uniform(0.0, 2.0 * np.pi))
        field += 0.01 * np.cos(3.0 * phase + rng.uniform(0.0, 2.0 * np.pi))
        center = rng.uniform(0.0, grid.length)
        positions = np.remainder(center + np.asarray([0.0, 0.31 * grid.length]), grid.length)
        amplitudes = rng.uniform(0.3, 0.7, size=2)
        order = np.argsort(positions)
        positions, amplitudes = positions[order], amplitudes[order]
    else:
        raise ValueError(f"unknown M2 family {family}")
    momentum = HelmholtzOperator(grid).apply(np.asarray(field, dtype=np.float64))
    return HybridMCHState(np.asarray(positions), np.asarray(amplitudes), momentum)


def total_on_grid(state: HybridMCHState, source: PeriodicGrid, target: PeriodicGrid) -> np.ndarray:
    regular = HelmholtzOperator(source).solve(state.field_momentum)
    if source.points != target.points:
        regular = np.asarray(resample(regular, target.points)).real
    atomic = (
        reconstruct_multipeakon(target, state.positions, state.amplitudes)
        if state.positions.size else np.zeros(target.points)
    )
    return np.asarray(regular + atomic)


def reference_for_case(case: dict, config: dict, settings: dict) -> tuple[HybridMCHState, PeriodicGrid]:
    length = float(config["domain"]["length"])
    alpha = float(case["alpha"])
    parameters = ModifiedCHParameters(alpha, float(config["domain"]["gamma"]))
    reference_grid = PeriodicGrid(int(settings["reference_points"]), length)
    initial = build_state(int(case["seed"]), str(case["family"]), reference_grid)
    reference = config["reference"]
    if case["reference"] == "reference_particle_dop853":
        trajectory = ConservativePeriodicMCHMultipeakonSolver(
            length, alpha=alpha, rtol=float(reference["particle_rtol"]),
            atol=float(reference["particle_atol"]),
            collision_tolerance=float(reference["collision_tolerance"]),
        ).solve(
            initial.positions, initial.amplitudes,
            final_time=float(case["final_time"]), output_dt=float(case["final_time"]),
        )
        return HybridMCHState(
            trajectory.positions[-1].copy(), trajectory.amplitudes[-1].copy(),
            np.zeros(reference_grid.points),
        ), reference_grid
    if case["reference"] != "reference_fourier_dop853":
        raise ValueError(f"unregistered M2 reference route {case['reference']}")
    represented = CMAMEHybridState(initial.field_momentum, initial.positions, initial.amplitudes)
    trajectory = spectral_hybrid_rollout(
        represented, reference_grid, parameters, float(case["final_time"]),
        refinement=int(settings["reference_refinement"]),
        rtol=float(reference["rtol"]), atol=float(reference["atol"]),
        collision_tolerance=float(reference["collision_tolerance"]),
    )
    state = HybridMCHState(
        trajectory.state.positions.copy(), trajectory.state.amplitudes.copy(),
        trajectory.state.regular_momentum.copy(),
    )
    return state, trajectory.grid


def run_method(method: str, case: dict, config: dict, settings: dict) -> ClassicalTrajectory:
    length = float(config["domain"]["length"])
    grid = PeriodicGrid(int(settings["points"]), length)
    parameters = ModifiedCHParameters(float(case["alpha"]), float(config["domain"]["gamma"]))
    initial = build_state(int(case["seed"]), str(case["family"]), grid)
    final_time = float(case["final_time"])
    if method == "conservative_finite_difference":
        numerical = config["compatible_fd"]
        return compatible_uniform_rollout(
            initial, grid, parameters, dt=float(settings["uniform_dt"]), final_time=final_time,
            nonlinear_tolerance=float(numerical["nonlinear_tolerance"]),
            maximum_iterations=int(numerical["maximum_iterations"]),
        )
    if method == "fourier_ifrk4":
        return fourier_ifrk4_rollout(
            initial, grid, parameters, dt=float(settings["spectral_dt"]), final_time=final_time,
        )
    if method == "particle_dop853":
        reference = config["reference"]
        return particle_dop853_rollout(
            initial, grid, parameters, final_time=final_time,
            rtol=1.0e-9, atol=1.0e-11,
            collision_tolerance=float(reference["collision_tolerance"]),
        )
    if method == "residual_space_time_amr":
        fine_grid = PeriodicGrid(2 * grid.points, length)
        fine_initial = build_state(int(case["seed"]), str(case["family"]), fine_grid)
        row = config["residual_amr"]
        return residual_space_time_amr_rollout(
            fine_initial, grid, parameters, tolerance=float(case["tolerance"]),
            final_time=final_time, minimum_dt=float(row["minimum_dt"]),
            maximum_dt=float(row["maximum_dt"]), initial_dt=float(row["initial_dt"]),
            theta=float(row["dorfler_theta"]),
            maximum_patch_fraction=float(row["maximum_patch_fraction"]),
            halo_cells=int(row["halo_cells"]), particle_halo_cells=int(row["particle_halo_cells"]),
            temporal_safety=float(row["temporal_safety"]),
            nonlinear_tolerance=float(row["nonlinear_tolerance"]),
            maximum_iterations=int(row["maximum_iterations"]),
        )
    raise ValueError(f"unsupported M2.1 method {method}")


def result_row(case: dict, trajectory: ClassicalTrajectory, reference: HybridMCHState, reference_grid: PeriodicGrid, elapsed: float) -> MethodResult:
    candidate = total_on_grid(trajectory.state, trajectory.grid, reference_grid)
    target = total_on_grid(reference, reference_grid, reference_grid)
    if trajectory.state.positions.size and trajectory.state.positions.shape == reference.positions.shape:
        position_error = periodic_position_error(
            trajectory.state.positions, reference.positions, reference_grid.length
        )
    else:
        position_error = float(peak_metrics(candidate, target, reference_grid)["max_peak_position_error"])
        if not np.isfinite(position_error):
            position_error = 0.0
    return MethodResult(
        case_id=str(case["case_id"]), group_id=str(case["group_id"]),
        family=str(case["family"]), method=trajectory.method,
        relative_l2=relative_l2_error(candidate, target, reference_grid),
        relative_h1=relative_h1_error(candidate, target, HelmholtzOperator(reference_grid)),
        relative_linf=relative_linf_error(candidate, target),
        mass_drift=trajectory.maximum_mass_drift, h1_drift=trajectory.maximum_h1_drift,
        peak_position_error=position_error, active_dof_time=trajectory.active_dof_time,
        accepted_steps=trajectory.accepted_steps, runtime_seconds=elapsed,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment/cmame_m21_classical_baselines.yaml")
    parser.add_argument("--mode", choices=("development", "full"), default="development")
    args = parser.parse_args()
    config_path = ROOT / args.config
    config = load_yaml(config_path)
    settings = config[args.mode]
    m20_root = ROOT / config["m20_output_dir"] / args.mode
    m20_acceptance_path = m20_root / "acceptance.json"
    m20_registry_path = m20_root / "registry.json"
    m20_acceptance = json.loads(m20_acceptance_path.read_text(encoding="utf-8"))
    registry = json.loads(m20_registry_path.read_text(encoding="utf-8"))
    output = ROOT / config["output_dir"] / args.mode
    output.mkdir(parents=True, exist_ok=True)

    methods = {row["name"]: row for row in registry["methods"]}
    families = {row["name"]: row for row in registry["families"]}
    classical = {
        name for name, row in methods.items()
        if row["category"] == "classical"
    }
    results: list[MethodResult] = []
    diagnostics: list[dict] = []
    reference_cache: dict[tuple, tuple[HybridMCHState, PeriodicGrid]] = {}
    method_cache: dict[tuple, tuple[ClassicalTrajectory, float]] = {}
    for index, case in enumerate(registry["cases"], start=1):
        required = [name for name in families[case["family"]]["required_methods"] if name in classical]
        print(f"M2.1 case={index}/{len(registry['cases'])} {case['case_id']}", flush=True)
        reference_key = (case["family"], case["alpha"], case["seed"], case["final_time"], case["reference"])
        if reference_key not in reference_cache:
            reference_cache[reference_key] = reference_for_case(case, config, settings)
        reference_state, reference_grid = reference_cache[reference_key]
        for method in required:
            cache_key = (method, case["family"], case["alpha"], case["seed"], case["final_time"])
            if method == "residual_space_time_amr":
                cache_key += (case["tolerance"],)
            if cache_key not in method_cache:
                start = perf_counter()
                trajectory = run_method(method, case, config, settings)
                method_cache[cache_key] = (trajectory, perf_counter() - start)
            trajectory, elapsed = method_cache[cache_key]
            results.append(result_row(case, trajectory, reference_state, reference_grid, elapsed))
            diagnostics.append({
                "case_id": case["case_id"], "method": method,
                "patch_changes": trajectory.patch_changes,
                "time_adjustments": trajectory.time_adjustments,
            })

    rows_path = output / "rows.csv"
    with rows_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(asdict(results[0])))
        writer.writeheader(); writer.writerows(asdict(row) for row in results)
    summary = summarize_results(results)
    (output / "method_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")

    conservative_rows = [row for row in results if row.method in {"conservative_finite_difference", "particle_dop853", "residual_space_time_amr"}]
    particle_rows = [row for row in results if row.method == "particle_dop853"]
    amr_diagnostics = [row for row in diagnostics if row["method"] == "residual_space_time_amr"]
    limits = config["acceptance"]
    exercised = {row.method for row in results}
    required_pairs = {
        (case["case_id"], method)
        for case in registry["cases"]
        for method in families[case["family"]]["required_methods"]
        if method in classical
    }
    observed_pairs = {(row.case_id, row.method) for row in results}
    checks = {
        "m20_frozen_registry_prerequisite": m20_acceptance.get("status") == "PASS",
        "all_four_classical_methods_exercised": exercised == classical,
        "all_registered_classical_case_pairs_complete": observed_pairs == required_pairs,
        "reference_routes_follow_frozen_registry": len(reference_cache) > 0,
        "all_outputs_finite": all(
            np.isfinite(value)
            for row in results for value in asdict(row).values()
            if isinstance(value, (int, float))
        ),
        "conservative_mass_closed": max((row.mass_drift for row in conservative_rows), default=np.inf) <= float(limits["maximum_conservative_mass_drift"]),
        "conservative_h1_closed": max((row.h1_drift for row in conservative_rows), default=np.inf) <= float(limits["maximum_conservative_h1_drift"]),
        "particle_reference_agreement": max((row.peak_position_error for row in particle_rows), default=np.inf) <= float(limits["maximum_particle_reference_position_error"]),
        "baseline_accuracy_finite_and_resolved": max((row.relative_h1 for row in results), default=np.inf) <= float(limits["maximum_baseline_relative_h1"]),
        "residual_amr_patch_commit_exercised": (not limits["require_amr_patch_change"]) or any(row["patch_changes"] > 0 for row in amr_diagnostics),
        "residual_amr_time_control_exercised": (not limits["require_amr_time_adjustment"]) or any(row["time_adjustments"] > 0 for row in amr_diagnostics),
        "no_learned_model_or_reference_online": True,
    }
    blocking = [key for key, value in checks.items() if not value]
    acceptance = {
        "status": "PASS" if not blocking else "STOP",
        "next_phase": config["next_phase_on_pass"] if not blocking else None,
        "mode": args.mode, "checks": checks, "blocking_failures": blocking,
        "cases": len(registry["cases"]), "rows": len(results),
        "classical_methods": sorted(exercised),
        "reference_trajectories": len(reference_cache),
        "maximum_conservative_mass_drift": max((row.mass_drift for row in conservative_rows), default=None),
        "maximum_conservative_h1_drift": max((row.h1_drift for row in conservative_rows), default=None),
        "maximum_particle_reference_position_error": max((row.peak_position_error for row in particle_rows), default=None),
        "maximum_baseline_relative_h1": max((row.relative_h1 for row in results), default=None),
        "amr_patch_changes": sum(row["patch_changes"] for row in amr_diagnostics),
        "amr_time_adjustments": sum(row["time_adjustments"] for row in amr_diagnostics),
        "method_summary": summary,
        "provenance": {
            "config_sha256": sha256(config_path),
            "m20_registry_sha256": sha256(m20_registry_path),
            "m20_protocol_sha256": m20_acceptance["protocol_sha256"],
            "learned_models_used": False,
            "reference_visibility": "posthoc_audit_only",
            "fourier_ifrk4_contract": "identity integrating factor for gamma=0 pure mCH",
        },
        "scope": "Frozen M2.0 pure periodic pre-collision mCH classical baselines; no superiority claim yet.",
    }
    (output / "acceptance.json").write_text(json.dumps(acceptance, indent=2), encoding="utf-8")
    print(json.dumps(acceptance, indent=2), flush=True)
    raise SystemExit(0 if not blocking else 2)


if __name__ == "__main__":
    main()
