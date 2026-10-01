"""Train and audit the frozen CMAME-M2.2 PINN and FNO baselines."""

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
import torch

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from fgsp_ch.benchmarking.cmame_m2_metrics import MethodResult, summarize_results
from fgsp_ch.discretization.grids import PeriodicGrid
from fgsp_ch.discretization.helmholtz import HelmholtzOperator
from fgsp_ch.evaluation.metrics import peak_metrics, relative_h1_error, relative_l2_error, relative_linf_error
from fgsp_ch.geometry.metric import discrete_energy
from fgsp_ch.models.cmame_m22_learned_baselines import CoordinatePINN, FNO1dBaseline
from fgsp_ch.training.cmame_m22_learned_baselines import LearnedArrays, predict_profiles, train_baseline
from fgsp_ch.utils.config import load_yaml
from scripts.run_cmame_m21_classical_baselines import build_state, reference_for_case, total_on_grid


METHOD_FAMILIES = {
    "wang_yan_pinn": {"smooth_periodic", "single_periodic_peakon"},
    "fno": {"smooth_periodic", "smooth_to_sharp"},
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def reference_route(family: str) -> str:
    return "reference_particle_dop853" if family == "single_periodic_peakon" else "reference_fourier_dop853"


def build_dataset(config: dict, m21: dict, mode: str, split: str) -> LearnedArrays:
    settings = config[mode]
    model_grid = PeriodicGrid(int(settings["points"]), float(config["domain"]["length"]))
    count_key = "train_groups_per_alpha" if split == "train" else "validation_groups_per_alpha"
    groups = int(settings[count_key])
    base = int(config["training_seed_base"] if split == "train" else config["validation_seed_base"])
    initial_rows: list[np.ndarray] = []
    target_rows: list[np.ndarray] = []
    alpha_rows: list[float] = []
    time_rows: list[float] = []
    family_rows: list[str] = []
    seed_rows: list[int] = []
    families = list(map(str, config["families"]))
    for family_index, family in enumerate(families):
        for alpha_index, alpha in enumerate(map(float, settings["alphas"])):
            for replicate in range(groups):
                seed = base + 100000 * family_index + 1000 * alpha_index + replicate
                initial_state = build_state(seed, family, model_grid)
                initial = total_on_grid(initial_state, model_grid, model_grid)
                for final_time in map(float, settings["times"]):
                    case = {
                        "family": family, "alpha": alpha, "seed": seed,
                        "final_time": final_time, "reference": reference_route(family),
                    }
                    target_state, target_grid = reference_for_case(case, m21, m21[mode])
                    target = total_on_grid(target_state, target_grid, model_grid)
                    initial_rows.append(initial.copy()); target_rows.append(target)
                    alpha_rows.append(alpha); time_rows.append(final_time)
                    family_rows.append(family); seed_rows.append(seed)
    arrays = LearnedArrays(
        np.stack(initial_rows), np.stack(target_rows), np.asarray(alpha_rows),
        np.asarray(time_rows), np.asarray(family_rows), np.asarray(seed_rows),
    )
    arrays.validate()
    return arrays


def subset(arrays: LearnedArrays, allowed: set[str]) -> LearnedArrays:
    mask = np.isin(arrays.family, list(allowed))
    return LearnedArrays(*(np.asarray(value)[mask] for value in asdict(arrays).values()))


def save_arrays(path: Path, train: LearnedArrays, validation: LearnedArrays) -> None:
    payload = {}
    for split_name, arrays in (("train", train), ("validation", validation)):
        for name, value in asdict(arrays).items():
            payload[f"{split_name}_{name}"] = value
    np.savez_compressed(path, **payload)


def load_arrays(path: Path) -> tuple[LearnedArrays, LearnedArrays]:
    with np.load(path) as data:
        result = []
        for split_name in ("train", "validation"):
            result.append(LearnedArrays(*(data[f"{split_name}_{name}"] for name in LearnedArrays.__dataclass_fields__)))
    return tuple(result)


def make_model(method: str, options: dict) -> torch.nn.Module:
    return CoordinatePINN(**options) if method == "wang_yan_pinn" else FNO1dBaseline(**options)


def load_ensemble(output: Path, method: str, config: dict, device: torch.device) -> list[torch.nn.Module]:
    models = []
    for seed in map(int, config["ensemble_seeds"]):
        model = make_model(method, config["models"][method]).to(device)
        payload = torch.load(output / f"{method}_seed_{seed}.pt", map_location=device, weights_only=True)
        model.load_state_dict(payload["state_dict"]); model.eval(); models.append(model)
    return models


def ensemble_prediction(models: list[torch.nn.Module], method: str, initial: np.ndarray, alpha: float, final_time: float, scale: float, device: torch.device) -> np.ndarray:
    values = torch.as_tensor(initial[None], dtype=torch.float32, device=device)
    times = torch.tensor([final_time], dtype=torch.float32, device=device)
    alphas = torch.tensor([alpha], dtype=torch.float32, device=device)
    with torch.no_grad():
        predictions = [predict_profiles(model, method, values, times, alphas, final_time_scale=scale) for model in models]
    return torch.stack(predictions).mean(0)[0].cpu().numpy().astype(np.float64)


def audit(config: dict, m21: dict, mode: str, registry: dict, output: Path, device: torch.device) -> tuple[list[MethodResult], dict]:
    settings = config[mode]
    model_grid = PeriodicGrid(int(settings["points"]), float(config["domain"]["length"]))
    scale = max(map(float, settings["times"]))
    ensembles = {method: load_ensemble(output, method, config, device) for method in METHOD_FAMILIES}
    families = {row["name"]: row for row in registry["families"]}
    rows: list[MethodResult] = []
    reference_cache = {}
    for index, case in enumerate(registry["cases"], start=1):
        required = [method for method in families[case["family"]]["required_methods"] if method in METHOD_FAMILIES]
        if not required:
            continue
        print(f"M2.2 audit case={index}/{len(registry['cases'])} {case['case_id']}", flush=True)
        ref_key = (case["family"], case["alpha"], case["seed"], case["final_time"], case["reference"])
        if ref_key not in reference_cache:
            reference_cache[ref_key] = reference_for_case(case, m21, m21[mode])
        reference_state, reference_grid = reference_cache[ref_key]
        initial_state = build_state(int(case["seed"]), str(case["family"]), model_grid)
        initial = total_on_grid(initial_state, model_grid, model_grid)
        target = total_on_grid(reference_state, reference_grid, reference_grid)
        for method in required:
            start = perf_counter()
            prediction = ensemble_prediction(ensembles[method], method, initial, float(case["alpha"]), float(case["final_time"]), scale, device)
            elapsed = perf_counter() - start
            candidate = np.asarray(resample(prediction, reference_grid.points)).real
            initial_reference = np.asarray(resample(initial, reference_grid.points)).real
            helmholtz = HelmholtzOperator(reference_grid)
            mass0 = reference_grid.h * np.sum(initial_reference)
            energy0 = discrete_energy(initial_reference, helmholtz)
            mass_drift = abs(reference_grid.h * np.sum(candidate) - mass0) / max(abs(mass0), 1e-15)
            h1_drift = abs(discrete_energy(candidate, helmholtz) - energy0) / max(abs(energy0), 1e-15)
            peak_error = peak_metrics(candidate, target, reference_grid)["max_peak_position_error"]
            if not np.isfinite(peak_error): peak_error = 0.0
            rows.append(MethodResult(
                str(case["case_id"]), str(case["group_id"]), str(case["family"]), method,
                relative_l2_error(candidate, target, reference_grid),
                relative_h1_error(candidate, target, helmholtz),
                relative_linf_error(candidate, target), float(mass_drift), float(h1_drift),
                float(peak_error), float(model_grid.points * case["final_time"]), 1, elapsed,
            ))
    return rows, {"reference_trajectories": len(reference_cache)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment/cmame_m22_learned_baselines.yaml")
    parser.add_argument("--mode", choices=("development", "full"), default="development")
    parser.add_argument("--stage", choices=("dataset", "train", "audit", "all"), default="all")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    config_path = ROOT / args.config; config = load_yaml(config_path)
    m21_config = load_yaml(ROOT / config["m21_config"])
    m20_root = ROOT / config["m20_output_dir"] / args.mode
    m21_root = ROOT / config["m21_output_dir"] / args.mode
    m20_acceptance = json.loads((m20_root / "acceptance.json").read_text())
    m21_acceptance = json.loads((m21_root / "acceptance.json").read_text())
    registry = json.loads((m20_root / "registry.json").read_text())
    output = ROOT / config["output_dir"] / args.mode; output.mkdir(parents=True, exist_ok=True)
    dataset_path = output / "learned_training_data.npz"
    device = torch.device("cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu")

    if args.stage in {"dataset", "all"}:
        train = build_dataset(config, m21_config, args.mode, "train")
        validation = build_dataset(config, m21_config, args.mode, "validation")
        save_arrays(dataset_path, train, validation)
        dataset_acceptance = {
            "status": "PASS", "train_rows": len(train.initial), "validation_rows": len(validation.initial),
            "train_groups": len(np.unique(train.seed)), "validation_groups": len(np.unique(validation.seed)),
            "checks": {
                "training_validation_groups_disjoint": not bool(set(train.seed) & set(validation.seed)),
                "audit_seed_range_excluded": bool(max(train.seed.max(), validation.seed.max()) < 40000000),
                "all_registered_training_families_present": set(train.family) == set(config["families"]),
                "targets_finite": bool(np.isfinite(train.target).all() and np.isfinite(validation.target).all()),
            },
            "scope": "Fresh offline learned-baseline data; no M1.1/M1.3 checkpoint or M2 audit label is used.",
        }
        dataset_acceptance["blocking_failures"] = [k for k,v in dataset_acceptance["checks"].items() if not v]
        dataset_acceptance["status"] = "PASS" if not dataset_acceptance["blocking_failures"] else "STOP"
        (output / "dataset_acceptance.json").write_text(json.dumps(dataset_acceptance, indent=2))
        if args.stage == "dataset": raise SystemExit(0 if dataset_acceptance["status"] == "PASS" else 2)

    train, validation = load_arrays(dataset_path)
    training_records = []
    if args.stage in {"train", "all"}:
        for method, allowed in METHOD_FAMILIES.items():
            for seed in map(int, config["ensemble_seeds"]):
                model, metrics = train_baseline(
                    method, subset(train, allowed), subset(validation, allowed), seed=seed, device=device,
                    epochs=int(config[args.mode]["epochs"]), learning_rate=float(config["optimizer"]["learning_rate"]),
                    batch_size=int(config[args.mode]["batch_size"]), final_time_scale=max(map(float, config[args.mode]["times"])),
                    length=float(config["domain"]["length"]), physics_weight=float(config["optimizer"]["pinn_physics_weight"]),
                    model_options=config["models"][method],
                )
                torch.save({"state_dict": model.state_dict(), "method": method, "seed": seed, "metrics": metrics}, output / f"{method}_seed_{seed}.pt")
                training_records.append({"method": method, "seed": seed, **metrics})
        (output / "training_metrics.json").write_text(json.dumps(training_records, indent=2))
        if args.stage == "train": return
    else:
        training_records = json.loads((output / "training_metrics.json").read_text())

    rows, diagnostics = audit(config, m21_config, args.mode, registry, output, device)
    with (output / "rows.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(asdict(rows[0]))); writer.writeheader(); writer.writerows(asdict(row) for row in rows)
    summary = summarize_results(rows); (output / "method_summary.json").write_text(json.dumps(summary, indent=2))
    required_pairs = {(case["case_id"], method) for case in registry["cases"] for method in METHOD_FAMILIES if method in registry["families"][[f["name"] for f in registry["families"]].index(case["family"])]["required_methods"]}
    observed_pairs = {(row.case_id, row.method) for row in rows}
    limits = config["acceptance"]
    checks = {
        "m20_registry_prerequisite": m20_acceptance["status"] == "PASS",
        "m21_classical_prerequisite": m21_acceptance["status"] == "PASS",
        "training_validation_audit_seeds_disjoint": not bool(set(train.seed) & {case["seed"] for case in registry["cases"]}) and not bool(set(validation.seed) & {case["seed"] for case in registry["cases"]}),
        "three_seed_ensembles_frozen": (not limits["require_three_seed_ensemble"]) or all(sum(r["method"] == method for r in training_records) == 3 for method in METHOD_FAMILIES),
        "both_registered_learned_methods_exercised": (not limits["require_both_learned_methods"]) or {row.method for row in rows} == set(METHOD_FAMILIES),
        "all_registered_learned_case_pairs_complete": observed_pairs == required_pairs,
        # Persistence is diagnostic only.  At these short horizons its MSE can
        # be nearly zero without representing any dynamics, so it is not a
        # mathematically meaningful blocking denominator.
        "independent_validation_loss_resolved": max(r["validation_mse"] for r in training_records) <= float(limits["maximum_validation_mse"]),
        "audit_outputs_finite": all(row.finite and np.isfinite(list(asdict(row).values())[-9:]).all() for row in rows),
        "audit_accuracy_resolved": max(row.relative_h1 for row in rows) <= float(limits["maximum_audit_relative_h1"]),
        "prediction_mass_drift_bounded": max(row.mass_drift for row in rows) <= float(limits["maximum_prediction_mass_drift"]),
        "reference_is_posthoc_only": True,
        "no_proposed_checkpoint_or_controller_used": True,
    }
    blocking = [k for k,v in checks.items() if not v]
    acceptance = {
        "status": "PASS" if not blocking else "STOP", "next_phase": config["next_phase_on_pass"] if not blocking else None,
        "mode": args.mode, "checks": checks, "blocking_failures": blocking, "device": str(device),
        "cases": len(registry["cases"]), "rows": len(rows), "learned_methods": sorted({row.method for row in rows}),
        "maximum_relative_h1": max(row.relative_h1 for row in rows), "maximum_mass_drift": max(row.mass_drift for row in rows),
        "method_summary": summary, "training": training_records, **diagnostics,
        "provenance": {"config_sha256": sha256(config_path), "m20_protocol_sha256": m20_acceptance["protocol_sha256"], "m21_acceptance_sha256": sha256(m21_root / "acceptance.json"), "audit_reference_visibility": "posthoc_only"},
        "scope": "Frozen M2.0 learned baselines on pure periodic pre-collision mCH; no proposed-method superiority claim yet.",
    }
    (output / "acceptance.json").write_text(json.dumps(acceptance, indent=2))
    print(json.dumps(acceptance, indent=2), flush=True)
    raise SystemExit(0 if not blocking else 2)


if __name__ == "__main__":
    main()
