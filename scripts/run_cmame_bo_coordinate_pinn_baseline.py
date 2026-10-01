"""Build, train, and freeze the independent BO Coordinate-PINN baseline."""

from __future__ import annotations

import argparse
from dataclasses import fields
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from fgsp_ch.discretization.grids import PeriodicGrid  # noqa: E402
from fgsp_ch.nonlocal_waves.initial_conditions.bo_families import bo_initial_condition  # noqa: E402
from fgsp_ch.nonlocal_waves.solvers.bo_reference import BOIFRK4Solver  # noqa: E402
from fgsp_ch.training.bo_coordinate_pinn import (  # noqa: E402
    BOCoordinatePINN, BOPINNArrays, train_bo_coordinate_pinn,
)
from fgsp_ch.training.cmame_m22_learned_baselines import predict_profiles  # noqa: E402
from fgsp_ch.utils.config import load_yaml  # noqa: E402


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def build_arrays(config: dict, mode: str, split: str) -> BOPINNArrays:
    settings = config[mode]
    grid = PeriodicGrid(int(settings["points"]), float(config["domain"]["length"]))
    groups_key = "groups_per_family" if split == "train" else "validation_groups_per_family"
    groups = int(settings[groups_key])
    split_offset = 0 if split == "train" else 500000
    rows = {name: [] for name in ("initial", "target", "amplitude", "time", "family", "group_id")}
    maximum_time = max(map(float, settings["times"]))
    for family_index, family in enumerate(map(str, config["families"])):
        for replicate in range(groups):
            seed = int(config["dataset_seed"]) + split_offset + 10000 * family_index + replicate
            rng = np.random.default_rng(seed)
            amplitude = float(rng.uniform(0.55, 1.45))
            phase = float(rng.uniform(0.0, grid.length))
            initial = bo_initial_condition(family, grid, amplitude=amplitude, phase_shift=phase)
            trajectory = BOIFRK4Solver(grid).solve(
                initial, final_time=maximum_time, dt=float(settings["reference_dt"])
            )
            for target_time in map(float, settings["times"]):
                index = int(np.argmin(np.abs(trajectory.times - target_time)))
                if not np.isclose(trajectory.times[index], target_time, atol=1e-13):
                    raise ValueError("BO training time must be a native reference time")
                rows["initial"].append(initial.copy())
                rows["target"].append(trajectory.states[index].copy())
                rows["amplitude"].append(amplitude)
                rows["time"].append(target_time)
                rows["family"].append(family)
                rows["group_id"].append(seed)
    arrays = BOPINNArrays(
        np.asarray(rows["initial"]), np.asarray(rows["target"]),
        np.asarray(rows["amplitude"]), np.asarray(rows["time"]),
        np.asarray(rows["family"]), np.asarray(rows["group_id"]),
    )
    arrays.validate()
    return arrays


def _save(path: Path, train: BOPINNArrays, validation: BOPINNArrays) -> None:
    payload = {}
    for split, arrays in (("train", train), ("validation", validation)):
        for field in fields(BOPINNArrays):
            payload[f"{split}_{field.name}"] = getattr(arrays, field.name)
    np.savez_compressed(path, **payload)


def _load(path: Path) -> tuple[BOPINNArrays, BOPINNArrays]:
    with np.load(path, allow_pickle=False) as data:
        return tuple(BOPINNArrays(*(data[f"{split}_{field.name}"] for field in fields(BOPINNArrays)))
                     for split in ("train", "validation"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment/cmame_bo_coordinate_pinn_baseline.yaml")
    parser.add_argument("--mode", choices=("development", "full"), default="development")
    parser.add_argument("--stage", choices=("dataset", "train", "audit", "all"), default="all")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    config_path = ROOT / args.config
    config = load_yaml(config_path)
    settings = config[args.mode]
    output = ROOT / config["output_dir"] / args.mode
    output.mkdir(parents=True, exist_ok=True)
    dataset_path = output / "bo_pinn_training_data.npz"
    if args.stage in {"dataset", "all"}:
        train = build_arrays(config, args.mode, "train")
        validation = build_arrays(config, args.mode, "validation")
        _save(dataset_path, train, validation)
        dataset_acceptance = {
            "status": "PASS",
            "checks": {
                "groups_disjoint": not bool(set(train.group_id) & set(validation.group_id)),
                "all_families_present": set(train.family) == set(config["families"]),
                "fields_finite": bool(np.isfinite(train.target).all() and np.isfinite(validation.target).all()),
                "native_reference_times": True,
            },
            "train_rows": len(train.initial), "validation_rows": len(validation.initial),
        }
        dataset_acceptance["blocking_failures"] = [k for k, v in dataset_acceptance["checks"].items() if not v]
        dataset_acceptance["status"] = "PASS" if not dataset_acceptance["blocking_failures"] else "STOP"
        (output / "dataset_acceptance.json").write_text(json.dumps(dataset_acceptance, indent=2))
        if args.stage == "dataset":
            raise SystemExit(0 if dataset_acceptance["status"] == "PASS" else 2)
    train, validation = _load(dataset_path)
    device = torch.device("cuda" if args.device == "cuda" or
                          (args.device == "auto" and torch.cuda.is_available()) else "cpu")
    records, trained_models = [], []
    for seed in map(int, config["ensemble_seeds"]):
        checkpoint = output / f"bo_pinn_seed_{seed}.pt"
        if args.stage in {"train", "all"}:
            model, metrics = train_bo_coordinate_pinn(
                train, validation, seed=seed, device=device,
                model_options=config["model"], epochs=int(settings["epochs"]),
                batch_size=int(settings["batch_size"]),
                learning_rate=float(config["optimizer"]["learning_rate"]),
                physics_weight=float(config["optimizer"]["physics_weight"]),
                length=float(config["domain"]["length"]),
                final_time_scale=max(map(float, settings["times"])),
            )
            torch.save({"state_dict": model.state_dict(), "seed": seed, "metrics": metrics}, checkpoint)
        else:
            payload = torch.load(checkpoint, map_location=device, weights_only=True)
            model = BOCoordinatePINN(**config["model"]).to(device)
            model.load_state_dict(payload["state_dict"]); model.eval()
            metrics = payload["metrics"]
        trained_models.append(model)
        records.append({"seed": seed, **metrics, "checkpoint_sha256": _sha256(checkpoint)})
    vi = torch.as_tensor(validation.initial, dtype=torch.float32, device=device)
    vt = torch.as_tensor(validation.target, dtype=torch.float32, device=device)
    va = torch.as_tensor(validation.amplitude, dtype=torch.float32, device=device)
    vtime = torch.as_tensor(validation.time, dtype=torch.float32, device=device)
    with torch.no_grad():
        ensemble_prediction = torch.stack([
            predict_profiles(
                model, "wang_yan_pinn", vi, vtime, va,
                final_time_scale=max(map(float, settings["times"])),
            ) for model in trained_models
        ]).mean(0)
        ensemble_mse = torch.mean((ensemble_prediction - vt).square()).item()
        persistence_mse = torch.mean((vi - vt).square()).item()
    ensemble_ratio = ensemble_mse / max(persistence_mse, 1e-30)
    checks = {
        "dataset_passed": json.loads((output / "dataset_acceptance.json").read_text())["status"] == "PASS",
        "three_members_frozen": len(records) == 3,
        "ensemble_validation_mse_bounded": ensemble_mse <= float(config["acceptance"]["maximum_validation_mse"]),
        "ensemble_learned_dynamics_better_than_persistence": ensemble_ratio <= float(config["acceptance"]["maximum_ratio_to_persistence"]),
        "all_metrics_finite": bool(np.isfinite([[row["validation_mse"], row["persistence_mse"]] for row in records]).all()),
        "bo_physics_residual_used": float(config["optimizer"]["physics_weight"]) > 0.0,
        "no_proposed_checkpoint_used": True,
    }
    acceptance = {
        "status": "PASS" if all(checks.values()) else "STOP", "checks": checks,
        "blocking_failures": [k for k, v in checks.items() if not v],
        "mode": args.mode, "device": str(device), "members": records,
        "ensemble_validation_mse": float(ensemble_mse),
        "ensemble_persistence_mse": float(persistence_mse),
        "ensemble_ratio_to_persistence": float(ensemble_ratio),
        "config_sha256": _sha256(config_path), "scope": config["scope"],
    }
    (output / "acceptance.json").write_text(json.dumps(acceptance, indent=2))
    print(json.dumps(acceptance, indent=2))
    raise SystemExit(0 if acceptance["status"] == "PASS" else 2)


if __name__ == "__main__":
    main()
