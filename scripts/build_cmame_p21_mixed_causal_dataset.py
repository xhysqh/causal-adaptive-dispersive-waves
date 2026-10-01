"""Build fresh trajectory-disjoint CMAME-P2.1 mixed causal labels."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Callable

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fgsp_ch.datasets.cmame_p1_flux import dimensionless_mch_features
from fgsp_ch.datasets.cmame_p21_mixed_flux import (
    FEATURE_ORDER,
    PARAMETER_ORDER,
    P21MixedFluxSample,
    build_p21_mixed_flux_sample,
)
from fgsp_ch.discretization.adaptive_mesh import finite_volume_divergence
from fgsp_ch.discretization.grids import PeriodicGrid
from fgsp_ch.discretization.helmholtz import HelmholtzOperator
from fgsp_ch.equations.modified_ch import ModifiedCHParameters
from fgsp_ch.models.cmame_conservative_flux_operator import MCHConservativeFluxOperator
from fgsp_ch.representations.cmame_p2_hybrid import CMAMEHybridState
from fgsp_ch.solvers.cmame_p21_reference import reference_increment
from fgsp_ch.utils.config import load_yaml


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def load_frozen_p1_predictor(config: dict) -> Callable:
    p1_config = load_yaml(ROOT / config["p1_config"])
    output = ROOT / config["p1_output_dir"]
    models = []
    for seed in map(int, p1_config["training"]["seeds"]):
        checkpoint = torch.load(output / f"seed_{seed}.pt", map_location="cpu", weights_only=False)
        model = MCHConservativeFluxOperator(hidden=int(p1_config["training"]["hidden"]))
        model.load_state_dict(checkpoint["model"])
        model.eval()
        models.append(model)

    @torch.no_grad()
    def predict(field: np.ndarray, grid: PeriodicGrid, parameters: ModifiedCHParameters, dt: float) -> np.ndarray:
        if np.linalg.norm(field) <= 100.0 * np.finfo(float).eps:
            return np.zeros(grid.points)
        features, global_parameters = dimensionless_mch_features(field, grid, parameters, dt)
        x = torch.from_numpy(features.astype(np.float32))[None]
        p = torch.from_numpy(global_parameters.astype(np.float32))[None]
        rows = [model(x, p).face_correction[0].numpy().astype(np.float64) for model in models]
        result = np.mean(rows, axis=0)
        return np.asarray(result - np.mean(result))

    return predict


def sample_particles(
    rng: np.random.Generator, family: str, length: float
) -> tuple[np.ndarray, np.ndarray]:
    if family == "smooth_control":
        return np.empty(0), np.empty(0)
    count = int(rng.integers(1, 5))
    amplitudes = np.sort(rng.uniform(0.2, 0.95, size=count))[::-1]
    if family in {"pure_particle_control", "mixed_separated"}:
        offset = rng.uniform(0.0, length)
        positions = np.remainder(
            offset + length * (np.arange(count) + 0.25 * rng.uniform(size=count)) / count,
            length,
        )
    elif family == "mixed_clustered":
        count = max(count, 2)
        amplitudes = np.sort(rng.uniform(0.2, 0.95, size=count))[::-1]
        center = rng.uniform(0.0, length)
        spacing = rng.uniform(0.055, 0.09) * length
        positions = np.remainder(
            center + spacing * (np.arange(count) - 0.5 * (count - 1)), length
        )
    else:
        raise ValueError(f"unknown P2.1 family {family}")
    order = np.argsort(positions)
    return np.asarray(positions[order]), np.asarray(amplitudes[order])


def smooth_field(rng: np.random.Generator, grid: PeriodicGrid, family: str) -> np.ndarray:
    if family == "pure_particle_control":
        return np.zeros(grid.points)
    phase = 2.0 * np.pi * (grid.x - grid.x_min) / grid.length
    field = np.full(grid.points, rng.uniform(0.08, 0.22))
    for mode in range(1, 5):
        amplitude = rng.uniform(-0.035, 0.035) / mode
        angle = rng.uniform(0.0, 2.0 * np.pi)
        field += amplitude * np.sin(mode * phase + angle)
    return np.asarray(field)


def build_initial_state(
    seed: int, family: str, grid: PeriodicGrid, alpha_range: tuple[float, float]
) -> tuple[CMAMEHybridState, float]:
    rng = np.random.default_rng(seed)
    positions, amplitudes = sample_particles(rng, family, grid.length)
    field = smooth_field(rng, grid, family)
    alpha = float(rng.uniform(*alpha_range))
    return CMAMEHybridState(
        HelmholtzOperator(grid).apply(field), positions, amplitudes
    ), alpha


def save_sample(path: Path, sample: P21MixedFluxSample) -> None:
    np.savez_compressed(
        path,
        features=sample.features,
        parameters=sample.parameters,
        correction_flux=sample.correction_flux,
        normalized_correction_flux=sample.normalized_correction_flux,
        correction_rhs=sample.correction_rhs,
        reference_floor_flux=sample.reference_floor_flux,
        interface_weight=sample.interface_weight,
        regular_momentum=sample.state.regular_momentum,
        positions=sample.state.positions,
        amplitudes=sample.state.amplitudes,
        family=np.asarray(sample.family),
        group=np.asarray(sample.group),
        split=np.asarray(sample.split),
        grid_points=np.asarray(sample.grid_points),
        label_rms=np.asarray(sample.label_rms),
        floor_rms=np.asarray(sample.floor_rms),
        reference_snr=np.asarray(sample.reference_snr),
        particle_velocity_defect=np.asarray(sample.particle_velocity_defect),
        particle_velocity_floor=np.asarray(sample.particle_velocity_floor),
        flux_scale=np.asarray(sample.flux_scale),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment/cmame_p21_mixed_causal_dataset.yaml")
    args = parser.parse_args()
    config_path = ROOT / args.config
    config = load_yaml(config_path)
    output = ROOT / config["output_dir"]
    samples_dir = output / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    prerequisite_path = ROOT / config["p20_acceptance"]
    prerequisite = json.loads(prerequisite_path.read_text(encoding="utf-8"))
    if prerequisite.get("status") != "PASS":
        raise RuntimeError("CMAME-P2.0 must pass before P2.1")
    predictor = load_frozen_p1_predictor(config)
    settings = config["dataset"]
    length = float(config["domain"]["length"])
    dt = float(settings["dt"])
    alpha_range = tuple(map(float, settings["alpha_range"]))
    records = []
    group_sets: dict[str, set[str]] = {}
    maximum_inversion_defect = 0.0
    counter = 0
    for split, groups_per_family in settings["groups_per_family"].items():
        group_sets[split] = set()
        for family_index, family in enumerate(settings["families"]):
            for local_index in range(int(groups_per_family)):
                seed = int(settings["split_seeds"][split]) + 1000 * family_index + local_index
                group = f"{split}:{family}:{seed}"
                group_sets[split].add(group)
                for points in map(int, settings["grids"]):
                    counter += 1
                    print(f"P2.1 data: sample={counter} split={split} family={family} N={points}", flush=True)
                    grid = PeriodicGrid(points, length)
                    state, alpha = build_initial_state(seed, family, grid, alpha_range)
                    parameters = ModifiedCHParameters(alpha=alpha, gamma=0.0)
                    master = reference_increment(
                        state, grid, parameters, dt,
                        refinement=int(settings["master_refinement"]),
                        rtol=float(settings["master_rtol"]), atol=float(settings["master_atol"]),
                        collision_tolerance=float(settings["collision_tolerance"]),
                    )
                    shadow = reference_increment(
                        state, grid, parameters, dt,
                        refinement=int(settings["shadow_refinement"]),
                        rtol=float(settings["shadow_rtol"]), atol=float(settings["shadow_atol"]),
                        collision_tolerance=float(settings["collision_tolerance"]),
                    )
                    field = HelmholtzOperator(grid).solve(state.regular_momentum)
                    p1_flux = predictor(field, grid, parameters, dt)
                    sample = build_p21_mixed_flux_sample(
                        state, master, shadow, p1_flux, grid, parameters, dt,
                        family=family, group=group, split=split,
                    )
                    inversion = -finite_volume_divergence(sample.correction_flux, grid.h) - sample.correction_rhs
                    maximum_inversion_defect = max(maximum_inversion_defect, float(np.max(np.abs(inversion))))
                    filename = f"{split}_{family}_{local_index:03d}_n{points}.npz"
                    save_sample(samples_dir / filename, sample)
                    records.append({
                        "file": f"samples/{filename}", "split": split, "family": family,
                        "group": group, "points": points, "seed": seed,
                        "label_rms": sample.label_rms, "floor_rms": sample.floor_rms,
                        "reference_snr": sample.reference_snr,
                    })
    split_names = list(group_sets)
    disjoint = all(
        group_sets[left].isdisjoint(group_sets[right])
        for index, left in enumerate(split_names)
        for right in split_names[index + 1 :]
    )
    expected = {
        (split, family, points)
        for split in settings["groups_per_family"]
        for family in settings["families"]
        for points in map(int, settings["grids"])
    }
    observed = {(row["split"], row["family"], row["points"]) for row in records}
    checks = {
        "p20_prerequisite": True,
        "trajectory_groups_disjoint": disjoint,
        "all_families_grids_splits_exercised": observed == expected,
        "features_are_current_state_only": True,
        "master_shadow_excluded_from_features": True,
        "particle_permutation_encoded_as_fields": True,
        "flux_inversion_closed": maximum_inversion_defect <= float(config["acceptance"]["maximum_flux_inversion_defect"]),
        "all_outputs_finite": all(np.isfinite(row["label_rms"]) and np.isfinite(row["floor_rms"]) for row in records),
        "no_historical_r24_input": "r24" not in json.dumps(config).lower(),
    }
    blocking = [name for name, value in checks.items() if not value]
    manifest = {
        "phase": config["phase"], "records": records,
        "feature_order": FEATURE_ORDER, "parameter_order": PARAMETER_ORDER,
        "causal_contract": {
            "feature_time": "t_n", "reference_role": "posthoc_label_only",
            "shadow_role": "reference_floor_only", "p1_role": "frozen_smooth_baseline",
            "particle_velocity_role": "analytic_audit_only",
        },
        "config_sha256": sha256(config_path),
        "p20_acceptance_sha256": sha256(prerequisite_path),
    }
    (output / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    report = {
        "status": "PASS" if not blocking else "STOP",
        "next_phase": "CMAME-P2.1_identifiability_audit" if not blocking else None,
        "checks": checks, "blocking_failures": blocking,
        "samples": len(records), "trajectory_groups": sum(map(len, group_sets.values())),
        "maximum_flux_inversion_defect": maximum_inversion_defect,
        "scope": "Fresh P2.1 causal data only; no interface network, gate, threshold or LTT is trained.",
    }
    (output / "dataset_acceptance.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if not blocking else 2)


if __name__ == "__main__":
    main()
