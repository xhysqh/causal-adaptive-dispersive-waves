"""Offline source-ledger evaluation of the 24/30/36/42 V2 history ablation."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from fgsp_ch.nonlocal_waves.adaptivity.v2_safety_shielded_controller import (  # noqa: E402
    FrozenV2PredictiveRiskEnsemble,
)
from fgsp_ch.utils.config import load_yaml  # noqa: E402


@torch.no_grad()
def predictions(ensemble, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    active = ensemble.manifest.get("active_feature_indices")
    projected = rows if active is None else rows[:, np.asarray(active, dtype=int)]
    tensor = torch.as_tensor(projected, dtype=torch.float32, device=ensemble.device)
    outputs = [model(tensor) for model in ensemble.models]
    mean = torch.stack([item.mean_log_risk for item in outputs]).cpu().numpy().mean(axis=0)
    upper_members = torch.stack([item.upper_log_risk for item in outputs]).cpu().numpy()
    upper = (
        upper_members.mean(axis=0)
        + float(ensemble.manifest["ensemble_std_multiplier"]) * upper_members.std(axis=0)
        + float(ensemble.manifest["conformal_log_risk_margin"])
    )
    return mean, upper


def scalar_metrics(truth_log: np.ndarray, mean_log: np.ndarray, upper_log: np.ndarray,
                   threshold: float) -> dict:
    truth = np.exp(np.clip(truth_log, -32.0, 32.0))
    upper = np.exp(np.clip(upper_log, -32.0, 32.0))
    unsafe = truth > threshold
    predicted_safe = upper <= threshold
    safe = ~unsafe
    return {
        "samples": int(len(truth)),
        "unsafe_events": int(unsafe.sum()),
        "mae_log_risk": float(np.mean(np.abs(mean_log - truth_log))),
        "false_safe_joint": float(np.mean(unsafe & predicted_safe)),
        "false_safe_conditional": float(np.mean(predicted_safe[unsafe])) if np.any(unsafe) else None,
        "false_alarm_conditional": float(np.mean((~predicted_safe)[safe])) if np.any(safe) else None,
        "upper_coverage": float(np.mean(truth_log <= upper_log)),
        "mean_log_sharpness": float(np.mean(upper_log - truth_log)),
    }


def warning_metrics(groups, prefixes, truth_log, upper_log, threshold: float) -> dict:
    threshold_log = np.log(threshold)
    leads, missed, event_groups = [], 0, 0
    for group in sorted(set(groups.astype(str))):
        indices = np.flatnonzero(groups.astype(str) == group)
        indices = indices[np.argsort(prefixes[indices])]
        crossings = indices[truth_log[indices] > threshold_log]
        if not len(crossings):
            continue
        event_groups += 1
        event_prefix = int(prefixes[crossings[0]])
        warnings = indices[upper_log[indices] > threshold_log]
        if not len(warnings):
            missed += 1
            continue
        leads.append(event_prefix - int(prefixes[warnings[0]]))
    array = np.asarray(leads, dtype=float)
    return {
        "event_groups": event_groups,
        "warning_groups": int(len(leads)),
        "missed_warning_groups": missed,
        "mean_warning_lead": float(array.mean()) if len(array) else None,
        "median_warning_lead": float(np.median(array)) if len(array) else None,
        "early_fraction": float(np.mean(array > 0)) if len(array) else None,
        "synchronous_fraction": float(np.mean(array == 0)) if len(array) else None,
        "late_fraction": float(np.mean(array < 0)) if len(array) else None,
    }


def bootstrap_intervals(groups, truth, mean, upper, threshold, replicates, seed):
    rng = np.random.default_rng(seed)
    unique = np.asarray(sorted(set(groups.astype(str))))
    values = {key: [] for key in (
        "mae_log_risk", "false_safe_conditional", "false_alarm_conditional", "upper_coverage"
    )}
    for _ in range(replicates):
        selected = rng.choice(unique, len(unique), replace=True)
        indices = np.concatenate([np.flatnonzero(groups.astype(str) == group) for group in selected])
        metrics = scalar_metrics(truth[indices], mean[indices], upper[indices], threshold)
        for key in values:
            if metrics[key] is not None:
                values[key].append(metrics[key])
    return {key: [float(np.quantile(rows, 0.025)), float(np.quantile(rows, 0.975))]
            for key, rows in values.items() if rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment/cmame_v2_nested_history_ablation.yaml")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    cfg = load_yaml(ROOT / args.config)
    root = ROOT / cfg["output_dir"]
    data = np.load(ROOT / cfg["source_dataset_root"] / "v2_predictive_risk_dataset.npz")
    mask = (data["split"].astype(str) == "selection") & (data["action"].astype(str) == "hold")
    features = np.asarray(data["features"][mask], dtype=np.float64)
    truth = np.asarray(data["target_log_future_risk"][mask], dtype=np.float64)
    equations = data["equation"][mask].astype(str)
    groups = data["group_id"][mask].astype(str)
    prefixes = np.asarray(data["prefix"][mask], dtype=int)
    threshold = float(cfg["threshold"])
    full_manifest = ROOT / cfg["source_dataset_root"] / "frozen_v2_predictive_risk_manifest.json"
    manifests = {"full": full_manifest}
    manifests.update({variant: root / "models" / variant / "frozen_v2_predictive_risk_manifest.json"
                      for variant in cfg["variants"] if variant != "full"})
    records, details = [], {}
    for variant in cfg["variants"]:
        path = manifests[variant]
        ensemble = FrozenV2PredictiveRiskEnsemble.load(
            path.parent, json.loads(path.read_text(encoding="utf-8")), device=args.device,
        )
        mean, upper = predictions(ensemble, features)
        details[variant] = {"mean_log_risk": mean, "upper_log_risk": upper}
        for equation in ("combined", "modified_camassa_holm", "benjamin_ono"):
            subset = np.ones(len(features), dtype=bool) if equation == "combined" else equations == equation
            record = {"variant": variant, "equation": equation,
                      "input_dimension": int(ensemble.models[0].input_width)}
            record.update(scalar_metrics(truth[subset], mean[subset], upper[subset], threshold))
            record.update(warning_metrics(groups[subset], prefixes[subset], truth[subset], upper[subset], threshold))
            record["bootstrap_95"] = bootstrap_intervals(
                groups[subset], truth[subset], mean[subset], upper[subset], threshold,
                int(cfg["bootstrap_replicates"]), int(cfg["bootstrap_seed"]),
            )
            records.append(record)
    fields = [key for key in records[0] if key != "bootstrap_95"]
    with (root / "source_offline_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows({key: row[key] for key in fields} for row in records)
    report = {
        "status": "PASS",
        "threshold": threshold,
        "selection_hold_samples": int(mask.sum()),
        "records": records,
        "scope": cfg["scope"],
    }
    (root / "source_offline_metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
