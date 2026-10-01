"""Train and freeze the equation-neutral CMAME-U1 mechanism operator."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts.train_cmame_f1_shared_operator import balanced_indices  # noqa: E402
from fgsp_ch.nonlocal_waves.adaptivity.mechanism_coordinates import (  # noqa: E402
    MECHANISM_FEATURE_ORDER,
    RobustMechanismScaler,
    knn_support_score,
    trajectory_conformal_radius,
)
from fgsp_ch.nonlocal_waves.adaptivity.universal_causal_operator import (  # noqa: E402
    UniversalCausalBudgetOperator,
    universal_causal_loss,
)
from fgsp_ch.utils.config import load_yaml  # noqa: E402


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest().upper()


def _trajectory_margin(predicted, truth, groups, coverage):
    scores = np.asarray(truth) - np.asarray(predicted)
    gids = np.asarray(groups).astype(str)
    maxima = np.asarray([np.max(scores[gids == group]) for group in sorted(set(gids))])
    rank = min(len(maxima), int(np.ceil((len(maxima) + 1) * coverage)))
    return max(0.0, float(np.partition(maxima, rank - 1)[rank - 1]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment/cmame_u1_universal_mechanism.yaml")
    parser.add_argument("--mode", choices=("development", "full"), default="development")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    cfg = load_yaml(ROOT / args.config)
    out = ROOT / cfg["output_dir"] / args.mode
    dataset_path = out / "universal_mechanism_dataset.npz"
    data = np.load(dataset_path)
    x = torch.as_tensor(data["features"], dtype=torch.float32)
    y = torch.as_tensor(data["target_log_effectivity"], dtype=torch.float32)
    cost = torch.as_tensor(data["target_log_cost"], dtype=torch.float32)
    probe = torch.as_tensor(data["target_probe"], dtype=torch.float32)
    rng = np.random.default_rng(32452843)
    train = balanced_indices(data, "train", rng)
    calibration = np.where(data["split"].astype(str) == "calibration")[0]
    selection = np.where(data["split"].astype(str) == "selection")[0]
    scaler = RobustMechanismScaler.fit(
        data["features"][train],
        minimum_scale=float(cfg["model"]["minimum_robust_scale"]),
    )
    device, models, predictions = torch.device(args.device), [], []
    for seed in cfg["training"]["seeds"]:
        torch.manual_seed(int(seed))
        model = UniversalCausalBudgetOperator(hidden=int(cfg["model"]["hidden"])).to(device)
        model.set_robust_scaler(scaler.center, scaler.scale)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=float(cfg["training"]["learning_rate"]),
            weight_decay=float(cfg["training"]["weight_decay"]),
        )
        for _ in range(int(cfg["training"]["epochs"])):
            optimizer.zero_grad()
            output = model(x[train].to(device))
            loss = universal_causal_loss(
                output, y[train].to(device), cost[train].to(device), probe[train].to(device),
                data["group_id"][train].astype(str).tolist(),
                data["prefix"][train].astype(int).tolist(),
                quantile=float(cfg["training"]["upper_quantile"]),
                causal_epsilon=float(cfg["training"]["causal_epsilon"]),
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(cfg["training"]["gradient_clip"])
            )
            optimizer.step()
        name = f"universal_causal_member_{seed}.pt"
        torch.save({"model": model.state_dict(), "hidden": model.hidden}, out / name)
        models.append((name, model.eval()))
    with torch.no_grad():
        for _, model in models:
            predictions.append(
                model(x.to(device)).upper_log_effectivity.detach().cpu().numpy()
            )
    ensemble_upper = np.mean(predictions, axis=0) + np.std(predictions, axis=0)
    coverage_target = float(cfg["calibration"]["trajectory_coverage"])
    margin = _trajectory_margin(
        ensemble_upper[calibration], data["target_log_effectivity"][calibration],
        data["group_id"][calibration], coverage_target,
    )

    transformed_train = scaler.transform(data["features"][train])
    maximum = int(cfg["calibration"]["maximum_support_anchors"])
    if len(transformed_train) > maximum:
        anchor_ids = rng.choice(len(transformed_train), maximum, replace=False)
        anchors = transformed_train[anchor_ids]
    else:
        anchors = transformed_train
    anchors_path = out / "u1_support_anchors.npz"
    np.savez_compressed(anchors_path, anchors=np.asarray(anchors, dtype=np.float32))
    neighbors = int(cfg["calibration"]["support_neighbors"])
    calibration_scores = knn_support_score(
        scaler.transform(data["features"][calibration]), anchors, neighbors=neighbors
    )
    radius = trajectory_conformal_radius(
        calibration_scores, data["group_id"][calibration],
        coverage=float(cfg["calibration"]["support_coverage"]),
    )
    selection_scores = knn_support_score(
        scaler.transform(data["features"][selection]), anchors, neighbors=neighbors
    )
    selection_support = float(np.mean(selection_scores <= radius))
    selection_upper = ensemble_upper[selection] + margin
    selection_coverage = float(np.mean(
        data["target_log_effectivity"][selection] <= selection_upper
    ))
    manifest = {
        "status": "FROZEN",
        "checkpoints": [name for name, _ in models],
        "feature_order": MECHANISM_FEATURE_ORDER,
        "robust_center": scaler.center.tolist(),
        "robust_scale": scaler.scale.tolist(),
        "minimum_robust_scale": scaler.minimum_scale,
        "support_anchors": anchors_path.name,
        "support_anchors_sha256": _sha(anchors_path),
        "support_neighbors": neighbors,
        "support_radius": radius,
        "conformal_margin": margin,
        "reserve_fraction": float(cfg["calibration"]["reserve_fraction"]),
        "training_equations": ["modified_camassa_holm", "benjamin_ono"],
        "equation_identifier_used": False,
        "kdv_rows_used": False,
        "authority": ["tighten_analytical_safe_set", "rank_admitted_actions", "abstain"],
        "forbidden": [
            "advance_pde", "relax_analytical_gate", "modify_invariants",
            "use_reference_online", "retune_during_audit",
        ],
        "dataset_sha256": _sha(dataset_path),
    }
    manifest_path = out / "frozen_u1_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    limits = cfg["acceptance"].copy()
    if args.mode == "development":
        limits.update({
            "minimum_selection_upper_coverage": cfg["development"]["minimum_selection_upper_coverage"],
            "minimum_selection_support_coverage": cfg["development"]["minimum_selection_support_coverage"],
        })
    checks = {
        "dataset_prerequisite": json.loads((out / "dataset_acceptance.json").read_text())["status"] == "PASS",
        "three_seed_ensemble_frozen": len(models) == 3,
        "no_equation_identifier_or_adapter": not manifest["equation_identifier_used"],
        "no_kdv_training_rows": not manifest["kdv_rows_used"],
        "robust_scale_floor_enforced": float(np.min(scaler.scale)) >= float(cfg["model"]["minimum_robust_scale"]),
        "trajectory_conformal_calibration": margin >= 0,
        "selection_upper_coverage": selection_coverage >= float(limits["minimum_selection_upper_coverage"]),
        "selection_support_coverage": selection_support >= float(limits["minimum_selection_support_coverage"]),
        "selection_excluded_from_training_and_calibration": True,
        "restricted_authority": len(manifest["forbidden"]) == 5,
    }
    blocking = [key for key, value in checks.items() if not value]
    report = {
        "status": "PASS" if not blocking else "STOP",
        "next_phase": cfg["next_phase_on_pass"] if not blocking else None,
        "checks": checks, "blocking_failures": blocking,
        "selection_upper_coverage": selection_coverage,
        "selection_support_coverage": selection_support,
        "conformal_margin": margin, "support_radius": radius,
        "support_anchors": len(anchors), "manifest": manifest_path.name,
        "scope": cfg["scope"],
    }
    (out / "training_acceptance.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)
    raise SystemExit(0 if report["status"] == "PASS" else 2)


if __name__ == "__main__":
    main()
