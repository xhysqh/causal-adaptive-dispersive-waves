"""Train and freeze the CMAME V2 stage-3 finite-horizon risk ensemble."""

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

from fgsp_ch.nonlocal_waves.adaptivity.mechanism_coordinates import (  # noqa: E402
    CAUSAL_HISTORY_FEATURE_ORDER, MECHANISM_FEATURE_ORDER,
    V2_CAUSAL_FEATURE_ORDER, knn_support_score, trajectory_conformal_radius,
)
from fgsp_ch.nonlocal_waves.adaptivity.v2_predictive_risk_operator import (  # noqa: E402
    V2PredictiveRiskOperator, predictive_risk_loss,
)
from fgsp_ch.utils.config import load_yaml  # noqa: E402


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def robust_scaler(rows: np.ndarray, minimum_scale: float) -> tuple[np.ndarray, np.ndarray]:
    center = np.median(rows, axis=0)
    mad = np.median(np.abs(rows - center), axis=0)
    return center, np.maximum(1.4826 * mad, float(minimum_scale))


def trajectory_margin(predicted, truth, groups, coverage: float) -> float:
    residual = np.asarray(truth) - np.asarray(predicted)
    group_id = np.asarray(groups).astype(str)
    maxima = np.asarray([np.max(residual[group_id == value]) for value in sorted(set(group_id))])
    rank = min(len(maxima), int(np.ceil((len(maxima) + 1) * float(coverage))))
    return max(0.0, float(np.partition(maxima, rank - 1)[rank - 1]))


@torch.no_grad()
def member_upper(model, features: torch.Tensor, device: torch.device) -> np.ndarray:
    return model(features.to(device)).upper_log_risk.detach().cpu().numpy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment/cmame_v2_stage3_predictive_risk_operator.yaml")
    parser.add_argument("--mode", choices=("development", "full"), default="development")
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--variant",
        choices=("full", "no_history", "instantaneous", "departure", "departure_velocity", "one_step"),
        default="full",
    )
    parser.add_argument("--output-dir")
    parser.add_argument("--dataset-root")
    args = parser.parse_args()
    config_path = ROOT / args.config
    cfg = load_yaml(config_path)
    dataset_root = ROOT / (args.dataset_root or str(Path(cfg["output_dir"]) / args.mode))
    output = ROOT / (args.output_dir or str(Path(cfg["output_dir"]) / args.mode))
    output.mkdir(parents=True, exist_ok=True)
    data_path = dataset_root / cfg["dataset_name"]
    acceptance_path = dataset_root / "dataset_acceptance.json"
    prerequisite = json.loads(acceptance_path.read_text(encoding="utf-8"))
    if prerequisite["status"] != "PASS":
        raise RuntimeError("V2 stage 3 requires a passing stages-0--2 dataset")
    data = np.load(data_path)
    features = np.asarray(data["features"], dtype=np.float32)
    base_width = len(MECHANISM_FEATURE_ORDER)
    block_width = len(CAUSAL_HISTORY_FEATURE_ORDER) // 3
    control_start = base_width + len(CAUSAL_HISTORY_FEATURE_ORDER)
    if args.variant in {"no_history", "instantaneous"}:
        active_indices = tuple(range(base_width)) + tuple(range(control_start, len(V2_CAUSAL_FEATURE_ORDER)))
        history_blocks = ()
    elif args.variant == "departure":
        active_indices = tuple(range(base_width + block_width)) + tuple(range(control_start, len(V2_CAUSAL_FEATURE_ORDER)))
        history_blocks = ("deviation",)
    elif args.variant == "departure_velocity":
        active_indices = tuple(range(base_width + 2 * block_width)) + tuple(range(control_start, len(V2_CAUSAL_FEATURE_ORDER)))
        history_blocks = ("deviation", "velocity")
    else:
        active_indices = tuple(range(len(V2_CAUSAL_FEATURE_ORDER)))
        history_blocks = ("deviation", "velocity", "acceleration")
    features = features[:, active_indices]
    if features.shape[1] != len(active_indices):
        raise RuntimeError("V2 stage-3 dataset violates its active feature contract")
    split = data["split"].astype(str)
    train = np.flatnonzero(split == "train")
    validation = np.flatnonzero(split == "validation")
    calibration = np.flatnonzero(split == "calibration")
    selection = np.flatnonzero(split == "selection")
    if not all(len(indices) for indices in (train, validation, calibration, selection)):
        raise RuntimeError("V2 stage 3 requires all four registered splits")
    center, scale = robust_scaler(features[train], float(cfg["model"]["minimum_robust_scale"]))
    x = torch.as_tensor(features, dtype=torch.float32)
    if args.variant == "one_step":
        one_step = np.asarray(data["rollout_relative_errors"], dtype=np.float64)[:, 0]
        budget = np.maximum(np.asarray(data["local_rollout_budget"], dtype=np.float64), 1.0e-12)
        risk_target = np.log(np.maximum(one_step / budget, 1.0e-16))
    else:
        risk_target = np.asarray(data["target_log_future_risk"], dtype=np.float64)
    risk = torch.as_tensor(risk_target, dtype=torch.float32)
    cost = torch.as_tensor(data["target_log_cost"], dtype=torch.float32)
    probe = torch.as_tensor(data["target_probe"], dtype=torch.float32)
    device = torch.device(args.device)
    members: list[tuple[str, V2PredictiveRiskOperator, float]] = []
    for seed in cfg["training"]["seeds"]:
        torch.manual_seed(int(seed))
        model = V2PredictiveRiskOperator(
            hidden=int(cfg["model"]["hidden"]), input_width=len(active_indices),
        ).to(device)
        model.set_robust_scaler(center, scale)
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["training"]["learning_rate"]),
                                      weight_decay=float(cfg["training"]["weight_decay"]))
        for _ in range(int(cfg["training"]["epochs"])):
            optimizer.zero_grad()
            output_member = model(x[train].to(device))
            loss = predictive_risk_loss(
                output_member, risk[train].to(device), cost[train].to(device), probe[train].to(device),
                upper_quantile=float(cfg["training"]["upper_quantile"]),
                underestimate_weight=float(cfg["training"]["underestimate_weight"]),
                cost_weight=float(cfg["training"]["cost_weight"]),
                probe_weight=float(cfg["training"]["probe_weight"]),
                probe_positive_weight=float(cfg["training"]["probe_positive_weight"]),
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["training"]["gradient_clip"]))
            optimizer.step()
        validation_loss = float(predictive_risk_loss(
            model(x[validation].to(device)), risk[validation].to(device), cost[validation].to(device), probe[validation].to(device),
            upper_quantile=float(cfg["training"]["upper_quantile"]),
            underestimate_weight=float(cfg["training"]["underestimate_weight"]),
            cost_weight=float(cfg["training"]["cost_weight"]),
            probe_weight=float(cfg["training"]["probe_weight"]),
            probe_positive_weight=float(cfg["training"]["probe_positive_weight"]),
        ).detach().cpu())
        name = f"v2_predictive_risk_member_{seed}.pt"
        torch.save({
            "model": model.state_dict(), "hidden": model.hidden,
            "input_width": model.input_width,
        }, output / name)
        members.append((name, model.eval(), validation_loss))
    member_predictions = np.stack([member_upper(model, x, device) for _, model, _ in members])
    raw_upper = member_predictions.mean(axis=0) + float(cfg["calibration"]["ensemble_std_multiplier"]) * member_predictions.std(axis=0)
    conformal_margin = trajectory_margin(raw_upper[calibration], risk_target[calibration],
                                         data["group_id"][calibration], float(cfg["calibration"]["trajectory_coverage"]))
    transformed_train = (features[train] - center) / scale
    rng = np.random.default_rng(317311)
    maximum = int(cfg["calibration"]["maximum_support_anchors"])
    anchors = transformed_train if len(train) <= maximum else transformed_train[rng.choice(len(train), maximum, replace=False)]
    anchors_path = output / "v2_support_anchors.npz"
    np.savez_compressed(anchors_path, anchors=anchors.astype(np.float32))
    neighbors = int(cfg["calibration"]["support_neighbors"])
    calibration_distance = knn_support_score((features[calibration] - center) / scale, anchors, neighbors=neighbors)
    support_radius = trajectory_conformal_radius(calibration_distance, data["group_id"][calibration],
                                                 coverage=float(cfg["calibration"]["support_coverage"]))
    selection_distance = knn_support_score((features[selection] - center) / scale, anchors, neighbors=neighbors)
    final_upper = raw_upper + conformal_margin
    selection_coverage = float(np.mean(risk_target[selection] <= final_upper[selection]))
    calibration_coverage = float(np.mean(risk_target[calibration] <= final_upper[calibration]))
    selection_support = float(np.mean(selection_distance <= support_radius))
    limits = cfg["acceptance"].copy()
    if args.mode == "development":
        limits.update(cfg["development"])
    manifest = {
        "status": "FROZEN", "phase": cfg["phase"],
        "checkpoints": [name for name, _, _ in members],
        "feature_order": V2_CAUSAL_FEATURE_ORDER,
        "active_feature_order": [V2_CAUSAL_FEATURE_ORDER[index] for index in active_indices],
        "active_feature_indices": list(active_indices),
        "input_dimension": len(active_indices),
        "history_blocks": list(history_blocks),
        "robust_center": center.tolist(), "robust_scale": scale.tolist(),
        "support_anchors": anchors_path.name, "support_anchors_sha256": sha256(anchors_path),
        "support_neighbors": neighbors, "support_radius": support_radius,
        "ensemble_std_multiplier": float(cfg["calibration"]["ensemble_std_multiplier"]),
        "conformal_log_risk_margin": conformal_margin,
        "horizon_steps": 1 if args.variant == "one_step" else 2,
        "continuation": "zero_order_hold_post_action_discretization",
        "ablation_variant": args.variant,
        "input_ablation": args.variant,
        "training_equations": ["modified_camassa_holm", "benjamin_ono"],
        "equation_identifier_used": False,
        "reference_used_online": False,
        "authority": ["predict_future_risk_upper_bound", "rank_prevalidated_candidates", "request_probe"],
        "forbidden": ["advance_pde", "relax_deterministic_safety_checks", "change_invariants", "use_reference_online", "online_retraining"],
        "probe_status": "diagnostic_only_sparse_positive_labels",
        "dataset_sha256": sha256(data_path), "config_sha256": sha256(config_path),
    }
    manifest_path = output / "frozen_v2_predictive_risk_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    checks = {
        "dataset_prerequisite": prerequisite["status"] == "PASS",
        "three_member_ensemble": len(members) == 3,
        "v2_feature_contract": features.shape[1] == len(active_indices),
        "equation_identifier_absent": not manifest["equation_identifier_used"],
        "selection_split_excluded_from_fit_and_calibration": True,
        "trajectory_group_conformal_calibration": conformal_margin >= 0.0,
        "selection_upper_coverage": selection_coverage >= float(limits["minimum_selection_upper_coverage"]),
        "selection_support_coverage": selection_support >= float(limits["minimum_selection_support_coverage"]),
        "restricted_authority": len(manifest["forbidden"]) == 5,
        "reference_excluded_from_online_features": not manifest["reference_used_online"],
    }
    blocking = [name for name, ok in checks.items() if not ok]
    report = {
        "status": "PASS" if not blocking else "STOP", "next_phase": cfg["next_phase_on_pass"] if not blocking else None,
        "checks": checks, "blocking_failures": blocking,
        "validation_member_losses": {name: value for name, _, value in members},
        "calibration_upper_coverage": calibration_coverage, "selection_upper_coverage": selection_coverage,
        "selection_support_coverage": selection_support, "conformal_log_risk_margin": conformal_margin,
        "support_radius": support_radius, "support_anchors": int(len(anchors)),
        "probe_positive_rates": {name: float(data["target_probe"][split == name].mean()) for name in sorted(set(split))},
        "manifest": manifest_path.name,
        "ablation_variant": args.variant,
        "scope": (
            cfg["scope"] if args.variant == "full" else
            f"Registered {args.variant} ablation using the unchanged mCH/BO trajectory groups, "
            "architecture, seeds, optimizer, calibration protocol, and restricted online authority."
        ),
    }
    (output / "stage3_training_acceptance.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    raise SystemExit(0 if not blocking else 2)


if __name__ == "__main__":
    main()
