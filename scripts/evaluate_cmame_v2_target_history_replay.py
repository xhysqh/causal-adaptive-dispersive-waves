"""Replay nested history models on one fixed Full-V2 ILW and power-law ledger."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts.evaluate_cmame_v2_nested_history_ablation import predictions, scalar_metrics  # noqa: E402
from fgsp_ch.nonlocal_waves.adaptivity.v2_safety_shielded_controller import (  # noqa: E402
    FrozenV2PredictiveRiskEnsemble,
)
from fgsp_ch.utils.config import load_yaml  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment/cmame_v2_nested_history_ablation.yaml")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    cfg = load_yaml(ROOT / args.config)
    root = ROOT / cfg["output_dir"]
    ledgers = {
        "ILW (delta=1)": np.load(root / "fixed_ledger/kdv_ilw/ilw_representative_trajectory.npz"),
        "power-law (p=2.6)": np.load(root / "fixed_ledger/power_law/full/representative_trajectories.npz"),
    }
    manifests = {"full": ROOT / cfg["source_dataset_root"] / "frozen_v2_predictive_risk_manifest.json"}
    manifests.update({variant: root / "models" / variant / "frozen_v2_predictive_risk_manifest.json"
                      for variant in cfg["variants"] if variant != "full"})
    records = []
    for equation, data in ledgers.items():
        if equation.startswith("ILW"):
            features = np.asarray(data["hold_features"], dtype=float)
            realized = np.asarray(data["realized_hold_risk"], dtype=float)[:len(features)]
        else:
            features = np.asarray(data["hold_features"], dtype=float)[0]
            realized = np.asarray(data["realized_hold_risk"], dtype=float)[0, :len(features)]
        finite = np.isfinite(realized)
        truth = np.log(np.maximum(realized[finite], 1.0e-16))
        for variant in cfg["variants"]:
            path = manifests[variant]
            ensemble = FrozenV2PredictiveRiskEnsemble.load(
                path.parent, json.loads(path.read_text(encoding="utf-8")), device=args.device,
            )
            mean, upper = predictions(ensemble, features[finite])
            row = {"equation": equation, "variant": variant,
                   "input_dimension": int(ensemble.models[0].input_width)}
            row.update(scalar_metrics(truth, mean, upper, float(cfg["threshold"])))
            records.append(row)
    report = {
        "status": "PASS", "records": records,
        "scope": "Fixed-ledger predictive replay; all variants observe identical Full-V2 states.",
    }
    (root / "target_fixed_ledger_metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
