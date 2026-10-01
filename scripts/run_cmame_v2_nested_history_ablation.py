"""Train and evaluate the registered 24/30/36/42 history ablation."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from fgsp_ch.utils.config import load_yaml  # noqa: E402


def run(command):
    completed = subprocess.run(command, cwd=ROOT, check=False)
    if completed.returncode:
        raise RuntimeError(f"nested ablation command failed: {' '.join(command)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment/cmame_v2_nested_history_ablation.yaml")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--skip-closed-loop", action="store_true")
    args = parser.parse_args()
    cfg = load_yaml(ROOT / args.config)
    root = ROOT / cfg["output_dir"]
    root.mkdir(parents=True, exist_ok=True)
    if not args.skip_training:
        for variant in cfg["variants"]:
            if variant == "full":
                continue
            destination = root / "models" / variant
            run([
                sys.executable, "scripts/train_cmame_v2_predictive_risk_operator.py",
                "--config", cfg["training_config"], "--mode", "full",
                "--variant", variant, "--device", args.device,
                "--dataset-root", cfg["source_dataset_root"],
                "--output-dir", str(destination.relative_to(ROOT)),
            ])
    run([sys.executable, "scripts/evaluate_cmame_v2_nested_history_ablation.py",
         "--config", args.config, "--device", args.device])
    if not args.skip_closed_loop:
        manifests = {
            "full": ROOT / cfg["source_dataset_root"] / "frozen_v2_predictive_risk_manifest.json",
            "instantaneous": root / "models/instantaneous/frozen_v2_predictive_risk_manifest.json",
            "departure": root / "models/departure/frozen_v2_predictive_risk_manifest.json",
            "departure_velocity": root / "models/departure_velocity/frozen_v2_predictive_risk_manifest.json",
        }
        for variant in cfg["closed_loop_variants"]:
            manifest = manifests[variant]
            run([
                sys.executable, "scripts/run_cmame_v2_kdv_ilw_showcases.py",
                "--config", cfg["kdv_ilw_config"], "--device", args.device,
                "--manifest", str(manifest.relative_to(ROOT)),
                "--output-dir", str((root / "closed_loop" / variant).relative_to(ROOT)),
            ])
    summary = {"status": "PASS", "variants": cfg["variants"],
               "closed_loop_variants": cfg["closed_loop_variants"], "scope": cfg["scope"]}
    (root / "acceptance.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
