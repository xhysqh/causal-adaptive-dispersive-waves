"""Train and run the registered minimal V2 causal-method ablation."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from fgsp_ch.utils.config import load_yaml  # noqa: E402


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def run(command: list[str]) -> None:
    completed = subprocess.run(command, cwd=ROOT, check=False)
    if completed.returncode:
        raise RuntimeError(f"ablation command failed ({completed.returncode}): {' '.join(command)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment/cmame_v2_causal_ablation.yaml")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--skip-runs", action="store_true")
    args = parser.parse_args()
    cfg = load_yaml(ROOT / args.config)
    output = ROOT / cfg["output_dir"]
    output.mkdir(parents=True, exist_ok=True)
    source = ROOT / cfg["source_dataset_root"]
    full_manifest = source / "frozen_v2_predictive_risk_manifest.json"
    full_hash_before = sha256(full_manifest)
    variants = tuple(cfg["variants"])

    if not args.skip_training:
        for variant in variants:
            if variant == "full":
                continue
            destination = output / "models" / variant
            run([
                sys.executable, "scripts/train_cmame_v2_predictive_risk_operator.py",
                "--config", cfg["training_config"], "--mode", "full",
                "--variant", variant, "--device", args.device,
                "--dataset-root", cfg["source_dataset_root"],
                "--output-dir", str(destination.relative_to(ROOT)),
            ])

    manifests = {
        "full": full_manifest,
        "no_history": output / "models/no_history/frozen_v2_predictive_risk_manifest.json",
        "one_step": output / "models/one_step/frozen_v2_predictive_risk_manifest.json",
    }
    if not args.skip_runs:
        for variant in variants:
            manifest = manifests[variant]
            if not manifest.exists():
                raise FileNotFoundError(manifest)
            run([
                sys.executable, "scripts/run_cmame_v2_kdv_ilw_showcases.py",
                "--config", cfg["kdv_ilw_config"], "--device", args.device,
                "--manifest", str(manifest.relative_to(ROOT)),
                "--output-dir", str((output / "closed_loop" / variant / "kdv_ilw").relative_to(ROOT)),
            ])
            run([
                sys.executable, "scripts/run_cmame_v2_power_law_showcase.py",
                "--config", cfg["power_law_config"], "--mode", "full",
                "--device", args.device, "--orders", str(cfg["transfer_cases"]["power_law_order"]),
                "--manifest", str(manifest.relative_to(ROOT)),
                "--output-dir", str((output / "closed_loop" / variant / "power_law").relative_to(ROOT)),
            ])

    record = {
        "phase": cfg["phase"],
        "variants": list(variants),
        "full_manifest": str(full_manifest.relative_to(ROOT)),
        "full_manifest_sha256_before": full_hash_before,
        "full_manifest_sha256_after": sha256(full_manifest),
        "full_manifest_unchanged": sha256(full_manifest) == full_hash_before,
        "manifests": {key: str(value.relative_to(ROOT)) for key, value in manifests.items()},
    }
    (output / "run_manifest.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    run([sys.executable, "scripts/analyze_cmame_v2_causal_ablation.py", "--config", args.config])


if __name__ == "__main__":
    main()
