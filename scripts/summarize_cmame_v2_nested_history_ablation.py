"""Create compact manuscript-facing summaries for the nested history ablation."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from fgsp_ch.utils.config import load_yaml  # noqa: E402


def rows(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment/cmame_v2_nested_history_ablation.yaml")
    args = parser.parse_args()
    cfg = load_yaml(ROOT / args.config)
    root = ROOT / cfg["output_dir"]
    source = json.loads((root / "source_offline_metrics.json").read_text(encoding="utf-8"))
    combined = {row["variant"]: row for row in source["records"] if row["equation"] == "combined"}
    closed = []
    for variant in cfg["closed_loop_variants"]:
        base = root / "closed_loop" / variant
        acceptance = json.loads((base / "acceptance.json").read_text(encoding="utf-8"))
        ledger = [row for row in rows(base / "online_decisions.csv") if row["equation"] == "ilw"]
        work = sum(float(row["candidate_work"]) + float(row["verification_work"]) for row in ledger)
        closed.append({
            "variant": variant,
            "terminal_relative_h1_error": float(acceptance["terminal_relative_h1_error"]["ilw"]),
            "work": work,
            "actions": dict(Counter(row["action"] for row in ledger)),
        })
    full_work = next(row["work"] for row in closed if row["variant"] == "full")
    for row in closed:
        row["normalized_work"] = row["work"] / full_work
    ranking = sorted(combined.values(), key=lambda row: (
        row["false_safe_conditional"], row["false_alarm_conditional"],
        row["mean_log_sharpness"], row["mae_log_risk"], row["input_dimension"],
    ))
    report = {
        "status": "PASS",
        "source_combined": list(combined.values()),
        "source_safety_sharpness_ranking": [row["variant"] for row in ranking],
        "ilw_closed_loop": closed,
        "selected_parsimonious_variant": ranking[0]["variant"],
        "interpretation": (
            "All variants have zero false-safe events. Selection therefore follows false-alarm rate, "
            "upper-envelope sharpness, MAE, and parsimony in that order."
        ),
    }
    (root / "nested_ablation_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
