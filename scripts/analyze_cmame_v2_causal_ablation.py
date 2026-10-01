"""Summarize the registered V2 causal-history and horizon ablation."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from fgsp_ch.utils.config import load_yaml  # noqa: E402


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def risk_metrics(realized: np.ndarray, actions: list[str], threshold: float) -> dict:
    values = np.asarray(realized, dtype=float)[:len(actions)]
    finite = np.isfinite(values)
    crossing = np.flatnonzero(finite & (values > threshold))
    corrective = np.asarray([index for index, action in enumerate(actions)
                             if action in {"refine_space", "shrink_time"}], dtype=int)
    lead = None
    if len(crossing) and len(corrective):
        eligible = corrective[corrective <= crossing[0]]
        if len(eligible):
            lead = int(crossing[0] - eligible[-1])
        else:
            lead = int(crossing[0] - corrective[0])
    return {
        "maximum_realized_hold_risk": float(np.max(values[finite])) if np.any(finite) else None,
        "realized_risk_violation_rate": float(np.mean(values[finite] > threshold)) if np.any(finite) else None,
        "first_corrective_lead_steps": lead,
    }


def ilw_row(root: Path, variant: str, threshold: float) -> dict:
    base = root / "closed_loop" / variant / "kdv_ilw"
    data = np.load(base / "ilw_representative_trajectory.npz")
    rows = [row for row in read_csv(base / "online_decisions.csv") if row["equation"] == "ilw"]
    actions = [row["action"] for row in rows]
    work = sum(float(row["candidate_work"]) + float(row["verification_work"]) for row in rows)
    acceptance = json.loads((base / "acceptance.json").read_text(encoding="utf-8"))
    result = {
        "equation": "ILW (delta=1)", "variant": variant,
        "terminal_relative_h1_error": float(acceptance["terminal_relative_h1_error"]["ilw"]),
        "online_work": float(work), "decisions": len(rows),
        **{f"action_{key}": value for key, value in Counter(actions).items()},
        **risk_metrics(data["realized_hold_risk"], actions, threshold),
    }
    return result


def power_row(root: Path, variant: str, threshold: float) -> dict:
    base = root / "closed_loop" / variant / "power_law" / "full"
    data = np.load(base / "representative_trajectories.npz")
    rows = read_csv(base / "online_decisions.csv")
    if len(data["dispersion_orders"]) != 1 or not np.isclose(data["dispersion_orders"][0], 2.6):
        raise RuntimeError("power-law ablation must contain only p=2.6")
    actions = [row["action"] for row in rows]
    work = sum(float(row["candidate_work"]) + float(row["verification_work"]) for row in rows)
    reference = np.asarray(data["reference"])[0, -1]
    adaptive = np.asarray(data["adaptive"])[0, -1]
    # The authoritative relative H1 value is already recorded by the run.
    acceptance = json.loads((base / "acceptance.json").read_text(encoding="utf-8"))
    terminal = float(acceptance["terminal_relative_h1_error"]["2.6"])
    if not np.all(np.isfinite(reference)) or not np.all(np.isfinite(adaptive)):
        raise RuntimeError("non-finite power-law terminal field")
    result = {
        "equation": "power-law (p=2.6)", "variant": variant,
        "terminal_relative_h1_error": terminal,
        "online_work": float(work), "decisions": len(rows),
        **{f"action_{key}": value for key, value in Counter(actions).items()},
        **risk_metrics(np.asarray(data["realized_hold_risk"])[0], actions, threshold),
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment/cmame_v2_causal_ablation.yaml")
    args = parser.parse_args()
    cfg = load_yaml(ROOT / args.config)
    root = ROOT / cfg["output_dir"]
    threshold = float(cfg["metrics"]["realized_risk_threshold"])
    variants = tuple(cfg["variants"])
    records = []
    for variant in variants:
        records.extend((ilw_row(root, variant, threshold), power_row(root, variant, threshold)))
    for equation in sorted({row["equation"] for row in records}):
        full_work = next(row["online_work"] for row in records
                         if row["equation"] == equation and row["variant"] == "full")
        for row in records:
            if row["equation"] == equation:
                row["normalized_work"] = row["online_work"] / full_work
    fields = sorted({key for row in records for key in row}, key=lambda key: (
        ["equation", "variant", "terminal_relative_h1_error", "online_work", "normalized_work",
         "maximum_realized_hold_risk", "realized_risk_violation_rate",
         "first_corrective_lead_steps", "decisions"].index(key)
        if key in ["equation", "variant", "terminal_relative_h1_error", "online_work", "normalized_work",
                   "maximum_realized_hold_risk", "realized_risk_violation_rate",
                   "first_corrective_lead_steps", "decisions"] else 100, key))
    with (root / "ablation_table.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(records)
    tolerance = float(cfg["metrics"]["tolerance"])
    run_manifest = json.loads((root / "run_manifest.json").read_text(encoding="utf-8"))
    checks = {
        "six_registered_runs_present": len(records) == 6,
        "all_metrics_finite": all(np.isfinite(row["terminal_relative_h1_error"])
                                  and np.isfinite(row["online_work"]) for row in records),
        "all_terminal_tolerances_met": all(row["terminal_relative_h1_error"] <= tolerance for row in records),
        "full_manifest_unchanged": bool(run_manifest["full_manifest_unchanged"]),
    }
    by_case = {(row["equation"], row["variant"]): row for row in records}
    interpretation = {}
    for equation in sorted({row["equation"] for row in records}):
        full = by_case[(equation, "full")]
        no_history = by_case[(equation, "no_history")]
        one_step = by_case[(equation, "one_step")]
        interpretation[equation] = {
            "h2_work_reduction_vs_h1": 1.0 - full["online_work"] / one_step["online_work"],
            "h2_error_ratio_vs_h1": full["terminal_relative_h1_error"] / one_step["terminal_relative_h1_error"],
            "full_work_ratio_vs_no_history": full["online_work"] / no_history["online_work"],
            "full_error_ratio_vs_no_history": (
                full["terminal_relative_h1_error"] / no_history["terminal_relative_h1_error"]
            ),
            "realized_risk_crossing_observed": bool(
                full["realized_risk_violation_rate"] and full["realized_risk_violation_rate"] > 0.0
            ),
        }
    history_benefit = all(
        interpretation[equation]["full_error_ratio_vs_no_history"] < 1.0
        and interpretation[equation]["full_work_ratio_vs_no_history"] <= 1.0
        for equation in interpretation
    )
    horizon_work_benefit = all(
        interpretation[equation]["h2_work_reduction_vs_h1"] > 0.0
        for equation in interpretation
    )
    lead_identifiable = any(
        interpretation[equation]["realized_risk_crossing_observed"]
        for equation in interpretation
    )
    manuscript_assessment = {
        "history_benefit_demonstrated": history_benefit,
        "two_step_work_reduction_demonstrated": horizon_work_benefit,
        "predictive_lead_identifiable_at_registered_tolerance": lead_identifiable,
        "recommended_claim": (
            ("The two-step horizon reduces executed work relative to the one-step variant in both "
             "registered transfer cases. " if horizon_work_benefit else
             "The work comparison between the two-step and one-step horizons is case dependent. ")
            + "All errors remain far below tolerance. The present smooth transfer cases do not "
              "demonstrate an independent benefit from the history block and do not identify warning "
              "lead because realized HOLD risk never crosses unity."
        ),
    }
    report = {
        "status": "PASS" if all(checks.values()) else "STOP",
        "checks": checks,
        "blocking_failures": [key for key, value in checks.items() if not value],
        "records": records,
        "interpretation": interpretation,
        "manuscript_assessment": manuscript_assessment,
        "scope": cfg["scope"],
    }
    (root / "ablation_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (root / "acceptance.json").write_text(json.dumps({
        "status": report["status"], "checks": checks,
        "blocking_failures": report["blocking_failures"],
    }, indent=2), encoding="utf-8")
    tex_rows = []
    labels = {"full": "Full V2", "no_history": "No history", "one_step": "One step"}
    for row in records:
        lead = "--" if row["first_corrective_lead_steps"] is None else str(row["first_corrective_lead_steps"])
        tex_rows.append(
            f'{row["equation"]} & {labels[row["variant"]]} & '
            f'{row["terminal_relative_h1_error"]:.3e} & {row["normalized_work"]:.3f} & '
            f'{row["maximum_realized_hold_risk"]:.3f} & '
            f'{row["realized_risk_violation_rate"]:.3f} & {lead} \\\\'
        )
    (root / "ablation_table.tex").write_text("\n".join(tex_rows) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    raise SystemExit(0 if report["status"] == "PASS" else 2)


if __name__ == "__main__":
    main()
