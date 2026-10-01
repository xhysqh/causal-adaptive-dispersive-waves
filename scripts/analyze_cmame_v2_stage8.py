"""Produce stage-8 physical-gating and accuracy/work evidence from frozen runs."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from fgsp_ch.utils.config import load_yaml  # noqa: E402


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def grouped(rows: list[dict], equation: str) -> list[dict]:
    return [row for row in rows if row["equation"] == equation]


def number(rows: list[dict], key: str) -> np.ndarray:
    return np.asarray([float(row[key]) for row in rows], dtype=float)


def physical_figure(rows: list[dict], controller: dict, target: Path) -> None:
    labels = {"modified_camassa_holm": "mCH", "benjamin_ono": "BO"}
    fig, axes = plt.subplots(2, 1, figsize=(7.0, 5.4), sharex=False, constrained_layout=True)
    for axis, equation in zip(axes, labels):
        data = grouped(rows, equation)
        time = number(data, "time")
        points = number(data, "points")
        grid_axis = axis.step(time, points, where="post", color="#1f4e79", linewidth=1.8, label=r"grid $N$")
        axis.set_ylabel(r"$N$", color="#1f4e79")
        axis.tick_params(axis="y", labelcolor="#1f4e79")
        axis.set_title(labels[equation], loc="left", fontsize=10, fontweight="bold")
        indicator = axis.twinx()
        tail = number(data, "spectral_tail") / float(controller["coarse_tail_limit"])
        sharp = number(data, "localized_sharpness") / float(controller["coarse_localized_sharpness_limit"])
        velocity = number(data, "history_velocity_norm") / float(controller["coarse_history_velocity_limit"])
        acceleration = number(data, "history_acceleration_norm") / float(controller["coarse_history_acceleration_limit"])
        curves = [
            indicator.plot(time, tail, color="#c85a17", label=r"$s_{tail}/s_{tail}^{c}$"),
            indicator.plot(time, sharp, color="#7a5195", label=r"$s_{loc}/s_{loc}^{c}$"),
            indicator.plot(time, velocity, color="#2a9d8f", label=r"$\|\Delta q\|/v_c$"),
            indicator.plot(time, acceleration, color="#e9c46a", label=r"$\|\Delta^2q\|/a_c$"),
        ]
        indicator.axhline(1.0, color="#555555", linewidth=.8, linestyle="--")
        indicator.set_ylabel("normalized stability diagnostics")
        for row in data:
            if row["action"] == "refine_space":
                axis.scatter(float(row["time"]), float(row["points"]), marker="^", color="#c1121f", zorder=4)
            elif row["action"] == "coarsen_space":
                axis.scatter(float(row["time"]), float(row["points"]), marker="v", color="#198754", zorder=4)
        if equation == "modified_camassa_holm":
            handles = [grid_axis[0], *(curve[0] for curve in curves)]
            axis.legend(handles=handles, loc="upper right", fontsize=7, ncol=2)
        axis.set_xlabel("time")
    fig.savefig(target, bbox_inches="tight")
    plt.close(fig)


def accuracy_work_figure(summary: list[dict], target: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0), constrained_layout=True)
    for axis, equation in zip(axes, ("modified_camassa_holm", "benjamin_ono")):
        rows = [row for row in summary if row["equation"] == equation]
        for row in rows:
            style = "o" if row["arm"] == "V2 adaptive" else "s"
            color = "#1f4e79" if row["arm"] == "V2 adaptive" else "#888888"
            axis.scatter(float(row["online_candidate_work"]), float(row["terminal_relative_h1_error"]),
                         marker=style, s=62, color=color, label=row["arm"])
        axis.set_xscale("log"); axis.set_yscale("log")
        axis.set_xlabel("online candidate work")
        axis.set_ylabel(r"terminal relative $H^1$ error")
        axis.set_title("mCH" if equation == "modified_camassa_holm" else "BO", fontsize=10)
        axis.grid(True, which="both", linewidth=.35, alpha=.45)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, fontsize=8)
    fig.savefig(target, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adaptive", default="results/cmame_v2_stage7_bidirectional_transient/full")
    parser.add_argument("--baseline", default="results/cmame_v2_stage8_fixed_fine_baseline/full")
    parser.add_argument("--controller-config", default="configs/experiment/cmame_v2_stage4_safety_shielded_controller.yaml")
    parser.add_argument("--experiment-config", default="configs/experiment/cmame_v2_stage7_bidirectional_transient.yaml")
    parser.add_argument("--output", default="results/cmame_v2_stage8_analysis")
    args = parser.parse_args()
    adaptive_root, baseline_root, output = ROOT / args.adaptive, ROOT / args.baseline, ROOT / args.output
    output.mkdir(parents=True, exist_ok=True)
    adaptive = json.loads((adaptive_root / "stage5_acceptance.json").read_text(encoding="utf-8"))
    baseline = json.loads((baseline_root / "stage5_acceptance.json").read_text(encoding="utf-8"))
    trace = read_csv(adaptive_root / "online_decisions.csv")
    controller = load_yaml(ROOT / args.controller_config)["controller"]
    experiment = load_yaml(ROOT / args.experiment_config)
    requested_tolerance = float(experiment["full"]["tolerance"])
    summary = []
    for equation in ("modified_camassa_holm", "benjamin_ono"):
        for arm, report in (("V2 adaptive", adaptive), ("fixed N=128", baseline)):
            summary.append({
                "equation": equation, "arm": arm,
                "terminal_relative_h1_error": report["terminal_relative_h1_error"][equation],
                "online_candidate_work": report["online_candidate_work"][equation],
            })
    with (output / "stage8_accuracy_work.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0])); writer.writeheader(); writer.writerows(summary)
    physical_figure(trace, controller, output / "figure_stage8_physical_gating.pdf")
    accuracy_work_figure(summary, output / "figure_stage8_accuracy_work.pdf")
    checks = {
        "adaptive_and_fixed_runs_pass": adaptive["status"] == "PASS" and baseline["status"] == "PASS",
        "adaptive_has_refinement": adaptive["space_refinement_decisions"] >= 1,
        "adaptive_has_verified_coarsening": adaptive["externally_verified_coarsening_decisions"] >= 1,
        "adaptive_work_below_fixed_grid": all(
            adaptive["online_candidate_work"][equation] < baseline["online_candidate_work"][equation]
            for equation in ("modified_camassa_holm", "benjamin_ono")
        ),
        "adaptive_error_within_registered_tolerance": all(
            adaptive["terminal_relative_h1_error"][equation] <= requested_tolerance
            for equation in ("modified_camassa_holm", "benjamin_ono")
        ),
    }
    blocking = [name for name, passed in checks.items() if not passed]
    report = {
        "status": "PASS" if not blocking else "STOP", "checks": checks,
        "blocking_failures": blocking, "summary": summary,
        "physical_gating_figure": "figure_stage8_physical_gating.pdf",
        "accuracy_work_figure": "figure_stage8_accuracy_work.pdf",
        "requested_tolerance": requested_tolerance,
        "scope": "Posthoc visualization and comparator audit only. No trace, reference, threshold, checkpoint, or frozen model is modified.",
    }
    (output / "stage8_acceptance.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    raise SystemExit(0 if not blocking else 2)


if __name__ == "__main__":
    main()
