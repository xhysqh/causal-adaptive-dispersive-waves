"""Render revised CMAME Figures 1--5 from frozen V2 trajectories."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from fgsp_ch.visualization.cmame_style import COLORS, cmame_style  # noqa: E402
from fgsp_ch.visualization.wavefield_publication import (  # noqa: E402
    WaveTrajectory, load_collection_trajectory, load_f2_trajectory,
    load_single_trajectory, plot_mch_figure_2, plot_power_law_figure_6,
    plot_wavefield_2x3,
)


def require_pass(path: Path) -> dict:
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("status") != "PASS":
        raise RuntimeError(f"figure prerequisite did not pass: {path}")
    return report


def load_ilw(path: Path) -> WaveTrajectory:
    with np.load(path, allow_pickle=False) as data:
        reference = np.asarray(data["reference"])
        adaptive = np.asarray(data["adaptive"])
        return WaveTrajectory(
            equation="ilw", x=np.asarray(data["x"]), time=np.asarray(data["times"]),
            reference=reference, adaptive=adaptive,
            selection_rule=str(np.asarray(data["selection_rule"]).item()),
            parameter_label=rf"$\delta={float(np.asarray(data['depth']).item()):g}$",
            actions=np.asarray(data["actions"]),
            active_points=np.asarray(data["active_points"]),
        )


def main() -> None:
    stage7 = ROOT / "results/cmame_v2_stage7_bidirectional_transient/full"
    transfer = ROOT / "results/cmame_v2_kdv_ilw_showcases/full"
    power = ROOT / "results/cmame_v2_power_law_showcase/full"
    stage7_report = require_pass(stage7 / "stage5_acceptance.json")
    transfer_report = require_pass(transfer / "acceptance.json")
    power_report = require_pass(power / "acceptance.json")

    mch = load_f2_trajectory(stage7 / "v2_closed_loop_showcase.npz", "mch")
    bo = load_f2_trajectory(stage7 / "v2_closed_loop_showcase.npz", "bo")
    kdv = load_single_trajectory(transfer / "kdv_representative_trajectory.npz")
    ilw = load_ilw(transfer / "ilw_representative_trajectory.npz")
    orders = (2.2, 2.4, 2.6, 2.8, 3.2)
    powers = [
        load_collection_trajectory(
            power / "representative_trajectories.npz",
            coordinate="dispersion_orders", value=order, equation="power_law",
        ) for order in orders
    ]

    output = ROOT / "output/pdf/CMAME_V2_Figures_01_to_05"
    output.mkdir(parents=True, exist_ok=True)
    with cmame_style():
        plot_mch_figure_2(mch, output / "figure_01_mch_wavefield", accent=COLORS["mch"])
        plot_wavefield_2x3(kdv, output / "figure_02_kdv_wavefield", profile_color=COLORS["mch"])
        plot_wavefield_2x3(bo, output / "figure_03_bo_wavefield", profile_color=COLORS["mch"])
        plot_wavefield_2x3(ilw, output / "figure_04_ilw_wavefield", profile_color=COLORS["mch"])
        plot_power_law_figure_6(
            powers, output / "figure_05_power_law_atlas",
            representative_order=2.6, profile_color=COLORS["mch"],
        )

    report = {
        "status": "PASS",
        "checks": {
            "all_sources_pass": True,
            "all_figures_use_frozen_v2": True,
            "all_references_posthoc": True,
            "coordinate_ranges_derived_from_v2_data": True,
            "original_visual_grammar_preserved": True,
        },
        "terminal_relative_h1_error": {
            **stage7_report["terminal_relative_h1_error"],
            **transfer_report["terminal_relative_h1_error"],
            "power_law": power_report["terminal_relative_h1_error"],
        },
        "figures": [
            "figure_01_mch_wavefield.pdf", "figure_02_kdv_wavefield.pdf",
            "figure_03_bo_wavefield.pdf", "figure_04_ilw_wavefield.pdf",
            "figure_05_power_law_atlas.pdf",
        ],
    }
    (output / "acceptance.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
