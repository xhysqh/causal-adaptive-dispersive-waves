from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

GROUPS = {
    "mechanism": (
        "plot_cmame_v2_figures_1_to_5.py",
        "plot_cmame_v2_figures_12_to_14.py",
        "plot_cmame_v2_figure_16_predictive_risk.py",
    ),
    "pinn": ("plot_cmame_v2_figures_6_to_11.py",),
}

PREREQUISITES = {
    "mechanism": (
        "results/cmame_v2_stage7_bidirectional_transient/full/v2_closed_loop_showcase.npz",
        "results/cmame_v2_kdv_ilw_showcases/full/kdv_representative_trajectory.npz",
        "results/cmame_v2_kdv_ilw_showcases/full/ilw_representative_trajectory.npz",
        "results/cmame_v2_power_law_showcase/full/representative_trajectories.npz",
    ),
    "pinn": (
        "results/cmame_v2_pinn_aligned_showcase/full/v2_closed_loop_showcase.npz",
        "results/cmame_v2_pinn_comparison/acceptance.json",
    ),
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate manuscript figures from locally produced results.")
    parser.add_argument(
        "--group", choices=("all", "mechanism", "pinn"), default="all",
        help="figure group to generate",
    )
    parser.add_argument("--check-only", action="store_true", help="only check required result files")
    args = parser.parse_args()

    selected = tuple(GROUPS) if args.group == "all" else (args.group,)
    missing = {
        group: [item for item in PREREQUISITES[group] if not (ROOT / item).is_file()]
        for group in selected
    }
    missing = {group: items for group, items in missing.items() if items}
    if missing:
        print("Missing locally generated prerequisites:")
        for group, items in missing.items():
            print(f"[{group}]")
            for item in items:
                print(f"  - {item}")
        print("Run the workflows in REPRODUCIBILITY.md first. No data are downloaded automatically.")
        return 2

    if args.check_only:
        print("All requested figure prerequisites are present.")
        return 0

    for group in selected:
        for script in GROUPS[group]:
            command = [sys.executable, str(ROOT / "scripts" / script)]
            print("+", " ".join(command), flush=True)
            subprocess.run(command, cwd=ROOT, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

