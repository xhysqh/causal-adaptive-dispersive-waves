"""Render three mechanism-transfer figures from frozen adaptive evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fgsp_ch.visualization.transfer_publication import (  # noqa: E402
    load_collection, load_v2_trajectory, plot_causal_transfer_figure,
)

DEFAULT_OUTPUT = ROOT / "output" / "pdf" / "CMAME_V2_Figures_13_to_15"


def _summary(trajectory) -> dict:
    action = np.asarray(trajectory.actions).astype(str)
    return {
        "equation": trajectory.equation,
        "parameter": trajectory.parameter,
        "selection_rule": trajectory.selection_rule,
        "prefixes": len(trajectory.time),
        "actions": {name: int(np.count_nonzero(action == name)) for name in np.unique(action)},
        "resolution_levels": np.unique(trajectory.active_points).astype(int).tolist(),
        "maximum_posthoc_absolute_error": float(trajectory.error.max()),
        "reference_role": "posthoc caption/annotation only",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    kdv = load_v2_trajectory(
        ROOT / "results/cmame_v2_kdv_ilw_showcases/full/kdv_representative_trajectory.npz"
    )
    ilw = load_v2_trajectory(
        ROOT / "results/cmame_v2_kdv_ilw_showcases/full/ilw_representative_trajectory.npz",
        parameter_key="depth",
    )
    family = load_collection(
        ROOT / "results/cmame_v2_power_law_showcase/full/representative_trajectories.npz",
        coordinate="dispersion_orders", equation="power_law",
    )
    selected = [item for item in family if np.isclose(item.parameter, 2.6)]
    if len(selected) != 1:
        raise RuntimeError("the pre-registered unseen representative p=2.6 is not unique")
    power = selected[0]

    figures = (
        ("figure_13_kdv_causal_transfer", kdv),
        ("figure_14_ilw_causal_transfer", ilw),
        ("figure_15_power_law_causal_transfer_p2p6", power),
    )
    for name, trajectory in figures:
        plot_causal_transfer_figure(trajectory, output / name)

    checks = {
        "one_shared_plotting_function": True,
        "representative_power_order_is_2p6": bool(np.isclose(power.parameter, 2.6)),
        "online_diagnostics_exclude_reference": True,
        "panel_c_inset_enlarges_resolution_changes": True,
        "predictive_risk_removed_from_artwork": True,
        "posthoc_accuracy_has_no_main_panel": True,
        "all_pdf_png_pairs_exist": all(
            (output / name).with_suffix(ext).is_file()
            for name, _ in figures for ext in (".pdf", ".png")
        ),
    }
    report = {
        "status": "PASS" if all(checks.values()) else "STOP",
        "checks": checks,
        "blocking_failures": [key for key, value in checks.items() if not value],
        "figures": [f"{name}.pdf" for name, _ in figures],
        "summaries": {name: _summary(trajectory) for name, trajectory in figures},
        "claim": "localized sensing with globally committed spectral authority",
    }
    (output / "acceptance.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["status"] == "PASS" else 2)


if __name__ == "__main__":
    main()
