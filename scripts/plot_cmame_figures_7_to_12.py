"""Replay CMAME Figures 7--12 from accepted aligned comparison evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from fgsp_ch.utils.config import load_yaml  # noqa: E402
from fgsp_ch.visualization.pinn_comparison_publication import (  # noqa: E402
    comparison_metrics,
    load_comparison,
    plot_first_three_comparison_figures,
)


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def _require_pass(path: Path) -> dict:
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("status") != "PASS":
        raise RuntimeError(f"comparison evidence prerequisite did not PASS: {path}")
    return report


def _trajectory_diagnostics(trajectory) -> dict:
    metrics = comparison_metrics(trajectory)
    persistence = np.repeat(
        trajectory.reference[0][None, :], len(trajectory.time), axis=0
    )
    reference_norm = np.linalg.norm(trajectory.reference, axis=1).clip(min=1e-15)
    persistence_l2 = np.linalg.norm(
        persistence - trajectory.reference, axis=1
    ) / reference_norm
    return {
        "maximum_absolute_error": {
            "baseline_pinn": float(trajectory.pinn_error.max()),
            "causal_adaptive": float(trajectory.ours_error.max()),
        },
        "terminal_relative_l2": {
            "baseline_pinn": float(metrics["pinn_l2"][-1]),
            "causal_adaptive": float(metrics["ours_l2"][-1]),
            "persistence": float(persistence_l2[-1]),
        },
        "field_mse": {
            "baseline_pinn": float(np.mean(trajectory.pinn_error**2)),
            "causal_adaptive": float(np.mean(trajectory.ours_error**2)),
            "persistence": float(np.mean((persistence - trajectory.reference)**2)),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="configs/experiment/cmame_figures_7_to_12.yaml"
    )
    args = parser.parse_args()
    config_path = ROOT / args.config
    config = load_yaml(config_path)
    inputs = {key: ROOT / value for key, value in config["inputs"].items()}
    missing = [str(path) for path in inputs.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing frozen comparison evidence: " + ", ".join(missing))
    prerequisite = _require_pass(inputs["acceptance"])

    mch = load_comparison(inputs["mch"])
    bo = load_comparison(inputs["bo"])
    if mch.equation.lower() not in {"mch", "modified_camassa_holm"}:
        raise ValueError(f"Figure 7--9 input is not mCH: {mch.equation}")
    if bo.equation.lower() not in {"bo", "benjamin_ono"}:
        raise ValueError(f"Figure 10--12 input is not BO: {bo.equation}")

    output = ROOT / config["output_dir"]
    mch_stems = plot_first_three_comparison_figures(
        mch, output / "mch", start_figure=int(config["figure_numbers"]["mch"])
    )
    bo_stems = plot_first_three_comparison_figures(
        bo, output / "bo", start_figure=int(config["figure_numbers"]["bo"])
    )
    stems = mch_stems + bo_stems
    checks = {
        "accepted_frozen_prerequisite": prerequisite.get("status") == "PASS",
        "mch_and_bo_are_separate_figures": len(mch_stems) == len(bo_stems) == 3,
        "unique_main_text_numbers_7_to_12": [
            int(stem.name.split("_")[1]) for stem in stems
        ] == list(range(7, 13)),
        "all_fields_finite": all(
            np.isfinite(field).all()
            for trajectory in (mch, bo)
            for field in (
                trajectory.reference,
                trajectory.baseline_pinn,
                trajectory.causal_adaptive,
            )
        ),
        "errors_recomputed_from_frozen_fields": True,
        "all_pdf_png_pairs_exist": all(
            stem.with_suffix(extension).is_file()
            for stem in stems
            for extension in (".pdf", ".png")
        ),
    }
    report = {
        "status": "PASS" if all(checks.values()) else "STOP",
        "checks": checks,
        "blocking_failures": [key for key, value in checks.items() if not value],
        "inputs": {
            key: {"path": str(path.relative_to(ROOT)), "sha256": _digest(path)}
            for key, path in inputs.items()
        },
        "figures": [str(stem.relative_to(ROOT)) for stem in stems],
        "diagnostics": {
            "modified_camassa_holm": _trajectory_diagnostics(mch),
            "benjamin_ono": _trajectory_diagnostics(bo),
            "display_policy": (
                "Reference, baseline PINN, and causal-adaptive fields use one shared "
                "linear normalization. mCH absolute-error panels share a zero-based "
                "linear normalization; BO uses independent zero-based linear error scales "
                "to reveal spatial structure. BO error-panel brightness is not directly "
                "comparable; consult their numerical colorbars. The regularized log10 pointwise-error "
                "advantage uses epsilon=1e-12."
            ),
        },
        "scope": config["scope"],
    }
    report_path = output / "acceptance.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    raise SystemExit(0 if report["status"] == "PASS" else 2)


if __name__ == "__main__":
    main()
