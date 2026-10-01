from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fgsp_ch.visualization.pinn_comparison_publication import (
    comparison_metrics,
    comparison_metrics_common_band,
    load_comparison,
    plot_figure_7_direct_comparison,
    plot_figure_8_profiles,
    plot_figure_9_error_propagation,
)


DEFAULT_SOURCE = ROOT / "results" / "cmame_v2_pinn_comparison"
DEFAULT_OUTPUT = ROOT / "output" / "pdf" / "CMAME_V2_Figures_06_to_11"


def _publication_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 9.1,
            "axes.labelsize": 9.4,
            "axes.titlesize": 10.0,
            "legend.fontsize": 8.1,
            "xtick.labelsize": 8.3,
            "ytick.labelsize": 8.3,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.45,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.035,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _save(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".pdf"), dpi=600)
    fig.savefig(stem.with_suffix(".png"), dpi=300)
    plt.close(fig)


def _panel(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.15,
        1.06,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10.2,
        fontweight="bold",
    )


def _main_composite(mch, bo, output: Path) -> None:
    _publication_style()
    fig, axes = plt.subplots(2, 3, figsize=(7.35, 4.55), constrained_layout=True)
    colors = {"reference": "#20252b", "pinn": "#d55e00", "v2": "#0072b2"}

    for row, (name, trajectory) in enumerate((("mCH", mch), ("BO", bo))):
        metrics, cutoff = comparison_metrics_common_band(trajectory)
        final = -1
        x = trajectory.x
        tau = trajectory.tau

        ax = axes[row, 0]
        ax.plot(x, trajectory.reference[final], color=colors["reference"], label="Reference")
        ax.plot(x, trajectory.baseline_pinn[final], color=colors["pinn"], linestyle="-.", label="PINN")
        ax.plot(x, trajectory.causal_adaptive[final], color=colors["v2"], linestyle="--", label="Adaptive")
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(rf"{name}: $u(x,T)$")
        ax.grid(alpha=0.16, linewidth=0.5)

        ax = axes[row, 1]
        pinn_error = np.maximum(np.abs(trajectory.baseline_pinn[final] - trajectory.reference[final]), 1.0e-14)
        v2_error = np.maximum(np.abs(trajectory.causal_adaptive[final] - trajectory.reference[final]), 1.0e-14)
        ax.semilogy(x, pinn_error, color=colors["pinn"], linestyle="-.")
        ax.semilogy(x, v2_error, color=colors["v2"], linestyle="--")
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$|u-u_{\rm ref}|$")
        ax.grid(alpha=0.16, linewidth=0.5, which="both")

        ax = axes[row, 2]
        start = 1 if len(tau) > 1 else 0
        ax.semilogy(tau[start:], np.maximum(metrics["pinn_h1"][start:], 1.0e-14), color=colors["pinn"], linestyle="-.")
        ax.semilogy(tau[start:], np.maximum(metrics["ours_h1"][start:], 1.0e-14), color=colors["v2"], linestyle="--")
        ax.set_xlabel(r"$\tau=t/T$")
        ax.set_ylabel(rf"common-band $H^1$ ($|k|\leq {cutoff}$)")
        ax.grid(alpha=0.16, linewidth=0.5, which="both")
        if trajectory.active_points is not None:
            changes = np.flatnonzero(np.diff(trajectory.active_points) != 0) + 1
            for change in changes:
                ax.axvline(tau[change], color="#6F7680", lw=0.75, ls=":")
                ax.text(tau[change], 0.98,
                        f"{trajectory.active_points[change-1]}→{trajectory.active_points[change]}",
                        transform=ax.get_xaxis_transform(), ha="center", va="top",
                        fontsize=7.0, color="#555B64")
            if len(changes) == 0:
                ax.text(0.5, 0.98, rf"$N={int(trajectory.active_points[0])}$ throughout",
                        transform=ax.transAxes, ha="center", va="top",
                        fontsize=7.0, color="#555B64")

    for col, title in enumerate(("Terminal profile", "Terminal absolute error", "Error evolution")):
        axes[0, col].set_title(title, pad=5)
    for label, ax in zip(("(a)", "(b)", "(c)", "(d)", "(e)", "(f)"), axes.flat):
        _panel(ax, label)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.035))
    _save(fig, output / "figure_main_pinn_v2_comparison")


def _metric_rows(equation: str, trajectory) -> list[dict[str, object]]:
    metrics, _ = comparison_metrics_common_band(trajectory)
    return [
        {
            "equation": equation,
            "method": method,
            "terminal_relative_l2": float(metrics[f"{key}_l2"][-1]),
            "terminal_relative_h1": float(metrics[f"{key}_h1"][-1]),
            "terminal_linf": float(metrics[f"{key}_linf"][-1]),
        }
        for method, key in (("PINN", "pinn"), ("V2 adaptive", "ours"))
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    source = args.source.resolve()
    output = args.output.resolve()

    acceptance_path = source / "acceptance.json"
    acceptance = json.loads(acceptance_path.read_text(encoding="utf-8"))
    if acceptance.get("status") != "PASS":
        raise RuntimeError(f"V2/PINN aligned evidence is not accepted: {acceptance_path}")

    mch = load_comparison(source / "aligned_comparison_evidence_mch.npz")
    bo = load_comparison(source / "aligned_comparison_evidence_bo.npz")
    output.mkdir(parents=True, exist_ok=True)

    plot_figure_7_direct_comparison(mch, output / "figure_06_mch_pinn_field")
    plot_figure_8_profiles(mch, output / "figure_07_mch_pinn_profiles")
    plot_figure_9_error_propagation(mch, output / "figure_08_mch_pinn_error")
    plot_figure_7_direct_comparison(bo, output / "figure_09_bo_pinn_field")
    plot_figure_8_profiles(bo, output / "figure_10_bo_pinn_profiles")
    plot_figure_9_error_propagation(bo, output / "figure_11_bo_pinn_error")
    _main_composite(mch, bo, output)

    rows = _metric_rows("mCH", mch) + _metric_rows("BO", bo)
    with (output / "comparison_table.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    payload = {
        "status": "PASS",
        "source": str(source),
        "main_text_figure": "figure_main_pinn_v2_comparison.pdf",
        "appendix_figures": [
            "figure_06_mch_pinn_field.pdf",
            "figure_07_mch_pinn_profiles.pdf",
            "figure_08_mch_pinn_error.pdf",
            "figure_09_bo_pinn_field.pdf",
            "figure_10_bo_pinn_profiles.pdf",
            "figure_11_bo_pinn_error.pdf",
        ],
        "metrics": rows,
    }
    (output / "acceptance.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
