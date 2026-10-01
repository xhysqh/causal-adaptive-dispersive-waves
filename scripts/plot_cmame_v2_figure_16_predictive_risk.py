"""Figure 16: finite-horizon risk linked to committed spatial and temporal authority."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

OUT = ROOT / "output" / "pdf" / "CMAME_V2_Figure_16"
HOLD = 0.65
CORRECTIVE = 1.0
BLUE = "#1769AA"
GREEN = "#009E73"
PURPLE = "#7B2CBF"
GREY = "#66717D"


def _style():
    return {
        "font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix", "font.size": 8.3,
        "axes.labelsize": 9.3, "axes.titlesize": 10.0,
        "xtick.labelsize": 7.8, "ytick.labelsize": 7.8,
        "legend.fontsize": 7.1, "axes.linewidth": 0.8,
        "xtick.direction": "in", "ytick.direction": "in",
        "axes.spines.top": False, "axes.spines.right": False,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    }


def _load_single(path: Path):
    with np.load(path, allow_pickle=False) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def _load_power_p26(path: Path):
    with np.load(path, allow_pickle=False) as data:
        index = int(np.flatnonzero(np.isclose(data["dispersion_orders"], 2.6))[0])
        return {
            "times": np.asarray(data["times"]),
            "upper_risk": np.asarray(data["upper_risk"][index]),
            "hold_upper_risk": np.asarray(data["hold_upper_risk"][index]),
            "realized_hold_risk": np.asarray(data["realized_hold_risk"][index]),
            "actions": np.asarray(data["actions"][index]).astype(str),
            "active_points": np.asarray(data["active_points"][index]),
        }


def _tau(times):
    return (times - times[0]) / (times[-1] - times[0])


def _events(actions):
    labels = np.asarray(actions).astype(str)
    return {
        "refine": np.flatnonzero(np.isin(labels, ("refine_space", "refine_representation"))),
        "shrink": np.flatnonzero(labels == "shrink_time"),
    }


def _action_lines(axes, tau, events):
    for index in events["refine"]:
        for axis in axes:
            axis.axvline(tau[index], color=GREEN, lw=0.55, alpha=0.16, zorder=0)
    for index in events["shrink"]:
        for axis in axes:
            axis.axvline(tau[index], color=PURPLE, lw=0.55, alpha=0.14, zorder=0)


def _plot_column(axes, record, title, panel_labels, *, legend=False):
    times = np.asarray(record["times"], float)
    tau = _tau(times)
    risk = np.asarray(record["hold_upper_risk"], float)
    realized = np.asarray(record["realized_hold_risk"], float)
    actions = np.asarray(record["actions"]).astype(str)
    points = np.asarray(record["active_points"], int)
    events = _events(actions)
    _action_lines(axes, tau, events)

    ax = axes[0]
    valid = np.isfinite(risk)
    log_risk = np.log10(np.maximum(risk[valid], 1e-14))
    ax.plot(tau[valid], log_risk, color=BLUE, lw=1.25,
            label=r"predicted $\log_{10}R^+_{\rm hold}(H)$")
    valid_realized = np.isfinite(realized)
    ax.plot(tau[valid_realized], np.log10(np.maximum(realized[valid_realized], 1e-14)),
            color="#32373D", lw=0.8, alpha=0.92,
            label=r"realized $\log_{10}R^{\rm real}_{\rm hold}(H)$")
    # The levels differ by only log10(1/0.65), so their values belong in the
    # shared legend rather than overlapping direct labels in the plot area.
    ax.axhline(np.log10(HOLD), color=GREY, lw=0.8, ls="--", label=r"hold threshold $R^+=0.65$")
    ax.axhline(np.log10(CORRECTIVE), color="#20242A", lw=0.8, ls=":", label=r"corrective threshold $R^+=1$")
    if len(events["refine"]):
        ax.scatter(tau[events["refine"]], np.log10(np.maximum(risk[events["refine"]], 1e-14)),
                   marker="D", s=28, color=GREEN, edgecolor="white", linewidth=0.4,
                   zorder=4, label="refinement")
    if len(events["shrink"]):
        ax.scatter(tau[events["shrink"]], np.log10(np.maximum(risk[events["shrink"]], 1e-14)),
                   marker="v", s=29, color=PURPLE, edgecolor="white", linewidth=0.4,
                   zorder=4, label="substep reduction")
    ax.set(title=title, ylabel=r"$\log_{10}R_{n,\mathrm{hold}}(H)$", xlim=(0, 1))
    ax.grid(axis="y", color="#D7DCE2", lw=0.45, alpha=0.7)
    ax.text(0.012, 0.992, panel_labels[0], transform=ax.transAxes, fontsize=9.4,
            fontweight="bold", va="top", ha="left", zorder=8,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.94, "pad": 0.35})

    ax = axes[1]
    ax.step(tau, points, where="post", color="#20242A", lw=1.5)
    if len(events["refine"]):
        ax.scatter(tau[events["refine"]], points[events["refine"]], marker="D", s=28,
                   color=GREEN, edgecolor="white", linewidth=0.4, zorder=4)
    ax.set(ylabel=r"$N(\tau)$", xlim=(0, 1))
    ax.set_yticks(np.unique(points))
    ax.grid(axis="y", color="#D7DCE2", lw=0.45, alpha=0.7)
    ax.text(0.012, 0.992, panel_labels[1], transform=ax.transAxes, fontsize=9.4,
            fontweight="bold", va="top", ha="left", zorder=8,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.94, "pad": 0.35})

    # The action changes the internal substep only for that macro interval;
    # it does not alter the physical macro-time horizon.
    ratio = np.ones_like(tau)
    ratio[actions == "shrink_time"] = 0.5
    ax = axes[2]
    ax.step(tau, ratio, where="pre", color="#20242A", lw=1.5)
    if len(events["shrink"]):
        ax.scatter(tau[events["shrink"]], ratio[events["shrink"]], marker="v", s=29,
                   color=PURPLE, edgecolor="white", linewidth=0.4, zorder=4)
    ax.set(xlabel=r"normalized time $\tau$", ylabel=r"$\Delta t_{\rm int}/\Delta t_0$",
           xlim=(0, 1), ylim=(0.42, 1.09), yticks=(0.5, 1.0),
           yticklabels=(r"$1/2$", r"$1$"))
    ax.grid(axis="y", color="#D7DCE2", lw=0.45, alpha=0.7)
    ax.text(0.012, 0.992, panel_labels[2], transform=ax.transAxes, fontsize=9.4,
            fontweight="bold", va="top", ha="left", zorder=8,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.94, "pad": 0.35})
    return {"refine": len(events["refine"]), "shrink": len(events["shrink"])}


def main():
    ilw = _load_single(ROOT / "results/cmame_v2_kdv_ilw_showcases/full/ilw_representative_trajectory.npz")
    power = _load_power_p26(ROOT / "results/cmame_v2_power_law_showcase/full/representative_trajectories.npz")
    OUT.mkdir(parents=True, exist_ok=True)
    with mpl.rc_context(_style()):
        fig, axes = plt.subplots(3, 2, figsize=(7.2, 5.65), sharex="col")
        fig.subplots_adjust(left=0.105, right=0.985, bottom=0.09, top=0.82,
                            hspace=0.16, wspace=0.29)
        ilw_actions = _plot_column(axes[:, 0], ilw, r"ILW: $\delta=1$", ("(a)", "(c)", "(e)"))
        power_actions = _plot_column(axes[:, 1], power, r"Power-law: $p=2.6$", ("(b)", "(d)", "(f)"))
        fig.suptitle("Finite-horizon predictive risk and committed discretization authority", y=0.99, fontsize=11)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.932),
                   ncol=3, frameon=False, fontsize=5.9, columnspacing=0.72,
                   handlelength=1.5, labelspacing=0.35)
        stem = OUT / "figure_16_finite_horizon_risk_authority"
        fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.025)
        fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight", pad_inches=0.025)
        plt.close(fig)
    report = {
        "status": "PASS", "figure": "figure_16_finite_horizon_risk_authority.pdf",
        "scope": "Frozen V2 predicted HOLD risk versus strict two-step, fixed-HOLD counterfactual risk, linked to committed actions.",
        "ILW_actions": ilw_actions, "power_p2p6_actions": power_actions,
        "interpretation": "refinement modifies global Fourier authority; shrink-time halves only the internal substep over its macro interval.",
    }
    (OUT / "acceptance.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
