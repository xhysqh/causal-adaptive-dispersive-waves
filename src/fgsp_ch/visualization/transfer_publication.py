"""CMAME transfer figures from frozen KdV, ILW, and power-law evidence.

The posthoc reference is used only for the reference and error panels.  Online
actions and active resolutions are read from the saved causal prefixes and are
never reconstructed from the reference solution.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
import csv
import json

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import colors, ticker
from matplotlib.lines import Line2D
import numpy as np


REFERENCE = "#20242A"
OURS = "#1769AA"
HOLD = "#8D99AE"
REFINE = "#009E73"
SHRINK = "#7B2CBF"
PROBE = "#D2B15B"
ACCENTS = {"kdv": "#158F83", "ilw": "#7357B3", "power_law": "#D99016"}
ACTION_COLORS = {
    "initial": "#D9DEE5", "hold": HOLD, "refine_space": REFINE,
    "refine_representation": REFINE, "shrink_time": SHRINK, "probe": PROBE,
}


@dataclass(frozen=True)
class TransferTrajectory:
    equation: str
    x: np.ndarray
    time: np.ndarray
    reference: np.ndarray
    adaptive: np.ndarray
    actions: np.ndarray
    active_points: np.ndarray
    selection_rule: str
    parameter: float | None = None
    upper_risk: np.ndarray | None = None
    probe_required: np.ndarray | None = None

    def __post_init__(self) -> None:
        x, time = np.asarray(self.x), np.asarray(self.time)
        shape = (len(time), len(x))
        if x.ndim != 1 or time.ndim != 1 or np.any(np.diff(x) <= 0) or np.any(np.diff(time) <= 0):
            raise ValueError("native coordinates must be strictly increasing vectors")
        for name in ("reference", "adaptive"):
            value = np.asarray(getattr(self, name), dtype=float)
            if value.shape != shape or not np.all(np.isfinite(value)):
                raise ValueError(f"{name} is not finite native trajectory evidence")
        if len(self.actions) != len(time) or len(self.active_points) != len(time):
            raise ValueError("online evidence must align with every saved causal prefix")
        if self.upper_risk is not None and len(self.upper_risk) != len(time):
            raise ValueError("predictive risk must align with every saved causal prefix")
        if self.probe_required is not None and len(self.probe_required) != len(time):
            raise ValueError("verification flags must align with every saved causal prefix")
        if not self.selection_rule:
            raise ValueError("representative selection rule is required")

    @property
    def tau(self) -> np.ndarray:
        return (self.time - self.time[0]) / (self.time[-1] - self.time[0])

    @property
    def error(self) -> np.ndarray:
        return np.abs(self.adaptive - self.reference)


def _scalar(data, key, default=""):
    return np.asarray(data[key]).item() if key in data.files else default


def load_kdv(path: Path) -> TransferTrajectory:
    with np.load(path, allow_pickle=False) as data:
        reference, adaptive = np.asarray(data["reference"]), np.asarray(data["adaptive"])
        if not np.allclose(data["absolute_error"], np.abs(adaptive - reference), rtol=1e-12, atol=1e-15):
            raise ValueError("KdV stored error is inconsistent with frozen fields")
        if str(_scalar(data, "reference_visibility")) != "posthoc_only":
            raise ValueError("KdV reference-visibility contract is not posthoc-only")
        return TransferTrajectory(
            equation="kdv", x=data["x"], time=data["times"], reference=reference,
            adaptive=adaptive, actions=data["actions"], active_points=data["active_points"],
            selection_rule=str(_scalar(data, "selection_rule")),
        )


def load_v2_trajectory(path: Path, *, parameter_key: str | None = None) -> TransferTrajectory:
    """Load one frozen adaptive trajectory with optional predictive-risk telemetry."""
    with np.load(path, allow_pickle=False) as data:
        reference, adaptive = np.asarray(data["reference"]), np.asarray(data["adaptive"])
        if not np.allclose(data["absolute_error"], np.abs(adaptive - reference), rtol=1e-12, atol=1e-15):
            raise ValueError("stored V2 error is inconsistent with frozen fields")
        parameter = (float(np.asarray(data[parameter_key]).item())
                     if parameter_key is not None and parameter_key in data.files else None)
        return TransferTrajectory(
            equation=str(_scalar(data, "equation")), x=data["x"], time=data["times"],
            reference=reference, adaptive=adaptive, actions=data["actions"],
            active_points=data["active_points"], selection_rule=str(_scalar(data, "selection_rule")),
            parameter=parameter,
            upper_risk=(np.asarray(data["upper_risk"], dtype=float)
                        if "upper_risk" in data.files else None),
            probe_required=(np.asarray(data["probe_required"], dtype=bool)
                            if "probe_required" in data.files else None),
        )


def load_f2(path: Path, ledger: Path, *, equation: str, initial_points: int) -> TransferTrajectory:
    """Align committed end-of-step F2 records with saved native snapshots.

    The initial budget is supplied by the frozen run configuration, not the
    visualization grid. mCH N counts the global field grid, not peakons.
    """
    names = {"mch": "modified_camassa_holm", "bo": "benjamin_ono"}
    with ledger.open(newline="", encoding="utf-8-sig") as handle:
        rows = [r for r in csv.DictReader(handle) if r["equation"] == names[equation]]
    rows.sort(key=lambda r: int(r["prefix"]))
    with np.load(path, allow_pickle=False) as data:
        time = data[f"{equation}_time"]
        actions = data[f"{equation}_actions"].astype(str)
        if len(rows) != len(time) - 1 or len(actions) != len(rows):
            raise ValueError("F2 ledger/snapshot length mismatch")
        if [int(r["prefix"]) for r in rows] != list(range(len(rows))):
            raise ValueError("F2 duplicate or missing prefix")
        if not np.allclose([float(r["time"]) for r in rows], time[1:], rtol=1e-10, atol=1e-13):
            raise ValueError("F2 ledger time mismatch")
        if not np.array_equal(actions, [r["action"] for r in rows]):
            raise ValueError("F2 ledger action mismatch")
        points = np.r_[initial_points, [int(r["points"]) for r in rows]]
        for i, action in enumerate(actions, 1):
            expected = points[i-1] * (2 if action == "refine_space" else 1)
            if action not in ("hold", "shrink_time", "refine_space") or points[i] != expected:
                raise ValueError("F2 committed resolution/action mismatch")
        return TransferTrajectory(
            equation, data["x"], time, data[f"{equation}_reference"],
            data[f"{equation}_field"], np.concatenate((["initial"], actions)), points,
            "Frozen F2 pre-registered showcase; ledger matched by equation, prefix, time and action",
        )


def load_collection(path: Path, *, coordinate: str, equation: str) -> list[TransferTrajectory]:
    result = []
    with np.load(path, allow_pickle=False) as data:
        reference, adaptive = np.asarray(data["reference"]), np.asarray(data["adaptive"])
        if not np.allclose(data["absolute_error"], np.abs(adaptive - reference), rtol=1e-12, atol=1e-15):
            raise ValueError(f"{equation} stored error is inconsistent with frozen fields")
        for index, value in enumerate(np.asarray(data[coordinate], dtype=float)):
            result.append(TransferTrajectory(
                equation=equation, x=data["x"], time=data["times"],
                reference=reference[index], adaptive=adaptive[index],
                actions=data["actions"][index], active_points=data["active_points"][index],
                selection_rule=str(_scalar(data, "selection_rule")), parameter=float(value),
                upper_risk=(np.asarray(data["upper_risk"][index], dtype=float)
                            if "upper_risk" in data.files else None),
                probe_required=(np.asarray(data["probe_required"][index], dtype=bool)
                                if "probe_required" in data.files else None),
            ))
    return result


def _style() -> dict:
    return {
        "text.usetex": False,
        "font.family": "serif",
        "font.serif": ["STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix", "font.size": 8.3,
        "axes.labelsize": 9.3, "axes.titlesize": 9.8,
        "xtick.labelsize": 7.7, "ytick.labelsize": 7.7, "legend.fontsize": 7.3,
        "xtick.direction": "in", "ytick.direction": "in", "axes.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    }


def _tag(axis, label: str) -> None:
    # Axes3D.text interprets the first three positional arguments as x, y, z;
    # use its dedicated 2-D annotation method so panel tags retain a fixed
    # publication position independently of the camera view.
    text = axis.text2D if hasattr(axis, "text2D") else axis.text
    text(-0.04, 1.02, label, transform=axis.transAxes, fontsize=10,
         fontweight="bold", ha="left", va="bottom")


def _native_indices(trajectory, fractions=(0.0, 0.5, 1.0)):
    return [int(np.argmin(np.abs(trajectory.tau - value))) for value in fractions]


def _field_norm(trajectory):
    lo = float(min(trajectory.reference.min(), trajectory.adaptive.min()))
    hi = float(max(trajectory.reference.max(), trajectory.adaptive.max()))
    if lo < 0 < hi:
        limit = max(abs(lo), abs(hi)); return colors.Normalize(-limit, limit)
    return colors.Normalize(lo, hi if hi > lo else lo + 1.0)


def _top_bar(fig, first, last, image, title, *, error=False):
    left, right = first.get_position(), last.get_position()
    axis = fig.add_axes([left.x0, left.y1 + 0.032, right.x1 - left.x0, 0.012])
    bar = fig.colorbar(image, cax=axis, orientation="horizontal")
    bar.ax.xaxis.set_ticks_position("top"); bar.ax.xaxis.set_label_position("top")
    bar.ax.tick_params(labelsize=8.3, length=2.5, pad=1.2)
    bar.ax.set_title(title, fontsize=9.3, pad=2.0)
    if error:
        formatter = ticker.ScalarFormatter(useMathText=True); formatter.set_powerlimits((0, 0))
        bar.formatter = formatter; bar.update_ticks()


def _heatmaps(fig, axes, trajectory, *, tags=("(a)", "(b)", "(c)")):
    extent = [trajectory.x[0], trajectory.x[-1], 0.0, 1.0]
    norm = _field_norm(trajectory)
    a = axes[0].imshow(trajectory.reference, origin="lower", aspect="auto", extent=extent,
                       cmap="turbo", norm=norm, interpolation="bilinear", rasterized=True)
    axes[1].imshow(trajectory.adaptive, origin="lower", aspect="auto", extent=extent,
                   cmap="turbo", norm=norm, interpolation="bilinear", rasterized=True)
    maximum = float(trajectory.error.max()) or 1.0
    c = axes[2].imshow(trajectory.error, origin="lower", aspect="auto", extent=extent,
                       cmap="magma", norm=colors.Normalize(0, maximum),
                       interpolation="bilinear", rasterized=True)
    for axis, title, tag in zip(axes, ("Reference solution", "Causal-adaptive solution", "Absolute error"), tags):
        axis.set(title=title, xlabel=r"$x$", ylabel=r"normalized time $\tau=t/T$")
        _tag(axis, tag)
    _top_bar(fig, axes[0], axes[1], a, r"wave field $u(x,t)$")
    _top_bar(fig, axes[2], axes[2], c, r"$|u_h-u_{\mathrm{ref}}|$", error=True)


def _profiles(axes, trajectory, *, fractions=(0.0, 0.5, 1.0), start_tag="(d)"):
    tags = [f"({chr(ord(start_tag[1]) + i)})" for i in range(3)]
    for axis, index, tag in zip(axes, _native_indices(trajectory, fractions), tags):
        axis.plot(trajectory.x, trajectory.reference[index], color=REFERENCE, lw=1.3,
                  label="Reference")
        axis.plot(trajectory.x, trajectory.adaptive[index], color=OURS, lw=1.3,
                  ls=(0, (3, 2)), label="Causal-adaptive")
        axis.set(xlabel=r"$x$", ylabel=r"$u(x,t)$",
                 title=rf"$\tau={trajectory.tau[index]:.2g}$")
        _tag(axis, tag)
    axes[0].legend(loc="upper right", frameon=False)


def _online_strip(axis, trajectory, *, title="Reference-free online adaptation"):
    accent = ACCENTS[trajectory.equation]
    axis.step(trajectory.tau, trajectory.active_points, where="post", color=accent, lw=1.55)
    axis.set(xlabel=r"normalized time $\tau$", ylabel="active points", title=title)
    upper = float(np.max(trajectory.active_points)); lower = float(np.min(trajectory.active_points))
    offset = max(1.0, 0.08 * max(upper - lower, 1.0))
    for action in np.unique(trajectory.actions):
        if action == "initial": continue
        mask = np.asarray(trajectory.actions) == action
        axis.scatter(trajectory.tau[mask], trajectory.active_points[mask] + offset,
                     s=12 if action == "hold" else 24, marker="o" if action == "hold" else "D",
                     color=ACTION_COLORS.get(str(action), HOLD), edgecolor="white", linewidth=0.35,
                     zorder=4, label=str(action).replace("_", " "))
    axis.legend(loc="upper right", frameon=False, ncol=2)


def _save(fig, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(target.with_suffix(".png"), dpi=600, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def online_diagnostic_components(adaptive: np.ndarray, dx: float) -> dict[str, np.ndarray]:
    """Return reference-free local indicators before display normalization."""
    field = np.asarray(adaptive, dtype=float)
    if field.ndim != 2 or not np.isfinite(field).all() or dx <= 0.0:
        raise ValueError("adaptive field and positive spatial spacing are required")
    count = field.shape[1]
    wave_numbers = 2.0 * np.pi * np.fft.rfftfreq(count, d=float(dx))
    coefficients = np.fft.rfft(field, axis=1)
    gradient = np.abs(np.fft.irfft(
        1j * wave_numbers[None, :] * coefficients, n=count, axis=1
    ))
    curvature = np.abs(np.fft.irfft(
        -(wave_numbers[None, :] ** 2) * coefficients, n=count, axis=1
    ))
    high_pass = coefficients.copy()
    high_pass[:, :max(2, count // 16)] = 0.0
    oscillation = np.abs(np.fft.irfft(high_pass, n=count, axis=1))
    return {"gradient": gradient, "curvature": curvature, "high_pass": oscillation}


def online_local_difficulty(adaptive: np.ndarray, dx: float) -> np.ndarray:
    """Compute a reference-free spatial diagnostic from causal adaptive fields.

    This is a *sensing* diagnostic, not a claimed local mesh-refinement mask.
    The transfer solvers retain a global Fourier representation; consequently
    the diagnostic may identify where a wave becomes difficult while the saved
    action still authorizes a global active resolution.  No reference field or
    posthoc error enters this calculation.
    """
    components = online_diagnostic_components(adaptive, dx)

    def normalize(values: np.ndarray) -> np.ndarray:
        scale = np.quantile(np.abs(values), 0.95, axis=1, keepdims=True)
        return np.abs(values) / np.maximum(scale, np.finfo(float).tiny)

    diagnostic = 0.50 * normalize(components["gradient"])
    diagnostic += 0.35 * normalize(components["curvature"])
    diagnostic += 0.15 * normalize(components["high_pass"])
    return diagnostic / np.maximum(diagnostic.max(axis=1, keepdims=True), np.finfo(float).tiny)


def _action_label(action: str) -> str:
    return str(action).replace("_", " ")


def _scientific_tex(value: float) -> str:
    if value == 0.0:
        return "0"
    exponent = int(np.floor(np.log10(abs(value))))
    mantissa = value / 10.0 ** exponent
    return rf"{mantissa:.2f}\times10^{{{exponent}}}"


def _assert_kdv_figure_contract(trajectory: TransferTrajectory) -> None:
    """Fail closed when the saved KdV ledger is internally inconsistent."""
    tau = trajectory.tau
    action = np.asarray(trajectory.actions).astype(str)
    committed = np.asarray(trajectory.active_points, dtype=int)
    assert len(tau) == 65
    assert committed[0] == 32
    before = np.r_[committed[0], committed[:-1]]
    changed = committed != before
    refine = action == "refine_space"
    assert np.array_equal(changed, refine)
    assert np.all(committed[refine] == 2 * before[refine])
    assert np.all(np.diff(committed) >= 0)


def _committed_action_label(action: str) -> str:
    return {
        "hold": "hold",
        "refine_space": "global spectral refinement",
        "refine_representation": "global representation refinement",
        "shrink_time": "substep reduction",
    }.get(str(action), str(action).replace("_", " "))


def _plot_transfer_manifold(trajectory: TransferTrajectory, target: Path) -> None:
    """Render localized sensing -> diagnosis -> committed global authority.

    The full transfer label belongs in the caption; the artwork retains only
    the shared three-panel semantic structure.
    """
    if trajectory.equation == "kdv":
        _assert_kdv_figure_contract(trajectory)

    tau = trajectory.tau
    action = np.asarray(trajectory.actions).astype(str)
    committed = np.asarray(trajectory.active_points, dtype=int)
    dx = float(np.mean(np.diff(trajectory.x)))
    eta = online_local_difficulty(trajectory.adaptive, dx)
    site_index = np.argmax(eta, axis=1)
    x_star = np.asarray(trajectory.x[site_index], dtype=float)
    u_star = np.asarray([
        np.interp(x_value, trajectory.x, trajectory.adaptive[row])
        for row, x_value in enumerate(x_star)
    ])
    decision_indices = np.flatnonzero(~np.isin(action, ("initial", "hold")))
    hold_indices = np.flatnonzero(action == "hold")
    show_hold = hold_indices[::6]
    if len(hold_indices) and hold_indices[-1] not in show_hold:
        show_hold = np.r_[show_hold, hold_indices[-1]]

    lo = float(trajectory.adaptive.min())
    hi = float(trajectory.adaptive.max())
    field_norm = colors.Normalize(lo, hi if hi > lo else lo + 1.0)
    X, TAU = np.meshgrid(trajectory.x, tau)

    with mpl.rc_context(_style()):
        fig = plt.figure(figsize=(7.2, 5.2))
        gs = fig.add_gridspec(
            2, 2,
            height_ratios=[1.35, 1.0],
            width_ratios=[1.35, 1.0],
            hspace=0.20,
            wspace=0.20,
        )
        ax3d = fig.add_subplot(gs[0, :], projection="3d", computed_zorder=False)
        ax2d = fig.add_subplot(gs[1, 0])
        axN = fig.add_subplot(gs[1, 1])
        fig.subplots_adjust(left=0.075, right=0.965, bottom=0.16, top=0.97)
        ax3d.set_position([0.03, 0.575, 0.90, 0.385])
        ax3d.set_anchor('C')

        surface = ax3d.plot_surface(
            X, TAU, trajectory.adaptive,
            cmap="turbo", norm=field_norm, alpha=0.88, linewidth=0,
            antialiased=True, rcount=min(65, len(tau)),
            ccount=min(256, len(trajectory.x)), shade=False, zorder=1,
        )
        ax3d.plot(x_star, tau, u_star, color="#59636E", lw=0.55, alpha=0.62,
                  label=r"localized online diagnostic sites $x_n^\star$", zorder=5)
        ax3d.scatter(
            x_star[show_hold], tau[show_hold], u_star[show_hold],
            s=10, color=HOLD, edgecolor="white", linewidth=0.25,
            depthshade=False, label="hold", zorder=6,
        )
        for index in decision_indices:
            current_action = action[index]
            action_color = ACTION_COLORS.get(current_action, REFINE)
            ax3d.scatter(
                x_star[index], tau[index], u_star[index],
                s=44 if current_action != "shrink_time" else 40, marker="D",
                color=action_color, edgecolor="white", linewidth=0.45,
                depthshade=False, label=_committed_action_label(current_action), zorder=8,
            )
        # Keep the coordinate names close to the projected 3D axes so the
        # large white margin around panel (a) remains visually quiet.
        ax3d.set_xlabel(r"$x$", labelpad=-5.0)
        ax3d.set_ylabel(r"$\tau=t/T$", labelpad=-3.0)
        ax3d.set_zlabel(r"$u_h(x,\tau)$", labelpad=-6.0)
        ax3d.set_title("3D causal response manifold", pad=1.5)
        ax3d.view_init(elev=21, azim=-61)
        ax3d.set_box_aspect((2.5, 1.25, 0.85), zoom=1.78)
        ax3d.set_aspect('auto')
        ax3d.set_proj_type("ortho")
        ax3d.tick_params(labelsize=7.4, pad=1.0)
        _tag(ax3d, "(a)")
        handles = [
            Line2D([], [], color="#59636E", lw=0.7, marker="o", markersize=2.1,
                   label=r"localized online diagnostic sites $x_n^\star$"),
            Line2D([], [], color=HOLD, marker="o", linestyle="none", markersize=3,
                   label="hold"),
        ]
        for current_action in dict.fromkeys(action[decision_indices]):
            handles.append(Line2D(
                [], [], color=ACTION_COLORS.get(current_action, REFINE),
                marker="D", linestyle="none", markersize=3.5,
                label=_committed_action_label(current_action),
            ))
        handles.append(Line2D(
            [], [], color=PROBE, lw=0.7, alpha=0.8,
            label="elevated-diagnostic contour",
        ))
        fig.legend(
            handles=handles, loc="upper left",
            bbox_to_anchor=(0.77, 0.965), frameon=False,
            borderpad=0.25, labelspacing=0.24, handletextpad=0.35,
            handlelength=1.00, fontsize=6.3,
        )

        image = ax2d.pcolormesh(
            X, TAU, trajectory.adaptive, shading="auto", cmap="turbo",
            norm=field_norm, rasterized=True,
        )
        ax2d.contour(
            X, TAU, eta, levels=[0.60], colors=[PROBE],
            alpha=0.42, linewidths=0.42,
        )
        ax2d.plot(x_star, tau, color="#59636E", lw=0.45, alpha=0.30, zorder=4)
        ax2d.scatter(
            x_star[::6], tau[::6], s=6, color=HOLD, alpha=0.80,
            label=r"localized online diagnostic sites $x_n^\star$",
            edgecolor="white", linewidth=0.20, zorder=5,
        )
        for index in decision_indices:
            current_action = action[index]
            action_color = ACTION_COLORS.get(current_action, REFINE)
            ax2d.scatter(
                x_star[index], tau[index], s=40, marker="D",
                color=action_color, edgecolor="white", linewidth=0.40,
                zorder=7, label=_committed_action_label(current_action),
            )
            ax2d.axhline(tau[index], color=action_color, lw=0.35,
                         alpha=0.22, zorder=3)
        ax2d.set_xlabel(r"$x$")
        ax2d.set_ylabel(r"normalized time $\tau$")
        # Preserve endpoint actions at their true normalized time while
        # leaving enough display margin for full marker glyphs at tau=1.
        ax2d.set_ylim(-0.02, 1.06)
        ax2d.set_yticks(np.linspace(0.0, 1.0, 6))
        ax2d.set_title("Projection: online diagnostic sites and causal decisions")
        _tag(ax2d, "(b)")
        # Panel (a) owns the compact semantic legend.  Keeping Panel (b)
        # legend-free preserves every part of the field projection; the
        # shared figure legend defines the yellow visualization-only contour.

        axN.step(tau, committed, where="post", color=REFERENCE, lw=1.6,
                 label=r"committed $N(\tau)$")
        axN.scatter(
            tau[show_hold], committed[show_hold], s=11, marker="o",
            color=HOLD, edgecolor="white", linewidth=0.25,
            label="hold", zorder=4,
        )
        for current_action in np.unique(action[decision_indices]):
            selected = decision_indices[action[decision_indices] == current_action]
            axN.scatter(
                tau[selected], committed[selected], s=42, marker="D",
                color=ACTION_COLORS.get(current_action, REFINE),
                edgecolor="white", linewidth=0.40,
                label=_committed_action_label(current_action), zorder=5,
            )
        resolution_levels = np.unique(committed)
        span = float(np.ptp(resolution_levels))
        pad = 0.19 * (span if span > 0 else float(resolution_levels.max()))
        axN.set_yticks(resolution_levels)
        axN.set_ylim(max(0.0, float(resolution_levels.min()) - pad),
                     float(resolution_levels.max()) + pad)
        axN.set_xlim(-0.02, 1.06)
        axN.set_xticks(np.linspace(0.0, 1.0, 6))
        axN.set_xlabel(r"normalized time $\tau$")
        axN.set_ylabel(r"committed global resolution $N(\tau)$")
        axN.set_title("Committed global field authority" if trajectory.equation == "mch"
                      else "Committed global spectral authority")
        _tag(axN, "(c)")

        transitions = []
        for index in decision_indices:
            if action[index] == "shrink_time":
                continue
            previous = int(committed[max(index - 1, 0)])
            current = int(committed[index])
            transitions.append((previous, current, float(tau[index])))
        if transitions:
            levels = [transitions[0][0], *[item[1] for item in transitions]]
            axN.text(
                0.30, 0.20,
                r"$N: " + r"\rightarrow".join(map(str, levels)) + r"$"
                + "\nthen retain $N$",
                transform=axN.transAxes, ha="center", va="center",
                fontsize=7.5, color=REFERENCE,
            )
        if not transitions:
            axN.text(0.5, 0.72, rf"global $N={committed[0]}$ unchanged",
                     transform=axN.transAxes, ha="center", fontsize=7.7,
                     bbox=dict(fc="white", ec="#C7CDD3", lw=0.5))
        spatial_events = np.flatnonzero(np.isin(action, ("refine_space", "refine_representation")))
        if len(spatial_events):
            last_spatial_tau = float(tau[spatial_events[-1]])
            zoom_end = min(1.0, max(0.05, 1.25 * last_spatial_tau))
            # Keep the enlarged early-transition ledger in the lower-right
            # whitespace so it never covers the main N(tau) staircase.
            inset = axN.inset_axes([.55, .14, .39, .25])
            inset.step(tau, committed, where='post', color=REFERENCE, lw=.8)
            inset.scatter(tau[spatial_events], committed[spatial_events],
                          color=REFINE, s=12, marker='D', edgecolor='white',
                          linewidth=.25, zorder=4)
            inset.set(xlim=(0, zoom_end), ylim=axN.get_ylim(),
                      xticks=[0, zoom_end], yticks=resolution_levels)
            inset.set_xticklabels(["0", f"{zoom_end:.2f}"])
            inset.tick_params(labelsize=5.7, pad=1, length=2)
            inset.set_xlabel(rf'$\tau\in[0,{zoom_end:.2f}]$', fontsize=6.7, labelpad=1.0)

        field_position = ax2d.get_position()
        colorbar_axis = fig.add_axes([
            field_position.x0 + 0.12 * field_position.width,
            field_position.y0 - 0.095,
            0.76 * field_position.width,
            0.012,
        ])
        colorbar = fig.colorbar(surface, cax=colorbar_axis, orientation="horizontal")
        colorbar.ax.xaxis.set_ticks_position("bottom")
        colorbar.ax.xaxis.set_label_position("bottom")
        colorbar.ax.tick_params(labelsize=6.9, length=2.0, pad=1.0)
        colorbar.ax.set_xlabel(r"$u_h(x,\tau)$", fontsize=7.7, labelpad=1.5)
        _save(fig, target)
    metadata = {
        'equation': trajectory.equation,
        'difficult_prefix_marker': 'removed: no causal selection rule was logged',
        'diagnostic_sites_rule': 'x_n^star = argmax_x eta of each adaptive snapshot; no smoothing; not neural attention',
        'display_contour': 0.60, 'contour_is_decision_threshold': False,
        'hold_marker_stride': 6, 'full_ledger_used': True,
        'actions': {a:int(np.count_nonzero(action==a)) for a in np.unique(action)},
        'committed_action_events': [
            {'action': str(action[index]), 'tau': float(tau[index]),
             'N_before': int(committed[max(index - 1, 0)]), 'N_after': int(committed[index])}
            for index in decision_indices
        ],
        'reference_used_for_selection': False,
    }
    target.with_suffix('.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    target.with_suffix('.txt').write_text(
        'Causal mechanism-transfer evidence. (a) Adaptive field with localized online diagnostic sites x*(tau), '
        'and the committed global spectral refinement(s). '
        '(b) x--tau projection with the online diagnostic contour. (c) The complete committed global spectral-resolution ledger. '
        'The 0.60 eta_max contour visualizes diagnostic concentration only; it is not an adaptive decision threshold. '
        'Diagnostic sites are recomputed from stored adaptive states, not logged neural attention or physical characteristics. '
        'Hold markers are subsampled; the complete saved ledger defines N(tau). Global spectral resolution is not local mesh refinement. '
        f'The reference is posthoc only: max |u_h-u_ref|={_scientific_tex(float(trajectory.error.max()))}. '
        'See the JSON sidecar for the exact selection rule and action counts.',encoding='utf-8')


def plot_kdv_transfer(trajectory: TransferTrajectory, target: Path) -> None:
    _plot_transfer_manifold(trajectory, target)


def plot_causal_transfer_figure(trajectory: TransferTrajectory, target: Path) -> None:
    """Shared artwork for source-equation response and held-out transfer."""
    _plot_transfer_manifold(trajectory, target)


def plot_trait_atlas(trajectories: list[TransferTrajectory], representative: float,
                     target: Path) -> None:
    """Plot the precommitted bridge trait; endpoint coverage remains tabulated."""
    matches = [item for item in trajectories if np.isclose(item.parameter, float(representative))]
    if len(matches) != 1:
        raise ValueError(f"precommitted representative trait {representative:g} must occur exactly once")
    _plot_transfer_manifold(matches[0], target)
