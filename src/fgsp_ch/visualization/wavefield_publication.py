"""Publication wave-field figures built only from frozen numerical evidence.

Posthoc reference fields, causal-adaptive numerical fields, and information
available online are kept separate.  Reference-derived quantities never
decorate an online decision panel.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import colors, ticker
from matplotlib.lines import Line2D
import numpy as np

from fgsp_ch.visualization.cmame_style import COLORS, panel_label


EQUATION_LABELS = {
    "mch": "mCH · localized peakon regime",
    "kdv": "KdV · zero-shot soliton interaction",
    "bo": "Benjamin–Ono · nonlocal dispersive regime",
    "ilw": "ILW · finite-depth transition",
    "power_law": "Power-law dispersion · unseen trait",
}

ACTION_COLORS = {
    "hold": COLORS["hold"],
    "shrink_time": COLORS["shrink_time"],
    "refine_space": COLORS["refine_space"],
    "refine_representation": COLORS["refine_space"],
    "coarsen_space": "#1B9E77",
    "probe": COLORS["abstain"],
    "revoke": "#C23B22",
    "fallback": "#C23B22",
}


def _optional_array(data, key, *, index=None):
    if key not in data.files:
        return None
    value = np.asarray(data[key])
    return value if index is None else value[index]


@dataclass(frozen=True)
class WaveTrajectory:
    equation: str
    x: np.ndarray
    time: np.ndarray
    reference: np.ndarray
    adaptive: np.ndarray
    selection_rule: str
    parameter_label: str = ""
    actions: np.ndarray | None = None
    active_points: np.ndarray | None = None
    feature_paths: np.ndarray | None = None
    spectrum: np.ndarray | None = None
    symbol_geometry: np.ndarray | None = None
    activation_mask: np.ndarray | None = None

    def __post_init__(self):
        x = np.asarray(self.x, dtype=float)
        time = np.asarray(self.time, dtype=float)
        reference = np.asarray(self.reference, dtype=float)
        adaptive = np.asarray(self.adaptive, dtype=float)
        if x.ndim != 1 or time.ndim != 1 or len(x) < 4 or len(time) < 3:
            raise ValueError("wave-field axes must be one-dimensional and nontrivial")
        if reference.shape != (len(time), len(x)) or adaptive.shape != reference.shape:
            raise ValueError("wave-field arrays must align with native time and space axes")
        if np.any(np.diff(x) <= 0) or np.any(np.diff(time) <= 0):
            raise ValueError("wave-field coordinates must be strictly increasing")
        if not np.all(np.isfinite(reference)) or not np.all(np.isfinite(adaptive)):
            raise ValueError("wave-field evidence must be finite")
        if not self.selection_rule:
            raise ValueError("a precommitted representative-selection rule is required")
        if self.actions is not None and len(self.actions) not in (len(time), len(time) - 1):
            raise ValueError("online actions must align with native saved prefixes")
        for name in ("active_points", "feature_paths", "spectrum", "symbol_geometry"):
            value = getattr(self, name)
            if value is not None and np.asarray(value).shape[0] != len(time):
                raise ValueError(f"{name} must align with native saved prefixes")
        if self.activation_mask is not None:
            mask = np.asarray(self.activation_mask)
            if mask.shape != reference.shape:
                raise ValueError("activation mask must align with the wave field")
            if not np.all(np.isfinite(mask)):
                raise ValueError("activation mask must be finite")

    @property
    def tau(self):
        return (self.time - self.time[0]) / (self.time[-1] - self.time[0])

    @property
    def error(self):
        result = np.abs(self.adaptive - self.reference)
        if not np.all(np.isfinite(result)):
            raise ValueError("absolute error must be finite")
        return result

    @property
    def aligned_actions(self):
        if self.actions is None:
            return np.full(len(self.time), "unrecorded", dtype="U16")
        actions = np.asarray(self.actions).astype("U32")
        if len(actions) == len(self.time) - 1:
            actions = np.concatenate((np.asarray(["initial"], dtype="U32"), actions))
        return actions


def native_snapshot_indices(time, fractions=(0.0, 0.5, 1.0)):
    values = np.asarray(time, dtype=float)
    tau = (values - values[0]) / (values[-1] - values[0])
    return tuple(int(np.argmin(np.abs(tau - fraction))) for fraction in fractions)


def _scalar(data, key, default=""):
    return str(np.asarray(data[key]).item()) if key in data.files else str(default)


def load_f2_trajectory(path: Path, equation: str) -> WaveTrajectory:
    stem = {"mch": "mch", "bo": "bo"}[equation]
    with np.load(path, allow_pickle=False) as data:
        return WaveTrajectory(
            equation=equation,
            x=np.asarray(data["x"]),
            time=np.asarray(data[f"{stem}_time"]),
            reference=np.asarray(data[f"{stem}_reference"]),
            adaptive=np.asarray(data[f"{stem}_field"]),
            selection_rule="frozen F2 precommitted closed-loop showcase",
            actions=_optional_array(data, f"{stem}_actions"),
            feature_paths=_optional_array(data, "mch_positions") if equation == "mch" else None,
            spectrum=_optional_array(data, "bo_spectrum") if equation == "bo" else None,
        )


def load_single_trajectory(path: Path) -> WaveTrajectory:
    with np.load(path, allow_pickle=False) as data:
        stored = np.asarray(data["absolute_error"])
        reference, adaptive = np.asarray(data["reference"]), np.asarray(data["adaptive"])
        if not np.allclose(stored, np.abs(adaptive - reference), rtol=1e-12, atol=1e-14):
            raise ValueError("stored representative error is inconsistent with the fields")
        return WaveTrajectory(
            equation=_scalar(data, "equation", "kdv"),
            x=np.asarray(data["x"]), time=np.asarray(data["times"]),
            reference=reference, adaptive=adaptive,
            selection_rule=_scalar(data, "selection_rule"),
            parameter_label=_scalar(data, "family").replace("_", "–"),
            actions=_optional_array(data, "actions"),
            active_points=_optional_array(data, "active_points"),
            activation_mask=_optional_array(data, "activation_mask"),
        )


def load_collection_trajectory(path: Path, *, coordinate: str, value: float,
                               equation: str) -> WaveTrajectory:
    with np.load(path, allow_pickle=False) as data:
        coordinates = np.asarray(data[coordinate], dtype=float)
        matches = np.flatnonzero(np.isclose(coordinates, float(value), rtol=0, atol=1e-12))
        if len(matches) != 1:
            raise ValueError(f"expected one precommitted {coordinate}={value}, found {len(matches)}")
        index = int(matches[0])
        reference = np.asarray(data["reference"][index])
        adaptive = np.asarray(data["adaptive"][index])
        stored = np.asarray(data["absolute_error"][index])
        if not np.allclose(stored, np.abs(adaptive - reference), rtol=1e-12, atol=1e-14):
            raise ValueError("stored collection error is inconsistent with the fields")
        label = rf"$\delta={value:g}$" if coordinate == "depths" else rf"$p={value:g}$"
        return WaveTrajectory(
            equation=equation, x=np.asarray(data["x"]), time=np.asarray(data["times"]),
            reference=reference, adaptive=adaptive,
            selection_rule=_scalar(data, "selection_rule"), parameter_label=label,
            actions=_optional_array(data, "actions", index=index),
            active_points=_optional_array(data, "active_points", index=index),
            symbol_geometry=_optional_array(data, "symbol_geometry", index=index),
            activation_mask=_optional_array(data, "activation_mask", index=index),
        )


def _edges(values):
    values = np.asarray(values, dtype=float)
    mid = 0.5 * (values[:-1] + values[1:])
    return np.r_[values[0] - 0.5 * (values[1] - values[0]), mid,
                 values[-1] + 0.5 * (values[-1] - values[-2])]


def _field_norm(reference, adaptive):
    lower = float(min(np.min(reference), np.min(adaptive)))
    upper = float(max(np.max(reference), np.max(adaptive)))
    if np.isclose(lower, upper):
        upper = lower + 1.0
    if lower < 0 < upper and min(abs(lower), abs(upper)) > 0.03 * (upper - lower):
        limit = max(abs(lower), abs(upper))
        return colors.Normalize(-limit, limit), "RdBu_r"
    return colors.Normalize(lower, upper), "viridis"


def _error_norm(error):
    maximum = float(np.max(error))
    return colors.Normalize(0.0, maximum if maximum > 0 else 1.0), error


def _error_cmap():
    # A pale zero distinguishes true zero error from missing or masked data.
    return colors.LinearSegmentedColormap.from_list(
        "cmame_absolute_error",
        ("#FFFDFC", "#FEE8C8", "#FDBB84", "#E34A33", "#7F0000"),
    )


def _tau_label(value):
    nearest_tenth = round(float(value), 1)
    return f"{nearest_tenth:.1f}" if abs(value - nearest_tenth) < 5e-3 else f"{value:.2f}"


def _format_colorbar(colorbar):
    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((-2, 2))
    colorbar.formatter = formatter
    colorbar.locator = ticker.MaxNLocator(5)
    colorbar.update_ticks()
    colorbar.ax.tick_params(length=3, pad=2)


def _action_color(action):
    return ACTION_COLORS.get(str(action), COLORS["hold"])


def _nontrivial_action_indices(trajectory):
    return [
        index for index, action in enumerate(trajectory.aligned_actions)
        if action not in ("initial", "hold", "unrecorded", "none")
    ]


def _plot_spacetime_evidence(axis, trajectory, *, cmap, norm):
    x_stride = max(1, len(trajectory.x) // 144)
    t_stride = max(1, len(trajectory.time) // 64)
    x = trajectory.x[::x_stride]
    tau = trajectory.tau[::t_stride]
    field = trajectory.adaptive[::t_stride, ::x_stride]
    xx, tt = np.meshgrid(x, tau)
    span = float(np.ptp(field)) or 1.0
    floor = float(np.min(field) - 0.18 * span)
    ceiling = float(np.max(field) + 0.05 * span)
    axis.plot_surface(
        xx, tt, field, cmap=cmap, norm=norm, linewidth=0, antialiased=True,
        alpha=0.90, rcount=len(tau), ccount=min(len(x), 144), rasterized=True,
    )
    levels = np.linspace(float(norm.vmin), float(norm.vmax), 12)
    axis.contourf(xx, tt, field, zdir="z", offset=floor, levels=levels,
                  cmap=cmap, norm=norm, alpha=0.42)

    actions = trajectory.aligned_actions
    for index in _nontrivial_action_indices(trajectory):
        xp = np.linspace(trajectory.x[0], trajectory.x[-1], 24)
        zp = np.linspace(floor, ceiling, 2)
        plane_x, plane_z = np.meshgrid(xp, zp)
        plane_t = np.full_like(plane_x, trajectory.tau[index])
        axis.plot_surface(plane_x, plane_t, plane_z, color=_action_color(actions[index]),
                          alpha=0.14, shade=False, linewidth=0)
        axis.plot(
            [trajectory.x[0], trajectory.x[-1], trajectory.x[-1], trajectory.x[0], trajectory.x[0]],
            [trajectory.tau[index]] * 5,
            [floor, floor, ceiling, ceiling, floor],
            color=_action_color(actions[index]), lw=0.85, alpha=0.85,
        )

    # Every gold point is a real saved causal prefix, not a decorative sample.
    axis.scatter(np.full(len(trajectory.tau), trajectory.x[0]), trajectory.tau,
                 np.full(len(trajectory.tau), floor), s=8, color=COLORS["abstain"],
                 depthshade=False, alpha=0.85)
    if trajectory.feature_paths is not None:
        paths = np.asarray(trajectory.feature_paths, dtype=float)
        if paths.ndim == 1:
            paths = paths[:, None]
        for path in paths.T:
            z = np.asarray([
                trajectory.adaptive[i, int(np.argmin(np.abs(trajectory.x - position)))]
                for i, position in enumerate(path)
            ])
            axis.plot(path, trajectory.tau, z + 0.025 * span,
                      color="#F7F7F7", lw=1.5, zorder=10)
            axis.plot(path, trajectory.tau, np.full(len(path), floor),
                      color="#0B6E4F", lw=1.05, alpha=0.95)
    if trajectory.activation_mask is not None:
        mask = np.ma.masked_where(np.asarray(trajectory.activation_mask) <= 0,
                                  np.asarray(trajectory.activation_mask))
        mask_x, mask_t = np.meshgrid(trajectory.x, trajectory.tau)
        axis.contourf(mask_x, mask_t, mask, zdir="z", offset=floor,
                      levels=(0.5, 1.5), colors=(COLORS["refine_space"],), alpha=0.35)

    axis.set(xlabel=r"$x$", ylabel=r"$\tau=t/T$", zlabel=r"$u_h(x,t)$",
             zlim=(floor, ceiling), title="Causal space–time evidence")
    axis.view_init(elev=24, azim=-58)
    axis.set_box_aspect((2.55, 1.15, 0.78), zoom=1.28)
    axis.grid(False)
    for pane in (axis.xaxis.pane, axis.yaxis.pane, axis.zaxis.pane):
        pane.set_facecolor((1, 1, 1, 0))
        pane.set_edgecolor("#D7DCE2")


def _plot_action_resolution(axis, trajectory, *, accent):
    if trajectory.active_points is not None:
        axis.step(trajectory.tau, trajectory.active_points, where="post",
                  color=accent, lw=1.8, label="active points")
        axis.set_ylabel("active points")
    else:
        axis.plot(trajectory.tau, np.ones_like(trajectory.tau), color="#C7CED8", lw=1.0)
        axis.set_yticks([])
    actions = trajectory.aligned_actions
    for index in _nontrivial_action_indices(trajectory):
        y = trajectory.active_points[index] if trajectory.active_points is not None else 1.0
        axis.scatter(trajectory.tau[index], y, s=38, marker="D",
                     color=_action_color(actions[index]), edgecolor="white", linewidth=0.6,
                     zorder=5, label=str(actions[index]).replace("_", " "))
    axis.set(xlabel=r"normalized time $\tau$", title="Committed online actions")
    handles, labels = axis.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    if unique:
        axis.legend(unique.values(), unique.keys(), fontsize=7, loc="best")


def _plot_equation_diagnostic(axis, trajectory, *, accent):
    if trajectory.equation == "mch" and trajectory.feature_paths is not None:
        paths = np.asarray(trajectory.feature_paths, dtype=float)
        if paths.ndim == 1:
            paths = paths[:, None]
        for index, path in enumerate(paths.T):
            axis.plot(trajectory.tau, path, marker="o", ms=2.7, lw=1.5,
                      color=(accent if index == 0 else "#0B6E4F"),
                      label=rf"peak path {index + 1}")
        axis.set(xlabel=r"normalized time $\tau$", ylabel=r"peak position $q_j$",
                 title="Tracked peakon geometry")
        axis.legend(fontsize=7, loc="best")
        return
    if trajectory.equation == "bo" and trajectory.spectrum is not None:
        spectrum = np.maximum(np.asarray(trajectory.spectrum, dtype=float),
                              np.finfo(float).tiny)
        modes = np.arange(spectrum.shape[1])
        indices = sorted(set(native_snapshot_indices(trajectory.time) +
                             (len(trajectory.time) // 4, 3 * len(trajectory.time) // 4)))
        time_cmap = plt.get_cmap("magma")
        for index in indices:
            normalized = spectrum[index] / max(float(np.max(spectrum[index])),
                                                np.finfo(float).tiny)
            axis.plot(modes, normalized, lw=1.15,
                      color=time_cmap(0.15 + 0.75 * trajectory.tau[index]),
                      label=rf"$\tau={_tau_label(trajectory.tau[index])}$")
        axis.set_yscale("log")
        axis.set(xlabel="Fourier mode", ylabel=r"$|\hat u_k|/\max_k|\hat u_k|$",
                 title="Nonlocal spectral curtain", ylim=(1e-12, 2))
        axis.legend(fontsize=6.5, ncol=2, loc="lower left")
        return
    if trajectory.equation == "ilw" and trajectory.symbol_geometry is not None:
        geometry = np.asarray(trajectory.symbol_geometry, dtype=float)
        active_order = 4.0 * geometry[:, 1]
        valid = np.isfinite(active_order)
        axis.plot(trajectory.tau[valid], active_order[valid], color=accent,
                  marker="o", ms=2.5, lw=1.5, label=r"active order $p_{eff}$")
        axis.set(xlabel=r"normalized time $\tau$", ylabel="effective dispersion order",
                 title="Finite-depth symbol response", ylim=(1.8, 3.2))
        if trajectory.active_points is not None:
            twin = axis.twinx()
            twin.step(trajectory.tau, trajectory.active_points, where="post",
                      color=COLORS["hold"], lw=1.2, alpha=0.9)
            twin.set_ylabel("active points", color=COLORS["hold"])
            twin.tick_params(axis="y", colors=COLORS["hold"])
        axis.legend(fontsize=7, loc="lower right")
        return
    _plot_action_resolution(axis, trajectory, accent=accent)


def _cmame_tag(axis, tag):
    draw = getattr(axis, "text2D", axis.text)
    draw(-0.095, 1.035, tag, transform=axis.transAxes, fontsize=10,
         fontweight="bold", ha="left", va="bottom")


def _add_top_colorbar(fig, first_axis, last_axis, mappable, label, *, ticks=None):
    """Add a 3%-height colorbar without adding a fourth GridSpec row."""
    fig.canvas.draw()
    left = first_axis.get_position()
    right = last_axis.get_position()
    height = 0.035 * left.height
    # Leave the ordinary axes title below the bar.  This explicit offset is
    # required for the compact 7.2-inch CMAME layout; anchoring the bar directly
    # to y1 makes TeX titles collide with its lower spine.
    bar = fig.add_axes([left.x0, left.y1 + 0.20 / fig.get_figheight(), right.x1 - left.x0, height])
    colorbar = fig.colorbar(mappable, cax=bar, orientation="horizontal")
    _format_colorbar(colorbar)
    if ticks is not None:
        # _format_colorbar installs a generic MaxNLocator, so the registered
        # Figure-2 ticks must be applied afterwards.
        colorbar.set_ticks(ticks)
        colorbar.update_ticks()
    bar.xaxis.set_ticks_position("top")
    bar.xaxis.set_label_position("top")
    bar.set_xlabel(label, fontsize=9.3, labelpad=3)
    bar.tick_params(labelsize=8.2)
    return colorbar


def plot_wavefield_2x3(
    trajectory: WaveTrajectory,
    target: Path,
    *,
    profile_color="#1769AA",
    state_limits=None,
    error_limit=None,
    state_ticks=None,
    error_ticks=None,
):
    """Render the common CMAME 2x3 field/profile grammar for Figures 2--5."""
    target.parent.mkdir(parents=True, exist_ok=True)

    style = {
        "text.usetex": False,
        "font.family": "serif",
        "font.serif": ["STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "pdf.fonttype": 42,
        "font.size": 9.0,
        "axes.labelsize": 9.3,
        "axes.titlesize": 10.0,
        "xtick.labelsize": 8.2,
        "ytick.labelsize": 8.2,
        "legend.fontsize": 7.4,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "axes.linewidth": 0.8,
    }
    with mpl.rc_context(style):
        if state_limits is None:
            state_lower = float(min(np.min(trajectory.reference), np.min(trajectory.adaptive)))
            state_upper = float(max(np.max(trajectory.reference), np.max(trajectory.adaptive)))
        else:
            state_lower, state_upper = map(float, state_limits)
        if not state_upper > state_lower:
            raise ValueError("state color limits must be strictly ordered")
        maximum_error = float(np.max(trajectory.error)) if error_limit is None else float(error_limit)
        if maximum_error <= 0.0:
            maximum_error = float(np.finfo(float).eps)
        state_norm = colors.Normalize(vmin=state_lower, vmax=state_upper, clip=True)
        error_norm = colors.Normalize(vmin=0.0, vmax=maximum_error, clip=True)
        fig = plt.figure(figsize=(7.2, 4.1))
        gs = fig.add_gridspec(
            2, 3, left=0.085, right=0.985, bottom=0.105, top=0.80,
            hspace=0.52, wspace=0.38,
        )
        fields = [fig.add_subplot(gs[0, column]) for column in range(3)]
        profiles = [fig.add_subplot(gs[1, column]) for column in range(3)]

        field_image = fields[0].pcolormesh(
            trajectory.x, trajectory.tau, trajectory.reference,
            shading="gouraud", cmap="turbo", norm=state_norm, rasterized=True,
        )
        fields[1].pcolormesh(
            trajectory.x, trajectory.tau, trajectory.adaptive,
            shading="gouraud", cmap="turbo", norm=state_norm, rasterized=True,
        )
        error_image = fields[2].pcolormesh(
            trajectory.x, trajectory.tau, trajectory.error,
            shading="gouraud", cmap="magma", norm=error_norm, rasterized=True,
        )
        for column, (axis, title) in enumerate(zip(
            fields, ("Reference solution", "Causal-adaptive solution", "Absolute error")
        )):
            # Compact mathematical labels retain each panel's coordinate axes.
            axis.set(
                xlabel=r"$x$",
                ylabel=r"$\tau=t/T$",
                title=title,
                xlim=(trajectory.x[0], trajectory.x[-1]),
                ylim=(0.0, 1.0),
            )
            axis.set_title(title, pad=2.0)
            axis.xaxis.labelpad = -1.0

        indices = native_snapshot_indices(trajectory.time)
        profile_indices = np.asarray(indices, dtype=int)
        profile_values = np.concatenate([
            trajectory.reference[profile_indices].ravel(),
            trajectory.adaptive[profile_indices].ravel(),
        ])
        profile_lower = float(np.min(profile_values))
        profile_upper = float(np.max(profile_values))
        profile_span = max(profile_upper - profile_lower, np.finfo(float).eps)
        profile_limits = (
            profile_lower - 0.05 * profile_span,
            profile_upper + 0.50 * profile_span,
        )
        for position, (axis, index) in enumerate(zip(profiles, indices)):
            axis.plot(trajectory.x, trajectory.reference[index], color="#20242A",
                      lw=1.3, label="Reference")
            axis.plot(trajectory.x, trajectory.adaptive[index], color=profile_color,
                      lw=1.3, dashes=(3.0, 2.0), label="Causal-adaptive")
            axis.set(xlabel=r"$x$", ylabel=r"$u(x,t)$",
                     title=rf"$\tau={_tau_label(trajectory.tau[index])}$",
                     ylim=profile_limits, xlim=(trajectory.x[0], trajectory.x[-1]))
            axis.set_title(rf"$\tau={_tau_label(trajectory.tau[index])}$", pad=1.0)
            if position == 0:
                axis.legend(loc="upper right", frameon=False, handlelength=2.8,
                            borderaxespad=0.35)

        for tag, axis in zip("abcdef", (*fields, *profiles)):
            _cmame_tag(axis, f"({tag})")
        _add_top_colorbar(
            fig, fields[0], fields[1], field_image, r"wave field $u(x,t)$",
            ticks=state_ticks,
        )
        _add_top_colorbar(fig, fields[2], fields[2], error_image,
                          r"$|u_h-u_{\mathrm{ref}}|$",
                          ticks=error_ticks)

        fig.savefig(target.with_suffix(".pdf"), dpi=300, bbox_inches="tight")
        fig.savefig(target.with_suffix(".png"), dpi=600, bbox_inches="tight")
        plt.close(fig)


def plot_mch_figure_2(trajectory: WaveTrajectory, target: Path, *, accent="#1769AA"):
    """CMAME Figure 2 with unclipped data-derived color limits."""
    if trajectory.equation != "mch":
        raise ValueError("plot_mch_figure_2 accepts only an mCH trajectory")
    plot_wavefield_2x3(
        trajectory,
        target,
        profile_color=accent,
        # Use full data ranges: the old 0.9/0.02 caps clipped the peak/error.
    )


def plot_six_panel(trajectory: WaveTrajectory, target: Path, *, accent: str):
    """Plot fields, native snapshots, and equation-specific causal evidence."""
    target.parent.mkdir(parents=True, exist_ok=True)
    norm, cmap = _field_norm(trajectory.reference, trajectory.adaptive)
    error_norm, visible_error = _error_norm(trajectory.error)
    fig = plt.figure(figsize=(12.2, 9.15))
    grid = fig.add_gridspec(
        4, 3, height_ratios=(0.055, 0.90, 0.68, 1.06),
        left=0.065, right=0.975, bottom=0.065, top=0.955,
        hspace=0.48, wspace=0.25,
    )
    wave_bar = fig.add_subplot(grid[0, :2])
    error_bar = fig.add_subplot(grid[0, 2])
    top = [fig.add_subplot(grid[1, index]) for index in range(3)]
    middle = [fig.add_subplot(grid[2, index]) for index in range(3)]
    evidence = fig.add_subplot(grid[3, :2], projection="3d")
    diagnostic = fig.add_subplot(grid[3, 2])

    # Gouraud is a rendering interpolation over the native vertices.  It does
    # not create data, select outcomes, or alter any reported metric; it avoids
    # presenting the storage grid as a false physical checkerboard.
    field_image = top[0].pcolormesh(trajectory.x, trajectory.tau, trajectory.reference,
                                    shading="gouraud", cmap=cmap, norm=norm, rasterized=True)
    top[1].pcolormesh(trajectory.x, trajectory.tau, trajectory.adaptive,
                      shading="gouraud", cmap=cmap, norm=norm, rasterized=True)
    error_image = top[2].pcolormesh(trajectory.x, trajectory.tau, visible_error,
                                    shading="gouraud", cmap=_error_cmap(), norm=error_norm,
                                    rasterized=True)
    for axis, title in zip(top, ("Reference solution", "Causal-adaptive solution",
                                 "Absolute error")):
        axis.set(title=title, xlabel=r"$x$", ylabel=r"normalized time $\tau=t/T$",
                 xlim=(trajectory.x[0], trajectory.x[-1]), ylim=(0, 1))
    wave_cb = fig.colorbar(field_image, cax=wave_bar, orientation="horizontal")
    error_cb = fig.colorbar(error_image, cax=error_bar, orientation="horizontal")
    _format_colorbar(wave_cb); _format_colorbar(error_cb)
    for bar in (wave_bar, error_bar):
        bar.xaxis.set_ticks_position("top"); bar.xaxis.set_label_position("top")
    wave_bar.set_xlabel(r"wave field $u(x,t)$", labelpad=5)
    error_bar.set_xlabel(r"posthoc $|u_h-u_{\mathrm{ref}}|$", labelpad=5)

    indices = native_snapshot_indices(trajectory.time)
    for position, (axis, index) in enumerate(zip(middle, indices)):
        axis.plot(trajectory.x, trajectory.reference[index], color="#20242A", lw=2.0,
                  label="Reference")
        axis.plot(trajectory.x, trajectory.adaptive[index], color=accent, lw=2.0,
                  dashes=(4.0, 2.0), label="Causal-adaptive")
        axis.set(xlabel=r"$x$", ylabel=r"$u(x,t)$",
                 title=rf"$\tau={_tau_label(trajectory.tau[index])}$")
        if position == 0:
            axis.legend(frameon=False, fontsize=7.5, loc="best")

    _plot_spacetime_evidence(evidence, trajectory, cmap=cmap, norm=norm)
    _plot_equation_diagnostic(diagnostic, trajectory, accent=accent)
    suffix = f" · {trajectory.parameter_label}" if trajectory.parameter_label else ""
    fig.text(0.065, 0.986, EQUATION_LABELS.get(trajectory.equation, trajectory.equation) + suffix,
             color=accent, fontsize=10.5, fontweight="bold", va="top")
    for tag, axis in zip("abcdef", (*top, *middle)):
        panel_label(axis, f"({tag})")
    evidence.text2D(-0.16, 1.04, "(g)", transform=evidence.transAxes,
                    fontsize=10, fontweight="bold", va="bottom")
    panel_label(diagnostic, "(h)")

    action_handles = [
        Line2D([0], [0], marker="s", color="none", markerfacecolor=ACTION_COLORS[key],
               markeredgecolor="none", markersize=6, label=label)
        for key, label in (("hold", "hold"), ("shrink_time", "shrink time"),
                           ("refine_space", "refine space/representation"),
                           ("probe", "saved causal prefix"))
    ]
    evidence.legend(handles=action_handles, loc="lower center", bbox_to_anchor=(0.5, -0.22),
                    ncol=4, fontsize=6.8, frameon=False)
    fig.savefig(target.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(target.with_suffix(".png"), dpi=500, bbox_inches="tight")
    plt.close(fig)


def _power_order(trajectory: WaveTrajectory) -> float:
    """Extract the registered power-law order from its immutable label."""
    if trajectory.equation != "power_law":
        raise ValueError("Figure 6 accepts only power-law trajectories")
    try:
        return float(trajectory.parameter_label.replace("$", "").split("=")[-1])
    except ValueError as exc:
        raise ValueError("power-law trajectory has no readable order label") from exc


def plot_power_law_figure_6(
    trajectories: list[WaveTrajectory],
    target: Path,
    *,
    representative_order: float = 2.6,
    profile_color: str = "#1769AA",
):
    """CMAME Figure 6: one detailed unseen trait plus a coefficient atlas.

    The detailed block repeats the Figure 2--5 field/profile grammar.  The
    atlas compares final profiles and absolute errors with explicitly separate
    zero-based scales; brightness across error columns is not comparable.
    """
    if len(trajectories) < 2:
        raise ValueError("Figure 6 requires a nontrivial coefficient atlas")
    ordered = sorted(trajectories, key=_power_order)
    orders = np.asarray([_power_order(row) for row in ordered], dtype=float)
    matches = np.flatnonzero(np.isclose(orders, float(representative_order), atol=1e-12))
    if len(matches) != 1:
        raise ValueError("representative power-law order must occur exactly once")
    representative = ordered[int(matches[0])]
    target.parent.mkdir(parents=True, exist_ok=True)

    state_lower = float(min(np.min(representative.reference),
                            np.min(representative.adaptive)))
    state_upper = float(max(np.max(representative.reference),
                            np.max(representative.adaptive)))
    state_norm = colors.Normalize(state_lower, state_upper, clip=True)
    representative_error_max = float(np.max(representative.error))
    error_norm = colors.Normalize(
        0.0, max(representative_error_max, np.finfo(float).eps), clip=True,
    )

    style = {
        "font.family": "serif",
        "font.serif": ["STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "text.usetex": False,
        "pdf.fonttype": 42,
        "font.size": 8.8,
        "axes.labelsize": 8.8,
        "axes.titlesize": 9.3,
        "xtick.labelsize": 7.7,
        "ytick.labelsize": 7.7,
        "legend.fontsize": 7.3,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "axes.linewidth": 0.8,
    }
    with mpl.rc_context(style):
        fig = plt.figure(figsize=(7.2, 7.0))
        mpl.rcParams.update({"axes.spines.top": False, "axes.spines.right": False})
        outer = fig.add_gridspec(
            2, 1, height_ratios=(1.12, 0.88), left=0.075, right=0.985,
            bottom=0.065, top=0.88, hspace=0.25,
        )
        main_grid = outer[0].subgridspec(2, 3, hspace=0.52, wspace=0.38)
        atlas_grid = outer[1].subgridspec(2, len(ordered), hspace=0.95, wspace=0.42)
        fields = [fig.add_subplot(main_grid[0, column]) for column in range(3)]
        profiles = [fig.add_subplot(main_grid[1, column]) for column in range(3)]
        atlas_profiles = [fig.add_subplot(atlas_grid[0, column])
                          for column in range(len(ordered))]
        atlas_errors = [fig.add_subplot(atlas_grid[1, column])
                        for column in range(len(ordered))]

        state_image = fields[0].pcolormesh(
            representative.x, representative.tau, representative.reference,
            shading="gouraud", cmap="turbo", norm=state_norm, rasterized=True,
        )
        fields[1].pcolormesh(
            representative.x, representative.tau, representative.adaptive,
            shading="gouraud", cmap="turbo", norm=state_norm, rasterized=True,
        )
        error_image = fields[2].pcolormesh(
            representative.x, representative.tau, representative.error,
            shading="gouraud", cmap="magma", norm=error_norm, rasterized=True,
        )
        top_titles = (
            rf"Reference solution ($p={representative_order:g}$)",
            "Causal-adaptive solution", "Absolute error",
        )
        for column, (axis, title) in enumerate(zip(fields, top_titles)):
            axis.set(
                xlabel=r"$x$",
                ylabel=r"$\tau=t/T$",
                title=title,
                xlim=(representative.x[0], representative.x[-1]),
                ylim=(0.0, 1.0),
            )
            axis.xaxis.labelpad = -1.0

        indices = native_snapshot_indices(representative.time)
        profile_indices = np.asarray(indices, dtype=int)
        main_values = np.concatenate((
            representative.reference[profile_indices].ravel(),
            representative.adaptive[profile_indices].ravel(),
        ))
        main_lower, main_upper = float(np.min(main_values)), float(np.max(main_values))
        main_span = max(main_upper - main_lower, np.finfo(float).eps)
        main_limits = (main_lower - 0.05 * main_span, main_upper + 0.50 * main_span)
        for position, (axis, index) in enumerate(zip(profiles, indices)):
            axis.plot(representative.x, representative.reference[index],
                      color="#20242A", lw=1.3, label="Reference")
            axis.plot(representative.x, representative.adaptive[index],
                      color=profile_color, lw=1.3, dashes=(3.0, 2.0),
                      label="Causal-adaptive")
            axis.set(xlabel=r"$x$", ylabel=r"$u(x,t)$", ylim=main_limits,
                     xlim=(representative.x[0], representative.x[-1]),
                     title=rf"$\tau={_tau_label(representative.tau[index])}$")
            if position == 0:
                axis.legend(loc="upper right", frameon=False, handlelength=2.6,
                            borderaxespad=0.35)
        for tag, axis in zip("abcdef", (*fields, *profiles)):
            _cmame_tag(axis, f"({tag})")

        final_values = np.concatenate([
            np.ravel(row.reference[-1]) for row in ordered
        ] + [np.ravel(row.adaptive[-1]) for row in ordered])
        atlas_lower, atlas_upper = float(np.min(final_values)), float(np.max(final_values))
        atlas_span = max(atlas_upper - atlas_lower, np.finfo(float).eps)
        atlas_limits = (atlas_lower - 0.05 * atlas_span,
                        atlas_upper + 0.12 * atlas_span)
        for column, (row, order, profile_axis, error_axis) in enumerate(
            zip(ordered, orders, atlas_profiles, atlas_errors)
        ):
            profile_axis.plot(row.x, row.reference[-1], color="#20242A", lw=1.05)
            profile_axis.plot(row.x, row.adaptive[-1], color=profile_color,
                              lw=1.05, dashes=(3.0, 2.0))
            profile_axis.set(title=rf"$p={order:g}$", xlim=(row.x[0], row.x[-1]),
                             ylim=atlas_limits)
            profile_axis.set(xlabel=r"$x$", ylabel=r"$u(x,T)$")
            profile_axis.tick_params(labelsize=6.7)
            profile_axis.set_xticks([-10, 0, 10])
            profile_axis.yaxis.set_major_locator(ticker.MaxNLocator(3))

            row_error_max = float(np.max(row.error))
            footprint_image = error_axis.pcolormesh(
                row.x, row.tau, row.error, shading="nearest", cmap="magma",
                norm=colors.Normalize(0, max(row_error_max, np.finfo(float).eps)), rasterized=True,
            )
            error_axis.set(xlim=(row.x[0], row.x[-1]), ylim=(0.0, 1.0),
                           xlabel=r"$x$", ylabel=r"$\tau$")
            error_axis.set_yticks([0, 0.5, 1])
            error_axis.set_xticks([-10, 0, 10])
            error_axis.tick_params(labelsize=6.7)
            _cmame_tag(profile_axis, f"(g{column + 1})")
            error_axis.text(0.03, 0.96, f"(h{column + 1})", transform=error_axis.transAxes,
                            va="top", color="white", fontsize=8, fontweight="bold")
            pos = error_axis.get_position()
            cax = fig.add_axes([pos.x0, pos.y1 + 0.024, pos.width, 0.006])
            cb = fig.colorbar(footprint_image, cax=cax, orientation="horizontal",
                              ticks=[0, row_error_max / 2, row_error_max])
            exponent = int(np.floor(np.log10(max(row_error_max, np.finfo(float).eps))))
            cb.formatter = ticker.FuncFormatter(lambda v, _, ex=exponent: f"{v / 10**ex:.1f}")
            cb.update_ticks()
            cax.xaxis.set_ticks_position("top")
            cax.tick_params(labelsize=6.5, pad=1, length=2)
            cax.set_title(rf"$|u_h-u_{{\mathrm{{ref}}}}|\;(\times10^{{{exponent}}})$", fontsize=7.1, pad=4)
        atlas_profiles[0].text(
            0.0, 1.24, "Registered dispersion family: terminal profiles and absolute errors",
            transform=atlas_profiles[0].transAxes, fontsize=9.3,
            fontweight="bold", ha="left", va="bottom",
        )

        _add_top_colorbar(fig, fields[0], fields[1], state_image,
                          r"wave field $u(x,t)$")
        _add_top_colorbar(fig, fields[2], fields[2], error_image,
                          r"$|u_h-u_{\mathrm{ref}}|$ at $p=2.6$")
        fig.savefig(target.with_suffix(".pdf"), dpi=300, bbox_inches="tight")
        fig.savefig(target.with_suffix(".png"), dpi=600, bbox_inches="tight")
        plt.close(fig)
        target.with_suffix('.txt').write_text(
            'Representative p=2.6: reference, adaptive solution, pointwise absolute error, '
            'and three native-time profiles. Lower atlas: terminal profiles and pointwise '
            'absolute errors with independent zero-based linear scales. Brightness is not '
            'comparable across error panels. Reference and adaptive trajectories start from '
            'the same 32-point band-limited initial field; errors exclude its discrepancy '
            'from the original initial-condition formula. The reference is a posthoc DOP853 '
            'solution. These are numerical differences, not certified continuous-PDE error bounds. '
            'Initial seeds and nonlinear transport coefficients vary across the registered '
            'dispersion family; this is not a controlled one-parameter sensitivity study.',
            encoding='utf-8')


def plot_power_law_atlas(trajectories: list[WaveTrajectory], target: Path):
    """Backward-compatible entry point for the unified Figure 6."""
    plot_power_law_figure_6(trajectories, target)
