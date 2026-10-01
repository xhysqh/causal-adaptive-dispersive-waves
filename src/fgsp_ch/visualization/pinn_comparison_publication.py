"""CMAME figures comparing a frozen coordinate PINN with the proposed solver.

Every panel is constructed from aligned, frozen fields.  The module never
creates a synthetic baseline and never changes either method's numerical
output for visual purposes.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
import json

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import colors, ticker
import numpy as np


PINN_COLOR = "#D55E00"
OURS_COLOR = "#1769AA"
REFERENCE_COLOR = "#20242A"
ADVANTAGE_EPSILON = 1.0e-12


@dataclass(frozen=True)
class PINNComparisonTrajectory:
    x: np.ndarray
    time: np.ndarray
    reference: np.ndarray
    baseline_pinn: np.ndarray
    causal_adaptive: np.ndarray
    equation: str = "mch"
    selection_rule: str = "pre-registered F2 mCH showcase"
    active_points: np.ndarray | None = None

    def __post_init__(self) -> None:
        x = np.asarray(self.x, dtype=float)
        time = np.asarray(self.time, dtype=float)
        shape = (len(time), len(x))
        if x.ndim != 1 or time.ndim != 1 or len(x) < 4 or len(time) < 3:
            raise ValueError("comparison axes must be nontrivial vectors")
        if np.any(np.diff(x) <= 0) or np.any(np.diff(time) <= 0):
            raise ValueError("comparison coordinates must be strictly increasing")
        for name in ("reference", "baseline_pinn", "causal_adaptive"):
            value = np.asarray(getattr(self, name), dtype=float)
            if value.shape != shape:
                raise ValueError(f"{name} must have shape {shape}")
            if not np.all(np.isfinite(value)):
                raise ValueError(f"{name} contains nonfinite values")
        if self.active_points is not None:
            active = np.asarray(self.active_points, dtype=int)
            if active.shape != time.shape or np.any(active < 4):
                raise ValueError("active_points must match the stored time axis")
        if not self.selection_rule:
            raise ValueError("representative selection must be precommitted")

    @property
    def tau(self) -> np.ndarray:
        return (self.time - self.time[0]) / (self.time[-1] - self.time[0])

    @property
    def pinn_error(self) -> np.ndarray:
        return np.abs(self.baseline_pinn - self.reference)

    @property
    def ours_error(self) -> np.ndarray:
        return np.abs(self.causal_adaptive - self.reference)


def load_comparison(path: Path) -> PINNComparisonTrajectory:
    with np.load(path, allow_pickle=False) as data:
        return PINNComparisonTrajectory(
            x=data["x"], time=data["time"], reference=data["reference"],
            baseline_pinn=data["baseline_pinn"],
            causal_adaptive=data["causal_adaptive"],
            equation=str(np.asarray(data["equation"]).item()),
            selection_rule=str(np.asarray(data["selection_rule"]).item()),
            active_points=(np.asarray(data["active_points"], dtype=int)
                           if "active_points" in data.files else None),
        )


def _style() -> dict:
    return {
        "text.usetex": False,
        "font.family": "serif",
        "font.serif": ["STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8.3,
        "axes.labelsize": 9.1,
        "axes.titlesize": 9.6,
        "xtick.labelsize": 7.7,
        "ytick.labelsize": 7.7,
        "legend.fontsize": 7.3,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }


def _tag(axis, text: str, *, inside: bool = False, color: str = REFERENCE_COLOR) -> None:
    if inside:
        axis.text(0.09, 0.95, text, transform=axis.transAxes, fontsize=10,
                  fontweight="bold", va="top", ha="left", color=color, zorder=10,
                  bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.94, "pad": 0.25})
    else:
        axis.text(-0.08, 1.01, text, transform=axis.transAxes, fontsize=10,
                  fontweight="bold", va="bottom", ha="right", color=color)


def _top_colorbar(fig, left_axis, right_axis, image, title, *, ticks=None,
                  pad=0.045, height=0.012) -> None:
    left = left_axis.get_position()
    right = right_axis.get_position()
    cax = fig.add_axes([left.x0, left.y1 + pad, right.x1 - left.x0, height])
    bar = fig.colorbar(image, cax=cax, orientation="horizontal", ticks=ticks)
    bar.ax.xaxis.set_ticks_position("top")
    bar.ax.xaxis.set_label_position("top")
    bar.ax.tick_params(labelsize=8.3, length=2.5, pad=1.5)
    magnitude = max(abs(float(image.norm.vmin)), abs(float(image.norm.vmax)))
    exponent = int(np.floor(np.log10(magnitude))) if magnitude > 0.0 else 0
    scientific = exponent <= -2 or exponent >= 2
    scale = 10.0 ** exponent if scientific else 1.0
    bar.formatter = ticker.FuncFormatter(lambda value, _: f"{value / scale:g}")
    bar.update_ticks()
    display_title = (
        title + rf" $\;(\times 10^{{{exponent}}})$" if scientific else title
    )
    bar.ax.set_title(display_title, fontsize=8.7, pad=5.0)


def _save(fig, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(target.with_suffix(".png"), dpi=600, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def _field_axis(
    axis, trajectory, values, title, image_options, *, xlabel=True, ylabel=True,
):
    image = axis.pcolormesh(
        trajectory.x, trajectory.tau, values, shading="gouraud",
        rasterized=True, **image_options,
    )
    axis.set(xlabel=(r"$x$" if xlabel else ""),
             ylabel=(r"normalized time $\tau=t/T$" if ylabel else ""), title=title,
             xlim=(trajectory.x[0], trajectory.x[-1]), ylim=(0.0, 1.0))
    axis.set_title(title, pad=2.0)
    return image


def _six_registered_indices(trajectory: PINNComparisonTrajectory) -> tuple[int, ...]:
    """Six predeclared, error-independent causal-prefix snapshots."""
    targets = np.array((0.10, 0.25, 0.40, 0.55, 0.75, 1.00))
    indices = tuple(int(np.argmin(np.abs(trajectory.tau - target))) for target in targets)
    if len(set(indices)) != 6 or any(right <= left for left, right in zip(indices, indices[1:])):
        raise ValueError("six profile prefixes must be unique and strictly ordered")
    if indices[-1] != len(trajectory.tau) - 1:
        raise ValueError("final profile must use the terminal registered prefix")
    return indices


def _regularized_advantage(trajectory: PINNComparisonTrajectory) -> np.ndarray:
    """Stable posthoc log error ratio; positive values favor causal adaptation."""
    advantage = np.log10(
        (trajectory.pinn_error + ADVANTAGE_EPSILON)
        / (trajectory.ours_error + ADVANTAGE_EPSILON)
    )
    if not np.all(np.isfinite(advantage)):
        raise ValueError("regularized pointwise advantage must remain finite")
    return advantage


def _periodic_derivative(values: np.ndarray, x: np.ndarray) -> np.ndarray:
    dx = float(np.mean(np.diff(x)))
    wave = 2.0 * np.pi * np.fft.fftfreq(len(x), d=dx)
    return np.fft.ifft(1j * wave[None, :] * np.fft.fft(values, axis=1), axis=1).real


def comparison_metrics(trajectory: PINNComparisonTrajectory) -> dict[str, np.ndarray]:
    reference = trajectory.reference
    ref_x = _periodic_derivative(reference, trajectory.x)
    denominator_l2 = np.linalg.norm(reference, axis=1).clip(min=1e-15)
    denominator_h1 = np.sqrt(
        np.sum(reference**2 + ref_x**2, axis=1)
    ).clip(min=1e-15)
    result: dict[str, np.ndarray] = {}
    for stem, field in (("pinn", trajectory.baseline_pinn),
                        ("ours", trajectory.causal_adaptive)):
        delta = field - reference
        delta_x = _periodic_derivative(delta, trajectory.x)
        result[f"{stem}_l2"] = np.linalg.norm(delta, axis=1) / denominator_l2
        result[f"{stem}_h1"] = np.sqrt(
            np.sum(delta**2 + delta_x**2, axis=1)
        ) / denominator_h1
        result[f"{stem}_linf"] = np.max(np.abs(delta), axis=1) / np.maximum(
            np.max(np.abs(reference), axis=1), 1e-15
        )
    return result


def comparison_metrics_common_band(trajectory: PINNComparisonTrajectory) -> tuple[dict[str, np.ndarray], int]:
    """Errors on one fixed Fourier band resolved by every adaptive prefix.

    A fixed band prevents a change of adaptive grid from changing the norm
    itself.  The two methods and the reference are filtered identically.
    """
    minimum_points = (int(np.min(trajectory.active_points))
                      if trajectory.active_points is not None else len(trajectory.x))
    cutoff = max(1, minimum_points // 3)
    modes = np.fft.fftfreq(len(trajectory.x)) * len(trajectory.x)
    mask = np.abs(modes) <= cutoff

    def filtered(values: np.ndarray) -> np.ndarray:
        spectrum = np.fft.fft(values, axis=1)
        spectrum[:, ~mask] = 0.0
        return np.fft.ifft(spectrum, axis=1).real

    common = PINNComparisonTrajectory(
        x=trajectory.x,
        time=trajectory.time,
        reference=filtered(trajectory.reference),
        baseline_pinn=filtered(trajectory.baseline_pinn),
        causal_adaptive=filtered(trajectory.causal_adaptive),
        equation=trajectory.equation,
        selection_rule=trajectory.selection_rule,
    )
    return comparison_metrics(common), cutoff


def plot_figure_7_direct_comparison(trajectory: PINNComparisonTrajectory,
                                    target: Path) -> None:
    """Reference, two solvers, their errors, and pointwise error advantage."""
    with mpl.rc_context(_style()):
        if not (trajectory.reference.shape == trajectory.baseline_pinn.shape == trajectory.causal_adaptive.shape):
            raise ValueError("Figure 7 requires one identical stored grid and time axis")
        state_min = float(min(trajectory.reference.min(), trajectory.baseline_pinn.min(),
                              trajectory.causal_adaptive.min()))
        state_max = float(max(trajectory.reference.max(), trajectory.baseline_pinn.max(),
                              trajectory.causal_adaptive.max()))
        state_norm = colors.Normalize(state_min, state_max)
        maximum_error = float(max(trajectory.pinn_error.max(), trajectory.ours_error.max()))
        error_norm = colors.Normalize(0.0, maximum_error or np.finfo(float).eps)
        independent_errors = trajectory.equation.lower() in {"bo", "benjamin_ono"}
        pinn_norm = (colors.Normalize(0.0, float(trajectory.pinn_error.max()) or np.finfo(float).eps)
                     if independent_errors else error_norm)
        ours_norm = (colors.Normalize(0.0, float(trajectory.ours_error.max()) or np.finfo(float).eps)
                     if independent_errors else error_norm)
        advantage = _regularized_advantage(trajectory)
        limit = max(float(np.nanpercentile(np.abs(advantage), 98.0)), 0.25)

        # Give the two heat-map rows enough vertical authority in the final
        # two-column layout; the previous 4.65-inch canvas looked compressed.
        fig = plt.figure(figsize=(7.2, 4.7))
        gs = fig.add_gridspec(2, 3, left=0.085, right=0.985, bottom=0.105,
                              top=0.82, wspace=0.18, hspace=0.58)
        axes = np.asarray([[fig.add_subplot(gs[row, col]) for col in range(3)]
                           for row in range(2)])
        state_options = {"cmap": "turbo", "norm": state_norm}
        state_image = _field_axis(axes[0, 0], trajectory, trajectory.reference,
                                  "Reference solution", state_options, xlabel=False)
        _field_axis(axes[0, 1], trajectory, trajectory.baseline_pinn,
                    "Baseline PINN", state_options, xlabel=False, ylabel=False)
        _field_axis(axes[0, 2], trajectory, trajectory.causal_adaptive,
                    "Adaptive operator", state_options, xlabel=False,
                    ylabel=False)
        pinn_error_image = _field_axis(
            axes[1, 0], trajectory, trajectory.pinn_error,
            "Baseline PINN absolute error",
            {"cmap": "magma", "norm": pinn_norm},
        )
        ours_error_image = _field_axis(
            axes[1, 1], trajectory, trajectory.ours_error,
            "Adaptive absolute error", {"cmap": "magma", "norm": ours_norm},
            ylabel=False,
        )
        advantage_image = _field_axis(
            axes[1, 2], trajectory, advantage, "Pointwise error advantage",
            {"cmap": "RdBu_r", "norm": colors.TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)},
            ylabel=False,
        )
        # Keep panel identifiers outside the image data.  White label boxes
        # obscure the field/error maps and look like temporary annotations in
        # the final two-column artwork.
        for tag, axis in zip("abcdef", axes.ravel()):
            _tag(axis, f"({tag})", inside=False, color=REFERENCE_COLOR)
        _top_colorbar(fig, axes[0, 0], axes[0, 2], state_image,
                      r"wave field $u(x,\tau)$", pad=0.055)
        # Preserve the manuscript notation exactly in both source equations:
        # e_PINN=|u_PINN-u_ref| and e_CA=|u_h-u_ref|.  For mCH the two bars
        # share the same normalization; for BO their zero-based scales remain
        # independent, as documented in the caption sidecar.
        _top_colorbar(fig, axes[1, 0], axes[1, 0], pinn_error_image,
                      r"$e_{\mathrm{PINN}}$", pad=0.050)
        _top_colorbar(fig, axes[1, 1], axes[1, 1], ours_error_image,
                      r"$e_{\mathrm{CA}}$", pad=0.050)
        _top_colorbar(fig, axes[1, 2], axes[1, 2], advantage_image,
                      r"$\log_{10}\!\left(\frac{e_{\mathrm{PINN}}+\varepsilon}{e_{\mathrm{CA}}+\varepsilon}\right)$",
                      pad=0.050)
        _save(fig, target)
        target.with_suffix('.txt').write_text(
            'Pointwise errors are e_PINN=|u_PINN-u_ref| and e_CA=|u_CA-u_ref|. '
            'The log10 ratio uses epsilon=1e-12; positive values favor causal adaptation. '
            f'The symmetric color range is [-{limit:.8g}, {limit:.8g}], selected by the '
            '98th percentile of absolute log ratios (minimum half-range 0.25); values outside saturate. '
            + (f'The absolute-error panels use independent zero-based linear scales: '
               f'PINN [0, {pinn_norm.vmax:.10g}], causal-adaptive [0, {ours_norm.vmax:.10g}]. '
               'Color brightness is not directly comparable between these panels; '
               'use the numerical colorbars and pointwise error ratio for magnitude comparisons. '
               if independent_errors else
               'Both absolute-error panels share one zero-based linear scale. ')
            +
            'These representative field-fidelity controls do not establish computational superiority.',
            encoding='utf-8')


def plot_figure_8_profiles(trajectory: PINNComparisonTrajectory, target: Path) -> None:
    """Pointwise absolute errors at six predeclared native prefixes."""
    with mpl.rc_context(_style()):
        indices = _six_registered_indices(trajectory)
        selected = np.asarray(indices, dtype=int)
        upper = max(float(trajectory.pinn_error[selected].max()),
                    float(trajectory.ours_error[selected].max()), 1e-15)
        fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.15))
        fig.subplots_adjust(left=0.085, right=0.985, bottom=0.11, top=0.84,
                            wspace=0.38, hspace=0.43)
        for axis, index in zip(axes.ravel(), indices):
            axis.plot(trajectory.x, trajectory.pinn_error[index], color=PINN_COLOR,
                      lw=1.25, label="Baseline PINN")
            axis.plot(trajectory.x, trajectory.ours_error[index], color=OURS_COLOR,
                      lw=1.25, dashes=(3, 2), label="Adaptive")
            axis.set(xlabel=r"$x$", ylabel="Absolute error",
                     title=rf"$\tau={trajectory.tau[index]:.2f}$",
                     xlim=(trajectory.x[0], trajectory.x[-1]),
                     ylim=(0.0, upper * 1.08))
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.995),
                   ncol=2, frameon=False, handlelength=2.6, columnspacing=1.5)
        for tag, axis in zip("abcdef", axes.ravel()):
            _tag(axis, f"({tag})")
        _save(fig, target)
        target.with_suffix('.json').write_text(json.dumps({
            'selection_targets': [0.10, 0.25, 0.40, 0.55, 0.75, 1.0],
            'selected_indices': list(indices), 'actual_tau': trajectory.tau[selected].tolist(),
            'selection_uses_reference_error': False, 'quantity': 'pointwise absolute error',
            'shared_ylim': [0.0, upper * 1.08],
        }, indent=2), encoding='utf-8')


def plot_figure_9_error_propagation(trajectory: PINNComparisonTrajectory,
                                    target: Path) -> None:
    """Three common-band norm-error panels after the initial trace.

    Resolution changes belong to the causal-mechanism figures, rather than to
    this PINN comparison: keeping this figure to the three error norms avoids
    treating the adaptive resolution history as a fourth comparison outcome.
    """
    metrics, cutoff = comparison_metrics_common_band(trajectory)
    with mpl.rc_context(_style()):
        fig, axes = plt.subplots(1, 3, figsize=(5.45, 2.5))
        fig.subplots_adjust(left=0.095, right=0.985, bottom=0.22, top=0.74, wspace=0.48)
        handles = None
        for column, (axis, suffix, label) in enumerate(zip(
            axes[:3], ("l2", "h1", "linf"),
            (r"$L^2$", r"$H^1$", r"$L^\infty$"),
        )):
            axis.plot(trajectory.tau[1:], metrics[f"pinn_{suffix}"][1:],
                          color=PINN_COLOR, lw=1.35, marker="o", ms=2.6, markevery=4,
                          label="Baseline PINN")
            axis.plot(trajectory.tau[1:], metrics[f"ours_{suffix}"][1:],
                          color=OURS_COLOR, lw=1.35, marker="s", ms=2.4, markevery=4,
                          dashes=(3, 2), label="Adaptive")
            axis.set(xlabel=r"normalized time $\tau$",
                     ylabel="Relative error", title=label, xlim=(0.0, 1.0))
            axis.grid(axis="y", color="#D7DCE2", linewidth=0.45, alpha=0.65)
            formatter = ticker.ScalarFormatter(useMathText=True)
            formatter.set_useOffset(False)
            values = np.r_[metrics[f"pinn_{suffix}"][1:], metrics[f"ours_{suffix}"][1:]]
            low, high = float(values.min()), float(values.max())
            padding = max(high - low, high * 0.05, 1e-15) * 0.12
            axis.set_ylim(max(0.0, low - padding), high + padding)
            exponent = int(np.floor(np.log10(high))) if high > 0 else 0
            formatter.set_powerlimits((exponent, exponent) if exponent != 0 else (-2, 2))
            axis.yaxis.set_major_formatter(formatter)
            if handles is None:
                handles = axis.get_legend_handles_labels()
        if handles is not None:
            fig.legend(
                *handles, loc="upper center", bbox_to_anchor=(0.5, 0.985),
                ncol=2, frameon=False, handlelength=2.7, columnspacing=1.8,
            )
        # Keep panel letters outside the data rectangle: the error histories
        # approach the upper edge in several panels, so in-axis lettering can
        # obscure a curve or a scientific-notation offset.
        for tag, axis in zip("abc", axes):
            axis.text(-0.18, 1.02, f"({tag})", transform=axis.transAxes,
                      ha="left", va="bottom", fontsize=10, fontweight="bold",
                      clip_on=False)
        _save(fig, target)
        target.with_suffix('.txt').write_text(
            'Relative L2, H1 and Linfinity errors use the same fixed common Fourier band for the reference, '
            f'PINN and V2 fields (|k| <= {cutoff}). H1 includes the derivative of the common-band field. '
            'Only prefixes after the initial trace are drawn. '
            'Linear y limits cover both methods at every displayed prefix with a 12% margin; they need not begin at zero. '
            'The initial error values (PINN, causal-adaptive) are: '
            + '; '.join(f'{key}: {metrics["pinn_"+key][0]:.8g}, {metrics["ours_"+key][0]:.8g}'
                        for key in ('l2', 'h1', 'linf')),
            encoding='utf-8')


def plot_first_three_comparison_figures(trajectory: PINNComparisonTrajectory,
                                        output: Path, *,
                                        start_figure: int = 7) -> list[Path]:
    """Render one equation's three comparison figures with unique paper numbers.

    ``start_figure`` is explicit because mCH and BO are separate main-text
    figures: mCH occupies Figures 7--9 and BO occupies Figures 10--12.
    """
    if start_figure < 1:
        raise ValueError("start_figure must be positive")
    output.mkdir(parents=True, exist_ok=True)
    stems = [
        output / f"figure_{start_figure}_full_field_comparison",
        output / f"figure_{start_figure + 1}_six_prefix_errors",
        output / f"figure_{start_figure + 2}_error_propagation",
    ]
    plot_figure_7_direct_comparison(trajectory, stems[0])
    plot_figure_8_profiles(trajectory, stems[1])
    plot_figure_9_error_propagation(trajectory, stems[2])
    return stems
