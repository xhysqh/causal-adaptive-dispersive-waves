"""Small, deterministic CMAME-oriented Matplotlib style helpers."""
from __future__ import annotations

from contextlib import contextmanager
import matplotlib as mpl


COLORS = {
    "mch": "#1769AA",
    "kdv": "#158F83",
    "bo": "#D54B5A",
    "ilw": "#7357B3",
    "power_law": "#D99016",
    "reference": "#111827",
    "error": "#F4A261",
    "shrink_time": "#7B2CBF",
    "hold": "#8D99AE",
    "refine_space": "#009E73",
    "abstain": "#E9C46A",
}


@contextmanager
def cmame_style():
    """Use a journal-safe style without mutating global settings permanently."""
    settings = {
        "font.family": "serif",
        "font.serif": ["STIX Two Text", "Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 10.0,
        "axes.titlesize": 11.0,
        "axes.labelsize": 10.0,
        "xtick.labelsize": 9.0,
        "ytick.labelsize": 9.0,
        "legend.fontsize": 8.0,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "legend.frameon": False,
        "axes.titlepad": 5.0,
        "axes.unicode_minus": True,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.transparent": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
    with mpl.rc_context(settings):
        yield


def panel_label(axis, label: str) -> None:
    draw = getattr(axis, "text2D", axis.text)
    draw(-0.10, 1.04, label, transform=axis.transAxes,
         fontsize=11, fontweight="bold", va="bottom")
