"""Pre-registered paired statistics for the CMAME-M2.3 paper comparison."""

from __future__ import annotations

from collections.abc import Iterable
import math

import numpy as np


GENERAL_BASELINES: dict[str, tuple[str, ...]] = {
    "smooth_periodic": (
        "conservative_finite_difference", "fourier_ifrk4",
        "residual_space_time_amr", "wang_yan_pinn", "fno",
    ),
    "smooth_to_sharp": (
        "conservative_finite_difference", "fourier_ifrk4",
        "residual_space_time_amr", "fno",
    ),
    "single_periodic_peakon": ("wang_yan_pinn",),
    "mixed_field_peakon": ("residual_space_time_amr",),
}

SPECIALIZED_CONTROLS: dict[str, tuple[str, ...]] = {
    "single_periodic_peakon": ("particle_dop853",),
    "separated_multipeakon": ("particle_dop853",),
    "clustered_multipeakon": ("particle_dop853",),
}


def grouped_median_bootstrap(
    values: Iterable[float], groups: Iterable[str], *, samples: int,
    confidence: float, seed: int = 20260827,
) -> tuple[float, float]:
    values = np.asarray(list(values), dtype=float)
    groups = np.asarray(list(groups), dtype=str)
    if values.ndim != 1 or values.size == 0 or values.shape != groups.shape:
        raise ValueError("paired values and trajectory groups must be aligned")
    if not np.all(np.isfinite(values)) or samples < 100 or not 0.0 < confidence < 1.0:
        raise ValueError("invalid grouped bootstrap settings")
    unique = np.unique(groups)
    rng = np.random.default_rng(seed)
    estimates = np.empty(samples)
    for index in range(samples):
        selected = rng.choice(unique, len(unique), replace=True)
        draw = np.concatenate([values[groups == group] for group in selected])
        estimates[index] = np.median(draw)
    tail = 0.5 * (1.0 - confidence)
    return tuple(map(float, np.quantile(estimates, [tail, 1.0 - tail])))


def paired_effect_records(rows: list[dict]) -> list[dict]:
    """Pair the proposed method with the best registered general baseline."""
    by_case: dict[str, list[dict]] = {}
    for row in rows:
        by_case.setdefault(str(row["case_id"]), []).append(row)
    records: list[dict] = []
    floor = np.finfo(float).tiny
    for case_id, case_rows in by_case.items():
        proposed = [row for row in case_rows if row["method"] == "proposed_m13"]
        if len(proposed) != 1:
            continue
        item = proposed[0]
        allowed = GENERAL_BASELINES.get(str(item["family"]), ())
        candidates = [row for row in case_rows if row["method"] in allowed]
        if not candidates:
            continue
        baseline = min(candidates, key=lambda row: float(row["relative_h1"]))
        ratio = float(item["relative_h1"]) / max(float(baseline["relative_h1"]), floor)
        records.append({
            "case_id": case_id,
            "group_id": str(item["group_id"]),
            "family": str(item["family"]),
            "tolerance": float(item["tolerance"]),
            "final_time": float(item["final_time"]),
            "proposed_relative_h1": float(item["relative_h1"]),
            "baseline_method": str(baseline["method"]),
            "baseline_relative_h1": float(baseline["relative_h1"]),
            "error_ratio": ratio,
            "log10_error_ratio": math.log10(max(ratio, floor)),
        })
    return records


def effect_summary(records: list[dict], *, samples: int, confidence: float) -> dict:
    if not records:
        raise ValueError("at least one paired general-baseline record is required")
    result = {}
    for family in ["all", *sorted({row["family"] for row in records})]:
        subset = records if family == "all" else [row for row in records if row["family"] == family]
        values = [row["log10_error_ratio"] for row in subset]
        interval = grouped_median_bootstrap(
            values, [row["group_id"] for row in subset], samples=samples,
            confidence=confidence, seed=20260827 + len(result),
        )
        result[family] = {
            "cases": len(subset),
            "trajectory_groups": len({row["group_id"] for row in subset}),
            "median_log10_error_ratio": float(np.median(values)),
            "median_error_ratio": float(10.0 ** np.median(values)),
            "group_bootstrap_log10_95": list(interval),
            "improvement_fraction": float(np.mean([row["error_ratio"] < 1.0 for row in subset])),
        }
    return result


def tolerance_monotone_fraction(proposed: list[dict]) -> float:
    pairs: dict[tuple[str, float], list[dict]] = {}
    for row in proposed:
        pairs.setdefault((str(row["group_id"]), float(row["final_time"])), []).append(row)
    monotone = []
    for values in pairs.values():
        ordered = sorted(values, key=lambda row: float(row["tolerance"]), reverse=True)
        monotone.append(all(
            float(ordered[index]["relative_h1"]) >= float(ordered[index + 1]["relative_h1"])
            for index in range(len(ordered) - 1)
        ))
    return float(np.mean(monotone)) if monotone else 0.0

