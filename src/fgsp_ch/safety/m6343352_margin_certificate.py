from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fgsp_ch.safety.m634334_group_ltt import clopper_pearson_upper


def monotone_horizon_upper(
    members: np.ndarray,
    sigma: float,
) -> np.ndarray:
    """Ensemble mean-plus-spread upper envelope, monotone in horizon."""
    values = np.asarray(members, dtype=np.float64)
    if values.ndim != 3 or not np.all(np.isfinite(values)):
        raise ValueError("members must have shape (ensemble, rows, horizon)")
    upper = np.clip(values.mean(axis=0) + sigma * values.std(axis=0), 0.0, 1.0)
    return np.maximum.accumulate(upper, axis=1)


def calibration_quantile_thresholds(
    score: np.ndarray,
    quantiles: tuple[float, ...] | list[float],
) -> tuple[float, ...]:
    values = np.asarray(score, dtype=np.float64)
    levels = np.asarray(quantiles, dtype=np.float64)
    if values.ndim != 1 or not np.all(np.isfinite(values)):
        raise ValueError("calibration scores must be a finite vector")
    if np.any(levels <= 0) or np.any(levels >= 1) or np.any(np.diff(levels) <= 0):
        raise ValueError("quantile levels must be strictly increasing in (0, 1)")
    thresholds = np.quantile(values, levels, method="higher")
    return tuple(float(value) for value in np.unique(thresholds))


@dataclass(frozen=True)
class MarginSelectionResult:
    threshold: float | None
    records: tuple[dict, ...]


def deployment_eligibility(
    support_ratio: np.ndarray,
    log_severity_upper: np.ndarray,
    log_direction_upper: np.ndarray,
    hard_current_violation: np.ndarray,
    *,
    severe_threshold: float,
    direction_threshold: float = 1.0,
) -> np.ndarray:
    """Shared offline/online deterministic pre-risk gate."""
    support = np.asarray(support_ratio, dtype=np.float64)
    severity = np.asarray(log_severity_upper, dtype=np.float64)
    direction = np.asarray(log_direction_upper, dtype=np.float64)
    hard = np.asarray(hard_current_violation, dtype=bool)
    if not (support.shape == severity.shape == direction.shape == hard.shape):
        raise ValueError("deployment gate arrays must have equal shape")
    return (
        np.isfinite(support) & np.isfinite(severity) & np.isfinite(direction)
        & (support <= 1.0)
        & (severity <= np.log1p(severe_threshold))
        & (direction <= np.log1p(direction_threshold))
        & ~hard
    )


def _group_failures(accepted, unsafe, groups) -> int:
    return sum(
        bool(np.any(accepted[groups == group] & unsafe[groups == group]))
        for group in np.unique(groups)
    )


def fixed_sequence_margin_selection(
    score: np.ndarray,
    eligible: np.ndarray,
    unsafe: np.ndarray,
    severe: np.ndarray,
    groups: np.ndarray,
    cells: np.ndarray,
    thresholds: tuple[float, ...] | list[float],
    *,
    global_alpha: float,
    global_delta: float,
    cell_alpha: float,
    cell_delta_total: float,
    minimum_cell_acceptances: int,
) -> MarginSelectionResult:
    score = np.asarray(score, dtype=np.float64)
    eligible = np.asarray(eligible, dtype=bool)
    unsafe = np.asarray(unsafe, dtype=bool)
    severe = np.asarray(severe, dtype=bool)
    groups = np.asarray(groups)
    cells = np.asarray(cells)
    if not (score.shape == eligible.shape == unsafe.shape == severe.shape == groups.shape == cells.shape):
        raise ValueError("all LTT row arrays must have equal shape")
    ordered = tuple(map(float, thresholds))
    if any(right <= left for left, right in zip(ordered, ordered[1:])):
        raise ValueError("thresholds must be strictly increasing")
    unique_groups = np.unique(groups)
    unique_cells = np.unique(cells)
    cell_delta = cell_delta_total / max(len(unique_cells), 1)
    chosen = None
    records: list[dict] = []
    for threshold in ordered:
        accepted = eligible & (score <= threshold)
        global_failures = _group_failures(accepted, unsafe, groups)
        global_upper = clopper_pearson_upper(
            global_failures, len(unique_groups), global_delta,
        )
        cell_records = {}
        cells_pass = True
        opportunities_pass = True
        for cell in unique_cells:
            mask = cells == cell
            cell_groups = groups[mask]
            failures = _group_failures(accepted[mask], unsafe[mask], cell_groups)
            trials = len(np.unique(cell_groups))
            upper = clopper_pearson_upper(failures, trials, cell_delta)
            cell_acceptances = int(np.sum(accepted[mask]))
            cell_records[str(cell)] = {
                "failures": int(failures), "trials": int(trials),
                "risk_upper": upper, "accepted": cell_acceptances,
            }
            cells_pass &= upper <= cell_alpha
            opportunities_pass &= cell_acceptances >= minimum_cell_acceptances
        severe_acceptances = int(np.sum(accepted & severe))
        risk_pass = global_upper <= global_alpha and cells_pass and severe_acceptances == 0
        records.append({
            "threshold": threshold,
            "global_failures": int(global_failures),
            "global_trials": int(len(unique_groups)),
            "global_risk_upper": global_upper,
            "severe_acceptances": severe_acceptances,
            "accepted": int(np.sum(accepted)),
            "cells": cell_records,
            "risk_pass": bool(risk_pass),
            "opportunities_pass": bool(opportunities_pass),
        })
        if not risk_pass:
            break
        if opportunities_pass and np.any(accepted):
            chosen = threshold
    return MarginSelectionResult(chosen, tuple(records))
