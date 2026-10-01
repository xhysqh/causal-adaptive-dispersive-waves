from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fgsp_ch.safety.m634334_group_ltt import clopper_pearson_upper


def joint_reachability_score(cumulative_risk, log_severity_upper, log_direction_upper):
    risk = np.asarray(cumulative_risk, dtype=np.float64)
    severity = np.asarray(log_severity_upper, dtype=np.float64)
    direction = np.asarray(log_direction_upper, dtype=np.float64)
    if not (risk.shape == severity.shape == direction.shape):
        raise ValueError("joint risk components must have equal shape")
    return np.maximum.reduce((
        risk,
        1.0 - np.exp(-np.maximum(severity, 0.0)),
        1.0 - np.exp(-np.maximum(direction, 0.0)),
    ))


def shrunken_cell_upper_margins(target, prediction, cells, quantile: float, shrinkage: float):
    target = np.asarray(target, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    cells = np.asarray(cells)
    if target.shape != prediction.shape or target.ndim != 2 or len(cells) != len(target):
        raise ValueError("margin arrays have incompatible shapes")
    residual = target - prediction
    global_margin = np.maximum(
        0.0, np.quantile(residual, quantile, axis=0, method="higher"),
    )
    margins = {}
    for cell in np.unique(cells):
        local = np.maximum(0.0, np.quantile(
            residual[cells == cell], quantile, axis=0, method="higher",
        ))
        margins[str(cell)] = shrinkage * global_margin + (1.0 - shrinkage) * local
    return global_margin, margins


@dataclass(frozen=True)
class ScaleThresholdSelection:
    threshold: float | None
    records: tuple[dict, ...]


@dataclass(frozen=True)
class PerCellScaleSelection:
    quantiles: dict[str, float] | None
    thresholds: dict[str, float] | None
    accepted: int
    unsafe_acceptances: int
    cell_acceptances: dict[str, int]
    records: dict[str, tuple[dict, ...]]


@dataclass(frozen=True)
class PerCellGroupScaleSelection:
    quantiles: dict[str, float] | None
    thresholds: dict[str, float] | None
    accepted: int
    unsafe_acceptances: int
    group_failures: int
    cell_acceptances: dict[str, int]
    cell_group_acceptances: dict[str, int]
    cell_risk_upper: dict[str, float]
    records: dict[str, tuple[dict, ...]]


@dataclass(frozen=True)
class FixedThresholdGroupCertificate:
    """Risk accounting for one threshold vector frozen before certification.

    ``policy`` treats every independently seeded trajectory as a trial.  The
    conditional ``learned`` rate instead conditions on a trajectory containing
    at least one learned candidate commit.  These are deliberately distinct:
    a conservative fallback can make the policy safe while leaving too little
    evidence about the learned branch itself.
    """

    policy_failures: int
    policy_trials: int
    policy_risk_upper: float
    learned_failures: int
    learned_trials: int
    learned_risk_upper: float
    cell_records: dict[str, dict]


def temperature_scaled_hazards(hazard_members, temperature: float) -> np.ndarray:
    """Apply one temperature to conditional first-exit hazards."""
    hazard = np.asarray(hazard_members, dtype=np.float64)
    if hazard.ndim != 3 or temperature <= 0.0:
        raise ValueError("hazards must be (ensemble, rows, horizon) and temperature positive")
    clipped = np.clip(hazard, 1.0e-6, 1.0 - 1.0e-6)
    logits = np.log(clipped / (1.0 - clipped)) / temperature
    return 1.0 / (1.0 + np.exp(-logits))


def temperature_scaled_cumulative(hazard_members, temperature: float) -> np.ndarray:
    """Return monotone cumulative exit probabilities after temperature scaling."""
    scaled = temperature_scaled_hazards(hazard_members, temperature)
    cumulative = 1.0 - np.cumprod(1.0 - scaled, axis=2)
    return cumulative


def select_zero_failure_scale_threshold(
    score, eligible, unsafe, cells, thresholds, *, minimum_cell_acceptances: int,
) -> ScaleThresholdSelection:
    score = np.asarray(score, dtype=np.float64)
    eligible = np.asarray(eligible, dtype=bool)
    unsafe = np.asarray(unsafe, dtype=bool)
    cells = np.asarray(cells)
    if not (score.shape == eligible.shape == unsafe.shape == cells.shape):
        raise ValueError("threshold arrays must have equal shape")
    chosen = None
    records = []
    for threshold in map(float, thresholds):
        accepted = eligible & (score <= threshold)
        failures = int(np.sum(accepted & unsafe))
        counts = {str(cell): int(np.sum(accepted & (cells == cell))) for cell in np.unique(cells)}
        opportunity = all(value >= minimum_cell_acceptances for value in counts.values())
        records.append({
            "threshold": threshold, "accepted": int(np.sum(accepted)),
            "unsafe_acceptances": failures, "cell_acceptances": counts,
            "opportunities_pass": bool(opportunity),
        })
        if failures:
            break
        if opportunity and np.any(accepted):
            chosen = threshold
    return ScaleThresholdSelection(chosen, tuple(records))


def select_zero_failure_cell_quantile(
    calibration_score, calibration_cells, score, eligible, unsafe, cells,
    quantiles, *, minimum_cell_acceptances: int,
) -> ScaleThresholdSelection:
    calibration_score = np.asarray(calibration_score, dtype=np.float64)
    calibration_cells = np.asarray(calibration_cells)
    score = np.asarray(score, dtype=np.float64)
    eligible = np.asarray(eligible, dtype=bool)
    unsafe = np.asarray(unsafe, dtype=bool)
    cells = np.asarray(cells)
    chosen, records = None, []
    unique_cells = np.unique(cells)
    if set(unique_cells) - set(np.unique(calibration_cells)):
        raise ValueError("every deployment cell needs calibration data")
    for quantile in map(float, quantiles):
        thresholds = {
            str(cell): float(np.quantile(
                calibration_score[calibration_cells == cell], quantile, method="higher",
            )) for cell in unique_cells
        }
        row_threshold = np.asarray([thresholds[str(cell)] for cell in cells])
        accepted = eligible & (score <= row_threshold)
        failures = int(np.sum(accepted & unsafe))
        counts = {str(cell): int(np.sum(accepted & (cells == cell))) for cell in unique_cells}
        opportunity = all(value >= minimum_cell_acceptances for value in counts.values())
        records.append({
            "quantile": quantile, "thresholds": thresholds,
            "accepted": int(np.sum(accepted)), "unsafe_acceptances": failures,
            "cell_acceptances": counts, "opportunities_pass": bool(opportunity),
        })
        if failures:
            break
        if opportunity and np.any(accepted):
            chosen = quantile
    return ScaleThresholdSelection(chosen, tuple(records))


def select_per_cell_zero_failure_quantiles(
    calibration_score, calibration_cells, score, eligible, unsafe, cells,
    quantiles, *, minimum_cell_acceptances: int,
) -> PerCellScaleSelection:
    calibration_score = np.asarray(calibration_score, dtype=np.float64)
    calibration_cells = np.asarray(calibration_cells)
    score = np.asarray(score, dtype=np.float64)
    eligible = np.asarray(eligible, dtype=bool)
    unsafe = np.asarray(unsafe, dtype=bool)
    cells = np.asarray(cells)
    chosen_quantiles, chosen_thresholds, records = {}, {}, {}
    for cell in np.unique(cells):
        calibration_values = calibration_score[calibration_cells == cell]
        mask = cells == cell
        chosen_quantile = chosen_threshold = None
        cell_records = []
        for quantile in map(float, quantiles):
            threshold = float(np.quantile(
                calibration_values, quantile, method="linear",
            ))
            accepted = mask & eligible & (score <= threshold)
            failures = int(np.sum(accepted & unsafe))
            count = int(np.sum(accepted))
            cell_records.append({
                "quantile": quantile, "threshold": threshold,
                "accepted": count, "unsafe_acceptances": failures,
            })
            if failures:
                break
            if count >= minimum_cell_acceptances:
                chosen_quantile, chosen_threshold = quantile, threshold
        records[str(cell)] = tuple(cell_records)
        if chosen_threshold is None:
            return PerCellScaleSelection(None, None, 0, 0, {}, records)
        chosen_quantiles[str(cell)] = chosen_quantile
        chosen_thresholds[str(cell)] = chosen_threshold
    row_threshold = np.asarray([chosen_thresholds[str(cell)] for cell in cells])
    accepted = eligible & (score <= row_threshold)
    counts = {str(cell): int(np.sum(accepted & (cells == cell))) for cell in np.unique(cells)}
    return PerCellScaleSelection(
        chosen_quantiles, chosen_thresholds, int(np.sum(accepted)),
        int(np.sum(accepted & unsafe)), counts, records,
    )


def select_per_cell_group_zero_failure_quantiles(
    calibration_score, calibration_cells, score, eligible, unsafe, cells, groups,
    quantiles, *, alpha: float, delta_total: float,
    minimum_cell_acceptances: int, minimum_cell_accepted_groups: int,
    require_zero_unsafe_acceptances: bool = False,
) -> PerCellGroupScaleSelection:
    """Freeze per-cell thresholds with trajectory groups as Bernoulli trials.

    Eight causal windows from one trajectory are deliberately not eight
    independent safety experiments: any unsafe accepted window makes the whole
    trajectory a failure.  The selector follows a fixed increasing sequence of
    calibration quantiles and stops at the first failed group-risk hypothesis.
    """
    calibration_score = np.asarray(calibration_score, dtype=np.float64)
    calibration_cells = np.asarray(calibration_cells)
    score = np.asarray(score, dtype=np.float64)
    eligible = np.asarray(eligible, dtype=bool)
    unsafe = np.asarray(unsafe, dtype=bool)
    cells = np.asarray(cells)
    groups = np.asarray(groups)
    if not (score.shape == eligible.shape == unsafe.shape == cells.shape == groups.shape):
        raise ValueError("deployment group-LTT arrays must have equal shape")
    unique_cells = np.unique(cells)
    if set(unique_cells) - set(np.unique(calibration_cells)):
        raise ValueError("every LTT cell needs calibration scores")
    if not 0.0 < alpha < 1.0 or not 0.0 < delta_total < 1.0:
        raise ValueError("group-LTT alpha and delta_total must lie in (0, 1)")
    delta = float(delta_total) / len(unique_cells)
    chosen_quantiles, chosen_thresholds, records = {}, {}, {}
    accepted_total = unsafe_total = failed_groups_total = 0
    counts, group_counts, upper_bounds = {}, {}, {}
    incomplete = False
    for cell in unique_cells:
        mask = cells == cell
        local_groups = np.unique(groups[mask])
        if not len(local_groups):
            raise ValueError("each LTT cell must contain trajectory groups")
        cell_records = []
        selected = None
        for quantile in map(float, quantiles):
            threshold = float(np.quantile(
                calibration_score[calibration_cells == cell], quantile, method="linear",
            ))
            accepted = mask & eligible & (score <= threshold)
            group_accepted = sum(bool(np.any(accepted[groups == group])) for group in local_groups)
            unsafe_acceptances = int(np.sum(accepted & unsafe))
            failures = sum(bool(np.any(accepted[groups == group] & unsafe[groups == group])) for group in local_groups)
            upper = clopper_pearson_upper(int(failures), len(local_groups), delta)
            record = {
                "quantile": quantile, "threshold": threshold,
                "accepted": int(np.sum(accepted)), "unsafe_acceptances": unsafe_acceptances,
                "accepted_groups": int(group_accepted),
                "group_failures": int(failures), "group_trials": int(len(local_groups)),
                "group_risk_upper": float(upper),
            }
            cell_records.append(record)
            # R2 policy selection has a strict selective-safety objective.  It
            # must never quietly inherit the older "risk below alpha" rule.
            if require_zero_unsafe_acceptances and unsafe_acceptances:
                break
            if upper > alpha:
                break
            if (record["accepted"] >= minimum_cell_acceptances
                    and record["accepted_groups"] >= minimum_cell_accepted_groups):
                selected = record
        records[str(cell)] = tuple(cell_records)
        if selected is None:
            # Keep auditing every cell.  A STOP must expose all scale blind
            # spots, rather than hiding later failures behind the first one.
            incomplete = True
            continue
        chosen_quantiles[str(cell)] = float(selected["quantile"])
        chosen_thresholds[str(cell)] = float(selected["threshold"])
        counts[str(cell)] = int(selected["accepted"])
        group_counts[str(cell)] = int(selected["accepted_groups"])
        upper_bounds[str(cell)] = float(selected["group_risk_upper"])
    if incomplete:
        return PerCellGroupScaleSelection(
            None, None, 0, 0, 0, {}, {}, {}, records,
        )
    row_threshold = np.asarray([chosen_thresholds[str(cell)] for cell in cells])
    accepted = eligible & (score <= row_threshold)
    for group in np.unique(groups):
        group_mask = groups == group
        failed_groups_total += bool(np.any(accepted[group_mask] & unsafe[group_mask]))
    return PerCellGroupScaleSelection(
        chosen_quantiles, chosen_thresholds, int(np.sum(accepted)),
        int(np.sum(accepted & unsafe)), int(failed_groups_total), counts,
        group_counts, upper_bounds, records,
    )


def fixed_threshold_group_certificate(
    accepted, unsafe, cells, groups, *, delta_total: float,
) -> FixedThresholdGroupCertificate:
    """Certify fixed thresholds without inspecting alternatives on LTT data.

    The caller must construct ``accepted`` using a threshold vector selected
    on a disjoint policy-selection split.  One trajectory is one Bernoulli
    trial, even when it yields many overlapping rollout windows.
    """
    accepted = np.asarray(accepted, dtype=bool)
    unsafe = np.asarray(unsafe, dtype=bool)
    cells = np.asarray(cells)
    groups = np.asarray(groups)
    if not (accepted.shape == unsafe.shape == cells.shape == groups.shape):
        raise ValueError("certificate arrays must have equal shape")
    if not 0.0 < float(delta_total) < 1.0:
        raise ValueError("delta_total must lie in (0, 1)")
    unique_cells = np.unique(cells)
    if not len(unique_cells):
        raise ValueError("certificate needs at least one cell")
    delta = float(delta_total) / len(unique_cells)
    policy_failures = policy_trials = learned_failures = learned_trials = 0
    records: dict[str, dict] = {}
    for cell in unique_cells:
        local_groups = np.unique(groups[cells == cell])
        policy = []
        learned = []
        for group in local_groups:
            mask = groups == group
            group_accepted = bool(np.any(accepted[mask]))
            group_failure = bool(np.any(accepted[mask] & unsafe[mask]))
            policy.append(group_failure)
            if group_accepted:
                learned.append(group_failure)
        failures = int(np.sum(policy))
        trials = int(len(policy))
        selected_failures = int(np.sum(learned))
        selected_trials = int(len(learned))
        policy_failures += failures
        policy_trials += trials
        learned_failures += selected_failures
        learned_trials += selected_trials
        records[str(cell)] = {
            "policy_failures": failures,
            "policy_trials": trials,
            "policy_risk_upper": float(clopper_pearson_upper(failures, trials, delta)),
            "learned_failures": selected_failures,
            "learned_trials": selected_trials,
            "learned_risk_upper": (
                float(clopper_pearson_upper(selected_failures, selected_trials, delta))
                if selected_trials else float("inf")
            ),
        }
    return FixedThresholdGroupCertificate(
        policy_failures, policy_trials,
        float(clopper_pearson_upper(policy_failures, policy_trials, float(delta_total))),
        learned_failures, learned_trials,
        (
            float(clopper_pearson_upper(learned_failures, learned_trials, float(delta_total)))
            if learned_trials else float("inf")
        ),
        records,
    )
