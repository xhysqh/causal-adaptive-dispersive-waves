from __future__ import annotations

from dataclasses import dataclass

import numpy as np


RISK_COMPONENT_ORDER = (
    "shadow_defect",
    "direction",
    "mass",
    "energy",
    "cfl",
    "patch_migration",
)
HARD_COMPONENT_INDEX = (2, 3, 4, 5)


@dataclass(frozen=True)
class FirstExitTargets:
    step_risk: np.ndarray
    exit_event: np.ndarray
    at_risk: np.ndarray
    first_exit: int
    cumulative_exit: np.ndarray
    cumulative_severity: np.ndarray
    severe_event: np.ndarray
    hard_violation: np.ndarray


def physical_risk_components(
    diagnostic,
    defect: float,
    direction: float,
    label: dict,
) -> np.ndarray:
    """Return the six dimensionless physical/shadow defect ratios."""
    direction_risk = max(
        0.0,
        (1.0 - float(direction))
        / (1.0 - float(label["minimum_direction_cosine"])),
    )
    values = np.asarray((
        float(defect) / float(label["maximum_shadow_defect_ratio"]),
        direction_risk,
        float(diagnostic.mass_drift) / float(label["maximum_mass_drift"]),
        float(diagnostic.relative_energy_drift)
        / float(label["maximum_relative_energy_drift"]),
        float(diagnostic.maximum_cfl) / float(label["maximum_cfl"]),
        (
            float(diagnostic.patch_migration_to_step_increment)
            / float(label["maximum_patch_migration_to_step_increment"])
            if diagnostic.patch_changed else 0.0
        ),
    ), dtype=np.float64)
    # A non-finite diagnostic is an unconditionally unsafe hard failure.
    return np.where(np.isfinite(values), values, np.inf)


def first_exit_targets(
    risk_components: np.ndarray,
    *,
    exit_threshold: float = 1.0,
    severe_threshold: float = 2.0,
) -> FirstExitTargets:
    components = np.asarray(risk_components, dtype=np.float64)
    if components.ndim != 2 or components.shape[1] != len(RISK_COMPONENT_ORDER):
        raise ValueError("risk_components must have shape (horizon, 6)")
    if not 0.0 < exit_threshold < severe_threshold:
        raise ValueError("thresholds must satisfy 0 < exit < severe")
    step_risk = np.max(components, axis=1)
    raw_exit = step_risk > exit_threshold
    previous_exit = np.concatenate((
        np.asarray([False]), np.maximum.accumulate(raw_exit[:-1]),
    ))
    at_risk = ~previous_exit
    exit_event = raw_exit & at_risk
    exit_indices = np.flatnonzero(exit_event)
    first_exit = int(exit_indices[0] + 1) if exit_indices.size else len(step_risk) + 1
    cumulative_exit = np.maximum.accumulate(raw_exit)
    cumulative_severity = np.maximum.accumulate(step_risk)
    severe_event = cumulative_severity > severe_threshold
    hard_violation = np.max(components[:, HARD_COMPONENT_INDEX], axis=1) > exit_threshold
    return FirstExitTargets(
        step_risk=step_risk,
        exit_event=exit_event,
        at_risk=at_risk,
        first_exit=first_exit,
        cumulative_exit=cumulative_exit,
        cumulative_severity=cumulative_severity,
        severe_event=severe_event,
        hard_violation=hard_violation,
    )


def horizon_columns(maximum_horizon: int, horizons=(2, 4, 8)) -> np.ndarray:
    columns = np.asarray(tuple(int(value) - 1 for value in horizons), dtype=np.int64)
    if np.any(columns < 0) or np.any(columns >= maximum_horizon):
        raise ValueError("requested horizons exceed the labelled rollout")
    return columns
