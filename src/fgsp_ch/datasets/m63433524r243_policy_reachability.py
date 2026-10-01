"""Policy-consistent reachable-risk targets for R2.4.3.

Collection policies may visit a state, but they are never allowed to define
the future used by its label.  Every labelled branch commits the candidate at
step zero and then follows one immutable target controller.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fgsp_ch.datasets.m634335_first_exit import FirstExitTargets, first_exit_targets


REGISTERED_HORIZONS = (1, 2, 4)
TARGET_POLICY_NAME = "candidate_then_frozen_r23"


@dataclass(frozen=True, slots=True)
class PolicyReachabilityTargets:
    first_exit: FirstExitTargets
    registered_exit: np.ndarray
    registered_severity: np.ndarray


def target_policy_commit(*, branch_step: int, frozen_controller_accepts: bool) -> bool:
    """Candidate is forced once; all later actions come from the frozen controller."""
    if branch_step < 0:
        raise ValueError("branch_step must be non-negative")
    return True if branch_step == 0 else bool(frozen_controller_accepts)


def policy_reachability_targets(
    risk_components: np.ndarray,
    *,
    target_actions: np.ndarray,
    horizons: tuple[int, ...] = REGISTERED_HORIZONS,
    exit_threshold: float = 1.0,
    severe_threshold: float = 2.0,
) -> PolicyReachabilityTargets:
    """Create first-exit targets from one explicitly recorded target-policy branch."""
    components = np.asarray(risk_components, dtype=np.float64)
    actions = np.asarray(target_actions, dtype=bool)
    if components.ndim != 2 or actions.shape != (len(components),):
        raise ValueError("target actions must align with the branch risk components")
    if not len(actions) or not bool(actions[0]):
        raise ValueError("a policy-conditioned candidate branch must commit candidate first")
    requested = tuple(map(int, horizons))
    if tuple(sorted(set(requested))) != requested or requested[0] < 1:
        raise ValueError("horizons must be unique, positive and increasing")
    if requested[-1] > len(components):
        raise ValueError("registered horizons exceed the labelled branch")
    first_exit = first_exit_targets(
        components,
        exit_threshold=exit_threshold,
        severe_threshold=severe_threshold,
    )
    columns = np.asarray(requested, dtype=np.int64) - 1
    return PolicyReachabilityTargets(
        first_exit=first_exit,
        registered_exit=first_exit.cumulative_exit[columns],
        registered_severity=first_exit.cumulative_severity[columns],
    )
