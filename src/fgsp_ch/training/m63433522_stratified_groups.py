from __future__ import annotations

"""Trajectory-level sampling for the scale-risk certificate.

All windows from a trajectory stay together.  The routines deliberately keep a
copy of every fitting trajectory in every ensemble member and only add
within-stratum bootstrap multiplicities.  This prevents a small bootstrap
sample from accidentally dropping an entire grid/time-step regime.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GroupStrata:
    groups: np.ndarray
    cells: dict[str, str]
    exit_classes: dict[str, str]

    def key(self, group: str) -> tuple[str, str]:
        return self.cells[str(group)], self.exit_classes[str(group)]


def _exit_class(cumulative: np.ndarray) -> str:
    """Classify a trajectory by its earliest exit over all causal windows."""
    horizon_exit = np.any(np.asarray(cumulative, dtype=bool), axis=0)
    if horizon_exit[0]:
        return "exit_1"
    if horizon_exit[min(3, len(horizon_exit) - 1)]:
        return "exit_2_4"
    if horizon_exit[-1]:
        return "exit_5_8"
    return "safe_8"


def build_group_strata(
    groups: np.ndarray,
    cells: np.ndarray,
    cumulative_exit: np.ndarray,
    mask: np.ndarray,
) -> GroupStrata:
    """Create one scale cell and first-exit class for every selected group."""
    groups, cells = np.asarray(groups), np.asarray(cells)
    cumulative_exit, mask = np.asarray(cumulative_exit), np.asarray(mask, dtype=bool)
    if not (len(groups) == len(cells) == len(cumulative_exit) == len(mask)):
        raise ValueError("group stratum inputs must have equal row counts")
    selected = np.unique(groups[mask])
    cell_map: dict[str, str] = {}
    exit_map: dict[str, str] = {}
    for group in selected:
        rows = mask & (groups == group)
        group_cells = np.unique(cells[rows])
        if len(group_cells) != 1:
            raise ValueError("a trajectory group must belong to one scale cell")
        cell_map[str(group)] = str(group_cells[0])
        exit_map[str(group)] = _exit_class(cumulative_exit[rows])
    return GroupStrata(selected.astype(str), cell_map, exit_map)


def stratified_internal_holdout(
    strata: GroupStrata,
    fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Hold out groups within each cell while retaining every cell in fitting.

    When a cell has more than one first-exit class, the chosen holdout starts
    with a round-robin draw across classes.  A singleton class is never allowed
    to consume the last fitting group of that cell.
    """
    if not 0.0 < fraction < 0.5:
        raise ValueError("internal validation fraction must lie in (0, 0.5)")
    rng = np.random.default_rng(seed)
    by_cell: dict[str, list[str]] = {}
    for group in strata.groups:
        by_cell.setdefault(strata.cells[str(group)], []).append(str(group))
    held_out: list[str] = []
    for cell, cell_groups in sorted(by_cell.items()):
        target = min(len(cell_groups) - 1, max(1, int(round(len(cell_groups) * fraction))))
        by_class: dict[str, list[str]] = {}
        for group in cell_groups:
            by_class.setdefault(strata.exit_classes[group], []).append(group)
        candidates: list[str] = []
        for exit_class in sorted(by_class):
            members = by_class[exit_class].copy()
            rng.shuffle(members)
            candidates.extend(members)
        # Round-robin order above makes early choices class-diverse.  Shuffle
        # only the remaining tail so the procedure is deterministic yet fair.
        if len(candidates) > len(by_class):
            head, tail = candidates[:len(by_class)], candidates[len(by_class):]
            rng.shuffle(tail)
            candidates = head + tail
        held_out.extend(candidates[:target])
    holdout = np.asarray(sorted(set(held_out)), dtype=str)
    fit = np.asarray(sorted(set(strata.groups) - set(holdout)), dtype=str)
    if not len(fit) or not len(holdout):
        raise ValueError("stratified split produced an empty partition")
    return fit, holdout


def coverage_bootstrap_groups(
    fit_groups: np.ndarray,
    strata: GroupStrata,
    seed: int,
) -> np.ndarray:
    """Return baseline coverage plus in-cell bootstrap multiplicities.

    Sampling is by cell, not raw row, so all rollout starts of a trajectory
    share exactly the same multiplicity and every member retains all nine
    registered scale cells.
    """
    fit_groups = np.asarray(fit_groups, dtype=str)
    if not len(fit_groups):
        raise ValueError("fit_groups must be nonempty")
    rng = np.random.default_rng(seed)
    by_cell: dict[str, list[str]] = {}
    for group in fit_groups:
        by_cell.setdefault(strata.cells[str(group)], []).append(str(group))
    extras: list[str] = []
    for _, members in sorted(by_cell.items()):
        extras.extend(rng.choice(members, size=len(members), replace=True).tolist())
    return np.concatenate((fit_groups, np.asarray(extras, dtype=str)))


def inverse_exit_class_weights(groups: np.ndarray, strata: GroupStrata) -> np.ndarray:
    """Mildly balance rare first-exit classes without changing cell mass."""
    groups = np.asarray(groups, dtype=str)
    keys = [strata.key(group) for group in groups]
    counts: dict[tuple[str, str], int] = {}
    for key in keys:
        counts[key] = counts.get(key, 0) + 1
    cell_counts: dict[str, int] = {}
    for cell, _ in keys:
        cell_counts[cell] = cell_counts.get(cell, 0) + 1
    weights = np.asarray([
        cell_counts[cell] / max(len({key for key in counts if key[0] == cell}) * counts[(cell, exit_class)], 1)
        for cell, exit_class in keys
    ], dtype=np.float32)
    return weights / max(float(np.mean(weights)), np.finfo(np.float32).eps)
