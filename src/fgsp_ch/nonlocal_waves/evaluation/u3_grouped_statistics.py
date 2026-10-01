"""Equation/family summaries for frozen multi-equation evidence."""

from __future__ import annotations

from collections import Counter

import numpy as np


def terminal_rows(rows: list[dict]) -> list[dict]:
    maximum = {}
    for row in rows:
        key = (row["source_id"], row["arm"], row["group_id"])
        maximum[key] = max(maximum.get(key, -1), int(row["prefix"]))
    return [row for row in rows if int(row["prefix"]) == maximum[
        (row["source_id"], row["arm"], row["group_id"])
    ]]


def equation_summary(rows: list[dict], proposed_arms: dict[str, str]) -> list[dict]:
    selected = [row for row in rows if row["arm"] == proposed_arms[row["source_id"]]]
    terminal = terminal_rows(selected)
    result = []
    for equation in sorted({row["equation"] for row in selected}):
        all_rows = [row for row in selected if row["equation"] == equation]
        final = [row for row in terminal if row["equation"] == equation]
        actions = Counter(row["action"] for row in all_rows)
        result.append({
            "equation": equation,
            "trajectory_groups": len({row["group_id"] for row in all_rows}),
            "causal_rows": len(all_rows),
            "terminal_tolerance_coverage": float(np.mean([
                row["tolerance_met"] for row in final
            ])),
            "median_terminal_error_over_tolerance": float(np.median([
                row["error_over_tolerance"] for row in final
            ])),
            "p95_terminal_error_over_tolerance": float(np.quantile([
                row["error_over_tolerance"] for row in final
            ], 0.95)),
            "maximum_invariant_drift": max(row["invariant_drift"] for row in all_rows),
            "hold": actions["hold"],
            "shrink_time": actions["shrink_time"],
            "refine_space": actions["refine_space"],
            "probe_count": sum(row["probe_count"] for row in all_rows),
            "abstentions": sum(row["abstained"] for row in all_rows),
        })
    return result


def paired_tolerance_monotone_fraction(rows: list[dict], proposed_arms: dict[str, str]) -> float:
    selected = terminal_rows([
        row for row in rows if row["arm"] == proposed_arms[row["source_id"]]
    ])
    groups = {}
    for row in selected:
        groups.setdefault(row["physical_group"], []).append(row)
    checks = []
    for group in groups.values():
        ordered = sorted(group, key=lambda row: row["tolerance"], reverse=True)
        for loose, tight in zip(ordered, ordered[1:]):
            checks.append(tight["relative_h1"] <= loose["relative_h1"] * 1.05 + 1e-15)
    return float(np.mean(checks)) if checks else 1.0
