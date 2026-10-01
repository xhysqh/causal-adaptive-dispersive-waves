"""Trajectory-group statistics for correlated adaptive-solver prefixes."""

from __future__ import annotations

import numpy as np


def grouped_median_log_ratio_interval(
    ratios,
    physical_groups,
    *,
    samples: int = 4000,
    confidence: float = 0.95,
    seed: int = 20260831,
):
    """Bootstrap the median after collapsing correlated rows per trajectory.

    Prefixes and tolerance levels from one physical trajectory are not treated
    as independent replicates.  Each trajectory contributes one median log10
    ratio before group resampling.
    """
    values = np.asarray(ratios, dtype=np.float64)
    groups = np.asarray(physical_groups).astype(str)
    if values.ndim != 1 or groups.shape != values.shape or not len(values):
        raise ValueError("grouped ratios must be non-empty aligned vectors")
    if np.any(~np.isfinite(values)) or np.any(values <= 0):
        raise ValueError("grouped ratios must be finite and positive")
    if samples < 100 or not 0.5 < confidence < 1.0:
        raise ValueError("invalid grouped-bootstrap configuration")
    unique = np.unique(groups)
    collapsed = np.asarray([
        np.median(np.log10(values[groups == group])) for group in unique
    ])
    rng = np.random.default_rng(seed)
    draws = rng.choice(collapsed, size=(samples, len(collapsed)), replace=True)
    bootstrap = np.median(draws, axis=1)
    alpha = (1.0 - confidence) / 2.0
    interval = np.quantile(bootstrap, [alpha, 1.0 - alpha])
    return {
        "physical_trajectory_groups": int(len(unique)),
        "median_log10_ratio": float(np.median(collapsed)),
        "median_ratio": float(10.0 ** np.median(collapsed)),
        "group_bootstrap_log10_interval": [float(interval[0]), float(interval[1])],
        "confidence": float(confidence),
        "bootstrap_samples": int(samples),
    }
