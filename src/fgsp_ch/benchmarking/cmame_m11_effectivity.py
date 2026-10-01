"""Calibration and audit metrics for CMAME-M1.1."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import nnls
from scipy.stats import spearmanr


def fit_nonnegative_estimator(
    features: NDArray[np.floating], truth: NDArray[np.floating], *, coverage_quantile: float
) -> tuple[NDArray[np.floating], float]:
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(truth, dtype=np.float64)
    if x.ndim != 2 or y.shape != (x.shape[0],) or x.shape[0] < x.shape[1]:
        raise ValueError("calibration features and truth have incompatible shapes")
    if not 0.5 <= coverage_quantile <= 1.0 or np.any(x < 0.0) or np.any(y < 0.0):
        raise ValueError("invalid nonnegative calibration data")
    weights, _ = nnls(x, y)
    prediction = x @ weights
    positive = prediction > np.finfo(float).eps
    if not np.any(positive):
        # A transparent non-degenerate fallback, still independent of audit labels.
        weights = np.ones(x.shape[1], dtype=np.float64)
        prediction = x @ weights
        positive = prediction > np.finfo(float).eps
    ratios = y[positive] / prediction[positive]
    safety = max(1.0, float(np.quantile(ratios, coverage_quantile)))
    return np.asarray(weights), safety


def calibrate_scalar_safety(
    raw_estimate: NDArray[np.floating], truth: NDArray[np.floating], *, coverage_quantile: float
) -> float:
    """Calibrate one scalar while retaining representation-invariant unit weights."""
    eta = np.asarray(raw_estimate, dtype=np.float64)
    error = np.asarray(truth, dtype=np.float64)
    if eta.shape != error.shape or eta.ndim != 1 or not eta.size:
        raise ValueError("raw estimates and truth must be aligned nonempty vectors")
    if np.any(eta <= 0.0) or np.any(error < 0.0) or not 0.5 <= coverage_quantile <= 1.0:
        raise ValueError("scalar calibration requires positive estimates and nonnegative truth")
    return max(1.0, float(np.quantile(error / eta, coverage_quantile)))


def estimator_metrics(estimate: NDArray[np.floating], truth: NDArray[np.floating]) -> dict[str, float]:
    eta = np.asarray(estimate, dtype=np.float64)
    error = np.asarray(truth, dtype=np.float64)
    if eta.shape != error.shape or eta.ndim != 1 or not eta.size:
        raise ValueError("estimate and truth must be aligned nonempty vectors")
    safe_error = np.maximum(error, np.finfo(float).eps)
    effectivity = eta / safe_error
    correlation = spearmanr(eta, error).statistic if eta.size > 1 else 1.0
    if not np.isfinite(correlation):
        correlation = 0.0
    return {
        "coverage": float(np.mean(eta >= error)),
        "spearman": float(correlation),
        "median_effectivity": float(np.median(effectivity)),
        "p95_effectivity": float(np.quantile(effectivity, 0.95)),
        "maximum_effectivity": float(np.max(effectivity)),
        "minimum_effectivity": float(np.min(effectivity)),
    }
