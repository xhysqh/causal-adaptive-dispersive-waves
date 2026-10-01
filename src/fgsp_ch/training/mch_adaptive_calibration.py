from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from fgsp_ch.models.mch_pinn_cnn_operator import MonotoneAdaptiveHead


@dataclass(frozen=True, slots=True)
class AdaptiveCalibrationResult:
    training_loss: float
    calibration_scale: float
    calibration_coverage: float
    calibration_size: int


def fit_and_calibrate_adaptive_head(
    head: MonotoneAdaptiveHead,
    training_indicators: np.ndarray,
    training_targets: np.ndarray,
    calibration_indicators: np.ndarray,
    calibration_targets: np.ndarray,
    *,
    epochs: int = 300,
    learning_rate: float = 2.0e-2,
    coverage: float = 0.9,
) -> AdaptiveCalibrationResult:
    """Fit on one trajectory group and one-sided calibrate on another.

    Targets are true local-error/tolerance ratios.  The conformal-style
    multiplicative quantile is restricted to at least one, so calibration can
    never make deployment less conservative than the fitted head.
    """
    arrays = [
        np.asarray(training_indicators, dtype=np.float32),
        np.asarray(training_targets, dtype=np.float32),
        np.asarray(calibration_indicators, dtype=np.float32),
        np.asarray(calibration_targets, dtype=np.float32),
    ]
    train_x, train_y, calibration_x, calibration_y = arrays
    if train_x.ndim != 2 or calibration_x.ndim != 2:
        raise ValueError("Adaptive indicators must be two-dimensional.")
    if train_x.shape[1] != head.raw_weights.numel() or calibration_x.shape[1] != train_x.shape[1]:
        raise ValueError("Adaptive indicator dimensions do not match the head.")
    if train_y.shape != (train_x.shape[0],) or calibration_y.shape != (calibration_x.shape[0],):
        raise ValueError("Adaptive targets must contain one value per row.")
    if min(train_x.shape[0], calibration_x.shape[0], epochs) < 1:
        raise ValueError("Training, calibration, and epochs must be non-empty.")
    if not 0.0 < coverage < 1.0 or learning_rate <= 0.0:
        raise ValueError("Invalid adaptive calibration settings.")
    if any(not np.all(np.isfinite(value)) for value in arrays) or np.any(train_y < 0.0) or np.any(calibration_y < 0.0):
        raise ValueError("Adaptive calibration data must be finite and non-negative.")
    head.train()
    head.set_calibration_scale(1.0)
    x = torch.from_numpy(train_x)
    target = torch.from_numpy(train_y)
    optimizer = torch.optim.Adam(head.parameters(), lr=learning_rate)
    final_loss = torch.tensor(float("inf"))
    for _ in range(epochs):
        optimizer.zero_grad()
        prediction, _ = head(x)
        log_error = torch.log1p(prediction) - torch.log1p(target)
        # Underprediction is more costly because it can cause an unsafe accept.
        weights = torch.where(log_error < 0.0, 4.0, 1.0)
        final_loss = torch.mean(weights * log_error.square())
        final_loss.backward()
        optimizer.step()
    head.eval()
    with torch.no_grad():
        prediction, _ = head(torch.from_numpy(calibration_x))
    predicted = prediction.cpu().numpy()
    ratios = calibration_y / np.maximum(predicted, np.finfo(np.float32).eps)
    rank = min(
        ratios.size - 1,
        int(np.ceil((ratios.size + 1) * coverage)) - 1,
    )
    scale = max(1.0, float(np.sort(ratios)[rank]))
    head.set_calibration_scale(scale)
    with torch.no_grad():
        calibrated, _ = head(torch.from_numpy(calibration_x))
    achieved = float(np.mean(calibrated.cpu().numpy() + 1e-7 >= calibration_y))
    return AdaptiveCalibrationResult(
        float(final_loss.item()), scale, achieved, int(calibration_y.size)
    )
