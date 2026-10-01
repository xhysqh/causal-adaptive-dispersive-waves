from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from fgsp_ch.atlas.mch_atlas import AtlasKind
from fgsp_ch.equations.modified_ch import ModifiedCH
from fgsp_ch.geometry.metric import a_inner, a_norm, discrete_energy
from fgsp_ch.solvers.mch_energy_projected import project_to_mass_h1


@dataclass(frozen=True, slots=True)
class MCHCorrectionGateConfig:
    maximum_correction_ratio: float = 0.5
    maximum_correction_to_embedded_error: float = 1.0e6
    maximum_relative_ensemble_uncertainty: float = 0.75
    minimum_embedded_alignment: float = -0.25
    minimum_confidence: float = 0.55
    maximum_ood_distance: float = 6.0
    maximum_measure_purity: float = 1.0e-6
    minimum_separation_cells: float = 1.5
    maximum_projection_scale_change: float = 0.25
    maximum_spectral_tail_factor: float = 4.0
    spectral_tail_fraction: float = 0.25
    backtracking: tuple[float, ...] = (1.0, 0.5, 0.25, 0.125)
    epsilon: float = 1.0e-12

    def __post_init__(self) -> None:
        positive = (
            self.maximum_correction_ratio,
            self.maximum_correction_to_embedded_error,
            self.maximum_relative_ensemble_uncertainty,
            self.minimum_confidence,
            self.maximum_ood_distance,
            self.maximum_measure_purity,
            self.minimum_separation_cells,
            self.maximum_projection_scale_change,
            self.maximum_spectral_tail_factor,
            self.spectral_tail_fraction,
            self.epsilon,
        )
        if any(not np.isfinite(value) or value <= 0.0 for value in positive):
            raise ValueError("MCH correction-gate thresholds must be positive and finite.")
        if self.minimum_confidence >= 1.0 or self.spectral_tail_fraction >= 1.0:
            raise ValueError("Confidence and spectral-tail fraction must be below one.")
        if not -1.0 <= self.minimum_embedded_alignment <= 1.0:
            raise ValueError("Embedded alignment threshold must lie in [-1, 1].")
        if not self.backtracking or any(
            not 0.0 < value <= 1.0 for value in self.backtracking
        ):
            raise ValueError("Backtracking factors must lie in (0, 1].")


@dataclass(frozen=True, slots=True)
class MCHCorrectionCertificate:
    state: NDArray[np.floating]
    correction: NDArray[np.floating]
    accepted: bool
    reason: str
    damping: float
    raw_correction_ratio: float
    mass_error: float
    energy_error: float
    projection_scale: float


def _tail_fraction(values: NDArray[np.floating], fraction: float) -> float:
    power = np.abs(np.fft.fft(values)) ** 2
    modes = np.abs(np.fft.fftfreq(values.size))
    cutoff = (1.0 - fraction) * float(np.max(modes))
    total = float(np.sum(power))
    return 0.0 if total == 0.0 else float(np.sum(power[modes >= cutoff]) / total)


def certify_mch_correction(
    equation: ModifiedCH,
    current: NDArray[np.floating],
    baseline: NDArray[np.floating],
    raw_correction: NDArray[np.floating],
    *,
    atlas: AtlasKind,
    confidence: float,
    ood_distance: float,
    measure_purity: float,
    separation_cells: float,
    embedded_error_norm: float | None = None,
    embedded_direction: NDArray[np.floating] | None = None,
    ensemble_relative_uncertainty: float = 0.0,
    config: MCHCorrectionGateConfig,
    enforce_confidence: bool = True,
    enforce_conservation: bool = True,
) -> MCHCorrectionCertificate:
    """Return a causal certificate or the exact baseline fallback."""
    for values in (current, baseline, raw_correction):
        equation.grid.validate_state(values)
    increment_norm = a_norm(baseline - current, equation.helmholtz)
    raw_norm = a_norm(raw_correction, equation.helmholtz)
    ratio = raw_norm / (increment_norm + config.epsilon)

    def fallback(reason: str) -> MCHCorrectionCertificate:
        return MCHCorrectionCertificate(
            baseline.copy(), np.zeros_like(baseline), False, reason, 0.0, ratio,
            0.0, 0.0, 1.0,
        )

    if not np.all(np.isfinite(raw_correction)):
        return fallback("nonfinite_correction")
    if enforce_confidence:
        if confidence < config.minimum_confidence:
            return fallback("low_confidence")
        if ood_distance > config.maximum_ood_distance:
            return fallback("outside_training_support")
        if ensemble_relative_uncertainty > config.maximum_relative_ensemble_uncertainty:
            return fallback("ensemble_disagreement")
        if (
            embedded_error_norm is not None
            and raw_norm / (embedded_error_norm + config.epsilon)
            > config.maximum_correction_to_embedded_error
        ):
            return fallback("correction_exceeds_embedded_support")
        if embedded_direction is not None:
            equation.grid.validate_state(embedded_direction)
            embedded_norm = a_norm(embedded_direction, equation.helmholtz)
            if raw_norm > config.epsilon and embedded_norm > config.epsilon:
                alignment = a_inner(
                    raw_correction, embedded_direction, equation.helmholtz
                ) / (raw_norm * embedded_norm)
                if alignment < config.minimum_embedded_alignment:
                    return fallback("embedded_direction_disagreement")
        if atlas == AtlasKind.SEPARATED and measure_purity > config.maximum_measure_purity:
            return fallback("mixed_measure_field")
        if atlas != AtlasKind.FIELD and separation_cells < config.minimum_separation_cells:
            return fallback("collision_floor")
    cap = min(
        1.0,
        config.maximum_correction_ratio * (increment_norm + config.epsilon)
        / (raw_norm + config.epsilon),
    )
    target_mass = float(equation.grid.h * np.sum(current))
    target_energy = discrete_energy(current, equation.helmholtz)
    baseline_tail = _tail_fraction(baseline, config.spectral_tail_fraction)
    for factor in config.backtracking:
        damping = float(cap * factor)
        trial = baseline + damping * raw_correction
        try:
            if enforce_conservation:
                projected = project_to_mass_h1(
                    trial, target_mass, target_energy, equation
                )
                state = projected.state
                scale = projected.scale
                if abs(scale - 1.0) > config.maximum_projection_scale_change:
                    continue
            else:
                state = np.asarray(trial)
                scale = 1.0
        except (ValueError, FloatingPointError):
            continue
        if not np.all(np.isfinite(state)):
            continue
        tail = _tail_fraction(state, config.spectral_tail_fraction)
        if tail > config.maximum_spectral_tail_factor * (
            baseline_tail + config.epsilon
        ):
            continue
        mass_error = abs(float(equation.grid.h * np.sum(state)) - target_mass)
        energy_error = abs(discrete_energy(state, equation.helmholtz) - target_energy)
        return MCHCorrectionCertificate(
            np.asarray(state), np.asarray(state - baseline), True, "accepted",
            damping, ratio, mass_error, energy_error, float(scale),
        )
    return fallback("certificate_rejected")
