"""Equation-name-free response layer for the CMAME-U2.2 stress audit.

The layer does not advance a PDE and has no trainable parameters.  It decides
when the expensive alternative candidates or a verification probe are needed
from pre-commit symbol/state diagnostics.  All releases remain subject to the
existing analytical envelope and online certificate.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray


def trait_response_vector(
    symbol_coordinates, *, tail_ratio: float, aliasing_defect: float,
    unresolved_increment: float, linear_phase: float, nonlinear_phase: float,
    localized_sharpness: float,
) -> NDArray[np.float64]:
    """Build dimensionless coordinates available before an action is committed."""
    symbol = np.asarray(symbol_coordinates, dtype=np.float64)
    if symbol.ndim != 1 or not np.all(np.isfinite(symbol)):
        raise ValueError("U2.2 symbol coordinates must be a finite vector")
    positive = np.asarray(
        [tail_ratio, aliasing_defect, unresolved_increment], dtype=np.float64
    )
    if np.any(~np.isfinite(positive)) or np.any(positive < 0):
        raise ValueError("U2.2 resolution indicators must be finite and nonnegative")
    phase_ratio = float(nonlinear_phase) / max(float(linear_phase), 1.0e-14)
    result = np.r_[
        symbol,
        np.log10(np.maximum(positive, 1.0e-16)),
        np.log1p(max(phase_ratio, 0.0)),
        float(localized_sharpness),
    ]
    if not np.all(np.isfinite(result)):
        raise ValueError("U2.2 trait-response vector is not finite")
    return np.asarray(result, dtype=np.float64)


@dataclass(frozen=True, slots=True)
class TraitResponseConfig:
    drift_scale: tuple[float, ...]
    soft_drift: float
    hard_drift: float
    lazy_hold_budget_fraction: float
    tail_trigger: float
    aliasing_trigger: float
    stable_prefixes_for_coarsening: int
    coarsening_budget_fraction: float
    coarsening_tail_limit: float
    coarsening_aliasing_limit: float

    def __post_init__(self):
        scale = np.asarray(self.drift_scale, dtype=np.float64)
        if scale.ndim != 1 or not len(scale) or np.any(~np.isfinite(scale)) or np.any(scale <= 0):
            raise ValueError("U2.2 drift scales must be finite and positive")
        if not 0 < self.soft_drift < self.hard_drift:
            raise ValueError("U2.2 drift thresholds must be ordered and positive")
        fractions = (
            self.lazy_hold_budget_fraction, self.coarsening_budget_fraction,
            self.tail_trigger, self.aliasing_trigger,
            self.coarsening_tail_limit, self.coarsening_aliasing_limit,
        )
        if any(not np.isfinite(value) or value <= 0 for value in fractions):
            raise ValueError("U2.2 response thresholds must be finite and positive")
        if self.stable_prefixes_for_coarsening < 1:
            raise ValueError("U2.2 coarsening hysteresis must be positive")


@dataclass(frozen=True, slots=True)
class TraitResponseDecision:
    drift: float
    force_probe: bool
    hard_change: bool
    evaluate_alternatives: bool
    stable_prefixes: int
    hold_budget_utilization: float


@dataclass(slots=True)
class TraitResponsiveState:
    config: TraitResponseConfig
    previous_vector: NDArray[np.float64] | None = None
    stable_prefixes: int = 0

    def observe(
        self, vector, *, hold_error: float, local_budget: float,
        tail_ratio: float, aliasing_defect: float,
    ) -> TraitResponseDecision:
        current = np.asarray(vector, dtype=np.float64)
        scale = np.asarray(self.config.drift_scale, dtype=np.float64)
        if current.shape != scale.shape or not np.all(np.isfinite(current)):
            raise ValueError("U2.2 response vector does not match its frozen scale")
        if self.previous_vector is None:
            drift = 0.0
        else:
            drift = float(np.sqrt(np.mean(((current - self.previous_vector) / scale) ** 2)))
        utilization = float(hold_error) / max(float(local_budget), 1.0e-16)
        hard = drift >= self.config.hard_drift
        force_probe = drift >= self.config.soft_drift
        alternatives = bool(
            hard
            or utilization >= self.config.lazy_hold_budget_fraction
            or tail_ratio >= self.config.tail_trigger
            or aliasing_defect >= self.config.aliasing_trigger
        )
        stable = bool(
            drift < 0.5 * self.config.soft_drift
            and tail_ratio <= self.config.coarsening_tail_limit
            and aliasing_defect <= self.config.coarsening_aliasing_limit
        )
        self.stable_prefixes = self.stable_prefixes + 1 if stable else 0
        self.previous_vector = current.copy()
        return TraitResponseDecision(
            drift, force_probe, hard, alternatives,
            self.stable_prefixes, utilization,
        )

    def coarsening_admissible(
        self, *, projection_defect: float, local_budget: float,
        current_points: int, minimum_points: int,
    ) -> bool:
        return bool(
            current_points > minimum_points
            and self.stable_prefixes >= self.config.stable_prefixes_for_coarsening
            and np.isfinite(projection_defect) and projection_defect >= 0
            and projection_defect
            <= self.config.coarsening_budget_fraction * max(local_budget, 1.0e-16)
        )


def trait_response_config(mapping) -> TraitResponseConfig:
    """Construct a frozen configuration from YAML-compatible data."""
    return TraitResponseConfig(
        drift_scale=tuple(map(float, mapping["drift_scale"])),
        soft_drift=float(mapping["soft_drift"]),
        hard_drift=float(mapping["hard_drift"]),
        lazy_hold_budget_fraction=float(mapping["lazy_hold_budget_fraction"]),
        tail_trigger=float(mapping["tail_trigger"]),
        aliasing_trigger=float(mapping["aliasing_trigger"]),
        stable_prefixes_for_coarsening=int(mapping["stable_prefixes_for_coarsening"]),
        coarsening_budget_fraction=float(mapping["coarsening_budget_fraction"]),
        coarsening_tail_limit=float(mapping["coarsening_tail_limit"]),
        coarsening_aliasing_limit=float(mapping["coarsening_aliasing_limit"]),
    )
