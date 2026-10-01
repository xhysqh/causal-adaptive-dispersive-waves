"""Equation-name-free diagnostics for unseen power-law dispersive waves."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fgsp_ch.nonlocal_waves.adaptivity.prospective_causal_operator import prospective_features
from fgsp_ch.nonlocal_waves.operators.power_law_fourier import PowerLawFourierOperator


@dataclass(frozen=True, slots=True)
class PowerLawIndicators:
    tail_ratio: float
    aliasing_defect: float
    unresolved_increment: float
    maximum_linear_phase: float
    nonlinear_phase: float
    localized_sharpness: float


def power_law_indicators(values, grid, traits, dt, *, tail_fraction=.65):
    operator = PowerLawFourierOperator(grid, traits)
    spectrum = operator.spectrum(values)
    weights = 1 + operator.wave_numbers**2
    cutoff = max(1, int(tail_fraction*(len(spectrum)-1)))
    total = np.sum(weights*np.abs(spectrum)**2)
    tail = np.sum(weights[cutoff:]*np.abs(spectrum[cutoff:])**2)
    derivative = np.abs(operator.derivative(values))
    amplitude = max(float(np.max(np.abs(values))), 1e-14)
    return PowerLawIndicators(
        float(np.sqrt(tail/max(total, np.finfo(float).eps))),
        operator.aliasing_defect(values),
        operator.unresolved_nonlinear_increment(values, dt),
        traits.maximum_phase(operator.wave_numbers, dt),
        float(abs(dt)*traits.nonlinear_transport*amplitude*np.max(operator.wave_numbers)),
        float(np.max(derivative)/max(np.linalg.norm(derivative), np.finfo(float).eps)),
    )


def power_law_features(indicators, *, temporal, migration, invariant_drift,
                       invariant_limit, local_budget, remaining_budget_fraction,
                       remaining_time_fraction, relative_work, dt_factor,
                       resolution_factor):
    # The frozen network keeps its original coordinate system.  The symbol
    # order therefore enters through a dimensionless phase competition rather
    # than through an equation id or a newly fitted embedding.
    phase_competition = indicators.nonlinear_phase / max(
        indicators.maximum_linear_phase, 1e-12
    )
    return prospective_features(
        temporal=temporal,
        spatial=max(indicators.tail_ratio, indicators.unresolved_increment),
        representation=indicators.aliasing_defect,
        migration=migration,
        invariant_drift=invariant_drift,
        invariant_limit=invariant_limit,
        stiffness=max(
            indicators.tail_ratio / max(indicators.aliasing_defect, 1e-12),
            phase_competition,
        ),
        local_budget=local_budget,
        remaining_budget_fraction=remaining_budget_fraction,
        remaining_time_fraction=remaining_time_fraction,
        relative_work=relative_work,
        previous_rejected=False,
        localized_sharpness=indicators.localized_sharpness,
        nonlocal_tail=indicators.tail_ratio,
        dt_factor=dt_factor,
        resolution_factor=resolution_factor,
        probe_requested=resolution_factor > 1,
        representation_change=resolution_factor != 1,
    )
