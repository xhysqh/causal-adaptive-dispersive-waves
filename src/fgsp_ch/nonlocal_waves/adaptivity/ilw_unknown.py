"""Equation-name-free symbol geometry and diagnostics for blind ILW transfer."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fgsp_ch.nonlocal_waves.adaptivity.prospective_causal_operator import prospective_features
from fgsp_ch.nonlocal_waves.operators.ilw_fourier import ILWFourierOperator


@dataclass(frozen=True, slots=True)
class ILWSymbolGeometry:
    effective_order_low: float
    effective_order_active: float
    effective_order_high: float
    logarithmic_curvature: float
    transition_overlap: float

    def mechanism_vector(self):
        return np.asarray([
            self.effective_order_low / 4.0,
            self.effective_order_active / 4.0,
            self.effective_order_high / 4.0,
            np.tanh(self.logarithmic_curvature),
            self.transition_overlap,
        ], dtype=np.float64)


@dataclass(frozen=True, slots=True)
class ILWIndicators:
    tail_ratio: float
    aliasing_defect: float
    unresolved_increment: float
    maximum_linear_phase: float
    nonlinear_phase: float
    localized_sharpness: float
    symbol: ILWSymbolGeometry


def _weighted_average(values, weights):
    return float(np.sum(values * weights) / max(np.sum(weights), np.finfo(float).eps))


def ilw_symbol_geometry(operator, coefficients):
    k = operator.wave_numbers[1:-1]
    omega = np.abs(operator.linear_symbol[1:-1])
    energy = (1.0 + k**2) * np.abs(np.asarray(coefficients)[1:-1]) ** 2
    valid = (k > 0.0) & (omega > np.finfo(float).tiny)
    k, omega, energy = k[valid], omega[valid], energy[valid]
    if len(k) < 4:
        return ILWSymbolGeometry(3.0, 2.5, 2.0, 0.0, 0.0)
    logk, logomega = np.log(k), np.log(omega)
    order = np.gradient(logomega, logk)
    curvature = np.gradient(order, logk)
    thirds = np.array_split(np.arange(len(k)), 3)
    low = _weighted_average(order[thirds[0]], energy[thirds[0]])
    active = _weighted_average(order, energy)
    high = _weighted_average(order[thirds[-1]], energy[thirds[-1]])
    transition = np.exp(-np.abs(np.log(np.maximum(operator.depth * k, 1e-14))))
    overlap = _weighted_average(transition, energy)
    return ILWSymbolGeometry(
        float(np.clip(low, 1.0, 4.0)),
        float(np.clip(active, 1.0, 4.0)),
        float(np.clip(high, 1.0, 4.0)),
        float(np.sqrt(_weighted_average(curvature**2, energy))),
        float(np.clip(overlap, 0.0, 1.0)),
    )


def ilw_indicators(values, grid, depth, dt, *, nonlinear_transport=1.0,
                   tail_fraction=0.65):
    operator = ILWFourierOperator(grid, depth, nonlinear_transport)
    spectrum = operator.spectrum(values)
    weights = 1.0 + operator.wave_numbers**2
    cutoff = max(1, int(tail_fraction * (len(spectrum) - 1)))
    total = np.sum(weights * np.abs(spectrum) ** 2)
    tail = np.sum(weights[cutoff:] * np.abs(spectrum[cutoff:]) ** 2)
    derivative = np.abs(operator.derivative(values))
    amplitude = max(float(np.max(np.abs(values))), 1e-14)
    return ILWIndicators(
        float(np.sqrt(tail / max(total, np.finfo(float).eps))),
        operator.aliasing_defect(values),
        operator.unresolved_nonlinear_increment(values, dt),
        float(abs(dt) * np.max(np.abs(operator.linear_symbol))),
        float(abs(dt) * nonlinear_transport * amplitude * np.max(operator.wave_numbers)),
        float(np.max(derivative) / max(np.linalg.norm(derivative), np.finfo(float).eps)),
        ilw_symbol_geometry(operator, spectrum),
    )


def ilw_features(indicators, *, temporal, migration, invariant_drift,
                 invariant_limit, local_budget, remaining_budget_fraction,
                 remaining_time_fraction, relative_work, dt_factor,
                 resolution_factor):
    phase_competition = indicators.nonlinear_phase / max(
        indicators.maximum_linear_phase, 1e-12
    )
    symbol_stiffness = indicators.symbol.logarithmic_curvature * (
        1.0 + indicators.symbol.transition_overlap
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
            symbol_stiffness,
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
        probe_requested=resolution_factor > 1.0,
        representation_change=resolution_factor != 1.0,
    )
