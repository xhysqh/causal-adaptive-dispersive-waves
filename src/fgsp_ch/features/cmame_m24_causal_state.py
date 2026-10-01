"""Causal pre-commit state features for CMAME-M2.4."""

from __future__ import annotations

import numpy as np

from fgsp_ch.discretization.difference import first_derivative
from fgsp_ch.discretization.grids import PeriodicGrid
from fgsp_ch.discretization.helmholtz import HelmholtzOperator
from fgsp_ch.estimators.mch_hybrid_aposteriori import HybridErrorComponents
from fgsp_ch.solvers.mch_compatible_adaptive import minimum_periodic_separation
from fgsp_ch.solvers.mch_hybrid_amr import HybridMCHState


STATE_FEATURE_NAMES = (
    "eta_time", "eta_field", "eta_particle", "eta_interface",
    "eta_hierarchical", "eta_local_max", "eta_local_mean", "spectral_tail",
    "field_ratio", "atomic_ratio", "minimum_separation_over_h", "curvature",
    "dt_over_h", "grid_inverse", "log_target_tolerance",
    "log_remaining_absolute_budget", "remaining_budget_fraction", "previous_route",
)


def causal_state_features(
    state: HybridMCHState, grid: PeriodicGrid, estimate: HybridErrorComponents,
    *, dt: float, tolerance: float, remaining_budget_fraction: float,
    previous_route: str | None,
) -> np.ndarray:
    """Return finite features available before the numerical action.

    The target tolerance is part of the mathematical problem specification,
    not an audit label.  Both its absolute scale and the remaining trajectory
    budget are needed to distinguish identical states requested at different
    accuracies.
    """
    if tolerance <= 0.0:
        raise ValueError("target tolerance must be positive")
    field = HelmholtzOperator(grid).solve(np.asarray(state.field_momentum, dtype=np.float64))
    spectrum = np.abs(np.fft.rfft(field)) ** 2
    cutoff = max(1, int(np.ceil(0.65 * spectrum.size)))
    tail = float(spectrum[cutoff:].sum() / max(spectrum.sum(), np.finfo(float).eps))
    field_mass = float(np.linalg.norm(state.field_momentum))
    atomic_mass = float(np.linalg.norm(state.amplitudes))
    total = max(field_mass + atomic_mass, np.finfo(float).eps)
    separation = minimum_periodic_separation(state.positions, grid.length) / grid.h
    slope = first_derivative(field, grid)
    curvature = first_derivative(slope, grid)
    route_code = {None: 0.0, "smooth": 1.0 / 3.0, "particle": 2.0 / 3.0, "hybrid_amr": 1.0}[previous_route]
    local = np.asarray(estimate.local_indicators, dtype=np.float64)
    values = np.asarray([
        estimate.temporal, estimate.field, estimate.particle, estimate.interface,
        estimate.hierarchical_total, float(np.max(local)), float(np.mean(local)), tail,
        field_mass / total, atomic_mass / total, separation,
        float(np.max(np.abs(curvature))) / max(float(np.max(np.abs(field))), 1.0e-12),
        dt / grid.h, 1.0 / grid.points, np.log(tolerance),
        np.log(max(tolerance * remaining_budget_fraction, np.finfo(float).tiny)),
        remaining_budget_fraction, route_code,
    ], dtype=np.float64)
    if not np.all(np.isfinite(values)) or values.shape != (len(STATE_FEATURE_NAMES),):
        raise FloatingPointError("M2.4 causal state features are non-finite")
    return values
