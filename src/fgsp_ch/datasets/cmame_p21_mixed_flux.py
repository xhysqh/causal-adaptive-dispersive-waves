"""Causal mixed-measure flux labels for CMAME-P2.1."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from fgsp_ch.datasets.cmame_p1_flux import (
    FEATURE_ORDER as P1_FEATURE_ORDER,
    dimensionless_mch_features,
    periodic_zero_mean_flux_from_rhs,
)
from fgsp_ch.discretization.adaptive_mesh import finite_volume_divergence
from fgsp_ch.discretization.difference import first_derivative
from fgsp_ch.discretization.grids import PeriodicGrid
from fgsp_ch.discretization.helmholtz import HelmholtzOperator
from fgsp_ch.equations.modified_ch import ModifiedCHParameters
from fgsp_ch.geometry.metric import discrete_inner
from fgsp_ch.representations.cmame_p2_hybrid import (
    CMAMEHybridState,
    atomic_measure_weight,
    periodic_interpolate,
    regular_state,
)
from fgsp_ch.solvers.cmame_cross_representation import (
    coupled_particle_velocity,
    minimum_periodic_separation,
    regular_momentum_rhs,
)
from fgsp_ch.solvers.cmame_p21_reference import P21ReferenceIncrement
from fgsp_ch.solvers.multipeakon import (
    periodic_peakon_kernel,
    periodic_peakon_kernel_derivative,
)


INTERACTION_FEATURE_ORDER = (
    "atomic_u_over_u_scale",
    "h_atomic_ux_over_u_scale",
    "green_proximity",
    "nearest_distance_over_length",
    "atomic_density_over_m_scale",
    "interaction_velocity_over_u2_scale",
    "interaction_flux_over_flux_scale",
    "particle_count_over_four",
    "minimum_separation_over_length",
    "interface_weight",
)
FEATURE_ORDER = P1_FEATURE_ORDER + INTERACTION_FEATURE_ORDER
PARAMETER_ORDER = (
    "alpha", "gamma", "h", "dt", "u_scale", "m_scale",
    "particle_count", "minimum_separation", "amplitude_sum", "amplitude_max",
)


@dataclass(frozen=True, slots=True)
class P21MixedFluxSample:
    features: NDArray[np.floating]
    parameters: NDArray[np.floating]
    correction_flux: NDArray[np.floating]
    normalized_correction_flux: NDArray[np.floating]
    correction_rhs: NDArray[np.floating]
    reference_floor_flux: NDArray[np.floating]
    interface_weight: NDArray[np.floating]
    state: CMAMEHybridState
    family: str
    group: str
    split: str
    grid_points: int
    label_rms: float
    floor_rms: float
    reference_snr: float
    particle_velocity_defect: float
    particle_velocity_floor: float
    flux_scale: float


def periodic_cic_deposit(
    grid: PeriodicGrid,
    positions: NDArray[np.floating],
    weights: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Cloud-in-cell deposition whose grid integral equals ``sum(weights)``."""
    q = np.asarray(positions, dtype=np.float64)
    values = np.asarray(weights, dtype=np.float64)
    if q.ndim != 1 or q.shape != values.shape:
        raise ValueError("positions and weights must have equal one-dimensional shapes")
    result = np.zeros(grid.points, dtype=np.float64)
    coordinate = (q - grid.x_min) % grid.length / grid.h
    left = np.floor(coordinate).astype(int) % grid.points
    fraction = coordinate - np.floor(coordinate)
    np.add.at(result, left, values * (1.0 - fraction) / grid.h)
    np.add.at(result, (left + 1) % grid.points, values * fraction / grid.h)
    return result


def hybrid_h1_gradient(
    state: CMAMEHybridState, grid: PeriodicGrid
) -> NDArray[np.floating]:
    """Gradient of half the representation-exact H1 functional w.r.t. regular momentum."""
    field = regular_state(state, grid)
    if not state.positions.size:
        return field
    atomic_momentum = periodic_cic_deposit(
        grid,
        state.positions,
        atomic_measure_weight(grid.length) * state.amplitudes,
    )
    return np.asarray(field + HelmholtzOperator(grid).solve(atomic_momentum))


def project_regular_rhs_to_hybrid_tangent(
    rhs: NDArray[np.floating], state: CMAMEHybridState, grid: PeriodicGrid
) -> NDArray[np.floating]:
    grid.validate_state(np.asarray(rhs))
    centered = np.asarray(rhs, dtype=np.float64) - np.mean(rhs)
    gradient = hybrid_h1_gradient(state, grid)
    direction = gradient - np.mean(gradient)
    denominator = discrete_inner(direction, direction, grid)
    if denominator <= 100.0 * np.finfo(float).eps:
        return centered
    coefficient = discrete_inner(direction, centered, grid) / denominator
    projected = centered - coefficient * direction
    return np.asarray(projected - np.mean(projected))


def interaction_features(
    state: CMAMEHybridState,
    grid: PeriodicGrid,
    parameters: ModifiedCHParameters,
    dt: float,
) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating], float]:
    field = regular_state(state, grid)
    p1_features, p1_parameters = dimensionless_mch_features(field, grid, parameters, dt)
    eps = np.finfo(float).eps
    u_scale = max(float(p1_parameters[4]), eps)
    m_scale = max(float(p1_parameters[5]), eps)
    if state.positions.size:
        delta = grid.x[:, None] - state.positions[None, :]
        kernels = periodic_peakon_kernel(delta, grid.length)
        derivatives = periodic_peakon_kernel_derivative(delta, grid.length)
        atomic = kernels @ state.amplitudes
        atomic_slope = derivatives @ state.amplitudes
        proximity = np.max(kernels, axis=1)
        distances = np.min(np.abs(grid.periodic_distance(grid.x[:, None], state.positions[None, :])), axis=1)
        density = periodic_cic_deposit(grid, state.positions, atomic_measure_weight(grid.length) * state.amplitudes)
        separation = minimum_periodic_separation(grid, state.positions)
    else:
        atomic = np.zeros(grid.points)
        atomic_slope = np.zeros(grid.points)
        proximity = np.zeros(grid.points)
        distances = np.full(grid.points, 0.5 * grid.length)
        density = np.zeros(grid.points)
        separation = grid.length
    field_slope = first_derivative(field, grid)
    total = field + atomic
    total_slope = field_slope + atomic_slope
    smooth_velocity = field**2 - field_slope**2
    interaction_velocity = total**2 - total_slope**2 - smooth_velocity
    interaction_flux = parameters.alpha * interaction_velocity * state.regular_momentum
    flux_scale = max(parameters.alpha * u_scale**2 * m_scale, eps)
    interface_weight = np.clip(proximity, 0.0, 1.0)
    extra = np.stack((
        atomic / u_scale,
        grid.h * atomic_slope / u_scale,
        proximity,
        distances / grid.length,
        density / m_scale,
        interaction_velocity / max(u_scale**2, eps),
        interaction_flux / flux_scale,
        np.full(grid.points, state.positions.size / 4.0),
        np.full(grid.points, separation / grid.length),
        interface_weight,
    ))
    parameters_out = np.concatenate((
        p1_parameters,
        np.asarray([
            float(state.positions.size),
            separation,
            float(np.sum(state.amplitudes)),
            float(np.max(state.amplitudes)) if state.amplitudes.size else 0.0,
        ]),
    ))
    return np.concatenate((p1_features, extra)), parameters_out, interface_weight, flux_scale


def build_p21_mixed_flux_sample(
    state: CMAMEHybridState,
    master: P21ReferenceIncrement,
    shadow: P21ReferenceIncrement,
    p1_flux: NDArray[np.floating],
    grid: PeriodicGrid,
    parameters: ModifiedCHParameters,
    dt: float,
    *,
    family: str,
    group: str,
    split: str,
) -> P21MixedFluxSample:
    state.validate(grid)
    p1 = np.asarray(p1_flux, dtype=np.float64)
    grid.validate_state(p1)
    base_rhs = regular_momentum_rhs(state, grid, parameters)
    p1_rhs = -finite_volume_divergence(p1, grid.h)
    raw = master.regular_momentum_rate - base_rhs - p1_rhs
    correction_rhs = project_regular_rhs_to_hybrid_tangent(raw, state, grid)
    correction_flux = periodic_zero_mean_flux_from_rhs(correction_rhs, grid.h)
    floor_rhs = project_regular_rhs_to_hybrid_tangent(
        shadow.regular_momentum_rate - master.regular_momentum_rate, state, grid
    )
    floor_flux = periodic_zero_mean_flux_from_rhs(floor_rhs, grid.h)
    features, global_parameters, interface_weight, flux_scale = interaction_features(
        state, grid, parameters, dt
    )
    label_rms = float(np.sqrt(np.mean(correction_flux**2)))
    floor_rms = float(np.sqrt(np.mean(floor_flux**2)))
    snr = label_rms / max(floor_rms, 100.0 * np.finfo(float).eps)
    analytic_velocity = coupled_particle_velocity(state, grid, alpha=parameters.alpha)
    velocity_defect = (
        float(np.sqrt(np.mean((master.particle_position_rate - analytic_velocity) ** 2)))
        if state.positions.size else 0.0
    )
    velocity_floor = (
        float(np.sqrt(np.mean((shadow.particle_position_rate - master.particle_position_rate) ** 2)))
        if state.positions.size else 0.0
    )
    return P21MixedFluxSample(
        features,
        global_parameters,
        correction_flux,
        correction_flux / flux_scale,
        correction_rhs,
        floor_flux,
        interface_weight,
        state,
        family,
        group,
        split,
        grid.points,
        label_rms,
        floor_rms,
        snr,
        velocity_defect,
        velocity_floor,
        flux_scale,
    )

