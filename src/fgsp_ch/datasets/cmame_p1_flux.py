"""Causal conservative-flux labels for CMAME-P1."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from fgsp_ch.discretization.adaptive_mesh import finite_volume_divergence
from fgsp_ch.discretization.difference import first_derivative, second_derivative
from fgsp_ch.discretization.grids import PeriodicGrid
from fgsp_ch.discretization.helmholtz import HelmholtzOperator
from fgsp_ch.equations.modified_ch import ModifiedCH, ModifiedCHParameters


FEATURE_ORDER = (
    "u_over_u_scale", "h_ux_over_u_scale", "h2_uxx_over_u_scale",
    "m_over_m_scale", "base_flux_over_flux_scale", "curvature_indicator",
    "jump_indicator", "alpha_courant", "gamma_dt", "dt_over_h",
)
PARAMETER_ORDER = ("alpha", "gamma", "h", "dt", "u_scale", "m_scale")


@dataclass(frozen=True, slots=True)
class ConservativeFluxSample:
    features: NDArray[np.floating]
    parameters: NDArray[np.floating]
    correction_flux: NDArray[np.floating]
    correction_rhs: NDArray[np.floating]
    state: NDArray[np.floating]
    next_reference: NDArray[np.floating]
    group: str
    split: str


def cell_to_face_flux(cell_flux: NDArray[np.floating]) -> NDArray[np.floating]:
    values = np.asarray(cell_flux, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("cell flux must be one-dimensional")
    return 0.5 * (values + np.roll(values, -1))


def periodic_zero_mean_flux_from_rhs(rhs: NDArray[np.floating], h: float) -> NDArray[np.floating]:
    """Solve ``-D_f flux = rhs`` with a zero-mean periodic face flux."""
    values = np.asarray(rhs, dtype=np.float64)
    if values.ndim != 1 or h <= 0.0 or not np.all(np.isfinite(values)):
        raise ValueError("rhs must be a finite vector and h positive")
    centered = values - np.mean(values)
    flux = np.empty_like(centered)
    flux[0] = 0.0
    for index in range(1, len(flux)):
        flux[index] = flux[index - 1] - h * centered[index]
    flux -= np.mean(flux)
    defect = -finite_volume_divergence(flux, h) - centered
    if np.linalg.norm(defect, ord=np.inf) > 1.0e-10 * max(1.0, np.linalg.norm(centered, ord=np.inf)):
        raise FloatingPointError("periodic flux inversion did not close")
    return flux


def dimensionless_mch_features(
    state: NDArray[np.floating], grid: PeriodicGrid,
    parameters: ModifiedCHParameters, dt: float,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    grid.validate_state(state)
    if dt <= 0.0:
        raise ValueError("dt must be positive")
    equation = ModifiedCH(grid, parameters)
    ux = first_derivative(state, grid)
    uxx = second_derivative(state, grid)
    momentum = equation.helmholtz.apply(state)
    base_flux = equation.flux(state)
    eps = np.finfo(float).eps
    u_scale = max(float(np.max(np.abs(state))), eps)
    m_scale = max(float(np.sqrt(np.mean(momentum**2))), eps)
    flux_scale = max(float(np.sqrt(np.mean(base_flux**2))), eps)
    jump = np.abs(np.roll(state, -1) - state) / u_scale
    alpha_courant = parameters.alpha * u_scale**2 * dt / grid.h
    features = np.stack(
        (
            state / u_scale,
            grid.h * ux / u_scale,
            grid.h**2 * uxx / u_scale,
            momentum / m_scale,
            base_flux / flux_scale,
            np.abs(grid.h**2 * uxx / u_scale),
            jump,
            np.full(grid.points, alpha_courant),
            np.full(grid.points, parameters.gamma * dt),
            np.full(grid.points, dt / grid.h),
        )
    )
    global_parameters = np.asarray(
        [parameters.alpha, parameters.gamma, grid.h, dt, u_scale, m_scale],
        dtype=np.float64,
    )
    return features, global_parameters


def build_conservative_flux_sample(
    state: NDArray[np.floating], next_reference: NDArray[np.floating],
    grid: PeriodicGrid, parameters: ModifiedCHParameters, dt: float,
    *, group: str, split: str,
) -> ConservativeFluxSample:
    grid.validate_state(state)
    grid.validate_state(next_reference)
    equation = ModifiedCH(grid, parameters)
    helmholtz = HelmholtzOperator(grid)
    target_rhs = (helmholtz.apply(next_reference) - helmholtz.apply(state)) / dt
    ux = first_derivative(state, grid)
    base_face = cell_to_face_flux(equation.flux(state))
    base_rhs = -finite_volume_divergence(base_face, grid.h) - parameters.gamma * ux
    correction_rhs = target_rhs - base_rhs
    # A periodic conservative correction cannot represent roundoff in the zero
    # mode. Removing it is the unique mass-compatible projection of the label.
    correction_rhs -= np.mean(correction_rhs)
    correction_flux = periodic_zero_mean_flux_from_rhs(correction_rhs, grid.h)
    features, global_parameters = dimensionless_mch_features(state, grid, parameters, dt)
    return ConservativeFluxSample(
        features, global_parameters, correction_flux, correction_rhs,
        np.asarray(state), np.asarray(next_reference), group, split,
    )

