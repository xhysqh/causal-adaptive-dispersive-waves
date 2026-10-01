from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fgsp_ch.discretization.helmholtz import HelmholtzOperator


@dataclass(frozen=True, slots=True)
class ProposalResidualFeatures:
    """Raw diagnostics and their dimensionless, increment-scaled counterpart."""

    raw: np.ndarray
    scale_consistent: np.ndarray


def _rms(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    return float(np.sqrt(np.mean(values**2))) if values.size else 0.0


def _dual_norm(values: np.ndarray, operator: HelmholtzOperator) -> float:
    values = np.asarray(values, dtype=np.float64)
    square = float(operator.grid.h * np.sum(values * operator.solve(values)))
    return float(np.sqrt(max(square, 0.0)))


def proposal_residual_features(
    *,
    eulerian_residual: np.ndarray,
    field_increment: np.ndarray,
    eulerian_rhs: np.ndarray,
    operator: HelmholtzOperator,
    dt: float,
    particle_residual: np.ndarray,
    particle_displacement: np.ndarray,
    particle_mid_velocity: np.ndarray,
    amplitude_increment: np.ndarray,
    amplitude_reference: np.ndarray,
    interface_mismatch: np.ndarray,
    maximum_cfl: float,
    active_fraction: float,
) -> ProposalResidualFeatures:
    r"""Build the M6.3.4.3.1 risk coordinates.

    The first five raw coordinates contain rates and therefore can grow like
    ``1 / dt`` when a learned proposal is evaluated away from its training
    time step.  The scale-consistent coordinates compare the *integrated*
    defect with the predicted and equation increments.  They are
    dimensionless and remain finite when ``dt`` is refined.
    """
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be finite and positive.")
    h = float(operator.grid.h)
    eps = float(np.sqrt(np.finfo(np.float64).eps))
    eulerian_hminus1 = _dual_norm(eulerian_residual, operator)
    eulerian_rms = _rms(eulerian_residual)
    particle_rms = _rms(particle_residual)
    amplitude_rate_rms = _rms(amplitude_increment / dt)
    interface_rms = _rms(interface_mismatch)
    raw = np.asarray((
        eulerian_hminus1, eulerian_rms, particle_rms,
        amplitude_rate_rms, interface_rms, maximum_cfl, active_fraction,
    ), dtype=np.float64)

    increment_dual = _dual_norm(field_increment, operator)
    rhs_dual = _dual_norm(eulerian_rhs, operator)
    increment_rms = _rms(field_increment)
    rhs_rms = _rms(eulerian_rhs)
    field_dual_scale = increment_dual + dt * rhs_dual + eps
    field_rms_scale = increment_rms + dt * rhs_rms + eps
    displacement_scale = (
        _rms(particle_displacement)
        + dt * _rms(particle_mid_velocity)
        + h * eps
    )
    amplitude_scale = _rms(amplitude_reference) + eps
    scaled = np.asarray((
        dt * eulerian_hminus1 / field_dual_scale,
        dt * eulerian_rms / field_rms_scale,
        dt * particle_rms / displacement_scale,
        _rms(amplitude_increment) / amplitude_scale,
        (dt / h) * interface_rms / field_rms_scale,
        maximum_cfl,
        active_fraction,
    ), dtype=np.float64)
    if not np.all(np.isfinite(raw)) or not np.all(np.isfinite(scaled)):
        raise FloatingPointError("Proposal residual features contain NaN or Inf.")
    return ProposalResidualFeatures(raw=raw, scale_consistent=scaled)


def select_residual_features(
    features: ProposalResidualFeatures, version: str,
) -> np.ndarray:
    if version == "raw_v1":
        return features.raw
    if version == "scale_consistent_v2":
        return features.scale_consistent
    raise ValueError(f"Unknown risk feature version: {version!r}.")
