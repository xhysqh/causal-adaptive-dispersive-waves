from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from fgsp_ch.discretization.helmholtz import HelmholtzOperator
from fgsp_ch.geometry.metric import a_norm
from fgsp_ch.geometry.sphere import exp_map, log_map, radial_align


@dataclass(frozen=True, slots=True)
class IntrinsicGeometryMetrics:
    angle: float
    relative_retraction_signal: float
    absolute_retraction_signal: float
    log_exp_relative_error: float
    radial_relative_error: float
    finite: bool


@dataclass(frozen=True, slots=True)
class IntrinsicReferenceFloor:
    relative_log_disagreement: float
    absolute_log_disagreement: float
    finite: bool


def intrinsic_reference_floor(
    base: NDArray[np.floating],
    master_reference: NDArray[np.floating],
    shadow_reference: NDArray[np.floating],
    helmholtz: HelmholtzOperator,
    *,
    epsilon: float = 1.0e-15,
) -> IntrinsicReferenceFloor:
    """Measure master-shadow uncertainty in the intrinsic Log target.

    Both independently resolved references are restricted to the audited grid
    before this function is called and are aligned to the same energy shell.
    The floor is the A-norm disagreement between their Log tangents, divided
    by the master Log-tangent norm.
    """
    master = log_map(base, master_reference, helmholtz)
    shadow = log_map(base, shadow_reference, helmholtz)
    disagreement = a_norm(master - shadow, helmholtz)
    relative = disagreement / max(a_norm(master, helmholtz), epsilon)
    values = np.asarray([relative, disagreement])
    return IntrinsicReferenceFloor(
        relative_log_disagreement=float(relative),
        absolute_log_disagreement=float(disagreement),
        finite=bool(np.isfinite(values).all()),
    )


def intrinsic_geometry_metrics(
    base: NDArray[np.floating],
    reference: NDArray[np.floating],
    helmholtz: HelmholtzOperator,
    *,
    epsilon: float = 1.0e-15,
) -> IntrinsicGeometryMetrics:
    """Measure the oracle distinction between Exp and radial retraction.

    The reference is first aligned to the energy shell through ``base``.
    Exp(Log(reference)) is the exact intrinsic oracle; radial retraction of
    the same Log tangent is the competing first-order geometry.
    """
    aligned = radial_align(reference, base, helmholtz)
    tangent = log_map(base, aligned, helmholtz)
    radius = a_norm(base, helmholtz)
    tangent_norm = a_norm(tangent, helmholtz)
    angle = tangent_norm / max(radius, epsilon)
    exponential = exp_map(base, tangent, helmholtz)
    radial = radial_align(base + tangent, base, helmholtz)
    defect_norm = a_norm(aligned - base, helmholtz)
    signal = a_norm(radial - aligned, helmholtz)
    exp_error = a_norm(exponential - aligned, helmholtz) / max(
        a_norm(aligned, helmholtz), epsilon
    )
    radial_error = signal / max(a_norm(aligned, helmholtz), epsilon)
    values = np.asarray(
        [angle, signal / max(defect_norm, epsilon), signal, exp_error, radial_error]
    )
    return IntrinsicGeometryMetrics(
        angle=float(angle),
        relative_retraction_signal=float(signal / max(defect_norm, epsilon)),
        absolute_retraction_signal=float(signal),
        log_exp_relative_error=float(exp_error),
        radial_relative_error=float(radial_error),
        finite=bool(np.isfinite(values).all()),
    )
