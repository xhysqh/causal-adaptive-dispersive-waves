from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from fgsp_ch.discretization.helmholtz import HelmholtzOperator
from fgsp_ch.geometry.metric import a_norm
from fgsp_ch.geometry.sphere import log_map


@dataclass(frozen=True, slots=True)
class ReferenceEnsembleMetrics:
    particle_tolerance_floor: float
    fourier_master_disagreement: float
    fourier_shadow_disagreement: float
    fourier_convergence_ratio: float
    finite: bool


def reference_ensemble_metrics(
    base: NDArray[np.floating],
    particle_master: NDArray[np.floating],
    particle_shadow: NDArray[np.floating],
    fourier_master: NDArray[np.floating],
    fourier_shadow: NDArray[np.floating],
    helmholtz: HelmholtzOperator,
    *,
    epsilon: float = 1.0e-15,
) -> ReferenceEnsembleMetrics:
    """Compare intrinsic labels from particle and Fourier references.

    The particle master/shadow difference is the numerical label floor.  The
    Fourier differences are reported separately as method bias and never used
    to inflate the particle uncertainty floor.
    """
    particle = log_map(base, particle_master, helmholtz)
    particle_tight = log_map(base, particle_shadow, helmholtz)
    fourier = log_map(base, fourier_master, helmholtz)
    fourier_tight = log_map(base, fourier_shadow, helmholtz)
    scale = max(a_norm(particle, helmholtz), epsilon)
    particle_floor = a_norm(particle - particle_tight, helmholtz) / scale
    master_error = a_norm(fourier - particle, helmholtz) / scale
    shadow_error = a_norm(fourier_tight - particle, helmholtz) / scale
    convergence = shadow_error / max(master_error, epsilon)
    values = np.asarray(
        [particle_floor, master_error, shadow_error, convergence]
    )
    return ReferenceEnsembleMetrics(
        particle_tolerance_floor=float(particle_floor),
        fourier_master_disagreement=float(master_error),
        fourier_shadow_disagreement=float(shadow_error),
        fourier_convergence_ratio=float(convergence),
        finite=bool(np.isfinite(values).all()),
    )
