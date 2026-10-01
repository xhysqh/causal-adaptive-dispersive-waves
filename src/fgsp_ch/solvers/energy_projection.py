from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from fgsp_ch.discretization.helmholtz import HelmholtzOperator
from fgsp_ch.geometry.metric import discrete_energy


@dataclass(frozen=True, slots=True)
class EnergyProjectionResult:
    state: NDArray[np.floating]
    scale: float
    source_energy: float
    target_energy: float


def project_to_energy(
    candidate: NDArray[np.floating],
    target_energy: float,
    helmholtz: HelmholtzOperator,
    *,
    atol: float = 1.0e-14,
) -> EnergyProjectionResult:
    """Radially project ``candidate`` onto ``E_h(U)=target_energy``.

    Zero target energy maps to the exact zero state. A positive target with a
    degenerate candidate is reported as an error instead of being clipped.
    """
    helmholtz.grid.validate_state(candidate)
    if not np.isfinite(target_energy) or target_energy < 0.0:
        raise ValueError("Target energy must be finite and non-negative.")
    source_energy = discrete_energy(candidate, helmholtz)
    if not np.isfinite(source_energy):
        raise FloatingPointError("Candidate energy is not finite.")
    if target_energy == 0.0:
        state = np.zeros_like(candidate)
        return EnergyProjectionResult(state, 0.0, source_energy, 0.0)
    if source_energy <= 0.0:
        raise FloatingPointError(
            "Cannot project a degenerate candidate to positive energy."
        )
    scale = float(np.sqrt(target_energy / source_energy))
    if not np.isfinite(scale):
        raise FloatingPointError(
            "Energy projection scale is not finite; source energy is degenerate."
        )
    state = np.asarray(scale * candidate, dtype=candidate.dtype)
    if not np.all(np.isfinite(state)):
        raise FloatingPointError("Energy projection produced NaN or Inf.")
    return EnergyProjectionResult(state, scale, source_energy, target_energy)
