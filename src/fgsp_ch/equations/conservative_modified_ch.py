from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from fgsp_ch.discretization.grids import PeriodicGrid
from fgsp_ch.discretization.helmholtz import HelmholtzOperator
from fgsp_ch.equations.modified_ch import ModifiedCH, ModifiedCHParameters
from fgsp_ch.geometry.metric import discrete_inner


@dataclass(slots=True)
class InvariantTangentModifiedCH:
    """Mass--H1 tangent semidiscretization of smooth cubic/FORQ mCH.

    The centered conservative-flux right-hand side is corrected only along the
    zero-mean radial direction.  Consequently its continuous-time ODE obeys
    ``d mass/dt = d H1^2/dt = 0`` algebraically.  This is a project-derived
    independent Eulerian discretization, not a peakon product definition.
    """

    grid: PeriodicGrid
    parameters: ModifiedCHParameters = field(default_factory=ModifiedCHParameters)
    base: ModifiedCH = field(init=False)
    helmholtz: HelmholtzOperator = field(init=False)

    def __post_init__(self) -> None:
        self.base = ModifiedCH(self.grid, self.parameters)
        self.helmholtz = self.base.helmholtz

    def raw_state_rhs(self, state: NDArray[np.floating]) -> NDArray[np.floating]:
        return self.base.state_rhs(state)

    def state_rhs(self, state: NDArray[np.floating]) -> NDArray[np.floating]:
        self.grid.validate_state(state)
        raw = self.raw_state_rhs(state)
        centered = state - np.mean(state)
        momentum = self.helmholtz.apply(state)
        denominator = discrete_inner(momentum, centered, self.grid)
        energy_rate = discrete_inner(momentum, raw, self.grid)
        scale = max(1.0, float(np.linalg.norm(state)) ** 2)
        tolerance = 100.0 * np.finfo(state.dtype).eps * scale
        if abs(denominator) <= tolerance:
            if abs(energy_rate) > tolerance:
                raise FloatingPointError(
                    "Degenerate constant state has a nonzero discrete energy rate."
                )
            result = raw
        else:
            result = raw - (energy_rate / denominator) * centered
        if not np.all(np.isfinite(result)):
            raise FloatingPointError("Invariant-tangent mCH RHS contains NaN or Inf.")
        return np.asarray(result, dtype=state.dtype)

    def invariant_rates(self, state: NDArray[np.floating]) -> tuple[float, float]:
        rhs = self.state_rhs(state)
        mass_rate = float(self.grid.h * np.sum(rhs))
        energy_rate = discrete_inner(self.helmholtz.apply(state), rhs, self.grid)
        return mass_rate, energy_rate
