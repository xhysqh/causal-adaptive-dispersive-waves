"""Closed-loop hybrid solver with frozen P1 and learned P2.2 interface fluxes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from numpy.typing import NDArray

from fgsp_ch.datasets.cmame_p21_mixed_flux import interaction_features, project_regular_rhs_to_hybrid_tangent
from fgsp_ch.discretization.adaptive_mesh import finite_volume_divergence
from fgsp_ch.discretization.grids import PeriodicGrid
from fgsp_ch.equations.modified_ch import ModifiedCHParameters
from fgsp_ch.representations.cmame_p2_hybrid import CMAMEHybridState, hybrid_h1_squared, hybrid_mass, regular_state
from fgsp_ch.solvers.cmame_cross_representation import (
    coupled_particle_velocity, minimum_periodic_separation,
    project_regular_component_to_hybrid_invariants, regular_momentum_rhs, wrap_positions,
)


P1Predictor = Callable[[NDArray[np.floating], PeriodicGrid, ModifiedCHParameters, float], NDArray[np.floating]]
P22Predictor = Callable[[NDArray[np.floating], NDArray[np.floating]], NDArray[np.floating]]


@dataclass(frozen=True, slots=True)
class P22HybridStep:
    state: CMAMEHybridState
    interface_flux_rms: float
    mass_drift: float
    relative_h1_drift: float
    minimum_separation: float


@dataclass(slots=True)
class CMAMEP22HybridSolver:
    grid: PeriodicGrid
    parameters: ModifiedCHParameters
    dt: float
    p1_predictor: P1Predictor
    interface_predictor: P22Predictor
    collision_tolerance: float = 1.0e-8

    def _rates(self, state: CMAMEHybridState) -> tuple[np.ndarray, np.ndarray, float]:
        field = regular_state(state, self.grid)
        p1_flux = np.asarray(self.p1_predictor(field, self.grid, self.parameters, self.dt))
        features, parameters, _, flux_scale = interaction_features(
            state, self.grid, self.parameters, self.dt
        )
        normalized_interface = np.asarray(self.interface_predictor(features, parameters))
        self.grid.validate_state(p1_flux); self.grid.validate_state(normalized_interface)
        interface_flux = flux_scale * normalized_interface
        raw = regular_momentum_rhs(state, self.grid, self.parameters)
        raw -= finite_volume_divergence(p1_flux + interface_flux, self.grid.h)
        momentum_rate = project_regular_rhs_to_hybrid_tangent(raw, state, self.grid)
        position_rate = coupled_particle_velocity(state, self.grid, alpha=self.parameters.alpha)
        return momentum_rate, position_rate, float(np.sqrt(np.mean(interface_flux**2)))

    def step(self, state: CMAMEHybridState) -> P22HybridStep:
        if self.dt <= 0.0:
            raise ValueError("dt must be positive")
        state.validate(self.grid)
        if state.positions.size and self.parameters.gamma != 0.0:
            raise ValueError("P2.2 mixed singular states require gamma=0")
        if minimum_periodic_separation(self.grid, state.positions) <= self.collision_tolerance:
            raise RuntimeError("state is outside the P2.2 pre-collision contract")
        mass0, h10 = hybrid_mass(state, self.grid), hybrid_h1_squared(state, self.grid)
        dm0, dq0, flux0 = self._rates(state)
        predictor = CMAMEHybridState(
            state.regular_momentum + self.dt * dm0,
            wrap_positions(self.grid, state.positions + self.dt * dq0),
            np.asarray(state.amplitudes).copy(),
        )
        dm1, dq1, flux1 = self._rates(predictor)
        candidate = CMAMEHybridState(
            state.regular_momentum + 0.5 * self.dt * (dm0 + dm1),
            wrap_positions(self.grid, state.positions + 0.5 * self.dt * (dq0 + dq1)),
            np.asarray(state.amplitudes).copy(),
        )
        projected = project_regular_component_to_hybrid_invariants(candidate, self.grid, mass0, h10)
        state1 = projected.state
        return P22HybridStep(
            state1, max(flux0, flux1), abs(hybrid_mass(state1, self.grid) - mass0),
            abs(hybrid_h1_squared(state1, self.grid) - h10) / max(abs(h10), np.finfo(float).eps),
            minimum_periodic_separation(self.grid, state1.positions),
        )

