"""U0 adapter for the frozen compatible mCH numerical backbone."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping

import numpy as np

from fgsp_ch.discretization.difference import first_derivative, second_derivative
from fgsp_ch.discretization.grids import PeriodicGrid
from fgsp_ch.equations.modified_ch import ModifiedCHParameters
from fgsp_ch.nonlocal_waves.actions import ActionKind, AdaptiveAction
from fgsp_ch.nonlocal_waves.contracts import OperatorSignature, ProblemFeatures
from fgsp_ch.nonlocal_waves.state import (
    OperatorStepResult,
    RepresentationKind,
    ResolutionState,
    WaveState,
)
from fgsp_ch.representations.cmame_p2_hybrid import (
    CMAMEHybridState,
    hybrid_h1_squared,
    hybrid_mass,
)
from fgsp_ch.solvers.mch_compatible_adaptive import (
    minimum_periodic_separation,
    representation_route,
)
from fgsp_ch.solvers.mch_compatible_invariant import CompatibleInvariantHybridMCHSolver
from fgsp_ch.solvers.mch_hybrid_amr import HybridMCHState, hybrid_total_state


def _kind(route: str) -> RepresentationKind:
    return {
        "eulerian": RepresentationKind.EULERIAN,
        "particle": RepresentationKind.PARTICLE,
        "hybrid": RepresentationKind.HYBRID,
    }[route]


@dataclass(slots=True)
class MCHCompatibleProblem:
    """Expose the M1.0.1 compatible step through the U0 contract.

    U0 supports the numerically identical ``HOLD`` action only.  Mesh and
    representation actions are intentionally left to the later adaptive
    operator stage; this adapter is a regression boundary, not a new solver.
    """

    parameters: ModifiedCHParameters
    nonlinear_tolerance: float = 1.0e-11
    maximum_iterations: int = 80
    damping: float = 1.0

    @property
    def signature(self) -> OperatorSignature:
        return OperatorSignature(
            equation="modified_camassa_holm",
            nonlocal_mechanism="helmholtz_inverse_with_hybrid_momentum",
            conservative_quantities=("mass", "half_h1_squared"),
            admissible_representations=("eulerian", "particle", "hybrid", "local_amr"),
        )

    @staticmethod
    def _grid(state: WaveState) -> PeriodicGrid:
        return PeriodicGrid(
            state.resolution.points,
            state.domain_length,
            float(state.metadata.get("x_min", 0.0)),
        )

    @staticmethod
    def _hybrid(state: WaveState) -> HybridMCHState:
        return HybridMCHState(
            state.positions.copy(), state.amplitudes.copy(), state.field.copy()
        )

    @staticmethod
    def _represented(state: WaveState) -> CMAMEHybridState:
        return CMAMEHybridState(
            state.field.copy(), state.positions.copy(), state.amplitudes.copy()
        )

    def validate_state(self, state: WaveState) -> None:
        if state.representation not in {
            RepresentationKind.EULERIAN,
            RepresentationKind.PARTICLE,
            RepresentationKind.HYBRID,
        }:
            raise ValueError("U0 mCH adapter accepts Eulerian, particle, or hybrid states")
        represented = self._represented(state)
        represented.validate(self._grid(state))
        actual = _kind(representation_route(self._hybrid(state), self._grid(state)))
        if state.representation is not actual:
            raise ValueError(
                f"declared representation {state.representation.value} disagrees with {actual.value}"
            )

    def invariants(self, state: WaveState) -> Mapping[str, float]:
        self.validate_state(state)
        represented = self._represented(state)
        grid = self._grid(state)
        return {
            "mass": hybrid_mass(represented, grid),
            "half_h1_squared": 0.5 * hybrid_h1_squared(represented, grid),
        }

    def extract_features(self, state: WaveState) -> ProblemFeatures:
        self.validate_state(state)
        grid = self._grid(state)
        hybrid = self._hybrid(state)
        total = hybrid_total_state(hybrid, grid)
        gradient = first_derivative(total, grid)
        curvature = second_derivative(total, grid)
        spectrum = np.fft.rfft(total)
        split = max(1, int(np.floor(0.75 * spectrum.size)))
        spectral_energy = np.abs(spectrum) ** 2
        tail_ratio = float(
            np.sum(spectral_energy[split:])
            / max(np.sum(spectral_energy), np.finfo(float).eps)
        )
        momentum = np.abs(state.field)
        concentration = float(np.max(momentum) / max(np.mean(momentum), np.finfo(float).eps))
        separation = minimum_periodic_separation(state.positions, grid.length)
        budget_ratio = (
            float(state.error_budget_remaining)
            if np.isfinite(state.error_budget_remaining)
            else 1.0
        )
        return ProblemFeatures(
            common={
                "field_rms": float(np.sqrt(np.mean(total**2))),
                "gradient_rms": float(np.sqrt(np.mean(gradient**2))),
                "curvature_rms": float(np.sqrt(np.mean(curvature**2))),
                "dt_over_dx": float(state.dt / grid.h),
                "error_budget_remaining": budget_ratio,
            },
            equation_specific={
                "momentum_concentration": concentration,
                "spectral_tail_ratio": tail_ratio,
                "minimum_particle_separation_over_dx": float(separation / grid.h),
                "particle_count": float(state.positions.size),
            },
        )

    def admissible_actions(self, state: WaveState) -> tuple[AdaptiveAction, ...]:
        self.validate_state(state)
        return (
            AdaptiveAction(
                action_id="mch-compatible-hold",
                kind=ActionKind.HOLD,
                representation=state.representation,
                dt=state.dt,
                points=state.resolution.points,
                estimated_work=float(state.resolution.points),
            ),
        )

    def advance(self, state: WaveState, action: AdaptiveAction) -> OperatorStepResult:
        self.validate_state(state)
        if action.kind is not ActionKind.HOLD or action.action_id != "mch-compatible-hold":
            raise ValueError("U0 mCH adapter only certifies the compatible HOLD action")
        if action.points != state.resolution.points or action.representation is not state.representation:
            raise ValueError("U0 HOLD action cannot change resolution or representation")
        grid = self._grid(state)
        solver = CompatibleInvariantHybridMCHSolver(
            grid,
            self.parameters,
            action.dt,
            nonlinear_tolerance=self.nonlinear_tolerance,
            maximum_iterations=self.maximum_iterations,
            damping=self.damping,
        )
        before = self.invariants(state)
        advanced, diagnostics = solver.step(self._hybrid(state))
        route = representation_route(advanced, grid)
        next_state = WaveState(
            time=state.time + action.dt,
            field=advanced.field_momentum,
            dt=action.dt,
            domain_length=state.domain_length,
            representation=_kind(route),
            resolution=ResolutionState(state.resolution.points),
            positions=advanced.positions,
            amplitudes=advanced.amplitudes,
            error_budget_remaining=state.error_budget_remaining,
            metadata=state.metadata,
        )
        after = self.invariants(next_state)
        row = asdict(diagnostics)
        row["route"] = route
        row["reference_used_online"] = False
        return OperatorStepResult(next_state, action.action_id, before, after, row)
