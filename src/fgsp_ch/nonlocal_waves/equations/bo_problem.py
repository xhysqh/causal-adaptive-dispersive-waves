"""U0-compatible fixed-resolution Benjamin--Ono problem adapter."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from fgsp_ch.discretization.grids import PeriodicGrid
from fgsp_ch.nonlocal_waves.actions import ActionKind, AdaptiveAction
from fgsp_ch.nonlocal_waves.contracts import OperatorSignature, ProblemFeatures
from fgsp_ch.nonlocal_waves.evaluation.bo_metrics import bo_invariants
from fgsp_ch.nonlocal_waves.operators.bo_fourier import BOFourierOperator
from fgsp_ch.nonlocal_waves.solvers.bo_reference import BOIFRK4Solver
from fgsp_ch.nonlocal_waves.state import (
    OperatorStepResult,
    RepresentationKind,
    ResolutionState,
    WaveState,
)


@dataclass(frozen=True, slots=True)
class BOFourierProblem:
    padding_factor: float = 1.5

    @property
    def signature(self) -> OperatorSignature:
        return OperatorSignature(
            equation="benjamin_ono",
            nonlocal_mechanism="hilbert_fourier_multiplier",
            conservative_quantities=("mass", "half_l2_squared", "hamiltonian"),
            admissible_representations=("spectral",),
        )

    @staticmethod
    def _grid(state: WaveState) -> PeriodicGrid:
        return PeriodicGrid(
            state.resolution.points,
            state.domain_length,
            float(state.metadata.get("x_min", 0.0)),
        )

    def validate_state(self, state: WaveState) -> None:
        if state.representation is not RepresentationKind.SPECTRAL:
            raise ValueError("BO-P0 requires the spectral representation")
        if state.positions.size or state.amplitudes.size:
            raise ValueError("BO-P0 does not accept particle coordinates")
        self._grid(state).validate_state(state.field)

    def invariants(self, state: WaveState) -> Mapping[str, float]:
        self.validate_state(state)
        return bo_invariants(state.field, self._grid(state)).as_dict()

    def extract_features(self, state: WaveState) -> ProblemFeatures:
        self.validate_state(state)
        grid = self._grid(state)
        operator = BOFourierOperator(grid, self.padding_factor)
        coefficients = operator.spectrum(state.field)
        energy = np.abs(coefficients) ** 2
        split = max(1, int(np.floor(0.75 * energy.size)))
        tail = float(np.sum(energy[split:]) / max(np.sum(energy), np.finfo(float).eps))
        nonlinear = operator.nonlinear_spectrum(coefficients)
        dispersive = operator.linear_symbol * coefficients
        ratio = float(
            np.linalg.norm(nonlinear)
            / max(np.linalg.norm(dispersive), np.finfo(float).eps)
        )
        budget = float(state.error_budget_remaining) if np.isfinite(state.error_budget_remaining) else 1.0
        return ProblemFeatures(
            common={
                "field_rms": float(np.sqrt(np.mean(state.field**2))),
                "gradient_rms": float(
                    np.sqrt(np.mean(operator.physical(1j * operator.wave_numbers * coefficients) ** 2))
                ),
                "dt_over_dx": float(state.dt / grid.h),
                "error_budget_remaining": budget,
            },
            equation_specific={
                "spectral_tail_ratio": tail,
                "aliasing_defect": operator.aliasing_defect(state.field),
                "nonlinearity_dispersion_ratio": ratio,
                "maximum_linear_phase_increment": float(
                    state.dt * np.max(np.abs(operator.linear_symbol))
                ),
            },
        )

    def admissible_actions(self, state: WaveState) -> tuple[AdaptiveAction, ...]:
        self.validate_state(state)
        return (
            AdaptiveAction(
                "bo-reference-hold",
                ActionKind.HOLD,
                RepresentationKind.SPECTRAL,
                state.dt,
                state.resolution.points,
                active_modes=state.resolution.points // 2,
                dealias_fraction=2.0 / 3.0,
                estimated_work=float(state.resolution.points * np.log2(state.resolution.points)),
            ),
        )

    def advance(self, state: WaveState, action: AdaptiveAction) -> OperatorStepResult:
        self.validate_state(state)
        if action.kind is not ActionKind.HOLD or action.action_id != "bo-reference-hold":
            raise ValueError("BO-P0 only certifies its fixed-resolution reference HOLD action")
        if action.points != state.resolution.points or action.representation is not RepresentationKind.SPECTRAL:
            raise ValueError("BO-P0 HOLD cannot change resolution or representation")
        before = self.invariants(state)
        final = BOIFRK4Solver(self._grid(state), self.padding_factor).step(state.field, action.dt)
        next_state = WaveState(
            state.time + action.dt,
            final,
            action.dt,
            state.domain_length,
            RepresentationKind.SPECTRAL,
            ResolutionState(state.resolution.points, state.resolution.points // 2),
            error_budget_remaining=state.error_budget_remaining,
            metadata=state.metadata,
        )
        after = self.invariants(next_state)
        diagnostics = {
            "aliasing_defect": BOFourierOperator(
                self._grid(state), self.padding_factor
            ).aliasing_defect(state.field),
            "reference_used_online": False,
        }
        return OperatorStepResult(next_state, action.action_id, before, after, diagnostics)
