from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
import torch

from fgsp_ch.atlas.mch_atlas import AtlasKind
from fgsp_ch.geometry.metric import a_norm, discrete_energy
from fgsp_ch.models.mch_pinn_cnn_operator import MonotoneAdaptiveHead
from fgsp_ch.solvers.safe_adaptive_mch import (
    SafeAdaptiveMCHRollout,
    SafeAdaptiveMCHSolver,
    SafeAdaptiveMCHStep,
)
from fgsp_ch.training.mch_weak_physics import (
    mch_space_time_weak_residual,
    periodic_test_bank,
)


@dataclass(frozen=True, slots=True)
class M6StepDiagnostics:
    embedded_error_ratio: float
    weak_residual_indicator: float
    correction_ratio: float
    learned_error_ratio: float
    learned_step_factor: float


def mch_adaptive_indicators(
    solver: SafeAdaptiveMCHSolver,
    current: NDArray[np.floating],
    full: NDArray[np.floating],
    fine: NDArray[np.floating],
    dt: float,
    correction_ratio: float,
    network_accepted: bool,
    tests: torch.Tensor,
) -> tuple[NDArray[np.floating], float, float]:
    """Build non-negative, dimensionless causal M6 controller features."""
    equation = solver.equation
    scale = max(a_norm(fine, equation.helmholtz), np.finfo(float).eps)
    embedded = a_norm(fine - full, equation.helmholtz) / scale
    residual = mch_space_time_weak_residual(
        torch.as_tensor(current[None, :], dtype=torch.float64),
        torch.as_tensor(fine[None, :], dtype=torch.float64),
        tests,
        h=equation.grid.h,
        dt=dt,
        alpha=equation.parameters.alpha,
        gamma=equation.parameters.gamma,
    )
    weak = float(torch.sqrt(torch.mean(residual.square())).item() * dt / scale)
    raw = np.asarray(
        [
            embedded / solver.adaptive.relative_tolerance,
            weak,
            max(float(correction_ratio), 0.0),
            0.0 if network_accepted else 1.0,
            dt / solver.adaptive.maximum_dt,
        ],
        dtype=np.float64,
    )
    # Log scaling retains monotonicity and prevents a single stress indicator
    # from numerically dominating calibration.
    return np.log1p(raw), float(embedded), weak


@dataclass(slots=True)
class MCHPINNCNNAdaptiveSolver:
    """M6 controller around the M5 certified CNN/geometry stepper.

    M5 remains the safety kernel.  The learned head sees only causal physics
    indicators and may reject or reduce a step; it cannot accept a step that
    M5 rejected, disable conservation projection, or enlarge the M5 proposal.
    """

    certified_solver: SafeAdaptiveMCHSolver
    adaptive_head: MonotoneAdaptiveHead = field(default_factory=MonotoneAdaptiveHead)
    weak_modes: int = 4
    localized_tests: int = 4
    diagnostics: list[M6StepDiagnostics] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        self.adaptive_head.eval()
        if self.weak_modes < 1 or self.localized_tests < 0:
            raise ValueError("Invalid M6 weak-test configuration.")

    def solve(
        self, initial: NDArray[np.floating], *, final_time: float
    ) -> SafeAdaptiveMCHRollout:
        base = self.certified_solver
        equation = base.equation
        equation.grid.validate_state(initial)
        if final_time <= 0.0:
            raise ValueError("final_time must be positive.")
        tests = periodic_test_bank(
            equation.grid.points,
            modes=self.weak_modes,
            localized_centers=self.localized_tests,
            dtype=torch.float64,
        )
        states = [np.asarray(initial).copy()]
        times = [0.0]
        records: list[SafeAdaptiveMCHStep] = []
        self.diagnostics.clear()
        previous_atlas: AtlasKind | None = None
        dt = base.adaptive.initial_dt
        rejected = 0
        initial_mass = float(equation.grid.h * np.sum(initial))
        initial_energy = discrete_energy(initial, equation.helmholtz)
        while times[-1] < final_time - 1e-14:
            trial_dt = min(dt, final_time - times[-1])
            retries = 0
            while True:
                previous = states[-2] if len(states) > 1 else states[-1]
                full, full_decision, full_certificate = base._candidate(
                    previous, states[-1], trial_dt, previous_atlas
                )
                half, half_decision, half_certificate = base._candidate(
                    previous, states[-1], 0.5 * trial_dt, previous_atlas
                )
                fine, fine_decision, fine_certificate = base._candidate(
                    states[-1], half, 0.5 * trial_dt, half_decision.kind
                )
                accepted_network = half_certificate.accepted or fine_certificate.accepted
                correction_ratio = max(
                    half_certificate.raw_correction_ratio,
                    fine_certificate.raw_correction_ratio,
                )
                indicators, embedded, weak = mch_adaptive_indicators(
                    base, states[-1], full, fine, trial_dt, correction_ratio,
                    accepted_network, tests,
                )
                with torch.no_grad():
                    learned_ratio, learned_factor = self.adaptive_head(
                        torch.as_tensor(indicators[None, :], dtype=torch.float32)
                    )
                learned_ratio_value = float(learned_ratio.item())
                learned_factor_value = float(learned_factor.item())
                classical_ratio = embedded / base.adaptive.relative_tolerance
                tolerance_met = max(classical_ratio, learned_ratio_value) <= 1.0
                at_floor = trial_dt <= base.adaptive.minimum_dt * (1.0 + 1e-12)
                if tolerance_met or at_floor:
                    break
                rejected += 1
                retries += 1
                if retries >= base.adaptive.maximum_retries:
                    raise RuntimeError("M6 adaptive step exceeded maximum retries.")
                classical_factor = max(
                    base.adaptive.minimum_shrink,
                    base.adaptive.safety_factor
                    * max(classical_ratio, 1e-15) ** (-1.0 / 3.0),
                )
                factor = min(classical_factor, learned_factor_value)
                trial_dt = max(base.adaptive.minimum_dt, trial_dt * factor)
            states.append(np.asarray(fine))
            times.append(times[-1] + trial_dt)
            mass_drift = abs(float(equation.grid.h * np.sum(fine)) - initial_mass)
            energy_drift = abs(discrete_energy(fine, equation.helmholtz) - initial_energy)
            classical_growth = base.adaptive.maximum_growth if embedded == 0.0 else min(
                base.adaptive.maximum_growth,
                base.adaptive.safety_factor
                * (base.adaptive.relative_tolerance / max(embedded, 1e-15)) ** (1.0 / 3.0),
            )
            # The learned component can only keep or reduce the certified step.
            growth = min(classical_growth, learned_factor_value)
            next_dt = min(
                base.adaptive.maximum_dt,
                max(base.adaptive.minimum_dt, trial_dt * growth),
            )
            records.append(
                SafeAdaptiveMCHStep(
                    states[-1], times[-1], trial_dt, next_dt, embedded, retries,
                    fine_decision.kind,
                    half_decision.switched or fine_decision.switched,
                    accepted_network,
                    f"half:{half_certificate.reason}|fine:{fine_certificate.reason}|m6_physics",
                    mass_drift, energy_drift,
                )
            )
            self.diagnostics.append(
                M6StepDiagnostics(
                    embedded / base.adaptive.relative_tolerance,
                    weak,
                    correction_ratio,
                    learned_ratio_value,
                    learned_factor_value,
                )
            )
            previous_atlas = fine_decision.kind
            dt = next_dt
        return SafeAdaptiveMCHRollout(
            np.asarray(times), np.stack(states), tuple(records), rejected,
            sum(not step.network_accepted for step in records),
            sum(step.switched for step in records),
            max((step.mass_drift for step in records), default=0.0),
            max((step.energy_drift for step in records), default=0.0),
        )
