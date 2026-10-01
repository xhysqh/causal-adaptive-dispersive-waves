from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.signal import resample
import torch

from fgsp_ch.datasets.m63_branch_dataset import (
    FieldBranchSample,
    ParticleBranchSample,
    normalized_mch_parameters,
    peak_tokens_from_particles,
)
from fgsp_ch.discretization.grids import PeriodicGrid
from fgsp_ch.discretization.spectral import fourier_restrict
from fgsp_ch.equations.modified_ch import ModifiedCH, ModifiedCHParameters
from fgsp_ch.geometry.metric import a_norm, discrete_energy
from fgsp_ch.models.mch_pmwno import CertifiedPeakMeasureWaveletOperator
from fgsp_ch.solvers.mch_energy_projected import (
    MassH1ProjectedMCHSolver,
    project_to_mass_h1,
)
from fgsp_ch.solvers.mch_multipeakon import (
    ConservativePeriodicMCHMultipeakonSolver,
    mch_particle_h1,
)
from fgsp_ch.solvers.multipeakon import reconstruct_multipeakon
from fgsp_ch.training.m63_ensemble_calibration import (
    M63Calibration,
    field_ensemble_prediction,
    particle_ensemble_prediction,
)


@dataclass(frozen=True, slots=True)
class M63AdaptiveConfig:
    initial_dt: float = 0.01
    minimum_dt: float = 0.0025
    maximum_dt: float = 0.01
    classical_tolerance: float = 1.0e-5
    deployment_time_tolerance: float = 1.0e-6
    deployment_space_tolerance: float = 5.0e-6
    label_tolerance: float = 1.0e-8
    particle_position_tolerance: float = 2.0e-3
    safety_factor: float = 0.9
    minimum_shrink: float = 0.5
    cluster_cells: float = 6.0
    conservation_tolerance: float = 1.0e-10
    maximum_steps: int = 1000
    maximum_refined_fraction: float = 0.25

    def __post_init__(self) -> None:
        positive = (
            self.initial_dt, self.minimum_dt, self.maximum_dt,
            self.classical_tolerance, self.deployment_time_tolerance,
            self.deployment_space_tolerance, self.label_tolerance,
            self.particle_position_tolerance, self.safety_factor,
            self.minimum_shrink, self.cluster_cells, self.conservation_tolerance,
        )
        if any(value <= 0.0 or not np.isfinite(value) for value in positive):
            raise ValueError("M6.3.2 adaptive controls must be positive and finite.")
        if not self.minimum_dt <= self.initial_dt <= self.maximum_dt:
            raise ValueError("M6.3.2 time-step bounds are inconsistent.")
        if not 0.0 < self.maximum_refined_fraction <= 1.0:
            raise ValueError("maximum_refined_fraction must lie in (0, 1].")


@dataclass(frozen=True, slots=True)
class M63AdaptiveStep:
    dt: float
    next_dt: float
    temporal_risk: float
    spatial_risk: float
    ensemble_disagreement: float
    refined_fraction: float
    rejected: int
    fallback: bool
    reason: str


@dataclass(frozen=True, slots=True)
class M63AdaptiveRollout:
    times: NDArray[np.floating]
    states: NDArray[np.floating]
    steps: tuple[M63AdaptiveStep, ...]
    maximum_mass_drift: float
    maximum_energy_drift: float


def _two_half(equation: ModifiedCH, state: NDArray[np.floating], dt: float) -> NDArray[np.floating]:
    stepper = MassH1ProjectedMCHSolver(equation, 0.5 * dt)
    return stepper.step(stepper.step(state).state).state


def _minimum_separation(positions: NDArray[np.floating], length: float) -> float:
    if positions.size < 2:
        return length
    distance = np.abs(
        (positions[:, None] - positions[None, :] + 0.5 * length) % length
        - 0.5 * length
    )
    distance[np.eye(positions.size, dtype=bool)] = np.inf
    return float(np.min(distance))


class CertifiedPMWNOAdaptiveSolver:
    """Three-model controller with hard projection and reference fallback.

    M6.3.2 validates safety and decisions.  Its spatial shadow path still uses
    a global fine solve before applying a local mask, so no speedup claim is
    made until a true local-kernel implementation replaces that audit path.
    """

    def __init__(
        self, models: list[CertifiedPeakMeasureWaveletOperator], calibration: M63Calibration,
        *, device: torch.device, config: M63AdaptiveConfig | None = None,
    ) -> None:
        if len(models) < 3:
            raise ValueError("Certified rollout requires a three-model ensemble.")
        self.models = models
        self.calibration = calibration
        self.device = device
        self.config = config or M63AdaptiveConfig()
        for model in self.models:
            model.eval()

    def _field_risk(
        self, equation: ModifiedCH, state: NDArray[np.floating], dt: float
    ) -> tuple[float, float, float, NDArray[np.floating]]:
        grid = equation.grid
        momentum = equation.helmholtz.apply(state)
        field = np.stack((momentum, state, np.zeros(grid.points), np.ones(grid.points)))
        parameters = normalized_mch_parameters(
            equation.parameters.alpha, equation.parameters.gamma,
            grid.length, dt, grid.h,
        )
        dummy_levels = tuple(np.zeros(grid.points // (2 ** (level + 1))) for level in range(self.models[0].wavelet_levels))
        sample = FieldBranchSample(
            "rollout", "deployment", field, parameters, np.zeros(grid.points),
            dummy_levels, 0.0, 0.0,
        )
        prediction = field_ensemble_prediction(self.models, [sample], self.device)
        tolerance_factor_t = self.config.label_tolerance / self.config.deployment_time_tolerance
        tolerance_factor_x = self.config.label_tolerance / self.config.deployment_space_tolerance
        temporal = self.calibration.temporal_scale * float(
            prediction["temporal_mean"][0] + prediction["temporal_std"][0]
        ) * tolerance_factor_t
        spatial = self.calibration.spatial_scale * float(
            prediction["spatial_mean"][0] + prediction["spatial_std"][0]
        ) * tolerance_factor_x
        local = self.calibration.local_scale * (
            prediction["local_mean"][0][0] + prediction["local_std"][0][0]
        ) * tolerance_factor_x
        disagreement = float(max(prediction["temporal_std"][0], prediction["spatial_std"][0]))
        return temporal, spatial, disagreement, np.asarray(local)

    def solve_field(
        self, equation: ModifiedCH, initial: NDArray[np.floating], *, final_time: float,
        enable_spatial: bool = True, enable_temporal: bool = True,
        enable_projection: bool = True,
    ) -> M63AdaptiveRollout:
        equation.grid.validate_state(initial)
        states = [np.asarray(initial).copy()]
        times = [0.0]
        records: list[M63AdaptiveStep] = []
        initial_mass = float(equation.grid.h * np.sum(initial))
        initial_energy = discrete_energy(initial, equation.helmholtz)
        dt = self.config.initial_dt
        while times[-1] < final_time - 1.0e-14:
            if len(records) >= self.config.maximum_steps:
                raise RuntimeError("M6.3.2 field rollout exceeded maximum steps.")
            trial = min(dt, final_time - times[-1])
            rejected = 0
            while True:
                current = states[-1]
                full = MassH1ProjectedMCHSolver(equation, trial).step(current).state
                half = _two_half(equation, current, trial)
                classical = a_norm(half - full, equation.helmholtz) / max(
                    a_norm(half, equation.helmholtz), np.finfo(float).eps
                ) / self.config.classical_tolerance
                temporal, spatial, disagreement, local = self._field_risk(equation, current, trial)
                total_time = max(classical, temporal) if enable_temporal else classical
                if total_time <= 1.0 or trial <= self.config.minimum_dt * (1.0 + 1.0e-12):
                    break
                factor = max(
                    self.config.minimum_shrink,
                    self.config.safety_factor * total_time ** (-1.0 / 3.0),
                )
                trial = max(self.config.minimum_dt, trial * factor)
                rejected += 1
            candidate = half
            refined_fraction = 0.0
            fallback = False
            reason = "coarse_two_half"
            if enable_spatial and max(spatial, float(np.max(local))) > 1.0:
                # Budgeted marking: when the calibrated global spatial risk is
                # unsafe but no cell individually crosses one, refine the top
                # risk quantile instead of silently taking no spatial action.
                quantile = float(np.quantile(local, 1.0 - self.config.maximum_refined_fraction))
                active = local >= max(1.0, quantile) if np.any(local > 1.0) else local >= quantile
                mask = np.repeat(active, 2)[: equation.grid.points]
                mask = mask | np.roll(mask, 1) | np.roll(mask, -1)
                refined_fraction = float(np.mean(mask))
                fine_grid = PeriodicGrid(2 * equation.grid.points, equation.grid.length)
                fine_equation = ModifiedCH(fine_grid, equation.parameters)
                fine_initial = np.asarray(resample(states[-1], fine_grid.points)).real
                fine = _two_half(fine_equation, fine_initial, trial)
                restricted = fourier_restrict(fine, equation.grid.points)
                blended = candidate + mask * (restricted - candidate)
                if enable_projection:
                    try:
                        blended = project_to_mass_h1(
                            blended, initial_mass, initial_energy, equation
                        ).state
                    except (ValueError, FloatingPointError):
                        blended = candidate
                        fallback = True
                        reason = "projection_fallback"
                if a_norm(blended - restricted, equation.helmholtz) <= a_norm(
                    candidate - restricted, equation.helmholtz
                ):
                    candidate = blended
                    reason = "local_h2_shadow"
                else:
                    fallback = True
                    reason = "shadow_direction_fallback"
            states.append(np.asarray(candidate))
            times.append(times[-1] + trial)
            learned_factor = min(1.0, self.config.safety_factor * max(temporal, 1.0e-12) ** (-1.0 / 3.0))
            next_dt = min(
                trial,
                max(self.config.minimum_dt, min(self.config.maximum_dt, trial * learned_factor)),
            )
            records.append(M63AdaptiveStep(
                trial, next_dt, temporal, spatial, disagreement,
                refined_fraction, rejected, fallback, reason,
            ))
            dt = next_dt
        mass_drift = max(abs(float(equation.grid.h * np.sum(state)) - initial_mass) for state in states)
        energy_drift = max(abs(discrete_energy(state, equation.helmholtz) - initial_energy) for state in states)
        return M63AdaptiveRollout(
            np.asarray(times), np.stack(states), tuple(records), mass_drift, energy_drift
        )

    def _particle_prediction(
        self, positions: NDArray[np.floating], amplitudes: NDArray[np.floating],
        *, length: float, alpha: float, dt: float,
    ) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        maximum = 8
        tokens, mask = peak_tokens_from_particles(
            positions, amplitudes, length, length / 128.0, maximum
        )
        padded_q, padded_p = np.zeros(maximum), np.zeros(maximum)
        padded_q[: positions.size], padded_p[: positions.size] = positions, amplitudes
        sample = ParticleBranchSample(
            "rollout", "deployment", tokens, padded_q, padded_p, mask,
            normalized_mch_parameters(alpha, 0.0, length, dt, length / 128.0),
            length, np.zeros(maximum), np.zeros(maximum),
        )
        mean, std, _ = particle_ensemble_prediction(self.models, [sample], self.device)
        upper = self.calibration.particle_scale * (std[0] + self.calibration.particle_floor)
        return mean[0, : positions.size], upper[: positions.size]

    def solve_particles(
        self, grid: PeriodicGrid, positions: NDArray[np.floating], amplitudes: NDArray[np.floating],
        *, alpha: float, final_time: float, enable_fallback: bool = True,
    ) -> M63AdaptiveRollout:
        q = np.asarray(positions, dtype=np.float64).copy()
        p = np.asarray(amplitudes, dtype=np.float64).copy()
        states = [reconstruct_multipeakon(grid, q, p)]
        times = [0.0]
        records: list[M63AdaptiveStep] = []
        initial_particle_mass = float(np.sum(p))
        equation = ModifiedCH(grid, ModifiedCHParameters(alpha, 0.0))
        initial_particle_h1 = mch_particle_h1(q, p, grid.length)
        invariant_drifts: list[float] = [0.0]
        dt = self.config.initial_dt
        while times[-1] < final_time - 1.0e-14:
            trial = min(dt, final_time - times[-1])
            learned_velocity, upper = self._particle_prediction(q, p, length=grid.length, alpha=alpha, dt=trial)
            risk = float(trial * np.max(upper) / self.config.particle_position_tolerance)
            clustered = _minimum_separation(q, grid.length) / grid.h < self.config.cluster_cells
            fallback_requested = bool(clustered or risk > 1.0)
            fallback = bool(fallback_requested and enable_fallback)
            if fallback:
                trajectory = ConservativePeriodicMCHMultipeakonSolver(
                    grid.length, alpha=alpha, rtol=1.0e-12, atol=1.0e-14
                ).solve(q, p, final_time=trial, output_dt=trial)
                q_next = trajectory.positions[-1]
                reason = "cluster_or_uncertainty_fallback"
            else:
                q_predict = np.remainder(q + trial * learned_velocity, grid.length)
                second_velocity, second_upper = self._particle_prediction(
                    q_predict, p, length=grid.length, alpha=alpha, dt=trial
                )
                risk = max(risk, float(trial * np.max(second_upper) / self.config.particle_position_tolerance))
                q_next = np.remainder(q + 0.5 * trial * (learned_velocity + second_velocity), grid.length)
                reason = "learned_peakon_heun"
                energy_drift = abs(
                    mch_particle_h1(q_next, p, grid.length) - initial_particle_h1
                )
                if enable_fallback and energy_drift > self.config.conservation_tolerance:
                    trajectory = ConservativePeriodicMCHMultipeakonSolver(
                        grid.length, alpha=alpha, rtol=1.0e-12, atol=1.0e-14
                    ).solve(q, p, final_time=trial, output_dt=trial)
                    q_next = trajectory.positions[-1]
                    fallback = True
                    reason = "h1_fallback"
            q = np.asarray(q_next)
            state = reconstruct_multipeakon(grid, q, p)
            states.append(state)
            invariant_drifts.append(
                abs(mch_particle_h1(q, p, grid.length) - initial_particle_h1)
            )
            times.append(times[-1] + trial)
            factor = min(1.0, self.config.safety_factor * max(risk, 1.0e-12) ** (-1.0 / 3.0))
            next_dt = min(
                trial,
                max(self.config.minimum_dt, min(self.config.maximum_dt, trial * factor)),
            )
            records.append(M63AdaptiveStep(
                trial, next_dt, risk, 0.0, float(np.max(upper)), 0.0,
                0, fallback, reason,
            ))
            dt = next_dt
        mass_drift = abs(float(np.sum(p)) - initial_particle_mass)
        energy_drift = max(invariant_drifts)
        return M63AdaptiveRollout(
            np.asarray(times), np.stack(states), tuple(records), mass_drift, energy_drift
        )
