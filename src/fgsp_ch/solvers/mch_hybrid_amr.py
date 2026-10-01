from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from numpy.typing import NDArray
from scipy.signal import resample

from fgsp_ch.discretization.adaptive_mesh import (
    PeriodicRefinementPatch,
    finite_volume_divergence,
    reflux_two_to_one,
)
from fgsp_ch.discretization.difference import first_derivative
from fgsp_ch.discretization.grids import PeriodicGrid
from fgsp_ch.discretization.helmholtz import HelmholtzOperator
from fgsp_ch.equations.modified_ch import ModifiedCH, ModifiedCHParameters
from fgsp_ch.geometry.metric import discrete_energy
from fgsp_ch.solvers.mch_energy_projected import project_to_mass_h1
from fgsp_ch.solvers.multipeakon import (
    periodic_peakon_kernel,
    periodic_peakon_kernel_derivative,
    reconstruct_multipeakon,
)


def fourier_prolong(values: NDArray[np.floating], points: int) -> NDArray[np.floating]:
    data = np.asarray(values, dtype=np.float64)
    if data.ndim != 1 or points < data.size:
        raise ValueError("fourier_prolong requires a larger one-dimensional grid.")
    return np.asarray(resample(data, points)).real


@dataclass(frozen=True, slots=True)
class CompositeMomentumState:
    coarse_momentum: NDArray[np.floating]
    fine_detail_momentum: NDArray[np.floating]
    patch: PeriodicRefinementPatch


@dataclass(slots=True)
class HierarchicalPeriodicGreenSolver:
    coarse_grid: PeriodicGrid

    @property
    def fine_grid(self) -> PeriodicGrid:
        return PeriodicGrid(
            2 * self.coarse_grid.points,
            self.coarse_grid.length,
            self.coarse_grid.x_min,
            self.coarse_grid.dtype,
        )

    def _fine_green(self) -> NDArray[np.floating]:
        fine = self.fine_grid
        modes = np.arange(fine.points, dtype=np.float64)
        eigenvalues = 1.0 + 4.0 * np.sin(np.pi * modes / fine.points) ** 2 / fine.h**2
        return np.fft.ifft(1.0 / eigenvalues).real

    def lifted_coarse_state(self, coarse_momentum: NDArray[np.floating]) -> NDArray[np.floating]:
        self.coarse_grid.validate_state(np.asarray(coarse_momentum))
        coarse_state = HelmholtzOperator(self.coarse_grid).solve(np.asarray(coarse_momentum))
        return fourier_prolong(coarse_state, self.fine_grid.points)

    def base_fine_momentum(self, coarse_momentum: NDArray[np.floating]) -> NDArray[np.floating]:
        return HelmholtzOperator(self.fine_grid).apply(
            self.lifted_coarse_state(coarse_momentum)
        )

    def local_detail_solve(
        self, detail: NDArray[np.floating], active: NDArray[np.bool_]
    ) -> NDArray[np.floating]:
        values = np.asarray(detail, dtype=np.float64)
        mask = np.asarray(active, dtype=bool)
        if values.shape != (self.fine_grid.points,) or mask.shape != values.shape:
            raise ValueError("Fine detail and active mask shapes are inconsistent.")
        if np.any(np.abs(values[~mask]) > 1.0e-14):
            raise ValueError("Local detail must vanish outside the active patch.")
        kernel = self._fine_green()
        correction = np.zeros_like(values)
        for index in np.flatnonzero(mask & (values != 0.0)):
            correction += values[index] * np.roll(kernel, int(index))
        return correction

    def solve(self, state: CompositeMomentumState) -> NDArray[np.floating]:
        if state.fine_detail_momentum.shape != (self.fine_grid.points,):
            raise ValueError("Composite fine detail has the wrong size.")
        return self.lifted_coarse_state(state.coarse_momentum) + self.local_detail_solve(
            state.fine_detail_momentum, state.patch.fine_mask
        )

    def reconstruct_fine_momentum(self, state: CompositeMomentumState) -> NDArray[np.floating]:
        return self.base_fine_momentum(state.coarse_momentum) + state.fine_detail_momentum

    def global_check(self, state: CompositeMomentumState) -> NDArray[np.floating]:
        return HelmholtzOperator(self.fine_grid).solve(
            self.reconstruct_fine_momentum(state)
        )


@dataclass(frozen=True, slots=True)
class HybridMCHState:
    positions: NDArray[np.floating]
    amplitudes: NDArray[np.floating]
    field_momentum: NDArray[np.floating]


def _periodic_interpolate(
    grid: PeriodicGrid, values: NDArray[np.floating], positions: NDArray[np.floating]
) -> NDArray[np.floating]:
    xp = np.concatenate((grid.x, [grid.x_min + grid.length]))
    fp = np.concatenate((np.asarray(values), [values[0]]))
    wrapped = (np.asarray(positions) - grid.x_min) % grid.length + grid.x_min
    return np.interp(wrapped, xp, fp)


def hybrid_total_state(state: HybridMCHState, grid: PeriodicGrid) -> NDArray[np.floating]:
    field = HelmholtzOperator(grid).solve(np.asarray(state.field_momentum))
    if state.positions.size:
        field = field + reconstruct_multipeakon(grid, state.positions, state.amplitudes)
    return np.asarray(field)


def hybrid_particle_velocity(
    state: HybridMCHState, grid: PeriodicGrid, *, alpha: float
) -> NDArray[np.floating]:
    if state.positions.size == 0:
        return np.empty(0, dtype=np.float64)
    field = HelmholtzOperator(grid).solve(np.asarray(state.field_momentum))
    field_slope = first_derivative(field, grid)
    q, p = np.asarray(state.positions), np.asarray(state.amplitudes)
    delta = q[:, None] - q[None, :]
    u_at_q = periodic_peakon_kernel(delta, grid.length) @ p
    u_at_q += _periodic_interpolate(grid, field, q)
    regular_slope = periodic_peakon_kernel_derivative(delta, grid.length) @ p
    regular_slope += _periodic_interpolate(grid, field_slope, q)
    self_slope = p * np.tanh(0.5 * grid.length)
    return np.asarray(alpha * (u_at_q**2 - regular_slope**2 - self_slope**2))


def hybrid_flux_and_source(
    state: HybridMCHState, grid: PeriodicGrid, parameters: ModifiedCHParameters
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    total = hybrid_total_state(state, grid)
    slope = first_derivative(total, grid)
    cell_flux = parameters.alpha * (total**2 - slope**2) * state.field_momentum
    face_flux = 0.5 * (cell_flux + np.roll(cell_flux, -1))
    source = -parameters.gamma * slope
    return np.asarray(face_flux), np.asarray(source)


@dataclass(slots=True)
class UniformHybridMCHReferenceSolver:
    grid: PeriodicGrid
    parameters: ModifiedCHParameters
    dt: float
    project_invariants: bool = True
    correction_flux_predictor: Callable[
        [HybridMCHState, PeriodicGrid, ModifiedCHParameters, float],
        NDArray[np.floating],
    ] | None = None

    def _flux_and_source(
        self, state: HybridMCHState
    ) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        flux, source = hybrid_flux_and_source(state, self.grid, self.parameters)
        if self.correction_flux_predictor is not None:
            correction = np.asarray(
                self.correction_flux_predictor(
                    state, self.grid, self.parameters, self.dt
                ), dtype=np.float64,
            )
            self.grid.validate_state(correction)
            flux = flux + correction
        return np.asarray(flux), np.asarray(source)

    def step(self, state: HybridMCHState) -> HybridMCHState:
        if self.dt <= 0.0:
            raise ValueError("Reference time step must be positive.")
        initial_total = hybrid_total_state(state, self.grid)
        target_mass = float(self.grid.h * np.sum(initial_total))
        target_energy = discrete_energy(initial_total, HelmholtzOperator(self.grid))
        flux0, source0 = self._flux_and_source(state)
        rhs0 = -finite_volume_divergence(flux0, self.grid.h) + source0
        velocity0 = hybrid_particle_velocity(state, self.grid, alpha=self.parameters.alpha)
        predictor = HybridMCHState(
            np.remainder(state.positions + self.dt * velocity0, self.grid.length),
            state.amplitudes.copy(),
            state.field_momentum + self.dt * rhs0,
        )
        flux1, source1 = self._flux_and_source(predictor)
        rhs1 = -finite_volume_divergence(flux1, self.grid.h) + source1
        velocity1 = hybrid_particle_velocity(predictor, self.grid, alpha=self.parameters.alpha)
        positions = np.remainder(
            state.positions + 0.5 * self.dt * (velocity0 + velocity1),
            self.grid.length,
        )
        field_momentum = state.field_momentum + 0.5 * self.dt * (rhs0 + rhs1)
        candidate = HybridMCHState(positions, state.amplitudes.copy(), field_momentum)
        if self.project_invariants:
            total = hybrid_total_state(candidate, self.grid)
            projected = project_to_mass_h1(
                total, target_mass, target_energy,
                ModifiedCH(self.grid, self.parameters),
            ).state
            atomic = (
                reconstruct_multipeakon(self.grid, positions, state.amplitudes)
                if positions.size else np.zeros(self.grid.points)
            )
            field_momentum = HelmholtzOperator(self.grid).apply(projected - atomic)
        return HybridMCHState(positions, state.amplitudes.copy(), field_momentum)

    def solve(self, initial: HybridMCHState, *, final_time: float) -> tuple[HybridMCHState, ...]:
        steps = round(final_time / self.dt)
        if steps < 1 or not np.isclose(steps * self.dt, final_time, atol=1.0e-12):
            raise ValueError("final_time must be a positive multiple of dt.")
        states = [initial]
        for _ in range(steps):
            states.append(self.step(states[-1]))
        return tuple(states)


@dataclass(frozen=True, slots=True)
class HybridAMRState:
    positions: NDArray[np.floating]
    amplitudes: NDArray[np.floating]
    composite_field: CompositeMomentumState


@dataclass(frozen=True, slots=True)
class HybridAMRStepDiagnostics:
    active_fraction: float
    local_green_sources: int
    full_fine_points: int
    backbone_mass_drift: float
    backbone_energy_drift: float
    composite_mass_drift: float
    composite_relative_energy_drift: float
    maximum_fine_cfl: float
    minimum_particle_separation: float


@dataclass(slots=True)
class HybridMCHAMRSolver:
    coarse_grid: PeriodicGrid
    parameters: ModifiedCHParameters
    dt: float
    maximum_cfl: float = 0.45
    collision_tolerance: float = 1.0e-8
    correction_flux_predictor: Callable[
        [HybridMCHState, PeriodicGrid, ModifiedCHParameters, float],
        NDArray[np.floating],
    ] | None = None

    def __post_init__(self) -> None:
        if self.dt <= 0.0 or not np.isfinite(self.dt):
            raise ValueError("AMR time step must be positive and finite.")
        if not 0.0 < self.maximum_cfl <= 1.0:
            raise ValueError("maximum_cfl must lie in (0, 1].")
        if self.collision_tolerance <= 0.0:
            raise ValueError("collision_tolerance must be positive.")

    def _minimum_separation(self, positions: NDArray[np.floating]) -> float:
        q = np.asarray(positions, dtype=np.float64)
        if q.size < 2:
            return self.coarse_grid.length
        distance = np.abs(
            (q[:, None] - q[None, :] + 0.5 * self.coarse_grid.length)
            % self.coarse_grid.length - 0.5 * self.coarse_grid.length
        )
        distance[np.eye(q.size, dtype=bool)] = np.inf
        return float(np.min(distance))

    def _flux_and_source(
        self, state: HybridMCHState, grid: PeriodicGrid, dt: float
    ) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        """Evaluate the conservative backbone plus an optional learned flux.

        The callback is deliberately flux-valued: its divergence telescopes on
        every periodic level, and the same callback is evaluated on coarse and
        fine faces before reflux.  A neural model can therefore change local
        transport but cannot inject momentum through an unconstrained source.
        """
        flux, source = hybrid_flux_and_source(state, grid, self.parameters)
        if self.correction_flux_predictor is None:
            return flux, source
        correction = np.asarray(
            self.correction_flux_predictor(state, grid, self.parameters, dt),
            dtype=np.float64,
        )
        grid.validate_state(correction)
        if not np.all(np.isfinite(correction)):
            raise FloatingPointError("learned AMR correction flux is not finite")
        return np.asarray(flux + correction), source

    def step(
        self, state: HybridAMRState
    ) -> tuple[HybridAMRState, HybridAMRStepDiagnostics]:
        green = HierarchicalPeriodicGreenSolver(self.coarse_grid)
        fine_grid = green.fine_grid
        patch = state.composite_field.patch
        if patch.coarse_mask.shape != (self.coarse_grid.points,):
            raise ValueError("The refinement patch does not match the coarse grid.")
        if self._minimum_separation(state.positions) <= self.collision_tolerance:
            raise RuntimeError("Hybrid AMR state is outside the pre-collision contract.")
        initial_fine_total = green.solve(state.composite_field)
        if state.positions.size:
            initial_fine_total = initial_fine_total + reconstruct_multipeakon(
                fine_grid, state.positions, state.amplitudes
            )
        target_composite_mass = float(fine_grid.h * np.sum(initial_fine_total))
        target_composite_energy = discrete_energy(
            initial_fine_total, HelmholtzOperator(fine_grid)
        )
        coarse_hybrid = HybridMCHState(
            state.positions, state.amplitudes, state.composite_field.coarse_momentum
        )
        initial_total = hybrid_total_state(coarse_hybrid, self.coarse_grid)
        target_mass = float(self.coarse_grid.h * np.sum(initial_total))
        target_energy = discrete_energy(initial_total, HelmholtzOperator(self.coarse_grid))
        coarse_flux, coarse_source = self._flux_and_source(
            coarse_hybrid, self.coarse_grid, self.dt
        )
        coarse_provisional = (
            state.composite_field.coarse_momentum
            - self.dt * finite_volume_divergence(coarse_flux, self.coarse_grid.h)
            + self.dt * coarse_source
        )

        fine_field_momentum = green.reconstruct_fine_momentum(state.composite_field)
        q = state.positions.copy()
        fine_fluxes: list[NDArray[np.floating]] = []
        maximum_fine_cfl = 0.0
        for _ in range(2):
            fine_hybrid = HybridMCHState(q, state.amplitudes, fine_field_momentum)
            flux, source = self._flux_and_source(
                fine_hybrid, fine_grid, 0.5 * self.dt
            )
            fine_fluxes.append(flux)
            total = hybrid_total_state(fine_hybrid, fine_grid)
            slope = first_derivative(total, fine_grid)
            characteristic_speed = np.max(
                np.abs(self.parameters.alpha * (total**2 - slope**2))
            )
            fine_cfl = float(0.5 * self.dt * characteristic_speed / fine_grid.h)
            maximum_fine_cfl = max(maximum_fine_cfl, fine_cfl)
            if fine_cfl > self.maximum_cfl:
                raise RuntimeError(
                    f"Fine-level CFL {fine_cfl:.6g} exceeds {self.maximum_cfl:.6g}."
                )
            rhs = -finite_volume_divergence(flux, fine_grid.h) + source
            fine_field_momentum = np.where(
                patch.fine_mask,
                fine_field_momentum + 0.5 * self.dt * rhs,
                fine_field_momentum,
            )
            velocity = hybrid_particle_velocity(
                fine_hybrid, fine_grid, alpha=self.parameters.alpha
            )
            q = np.remainder(q + 0.5 * self.dt * velocity, fine_grid.length)
            if self._minimum_separation(q) <= self.collision_tolerance:
                raise RuntimeError("Hybrid AMR step reached the collision guard.")
        time_averaged_flux = 0.5 * (fine_fluxes[0] + fine_fluxes[1])
        synchronized = reflux_two_to_one(
            coarse_provisional, fine_field_momentum, coarse_flux,
            time_averaged_flux, patch.coarse_mask,
            dt=self.dt, coarse_h=self.coarse_grid.h,
        )
        candidate = HybridMCHState(q, state.amplitudes.copy(), synchronized)
        total = hybrid_total_state(candidate, self.coarse_grid)
        projected = project_to_mass_h1(
            total, target_mass, target_energy,
            ModifiedCH(self.coarse_grid, self.parameters),
        ).state
        atomic = (
            reconstruct_multipeakon(self.coarse_grid, q, state.amplitudes)
            if q.size else np.zeros(self.coarse_grid.points)
        )
        projected_coarse_momentum = HelmholtzOperator(self.coarse_grid).apply(
            projected - atomic
        )
        new_base = HierarchicalPeriodicGreenSolver(self.coarse_grid).base_fine_momentum(
            projected_coarse_momentum
        )
        detail = np.where(patch.fine_mask, fine_field_momentum - new_base, 0.0)
        active_points = int(np.count_nonzero(patch.fine_mask))
        if active_points:
            provisional_composite = CompositeMomentumState(
                projected_coarse_momentum, detail, patch
            )
            provisional_total = green.solve(provisional_composite)
            if q.size:
                provisional_total = provisional_total + reconstruct_multipeakon(
                    fine_grid, q, state.amplitudes
                )
            mass_defect = target_composite_mass - float(
                fine_grid.h * np.sum(provisional_total)
            )
            # The periodic Helmholtz inverse preserves the zero Fourier mode.
            # A patch-constant momentum correction therefore closes composite
            # mass without introducing a nonlocal fine-momentum source.
            detail[patch.fine_mask] += mass_defect / (fine_grid.h * active_points)
        next_state = HybridAMRState(
            q, state.amplitudes.copy(),
            CompositeMomentumState(projected_coarse_momentum, detail, patch),
        )
        final_total = hybrid_total_state(
            HybridMCHState(q, state.amplitudes, projected_coarse_momentum),
            self.coarse_grid,
        )
        backbone_mass_drift = abs(float(self.coarse_grid.h * np.sum(final_total)) - target_mass)
        backbone_energy_drift = abs(
            discrete_energy(final_total, HelmholtzOperator(self.coarse_grid)) - target_energy
        )
        final_fine_total = green.solve(next_state.composite_field)
        if q.size:
            final_fine_total = final_fine_total + reconstruct_multipeakon(
                fine_grid, q, state.amplitudes
            )
        composite_mass_drift = abs(
            float(fine_grid.h * np.sum(final_fine_total)) - target_composite_mass
        )
        composite_relative_energy_drift = abs(
            discrete_energy(final_fine_total, HelmholtzOperator(fine_grid))
            - target_composite_energy
        ) / max(target_composite_energy, np.finfo(float).eps)
        return next_state, HybridAMRStepDiagnostics(
            patch.active_fraction,
            int(np.count_nonzero(detail)),
            fine_grid.points,
            backbone_mass_drift,
            backbone_energy_drift,
            composite_mass_drift,
            composite_relative_energy_drift,
            maximum_fine_cfl,
            self._minimum_separation(q),
        )
