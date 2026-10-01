"""Hierarchical a-posteriori indicators for periodic hybrid mCH states.

Only compatible M1.0.1 advances enter the online estimator.  Refined
Fourier--DOP853 trajectories are deliberately absent from this module: they
are offline truth used by the experiment driver, never an estimator input.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.signal import resample

from fgsp_ch.discretization.difference import first_derivative
from fgsp_ch.discretization.grids import PeriodicGrid
from fgsp_ch.discretization.helmholtz import HelmholtzOperator
from fgsp_ch.equations.modified_ch import ModifiedCHParameters
from fgsp_ch.solvers.mch_compatible_invariant import CompatibleInvariantHybridMCHSolver
from fgsp_ch.solvers.mch_hybrid_amr import HybridMCHState
from fgsp_ch.solvers.multipeakon import reconstruct_multipeakon


@dataclass(frozen=True, slots=True)
class HybridErrorComponents:
    temporal: float
    field: float
    particle: float
    interface: float
    hierarchical_total: float
    local_indicators: NDArray[np.floating]

    def feature_vector(self) -> NDArray[np.floating]:
        return np.asarray(
            [self.temporal, self.field, self.particle, self.interface],
            dtype=np.float64,
        )


def _field_on_grid(
    state: HybridMCHState, source: PeriodicGrid, target: PeriodicGrid
) -> NDArray[np.floating]:
    field = HelmholtzOperator(source).solve(np.asarray(state.field_momentum))
    if source.points != target.points:
        field = np.asarray(resample(field, target.points)).real
    return np.asarray(field, dtype=np.float64)


def _particle_on_grid(state: HybridMCHState, target: PeriodicGrid) -> NDArray[np.floating]:
    if not state.positions.size:
        return np.zeros(target.points, dtype=np.float64)
    return reconstruct_multipeakon(target, state.positions, state.amplitudes)


def transfer_hybrid_state(
    state: HybridMCHState, source: PeriodicGrid, target: PeriodicGrid
) -> HybridMCHState:
    """Transfer only the regular field; the atomic measure remains exact."""
    field = _field_on_grid(state, source, target)
    momentum = HelmholtzOperator(target).apply(field)
    return HybridMCHState(
        np.asarray(state.positions, dtype=np.float64).copy(),
        np.asarray(state.amplitudes, dtype=np.float64).copy(),
        np.asarray(momentum, dtype=np.float64),
    )


def h1_norm(values: NDArray[np.floating], grid: PeriodicGrid) -> float:
    data = np.asarray(values, dtype=np.float64)
    grid.validate_state(data)
    slope = first_derivative(data, grid)
    return float(np.sqrt(max(grid.h * np.sum(data * data + slope * slope), 0.0)))


def hybrid_total_h1_norm(
    state: HybridMCHState,
    source_grid: PeriodicGrid,
    evaluation_grid: PeriodicGrid | None = None,
) -> float:
    """Return the total represented-state ``H1`` norm on one audit grid.

    Both the regular field and the atomic peakon field are included.  This is
    intentionally different from measuring the distance to an atomic-only
    copy of ``state``: that legacy construction vanishes for pure-particle
    states and is therefore not a valid relative-error denominator.
    """
    target = source_grid if evaluation_grid is None else evaluation_grid
    total = _field_on_grid(state, source_grid, target) + _particle_on_grid(state, target)
    result = h1_norm(total, target)
    if not np.isfinite(result) or result <= np.finfo(float).eps:
        raise ValueError("hybrid reference state has no resolvable total H1 scale")
    return result


def component_distance(
    left: HybridMCHState,
    left_grid: PeriodicGrid,
    right: HybridMCHState,
    right_grid: PeriodicGrid,
    evaluation_grid: PeriodicGrid,
) -> dict[str, float | NDArray[np.floating]]:
    """Return regular, atomic, total and interaction H1 discrepancies."""
    regular_error = _field_on_grid(left, left_grid, evaluation_grid) - _field_on_grid(
        right, right_grid, evaluation_grid
    )
    particle_error = _particle_on_grid(left, evaluation_grid) - _particle_on_grid(
        right, evaluation_grid
    )
    total_error = regular_error + particle_error
    regular = h1_norm(regular_error, evaluation_grid)
    particle = h1_norm(particle_error, evaluation_grid)
    total = h1_norm(total_error, evaluation_grid)
    # Exact polarization remainder.  It detects cancellation/amplification at
    # the Eulerian--particle interface without pretending orthogonality.
    cross = total * total - regular * regular - particle * particle
    interface = float(np.sqrt(abs(cross)))
    return {
        "regular": regular,
        "particle": particle,
        "total": total,
        "interface": interface,
        "total_error": total_error,
    }


def compatible_rollout(
    initial: HybridMCHState,
    grid: PeriodicGrid,
    parameters: ModifiedCHParameters,
    dt: float,
    final_time: float,
    nonlinear: dict[str, float | int],
) -> HybridMCHState:
    solver = CompatibleInvariantHybridMCHSolver(
        grid,
        parameters,
        dt,
        nonlinear_tolerance=float(nonlinear["tolerance"]),
        maximum_iterations=int(nonlinear["maximum_iterations"]),
        damping=float(nonlinear.get("damping", 1.0)),
    )
    states, _ = solver.solve(initial, final_time=final_time)
    return states[-1]


def local_h1_indicators(error: NDArray[np.floating], fine_grid: PeriodicGrid) -> NDArray[np.floating]:
    """Aggregate the fine H1 error density into conservative 2:1 coarse cells."""
    data = np.asarray(error, dtype=np.float64)
    fine_grid.validate_state(data)
    if fine_grid.points % 2:
        raise ValueError("local 2:1 indicators require an even fine grid")
    density = fine_grid.h * (data * data + first_derivative(data, fine_grid) ** 2)
    return np.sqrt(np.maximum(density.reshape(-1, 2).sum(axis=1), 0.0))


def dorfler_mark(indicators: NDArray[np.floating], theta: float) -> NDArray[np.bool_]:
    values = np.asarray(indicators, dtype=np.float64)
    if values.ndim != 1 or not values.size or np.any(values < 0.0) or not np.all(np.isfinite(values)):
        raise ValueError("indicators must be a nonempty finite nonnegative vector")
    if not 0.0 < theta <= 1.0:
        raise ValueError("theta must lie in (0,1]")
    energy = values * values
    total = float(np.sum(energy))
    mask = np.zeros(values.size, dtype=bool)
    if total == 0.0:
        mask[int(np.argmax(values))] = True
        return mask
    order = np.argsort(energy)[::-1]
    count = int(np.searchsorted(np.cumsum(energy[order]), theta * total, side="left")) + 1
    mask[order[:count]] = True
    return mask


def hierarchical_estimate(
    initial: HybridMCHState,
    grid: PeriodicGrid,
    parameters: ModifiedCHParameters,
    dt: float,
    final_time: float,
    nonlinear: dict[str, float | int],
    *,
    spatial_factor: int = 2,
    time_order: float = 2.0,
    spatial_order: float = 1.0,
) -> tuple[HybridMCHState, HybridErrorComponents]:
    """Compute a four-component estimator without reference-solution access."""
    if spatial_factor != 2:
        raise ValueError("M1.1 currently certifies only conservative 2:1 nesting")
    coarse = compatible_rollout(initial, grid, parameters, dt, final_time, nonlinear)
    half = compatible_rollout(initial, grid, parameters, 0.5 * dt, final_time, nonlinear)
    fine_grid = PeriodicGrid(2 * grid.points, grid.length, grid.x_min)
    fine_initial = transfer_hybrid_state(initial, grid, fine_grid)
    fine = compatible_rollout(fine_initial, fine_grid, parameters, dt, final_time, nonlinear)

    time_distance = component_distance(coarse, grid, half, grid, grid)
    space_distance = component_distance(coarse, grid, fine, fine_grid, fine_grid)
    reference_scale = max(
        h1_norm(
            _field_on_grid(fine, fine_grid, fine_grid) + _particle_on_grid(fine, fine_grid),
            fine_grid,
        ),
        np.finfo(float).eps,
    )
    temporal = float(time_distance["total"]) / (2.0**time_order - 1.0) / reference_scale
    denominator = 2.0**spatial_order - 1.0
    field = float(space_distance["regular"]) / denominator / reference_scale
    particle = float(space_distance["particle"]) / denominator / reference_scale
    interface = float(space_distance["interface"]) / denominator / reference_scale
    hierarchical_total = float(space_distance["total"]) / denominator / reference_scale
    local = local_h1_indicators(np.asarray(space_distance["total_error"]), fine_grid) / reference_scale
    values = np.asarray([temporal, field, particle, interface, hierarchical_total, *local])
    if not np.all(np.isfinite(values)):
        raise FloatingPointError("M1.1 estimator produced NaN or Inf")
    return coarse, HybridErrorComponents(temporal, field, particle, interface, hierarchical_total, local)
