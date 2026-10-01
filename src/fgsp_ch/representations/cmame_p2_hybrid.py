"""Exact bookkeeping for the CMAME Eulerian--peakon representation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from fgsp_ch.discretization.grids import PeriodicGrid
from fgsp_ch.discretization.helmholtz import HelmholtzOperator
from fgsp_ch.geometry.metric import discrete_inner
from fgsp_ch.solvers.mch_multipeakon import mch_particle_h1
from fgsp_ch.solvers.multipeakon import reconstruct_multipeakon


def atomic_measure_weight(length: float) -> float:
    """Mass of the unit-peak periodic Green profile.

    For the project's normalized kernel ``K_L(0)=1``, the distributional
    identity is ``(1-d_xx)K_L = c_L delta_0`` with
    ``c_L = 2 tanh(L/2)``.
    """
    if not np.isfinite(length) or length <= 0.0:
        raise ValueError("length must be positive and finite")
    return float(2.0 * np.tanh(0.5 * length))


def periodic_interpolate(
    grid: PeriodicGrid,
    values: NDArray[np.floating],
    positions: NDArray[np.floating],
) -> NDArray[np.floating]:
    grid.validate_state(np.asarray(values))
    q = np.asarray(positions, dtype=np.float64)
    if q.ndim != 1 or not np.all(np.isfinite(q)):
        raise ValueError("positions must be a finite one-dimensional array")
    xp = np.concatenate((grid.x, [grid.x_min + grid.length]))
    fp = np.concatenate((np.asarray(values, dtype=np.float64), [values[0]]))
    wrapped = (q - grid.x_min) % grid.length + grid.x_min
    return np.asarray(np.interp(wrapped, xp, fp))


@dataclass(frozen=True, slots=True)
class CMAMEHybridState:
    """Regular grid momentum plus an atomic positive-peakon measure."""

    regular_momentum: NDArray[np.floating]
    positions: NDArray[np.floating]
    amplitudes: NDArray[np.floating]

    def validate(self, grid: PeriodicGrid) -> None:
        grid.validate_state(np.asarray(self.regular_momentum))
        q = np.asarray(self.positions)
        a = np.asarray(self.amplitudes)
        if q.ndim != 1 or q.shape != a.shape:
            raise ValueError("positions and amplitudes must be equal 1-D arrays")
        if not np.all(np.isfinite(q)) or not np.all(np.isfinite(a)):
            raise FloatingPointError("particle state contains NaN or Inf")
        if np.any(a <= 0.0):
            raise ValueError("P2.0 is restricted to positive pre-collision peakons")


def regular_state(state: CMAMEHybridState, grid: PeriodicGrid) -> NDArray[np.floating]:
    state.validate(grid)
    return HelmholtzOperator(grid).solve(np.asarray(state.regular_momentum))


def atomic_state(state: CMAMEHybridState, grid: PeriodicGrid) -> NDArray[np.floating]:
    state.validate(grid)
    if not state.positions.size:
        return np.zeros(grid.points, dtype=np.float64)
    return reconstruct_multipeakon(grid, state.positions, state.amplitudes)


def total_state(state: CMAMEHybridState, grid: PeriodicGrid) -> NDArray[np.floating]:
    return np.asarray(regular_state(state, grid) + atomic_state(state, grid))


def hybrid_mass(state: CMAMEHybridState, grid: PeriodicGrid) -> float:
    """Representation-exact action of the total momentum on the constant 1."""
    state.validate(grid)
    regular = grid.h * np.sum(state.regular_momentum)
    atomic = atomic_measure_weight(grid.length) * np.sum(state.amplitudes)
    return float(regular + atomic)


def hybrid_h1_squared(state: CMAMEHybridState, grid: PeriodicGrid) -> float:
    """Representation-exact ``<u,m>`` including regular--atomic cross terms."""
    state.validate(grid)
    field = regular_state(state, grid)
    regular = discrete_inner(field, np.asarray(state.regular_momentum), grid)
    if not state.positions.size:
        return float(regular)
    weight = atomic_measure_weight(grid.length)
    cross = 2.0 * weight * np.dot(
        state.amplitudes, periodic_interpolate(grid, field, state.positions)
    )
    particle = mch_particle_h1(state.positions, state.amplitudes, grid.length)
    result = float(regular + cross + particle)
    if result < -100.0 * np.finfo(float).eps or not np.isfinite(result):
        raise FloatingPointError("hybrid H1 functional is invalid")
    return result


def hybrid_momentum_action(
    state: CMAMEHybridState,
    grid: PeriodicGrid,
    test_values: NDArray[np.floating],
) -> float:
    """Weak action ``<m,phi>`` without smearing the atomic measure."""
    state.validate(grid)
    grid.validate_state(np.asarray(test_values))
    regular = discrete_inner(state.regular_momentum, test_values, grid)
    atomic = atomic_measure_weight(grid.length) * np.dot(
        state.amplitudes,
        periodic_interpolate(grid, test_values, state.positions),
    )
    return float(regular + atomic)

