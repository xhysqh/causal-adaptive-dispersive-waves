from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from fgsp_ch.discretization.grids import PeriodicGrid
from fgsp_ch.equations.modified_ch import ModifiedCH, ModifiedCHParameters
from fgsp_ch.solvers.mch_energy_projected import MassH1ProjectedMCHSolver
from fgsp_ch.solvers.mch_multipeakon import conservative_periodic_mch_velocity


def normalized_mch_parameters(
    alpha: float, gamma: float, length: float, dt: float, h: float
) -> NDArray[np.floating]:
    """Eleven causal, dimensionless global channels used by C-PMWNO."""
    return np.asarray(
        [alpha, gamma, length / (2.0 * np.pi), dt / 0.01, h / 0.1,
         h / 0.1, 1.0, 1.0, 1.0, 0.0, 0.0], dtype=np.float64
    )


def peak_tokens_from_particles(
    positions: NDArray[np.floating], amplitudes: NDArray[np.floating],
    length: float, h: float, maximum_atoms: int,
) -> tuple[NDArray[np.floating], NDArray[np.bool_]]:
    count = positions.size
    if count > maximum_atoms:
        raise ValueError("maximum_atoms would truncate a particle sample.")
    tokens = np.zeros((maximum_atoms, 8), dtype=np.float64)
    mask = np.zeros(maximum_atoms, dtype=bool)
    scale = max(float(np.max(np.abs(amplitudes))), np.finfo(float).eps)
    distance = np.abs(
        (positions[:, None] - positions[None, :] + 0.5 * length) % length
        - 0.5 * length
    )
    distance[np.eye(count, dtype=bool)] = np.inf
    nearest = np.min(distance, axis=1) if count > 1 else np.full(count, length)
    phase = 2.0 * np.pi * positions / length
    tokens[:count, 0] = np.sin(phase)
    tokens[:count, 1] = np.cos(phase)
    tokens[:count, 2] = amplitudes / scale
    tokens[:count, 3] = np.sign(amplitudes) * np.log1p(np.abs(amplitudes) / scale)
    tokens[:count, 4:6] = nearest[:, None] / h
    mask[:count] = True
    return tokens, mask


def cluster_token(
    positions: NDArray[np.floating], amplitudes: NDArray[np.floating],
    length: float, h: float,
) -> NDArray[np.floating]:
    weights = np.abs(amplitudes)
    angles = 2.0 * np.pi * positions / length
    vector = np.sum(weights * np.exp(1j * angles))
    center = (np.angle(vector) % (2.0 * np.pi)) * length / (2.0 * np.pi)
    displacement = (positions - center + 0.5 * length) % length - 0.5 * length
    width = max(float(np.sqrt(np.average(displacement**2, weights=weights))), h)
    moments = [float(np.sum(amplitudes * (displacement / width) ** order)) for order in range(1, 5)]
    return np.asarray(
        [np.sin(2.0 * np.pi * center / length), np.cos(2.0 * np.pi * center / length),
         np.sum(amplitudes), np.log1p(width / h), positions.size, *moments],
        dtype=np.float64,
    )


def _center_from_token(token: NDArray[np.floating], length: float) -> float:
    return float((np.arctan2(token[0], token[1]) % (2.0 * np.pi)) * length / (2.0 * np.pi))


@dataclass(frozen=True, slots=True)
class ParticleBranchSample:
    group: str
    split: str
    tokens: NDArray[np.floating]
    positions: NDArray[np.floating]
    amplitudes: NDArray[np.floating]
    mask: NDArray[np.bool_]
    parameters: NDArray[np.floating]
    length: float
    position_rate: NDArray[np.floating]
    amplitude_rate: NDArray[np.floating]


@dataclass(frozen=True, slots=True)
class ClusterBranchSample:
    group: str
    split: str
    token: NDArray[np.floating]
    parameters: NDArray[np.floating]
    target: NDArray[np.floating]


@dataclass(frozen=True, slots=True)
class FieldBranchSample:
    group: str
    split: str
    field: NDArray[np.floating]
    parameters: NDArray[np.floating]
    target_momentum_rate: NDArray[np.floating]
    spatial_error_levels: tuple[NDArray[np.floating], ...]
    temporal_error_ratio: float
    spatial_error_ratio: float


def _separated_positions(rng: np.random.Generator, count: int, length: float) -> NDArray[np.floating]:
    for _ in range(1000):
        values = np.sort(rng.uniform(0.0, length, count))
        distance = np.abs((values[:, None] - values[None, :] + 0.5 * length) % length - 0.5 * length)
        distance[np.eye(count, dtype=bool)] = np.inf
        if np.min(distance) > 0.45:
            return values
    raise RuntimeError("Unable to sample separated particles.")


def build_particle_branch_samples(
    *, seed: int, counts: dict[str, int], maximum_atoms: int = 8,
) -> list[ParticleBranchSample]:
    rng = np.random.default_rng(seed)
    rows: list[ParticleBranchSample] = []
    for split, samples in counts.items():
        for index in range(samples):
            length = float(rng.uniform(2.0 * np.pi, 9.0))
            alpha = float(rng.uniform(0.7, 1.4))
            count = int(rng.integers(1, 5))
            positions = _separated_positions(rng, count, length)
            amplitudes = rng.uniform(0.35, 1.2, count)
            h, dt = length / 128.0, 0.005
            tokens, mask = peak_tokens_from_particles(
                positions, amplitudes, length, h, maximum_atoms
            )
            padded_q = np.zeros(maximum_atoms)
            padded_p = np.zeros(maximum_atoms)
            target = np.zeros(maximum_atoms)
            padded_q[:count], padded_p[:count] = positions, amplitudes
            target[:count] = conservative_periodic_mch_velocity(
                positions, amplitudes, length, alpha=alpha
            )
            rows.append(ParticleBranchSample(
                f"particle_{split}_{index}", split, tokens, padded_q, padded_p, mask,
                normalized_mch_parameters(alpha, 0.0, length, dt, h), length,
                target, np.zeros(maximum_atoms),
            ))
    return rows


def build_cluster_branch_samples(
    *, seed: int, counts: dict[str, int],
) -> list[ClusterBranchSample]:
    rng = np.random.default_rng(seed)
    rows: list[ClusterBranchSample] = []
    dt = 2.5e-4
    for split, samples in counts.items():
        for index in range(samples):
            length = float(rng.uniform(2.0 * np.pi, 9.0))
            alpha = float(rng.uniform(0.7, 1.4))
            count = int(rng.integers(2, 4))
            center = float(rng.uniform(0.0, length))
            offsets = np.linspace(-1.0, 1.0, count) * rng.uniform(0.04, 0.12)
            positions = np.remainder(center + offsets, length)
            amplitudes = rng.uniform(0.4, 1.1, count)
            h = length / 256.0
            current = cluster_token(positions, amplitudes, length, h)
            velocity = conservative_periodic_mch_velocity(
                positions, amplitudes, length, alpha=alpha
            )
            future = cluster_token(
                np.remainder(positions + dt * velocity, length), amplitudes, length, h
            )
            center_rate = (
                (_center_from_token(future, length) - _center_from_token(current, length)
                 + 0.5 * length) % length - 0.5 * length
            ) / dt
            target = np.concatenate(([center_rate, (future[3] - current[3]) / dt], (future[5:] - current[5:]) / dt))
            rows.append(ClusterBranchSample(
                f"cluster_{split}_{index}", split, current,
                normalized_mch_parameters(alpha, 0.0, length, dt, h), target,
            ))
    return rows


def _smooth_state(grid: PeriodicGrid, coefficients: NDArray[np.floating]) -> NDArray[np.floating]:
    phase = 2.0 * np.pi * grid.x / grid.length
    state = np.full(grid.points, coefficients[0])
    modes = (1, 2, 3)
    for offset, mode in enumerate(modes):
        state += coefficients[1 + 2 * offset] * np.sin(mode * phase)
        state += coefficients[2 + 2 * offset] * np.cos(mode * phase)
    return np.asarray(state)


def _two_half_steps(equation: ModifiedCH, state: NDArray[np.floating], dt: float) -> NDArray[np.floating]:
    solver = MassH1ProjectedMCHSolver(equation, 0.5 * dt)
    return solver.step(solver.step(state).state).state


def _average_pairs(values: NDArray[np.floating], levels: int) -> tuple[NDArray[np.floating], ...]:
    rows: list[NDArray[np.floating]] = []
    current = values
    for _ in range(levels):
        current = 0.5 * (current[0::2] + current[1::2])
        rows.append(np.asarray(current))
    return tuple(rows)


def build_field_branch_samples(
    *, seed: int, counts: dict[str, int], points: int = 64,
    wavelet_levels: int = 2, tolerance: float = 1.0e-4,
) -> list[FieldBranchSample]:
    rng = np.random.default_rng(seed)
    rows: list[FieldBranchSample] = []
    for split, samples in counts.items():
        for index in range(samples):
            length = float(rng.uniform(2.0 * np.pi, 9.0))
            alpha = float(rng.uniform(0.7, 1.4))
            gamma = float(rng.uniform(0.0, 0.2))
            dt = float(rng.choice([0.0025, 0.005, 0.0075]))
            coefficients = np.concatenate(([rng.uniform(0.08, 0.22)], rng.uniform(-0.08, 0.08, 6)))
            coarse_grid = PeriodicGrid(points, length)
            fine_grid = PeriodicGrid(2 * points, length)
            coarse_equation = ModifiedCH(coarse_grid, ModifiedCHParameters(alpha, gamma))
            fine_equation = ModifiedCH(fine_grid, ModifiedCHParameters(alpha, gamma))
            coarse_state = _smooth_state(coarse_grid, coefficients)
            fine_state = _smooth_state(fine_grid, coefficients)
            coarse_full = MassH1ProjectedMCHSolver(coarse_equation, dt).step(coarse_state).state
            coarse_half = _two_half_steps(coarse_equation, coarse_state, dt)
            fine_half = _two_half_steps(fine_equation, fine_state, dt)
            reference = fine_half[::2]
            reference_momentum = coarse_equation.helmholtz.apply(reference)
            current_momentum = coarse_equation.helmholtz.apply(coarse_state)
            target_rate = (reference_momentum - current_momentum) / dt
            state_scale = max(float(np.sqrt(np.mean(reference**2))), np.finfo(float).eps)
            spatial_map = np.abs(reference - coarse_half) / (state_scale * tolerance)
            temporal = float(np.sqrt(np.mean((coarse_half - coarse_full) ** 2)) / (state_scale * tolerance))
            spatial = float(np.sqrt(np.mean((reference - coarse_half) ** 2)) / (state_scale * tolerance))
            field = np.stack((
                current_momentum, coarse_state,
                np.zeros(points), np.ones(points),
            ))
            rows.append(FieldBranchSample(
                f"field_{split}_{index}", split, field,
                normalized_mch_parameters(alpha, gamma, length, dt, coarse_grid.h),
                target_rate, _average_pairs(spatial_map, wavelet_levels), temporal, spatial,
            ))
    return rows
