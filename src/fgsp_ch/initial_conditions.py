from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from fgsp_ch.discretization.grids import PeriodicGrid


def periodic_peakon(
    grid: PeriodicGrid,
    *,
    amplitude: float,
    center: float,
    width: float = 1.0,
) -> NDArray[np.floating]:
    """Periodic Helmholtz peakon.

    The profile is
    ``a cosh((L/2-d_L(x,c))/width) / cosh(L/(2 width))``.  For
    ``width=1`` it is the periodic Green-function shape of ``I-d_xx``,
    normalized to attain ``amplitude`` at the peak.
    """
    if width <= 0.0 or not np.isfinite(width):
        raise ValueError("Peakon width must be positive and finite.")
    distance = np.abs(grid.periodic_distance(grid.x, center))
    numerator = np.cosh((0.5 * grid.length - distance) / width)
    denominator = np.cosh(0.5 * grid.length / width)
    return np.asarray(amplitude * numerator / denominator, dtype=grid.dtype)


def periodic_exponential_cusp(
    grid: PeriodicGrid,
    *,
    amplitude: float,
    center: float,
    width: float = 1.0,
) -> NDArray[np.floating]:
    """Periodicized exponential cusp used only as a non-smooth stress test."""
    if width <= 0.0 or not np.isfinite(width):
        raise ValueError("Cusp width must be positive and finite.")
    distance = np.abs(grid.periodic_distance(grid.x, center))
    return np.asarray(amplitude * np.exp(-distance / width), dtype=grid.dtype)


def multi_peakon(
    grid: PeriodicGrid,
    amplitudes: Sequence[float],
    centers: Sequence[float],
    *,
    width: float = 1.0,
) -> NDArray[np.floating]:
    """Superpose one or more periodic peakons."""
    if len(amplitudes) != len(centers) or len(amplitudes) == 0:
        raise ValueError("Amplitudes and centers must have equal nonzero length.")
    state = np.zeros(grid.points, dtype=grid.dtype)
    for amplitude, center in zip(amplitudes, centers, strict=True):
        state += periodic_peakon(
            grid, amplitude=float(amplitude), center=float(center), width=width
        )
    return state


def gaussian(
    grid: PeriodicGrid,
    *,
    amplitude: float,
    center: float,
    width: float,
) -> NDArray[np.floating]:
    """Periodic Gaussian based on shortest periodic distance."""
    if width <= 0.0 or not np.isfinite(width):
        raise ValueError("Gaussian width must be positive and finite.")
    distance = grid.periodic_distance(grid.x, center)
    return np.asarray(
        amplitude * np.exp(-0.5 * (distance / width) ** 2),
        dtype=grid.dtype,
    )


def perturbed_peakon(
    grid: PeriodicGrid,
    *,
    amplitude: float,
    center: float,
    width: float,
    perturbation_amplitudes: Sequence[float],
    perturbation_modes: Sequence[int],
    perturbation_phases: Sequence[float] | None = None,
) -> NDArray[np.floating]:
    """Return a periodic peakon with a deterministic smooth Fourier perturbation."""
    if len(perturbation_amplitudes) != len(perturbation_modes):
        raise ValueError("Perturbation amplitudes and modes must have equal length.")
    phases = (
        [0.0] * len(perturbation_modes)
        if perturbation_phases is None
        else list(perturbation_phases)
    )
    if len(phases) != len(perturbation_modes):
        raise ValueError("Perturbation phases and modes must have equal length.")
    state = periodic_peakon(
        grid, amplitude=amplitude, center=center, width=width
    )
    coordinate = 2.0 * np.pi * (grid.x - grid.x_min) / grid.length
    for coefficient, mode, phase in zip(
        perturbation_amplitudes, perturbation_modes, phases, strict=True
    ):
        if int(mode) < 1:
            raise ValueError("Perturbation modes must be positive integers.")
        state += float(coefficient) * np.cos(int(mode) * coordinate + float(phase))
    return np.asarray(state, dtype=grid.dtype)


def random_smooth(
    grid: PeriodicGrid,
    *,
    amplitude: float,
    modes: int,
    decay: float,
    seed: int,
) -> NDArray[np.floating]:
    """Create a reproducible mean-zero smooth periodic Fourier field.

    Mode coefficients are independent standard normals multiplied by
    ``mode**(-decay)``.  The result is normalized to the requested maximum
    absolute amplitude, so the random seed changes shape but not scale.
    """
    if modes < 1 or modes >= grid.points // 2:
        raise ValueError("Random smooth modes must lie in [1, N/2).")
    if decay <= 0.0 or not np.isfinite(decay):
        raise ValueError("Random smooth decay must be positive and finite.")
    if amplitude <= 0.0 or not np.isfinite(amplitude):
        raise ValueError("Random smooth amplitude must be positive and finite.")
    rng = np.random.default_rng(seed)
    coordinate = 2.0 * np.pi * (grid.x - grid.x_min) / grid.length
    state = np.zeros(grid.points, dtype=np.float64)
    for mode in range(1, modes + 1):
        scale = mode ** (-decay)
        cosine, sine = rng.normal(size=2)
        state += scale * (
            cosine * np.cos(mode * coordinate)
            + sine * np.sin(mode * coordinate)
        )
    maximum = float(np.max(np.abs(state)))
    if maximum <= np.finfo(float).eps:
        raise FloatingPointError("Random smooth field has numerically zero amplitude.")
    return np.asarray(amplitude * state / maximum, dtype=grid.dtype)


def make_initial_condition(
    grid: PeriodicGrid, config: dict[str, object]
) -> NDArray[np.floating]:
    """Construct a configured Gaussian, peakon, multi-peakon, or cusp."""
    kind = str(config.get("kind", ""))
    width = float(config.get("width", 1.0))
    if kind == "gaussian":
        return gaussian(
            grid,
            amplitude=float(config["amplitude"]),
            center=float(config["center"]),
            width=width,
        )
    if kind == "single_peakon":
        return periodic_peakon(
            grid,
            amplitude=float(config["amplitude"]),
            center=float(config["center"]),
            width=width,
        )
    if kind == "multi_peakon":
        amplitudes = [float(value) for value in config["amplitudes"]]  # type: ignore[union-attr]
        centers = [float(value) for value in config["centers"]]  # type: ignore[union-attr]
        return multi_peakon(grid, amplitudes, centers, width=width)
    if kind == "exponential_cusp":
        return periodic_exponential_cusp(
            grid,
            amplitude=float(config["amplitude"]),
            center=float(config["center"]),
            width=width,
        )
    if kind == "perturbed_peakon":
        return perturbed_peakon(
            grid,
            amplitude=float(config["amplitude"]),
            center=float(config["center"]),
            width=width,
            perturbation_amplitudes=[
                float(value) for value in config["perturbation_amplitudes"]  # type: ignore[union-attr]
            ],
            perturbation_modes=[
                int(value) for value in config["perturbation_modes"]  # type: ignore[union-attr]
            ],
            perturbation_phases=[
                float(value)
                for value in config.get("perturbation_phases", [])  # type: ignore[union-attr]
            ]
            or None,
        )
    if kind == "random_smooth":
        return random_smooth(
            grid,
            amplitude=float(config["amplitude"]),
            modes=int(config["modes"]),
            decay=float(config["decay"]),
            seed=int(config["seed"]),
        )
    raise ValueError(f"Unsupported initial condition kind: {kind!r}.")
