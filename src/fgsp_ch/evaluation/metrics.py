from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from fgsp_ch.discretization.grids import PeriodicGrid
from fgsp_ch.discretization.helmholtz import HelmholtzOperator
from fgsp_ch.discretization.difference import first_derivative
from fgsp_ch.geometry.metric import a_norm, discrete_energy


def _relative(numerator: float, denominator: float) -> float:
    eps = np.finfo(np.float64).eps
    return float(numerator / max(denominator, eps))


def relative_l2_error(
    approximation: NDArray[np.floating],
    reference: NDArray[np.floating],
    grid: PeriodicGrid,
) -> float:
    return _relative(
        np.sqrt(grid.h * np.sum((approximation - reference) ** 2)),
        np.sqrt(grid.h * np.sum(reference**2)),
    )


def relative_linf_error(
    approximation: NDArray[np.floating],
    reference: NDArray[np.floating],
) -> float:
    return _relative(
        float(np.max(np.abs(approximation - reference))),
        float(np.max(np.abs(reference))),
    )


def relative_h1_error(
    approximation: NDArray[np.floating],
    reference: NDArray[np.floating],
    helmholtz: HelmholtzOperator,
) -> float:
    return _relative(
        a_norm(approximation - reference, helmholtz),
        a_norm(reference, helmholtz),
    )


def energy_relative_drift(
    state: NDArray[np.floating],
    initial_state: NDArray[np.floating],
    helmholtz: HelmholtzOperator,
) -> float:
    initial = discrete_energy(initial_state, helmholtz)
    return _relative(abs(discrete_energy(state, helmholtz) - initial), initial)


def local_maxima_indices(values: NDArray[np.floating]) -> NDArray[np.int_]:
    """Hard periodic peak detection for evaluation only."""
    return np.flatnonzero(
        (values >= np.roll(values, 1)) & (values > np.roll(values, -1))
    )


def peak_metrics(
    state: NDArray[np.floating],
    reference: NDArray[np.floating],
    grid: PeriodicGrid,
) -> dict[str, float | int]:
    """Amplitude, periodic-position, and spurious-peak diagnostics."""
    state_peaks = local_maxima_indices(state)
    reference_peaks = local_maxima_indices(reference)
    state_order = state_peaks[np.argsort(state[state_peaks])[::-1]]
    ref_order = reference_peaks[np.argsort(reference[reference_peaks])[::-1]]
    metrics: dict[str, float | int] = {
        "spurious_peak_count": max(0, len(state_peaks) - len(reference_peaks)),
        "max_peak_amplitude_error": abs(float(np.max(state) - np.max(reference))),
    }
    pair_count = min(2, len(state_order), len(ref_order))
    position_errors: list[float] = []
    for rank in range(pair_count):
        s_idx, r_idx = int(state_order[rank]), int(ref_order[rank])
        metrics[f"peak_{rank + 1}_amplitude_error"] = abs(
            float(state[s_idx] - reference[r_idx])
        )
        position_error = abs(
            float(grid.periodic_distance(grid.x[s_idx], grid.x[r_idx]))
        )
        metrics[f"peak_{rank + 1}_position_error"] = position_error
        position_errors.append(position_error)
    metrics["max_peak_position_error"] = max(position_errors, default=np.nan)
    return metrics


def tail_energy_ratio(
    state: NDArray[np.floating],
    grid: PeriodicGrid,
    *,
    radius: float,
    peak_count: int = 2,
) -> float:
    """Fraction of discrete H1 density outside dominant-peak neighborhoods."""
    grid.validate_state(state)
    if radius <= 0.0:
        raise ValueError("Tail radius must be positive.")
    peak_indices = local_maxima_indices(state)
    if len(peak_indices) == 0:
        return 1.0
    order = peak_indices[np.argsort(np.abs(state[peak_indices]))[::-1]]
    selected = order[:peak_count]
    near_peak = np.zeros(grid.points, dtype=bool)
    for index in selected:
        distance = np.abs(grid.periodic_distance(grid.x, grid.x[int(index)]))
        near_peak |= distance <= radius
    gradient = first_derivative(state, grid)
    density = state**2 + gradient**2
    total = float(grid.h * np.sum(density))
    if total <= 0.0:
        return 0.0
    return float(grid.h * np.sum(density[~near_peak]) / total)


def high_frequency_energy_ratio(
    state: NDArray[np.floating], *, high_frequency_fraction: float
) -> float:
    """Fraction of Fourier power in the highest configured mode fraction."""
    if not 0.0 < high_frequency_fraction < 1.0:
        raise ValueError("high_frequency_fraction must lie in (0, 1).")
    spectrum = np.abs(np.fft.fft(state)) ** 2
    frequencies = np.abs(np.fft.fftfreq(len(state)))
    threshold = np.quantile(frequencies, 1.0 - high_frequency_fraction)
    total = float(np.sum(spectrum))
    if total <= 0.0:
        return 0.0
    return float(np.sum(spectrum[frequencies >= threshold]) / total)
