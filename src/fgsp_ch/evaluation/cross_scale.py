from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from fgsp_ch.discretization.spectral import fourier_restrict


def increment_rate(
    next_state: NDArray[np.floating],
    current_state: NDArray[np.floating],
    dt: float,
) -> NDArray[np.floating]:
    """State increment per unit time, invariant to a static grid offset."""
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("Time step must be positive and finite.")
    if next_state.shape != current_state.shape:
        raise ValueError("Current and next states must have matching shapes.")
    return (next_state - current_state) / dt


def restrict_batch(
    values: NDArray[np.floating], coarse_points: int
) -> NDArray[np.floating]:
    """Fourier-restrict a state or a batch of periodic states."""
    if values.shape[-1] == coarse_points:
        return np.asarray(values).copy()
    return fourier_restrict(values, coarse_points)


def common_mode_project(
    values: NDArray[np.floating], mode_cutoff: int
) -> NDArray[np.floating]:
    """Project states onto a grid-independent set of Fourier mode numbers."""
    points = values.shape[-1]
    if mode_cutoff < 0 or mode_cutoff >= points // 2:
        raise ValueError("Mode cutoff must satisfy 0 <= cutoff < N/2.")
    modes = np.fft.fftfreq(points) * points
    spectrum = np.fft.fft(values, axis=-1)
    spectrum[..., np.abs(modes) > mode_cutoff] = 0.0
    projected = np.fft.ifft(spectrum, axis=-1).real
    return np.asarray(projected, dtype=values.dtype)


def represent_target(
    values: NDArray[np.floating],
    *,
    kind: str,
    absolute_cutoff: int | None = None,
    fraction: float | None = None,
    nyquist_margin: int | None = None,
) -> tuple[NDArray[np.floating], int]:
    """Apply a pre-registered grid-aware target representation."""
    points = values.shape[-1]
    if kind == "full":
        return np.asarray(values).copy(), points // 2
    if kind == "absolute":
        if absolute_cutoff is None:
            raise ValueError("absolute representation requires a cutoff.")
        cutoff = int(absolute_cutoff)
    elif kind == "relative":
        if fraction is None or not 0.0 < fraction < 0.5:
            raise ValueError("relative representation requires fraction in (0, 0.5).")
        cutoff = int(np.floor(points * fraction))
    elif kind == "near_full":
        if nyquist_margin is None or nyquist_margin < 1:
            raise ValueError("near_full representation requires a positive margin.")
        cutoff = points // 2 - int(nyquist_margin)
    else:
        raise ValueError(f"Unknown target representation: {kind}.")
    return common_mode_project(values, cutoff), cutoff


def matched_time_indices(
    left_times: NDArray[np.floating],
    right_times: NDArray[np.floating],
    *,
    tolerance: float = 1.0e-12,
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Return monotonically aligned indices for two physical-time arrays."""
    left = np.asarray(left_times, dtype=float)
    right = np.asarray(right_times, dtype=float)
    i = j = 0
    left_indices: list[int] = []
    right_indices: list[int] = []
    while i < len(left) and j < len(right):
        difference = left[i] - right[j]
        if abs(difference) <= tolerance:
            left_indices.append(i)
            right_indices.append(j)
            i += 1
            j += 1
        elif difference < 0.0:
            i += 1
        else:
            j += 1
    return (
        np.asarray(left_indices, dtype=np.int64),
        np.asarray(right_indices, dtype=np.int64),
    )


def richardson_spatial_component(
    coarse_rate: NDArray[np.floating],
    fine_rate_on_coarse: NDArray[np.floating],
    coarse_h: float,
    fine_h: float,
    spatial_order: float,
) -> NDArray[np.floating]:
    denominator = coarse_h**spatial_order - fine_h**spatial_order
    if abs(denominator) <= np.finfo(float).eps:
        raise ValueError("Spatial Richardson scales are singular.")
    return (coarse_rate - fine_rate_on_coarse) / denominator


def richardson_temporal_component(
    coarse_rate: NDArray[np.floating],
    fine_rate: NDArray[np.floating],
    coarse_dt: float,
    fine_dt: float,
    temporal_order: float,
) -> NDArray[np.floating]:
    denominator = coarse_dt**temporal_order - fine_dt**temporal_order
    if abs(denominator) <= np.finfo(float).eps:
        raise ValueError("Temporal Richardson scales are singular.")
    return (coarse_rate - fine_rate) / denominator


def fit_scale_components(
    rates: NDArray[np.floating],
    h_values: NDArray[np.floating],
    dt_values: NDArray[np.floating],
    *,
    spatial_order: float,
    temporal_order: float,
) -> tuple[NDArray[np.floating], NDArray[np.floating], float, float]:
    """Fit ``g = h**q D_x + dt**p D_t`` at every time/grid point.

    ``rates`` has shape ``(scales, times, points)``.  Returned components
    have shape ``(times, points)``.  Both raw and column-normalized design
    condition numbers are reported so units cannot hide ill-conditioning.
    """
    design = np.column_stack(
        (np.asarray(h_values) ** spatial_order,
         np.asarray(dt_values) ** temporal_order)
    )
    if rates.shape[0] != design.shape[0]:
        raise ValueError("Rate and scale counts do not match.")
    raw_condition = float(np.linalg.cond(design))
    column_norms = np.linalg.norm(design, axis=0)
    if np.any(column_norms <= np.finfo(float).eps):
        raise ValueError("Scale design has a zero column.")
    scaled_condition = float(np.linalg.cond(design / column_norms))
    coefficients = np.linalg.lstsq(
        design, rates.reshape(rates.shape[0], -1), rcond=None
    )[0]
    shape = rates.shape[1:]
    return (
        coefficients[0].reshape(shape),
        coefficients[1].reshape(shape),
        raw_condition,
        scaled_condition,
    )


def reconstruct_rate(
    spatial: NDArray[np.floating],
    temporal: NDArray[np.floating],
    h: float,
    dt: float,
    *,
    spatial_order: float,
    temporal_order: float,
) -> NDArray[np.floating]:
    return h**spatial_order * spatial + dt**temporal_order * temporal


def cosine_similarity(
    left: NDArray[np.floating],
    right: NDArray[np.floating],
    *,
    epsilon: float = 1.0e-15,
) -> float:
    numerator = float(np.vdot(left.ravel(), right.ravel()).real)
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return numerator / max(denominator, epsilon)
