"""Shared real-FFT spectral norms.

NumPy's ``rfft`` stores only the non-negative half of a real spectrum.  The
strictly positive, non-Nyquist modes therefore represent both ``+k`` and
``-k`` and must carry multiplicity two in Parseval sums.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def rfft_multiplicity(points: int) -> NDArray[np.float64]:
    """Return Parseval multiplicities for an ``rfft`` of ``points`` samples."""
    if points < 2:
        raise ValueError("an rFFT norm requires at least two samples")
    multiplicity = np.full(points // 2 + 1, 2.0, dtype=np.float64)
    multiplicity[0] = 1.0
    if points % 2 == 0:
        multiplicity[-1] = 1.0  # the Nyquist coefficient is self-conjugate
    return multiplicity


def relative_rfft_h1(
    approximation: NDArray[np.floating],
    reference: NDArray[np.floating],
    *,
    spacing: float,
) -> float:
    """Return the discrete periodic relative H1 norm for real samples."""
    approx = np.asarray(approximation, dtype=np.float64)
    ref = np.asarray(reference, dtype=np.float64)
    if approx.ndim != 1 or approx.shape != ref.shape:
        raise ValueError("relative H1 inputs must be aligned one-dimensional states")
    if spacing <= 0:
        raise ValueError("relative H1 spacing must be positive")
    points = len(ref)
    wave_numbers = 2.0 * np.pi * np.fft.rfftfreq(points, d=spacing)
    weights = rfft_multiplicity(points) * (1.0 + wave_numbers**2)
    difference_hat = np.fft.rfft(approx - ref)
    reference_hat = np.fft.rfft(ref)
    numerator = float(np.sqrt(np.sum(weights * np.abs(difference_hat) ** 2)))
    denominator = float(np.sqrt(np.sum(weights * np.abs(reference_hat) ** 2)))
    return numerator / max(denominator, np.finfo(float).eps)
