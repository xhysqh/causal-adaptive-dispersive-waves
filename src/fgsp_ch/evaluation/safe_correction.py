from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from fgsp_ch.equations.generalized_ch import GeneralizedCH
from fgsp_ch.features.a1 import baseline_residual
from fgsp_ch.geometry.metric import a_norm, discrete_energy
from fgsp_ch.solvers.energy_projection import project_to_energy


@dataclass(frozen=True, slots=True)
class SafeCorrectionResult:
    state: NDArray[np.floating]
    correction: NDArray[np.floating]
    alpha: float
    accepted: bool
    reason: str
    baseline_residual_norm: float
    accepted_residual_norm: float
    raw_correction_ratio: float


def spectral_tail_fraction(values: NDArray[np.floating], fraction: float) -> float:
    if not 0.0 < fraction < 1.0:
        raise ValueError("Tail fraction must lie in (0, 1).")
    spectrum = np.abs(np.fft.fft(values)) ** 2
    modes = np.abs(np.fft.fftfreq(len(values)))
    cutoff = (1.0 - fraction) * float(modes.max())
    total = float(spectrum.sum())
    return 0.0 if total == 0.0 else float(spectrum[modes >= cutoff].sum() / total)


def accept_safe_correction(
    equation: GeneralizedCH,
    current: NDArray[np.floating],
    baseline_next: NDArray[np.floating],
    proposed_correction: NDArray[np.floating],
    dt: float,
    *,
    rho: float,
    backtracking: tuple[float, ...],
    residual_factor: float,
    correction_rate_allowance: float,
    spectral_tail_factor: float,
    spectral_tail_fraction_value: float,
    maximum_new_peaks: int,
    peak_reference: NDArray[np.floating] | None = None,
    epsilon: float = 1.0e-12,
) -> SafeCorrectionResult:
    """Causally accept, damp, or reject a learned correction.

    Every trial is projected onto the current discrete-energy shell.  It is
    accepted only if its midpoint residual and high-frequency tail remain
    bounded relative to the already available baseline candidate.
    """
    for value in (current, baseline_next, proposed_correction):
        equation.grid.validate_state(value)
    if not np.all(np.isfinite(proposed_correction)):
        proposed_correction = np.zeros_like(baseline_next)
    increment = baseline_next - current
    baseline_increment_norm = a_norm(increment, equation.helmholtz)
    raw_norm = a_norm(proposed_correction, equation.helmholtz)
    cap = min(1.0, rho * (baseline_increment_norm + epsilon) / (raw_norm + epsilon))
    bounded = cap * proposed_correction
    baseline_r = baseline_residual(equation, current, baseline_next, dt)
    baseline_r_norm = a_norm(
        equation.helmholtz.solve(baseline_r), equation.helmholtz
    )
    baseline_tail = spectral_tail_fraction(
        baseline_next, spectral_tail_fraction_value
    )
    peak_anchor = baseline_next if peak_reference is None else peak_reference
    equation.grid.validate_state(peak_anchor)
    baseline_peaks = int(np.count_nonzero(
        (peak_anchor >= np.roll(peak_anchor, 1))
        & (peak_anchor > np.roll(peak_anchor, -1))
    ))
    target_energy = discrete_energy(current, equation.helmholtz)
    for alpha in backtracking:
        if not 0.0 < alpha <= 1.0:
            raise ValueError("Backtracking factors must lie in (0, 1].")
        trial = baseline_next + float(alpha) * bounded
        try:
            projected = project_to_energy(
                trial, target_energy, equation.helmholtz
            ).state
        except (FloatingPointError, ValueError):
            continue
        residual = baseline_residual(equation, current, projected, dt)
        residual_norm = a_norm(
            equation.helmholtz.solve(residual), equation.helmholtz
        )
        tail = spectral_tail_fraction(projected, spectral_tail_fraction_value)
        trial_peaks = int(np.count_nonzero(
            (projected >= np.roll(projected, 1))
            & (projected > np.roll(projected, -1))
        ))
        # A state correction changes the discrete time residual at leading
        # order by A(delta)/dt.  Comparing only with the baseline residual
        # therefore rejects every useful correction when the baseline scheme
        # itself has a small residual.  The A^-1 norm of A(delta)/dt is
        # exactly ||delta||_A/dt, giving this dimensionally consistent bound.
        correction_rate = a_norm(
            float(alpha) * bounded, equation.helmholtz
        ) / dt
        residual_bound = (
            residual_factor * (baseline_r_norm + epsilon)
            + correction_rate_allowance * correction_rate
        )
        residual_ok = residual_norm <= residual_bound
        tail_ok = tail <= spectral_tail_factor * (baseline_tail + epsilon)
        peaks_ok = trial_peaks <= baseline_peaks + maximum_new_peaks
        if np.all(np.isfinite(projected)) and residual_ok and tail_ok and peaks_ok:
            correction = projected - baseline_next
            return SafeCorrectionResult(
                projected,
                correction,
                float(alpha * cap),
                True,
                "accepted",
                baseline_r_norm,
                residual_norm,
                raw_norm / (baseline_increment_norm + epsilon),
            )
    fallback = baseline_next if peak_reference is None else peak_reference
    return SafeCorrectionResult(
        fallback.copy(),
        fallback - baseline_next,
        0.0,
        False,
        "fallback_baseline",
        baseline_r_norm,
        baseline_r_norm,
        raw_norm / (baseline_increment_norm + epsilon),
    )
