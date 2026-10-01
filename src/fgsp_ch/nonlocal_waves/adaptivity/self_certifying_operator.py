"""Reference-free online certification for an unseen equation.

The frozen neural model may propose actions outside its static support, but it
never releases them directly.  A hierarchical numerical probe establishes a
rolling multiplicative certificate.  Network weights and offline conformal
quantities remain immutable.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True, slots=True)
class SelfCertificationConfig:
    warmup_probes: int = 3
    rolling_window: int = 8
    release_horizon: int = 4
    verification_interval: int = 4
    probe_safety: float = 1.50
    maximum_log_jump: float = 1.50

    def __post_init__(self):
        if min(self.warmup_probes, self.rolling_window,
               self.release_horizon, self.verification_interval) < 1:
            raise ValueError("U1.1 certification counts must be positive")
        if self.probe_safety < 1 or self.maximum_log_jump <= 0:
            raise ValueError("U1.1 safety constants are invalid")


@dataclass(frozen=True, slots=True)
class CertificationDecision:
    action_id: str | None
    authority: bool
    probe_required: bool
    certified_upper: float
    state: str
    revoked: bool
    raw_effectivity: float
    support_distance: float


@dataclass(slots=True)
class ReferenceFreeCertificate:
    config: SelfCertificationConfig
    log_ratios: list[float] = field(default_factory=list)
    release_remaining: int = 0
    decisions_since_probe: int = 0
    revocations: int = 0

    @property
    def ready(self) -> bool:
        return len(self.log_ratios) >= self.config.warmup_probes

    @property
    def multiplier(self) -> float:
        if not self.ready:
            return np.inf
        window = self.log_ratios[-self.config.rolling_window:]
        return float(self.config.probe_safety * np.exp(max(window)))

    def requires_probe(self, *, force_probe: bool = False) -> bool:
        return bool(
            force_probe or not self.ready or self.release_remaining <= 0
            or self.decisions_since_probe >= self.config.verification_interval
        )

    def observe(self, *, predicted_upper: float, probe_error: float) -> bool:
        if not (np.isfinite(predicted_upper) and predicted_upper > 0
                and np.isfinite(probe_error) and probe_error >= 0):
            raise ValueError("invalid U1.1 probe observation")
        ratio = float(np.log(max(probe_error, 1e-16) / max(predicted_upper, 1e-16)))
        revoked = False
        if self.ready:
            previous = self.log_ratios[-self.config.rolling_window:]
            if ratio > max(previous) + self.config.maximum_log_jump:
                self.release_remaining = 0
                self.revocations += 1
                revoked = True
        self.log_ratios.append(ratio)
        self.log_ratios = self.log_ratios[-self.config.rolling_window:]
        self.decisions_since_probe = 0
        if self.ready and not revoked:
            self.release_remaining = self.config.release_horizon
        return revoked

    def decide(self, *, action_id: str | None, raw_effectivity: float,
               deterministic_error: float, remaining_budget: float,
               support_distance: float, probe_error: float | None = None,
               reserve_fraction: float = 0.1,
               force_probe: bool = False) -> CertificationDecision:
        if action_id is None:
            return CertificationDecision(
                None, False, False, np.inf, "fallback", False,
                raw_effectivity, support_distance,
            )
        predicted = max(raw_effectivity * deterministic_error, 1e-16)
        probe_required = self.requires_probe(force_probe=force_probe)
        revoked = False
        if probe_required:
            if probe_error is None:
                return CertificationDecision(
                    action_id, False, True, np.inf, "probe", False,
                    raw_effectivity, support_distance,
                )
            revoked = self.observe(predicted_upper=predicted, probe_error=probe_error)
        certified = self.multiplier * predicted
        authority = bool(
            self.ready and not revoked and np.isfinite(certified)
            and certified <= (1 - reserve_fraction) * remaining_budget
        )
        if authority:
            self.release_remaining = max(self.release_remaining - 1, 0)
            self.decisions_since_probe += 1
        state = "certified_release" if authority else ("revoked" if revoked else "probe")
        return CertificationDecision(
            action_id, authority, probe_required, float(certified), state,
            revoked, raw_effectivity, support_distance,
        )
