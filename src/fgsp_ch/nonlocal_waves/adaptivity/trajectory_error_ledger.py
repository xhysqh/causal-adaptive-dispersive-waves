"""Reference-free trajectory error-budget ledger used by CMAME-U1."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class TrajectoryErrorLedger:
    tolerance: float
    final_time: float
    reserve_fraction: float = 0.10
    spent_upper: float = 0.0

    def __post_init__(self):
        if self.tolerance <= 0 or self.final_time <= 0:
            raise ValueError("U1 ledger requires positive tolerance and final time")
        if not 0 <= self.reserve_fraction < 1:
            raise ValueError("U1 ledger reserve fraction is invalid")

    @property
    def remaining(self) -> float:
        return max(self.tolerance - self.spent_upper, 0.0)

    def local_budget(self, time: float, dt: float) -> float:
        if not 0 <= time < self.final_time or dt <= 0:
            raise ValueError("U1 ledger time request is invalid")
        remaining_time = max(self.final_time - time, dt)
        return max(self.remaining * min(dt / remaining_time, 1.0), 1e-16)

    def admissible(self, upper_error: float) -> bool:
        return bool(
            np.isfinite(upper_error) and upper_error >= 0
            and upper_error <= (1 - self.reserve_fraction) * self.remaining
        )

    def commit(self, upper_error: float) -> None:
        if not self.admissible(upper_error):
            raise ValueError("cannot commit an action outside the U1 trajectory budget")
        self.spent_upper += float(upper_error)
