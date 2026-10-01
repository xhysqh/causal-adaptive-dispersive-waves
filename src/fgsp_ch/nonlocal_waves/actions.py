from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
from numpy.typing import NDArray

from fgsp_ch.nonlocal_waves.state import RepresentationKind


class ActionKind(str, Enum):
    """Equation-independent adaptive intentions.

    Problems translate these intentions into equation-specific compatible
    operators.  For example, ``REFINE`` denotes a physical patch for mCH but
    a larger retained Fourier band for BO.
    """

    HOLD = "hold"
    REFINE = "refine"
    COARSEN = "coarsen"
    CHANGE_REPRESENTATION = "change_representation"
    FALLBACK = "fallback"


@dataclass(frozen=True, slots=True)
class AdaptiveAction:
    action_id: str
    kind: ActionKind
    representation: RepresentationKind
    dt: float
    points: int
    active_modes: int | None = None
    refine_mask: NDArray[np.bool_] | None = None
    dealias_fraction: float | None = None
    estimated_work: float = 1.0

    def __post_init__(self) -> None:
        if not self.action_id:
            raise ValueError("action_id must be nonempty")
        if not np.isfinite(self.dt) or self.dt <= 0.0:
            raise ValueError("action dt must be positive and finite")
        if self.points < 8:
            raise ValueError("an adaptive action requires at least eight points")
        if self.active_modes is not None and not 0 < self.active_modes <= self.points:
            raise ValueError("active_modes must lie in (0, points]")
        if self.dealias_fraction is not None and not 0.0 < self.dealias_fraction <= 1.0:
            raise ValueError("dealias_fraction must lie in (0, 1]")
        if not np.isfinite(self.estimated_work) or self.estimated_work <= 0.0:
            raise ValueError("estimated_work must be positive and finite")
        if self.refine_mask is not None:
            mask = np.asarray(self.refine_mask, dtype=bool)
            if mask.shape != (self.points,):
                raise ValueError("refine_mask must match action.points")
            object.__setattr__(self, "refine_mask", mask.copy())
