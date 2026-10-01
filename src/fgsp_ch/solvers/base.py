from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class StepResult:
    """Diagnostics for one baseline step."""

    state: NDArray[np.floating]
    provisional_state: NDArray[np.floating]
    energy_before: float
    provisional_energy: float
    energy_after: float
    projection_scale: float

