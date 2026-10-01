from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class PeriodicGrid:
    """Uniform periodic grid on ``[x_min, x_min + length)``."""

    points: int
    length: float
    x_min: float = 0.0
    dtype: np.dtype = np.dtype(np.float64)

    def __post_init__(self) -> None:
        if self.points < 3:
            raise ValueError("A periodic grid requires at least three points.")
        if not np.isfinite(self.length) or self.length <= 0.0:
            raise ValueError("Grid length must be positive and finite.")
        object.__setattr__(self, "dtype", np.dtype(self.dtype))

    @property
    def h(self) -> float:
        return self.length / self.points

    @property
    def x(self) -> NDArray[np.floating]:
        return np.asarray(
            self.x_min + self.h * np.arange(self.points), dtype=self.dtype
        )

    def validate_state(self, values: NDArray[np.floating]) -> None:
        if values.shape != (self.points,):
            raise ValueError(
                f"Expected state shape {(self.points,)}, got {values.shape}."
            )
        if not np.all(np.isfinite(values)):
            raise FloatingPointError("State contains NaN or Inf.")

    def periodic_distance(
        self, position: NDArray[np.floating] | float, center: float
    ) -> NDArray[np.floating]:
        """Signed shortest distance from ``center`` on the periodic domain."""
        delta = np.asarray(position, dtype=self.dtype) - center
        return (delta + 0.5 * self.length) % self.length - 0.5 * self.length

