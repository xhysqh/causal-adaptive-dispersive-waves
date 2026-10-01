from __future__ import annotations

from fgsp_ch.nonlocal_waves.contracts import NonlocalWaveProblem


class ProblemRegistry:
    """Explicit registry; importing U0 never silently constructs a solver."""

    def __init__(self) -> None:
        self._problems: dict[str, NonlocalWaveProblem] = {}

    def register(self, name: str, problem: NonlocalWaveProblem) -> None:
        key = name.strip().lower()
        if not key:
            raise ValueError("problem registry name must be nonempty")
        if key in self._problems:
            raise KeyError(f"problem already registered: {key}")
        if not isinstance(problem, NonlocalWaveProblem):
            raise TypeError("problem does not satisfy NonlocalWaveProblem")
        self._problems[key] = problem

    def get(self, name: str) -> NonlocalWaveProblem:
        key = name.strip().lower()
        if key not in self._problems:
            raise KeyError(f"unknown nonlocal-wave problem: {key}")
        return self._problems[key]

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._problems))
