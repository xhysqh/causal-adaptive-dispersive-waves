from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Protocol, runtime_checkable

import numpy as np

from fgsp_ch.nonlocal_waves.actions import AdaptiveAction
from fgsp_ch.nonlocal_waves.state import OperatorStepResult, WaveState


@dataclass(frozen=True, slots=True)
class OperatorSignature:
    equation: str
    nonlocal_mechanism: str
    conservative_quantities: tuple[str, ...]
    admissible_representations: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.equation or not self.nonlocal_mechanism:
            raise ValueError("operator signature names must be nonempty")
        if not self.conservative_quantities or not self.admissible_representations:
            raise ValueError("operator signature must register invariants and representations")


@dataclass(frozen=True, slots=True)
class ProblemFeatures:
    common: Mapping[str, float]
    equation_specific: Mapping[str, float]

    def __post_init__(self) -> None:
        common = {str(key): float(value) for key, value in self.common.items()}
        specific = {str(key): float(value) for key, value in self.equation_specific.items()}
        if not common or not specific:
            raise ValueError("common and equation-specific feature groups must be nonempty")
        if not all(np.isfinite(value) for value in (*common.values(), *specific.values())):
            raise ValueError("problem features must be finite")
        object.__setattr__(self, "common", MappingProxyType(common))
        object.__setattr__(self, "equation_specific", MappingProxyType(specific))

    def flattened(self) -> dict[str, float]:
        result = {f"common:{key}": value for key, value in self.common.items()}
        result.update({f"specific:{key}": value for key, value in self.equation_specific.items()})
        return result


@runtime_checkable
class NonlocalWaveProblem(Protocol):
    """Contract for compatible feature-conditioned nonlocal-wave evolution."""

    @property
    def signature(self) -> OperatorSignature: ...

    def validate_state(self, state: WaveState) -> None: ...

    def invariants(self, state: WaveState) -> Mapping[str, float]: ...

    def extract_features(self, state: WaveState) -> ProblemFeatures: ...

    def admissible_actions(self, state: WaveState) -> tuple[AdaptiveAction, ...]: ...

    def advance(self, state: WaveState, action: AdaptiveAction) -> OperatorStepResult: ...
