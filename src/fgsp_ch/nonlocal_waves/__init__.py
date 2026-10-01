"""Unified contracts for feature-conditioned nonlocal-wave solvers.

U0 deliberately contains no Benjamin--Ono time integrator.  It defines the
equation-independent state/action/result contracts and registers the existing
compatible mCH backbone through an adapter.  BO enters only after an
independent reference solver has passed its own convergence audit.
"""

from fgsp_ch.nonlocal_waves.actions import AdaptiveAction, ActionKind
from fgsp_ch.nonlocal_waves.contracts import (
    NonlocalWaveProblem,
    OperatorSignature,
    ProblemFeatures,
)
from fgsp_ch.nonlocal_waves.registry import ProblemRegistry
from fgsp_ch.nonlocal_waves.state import (
    OperatorStepResult,
    RepresentationKind,
    ResolutionState,
    WaveState,
)

__all__ = [
    "ActionKind",
    "AdaptiveAction",
    "NonlocalWaveProblem",
    "OperatorSignature",
    "OperatorStepResult",
    "ProblemFeatures",
    "ProblemRegistry",
    "RepresentationKind",
    "ResolutionState",
    "WaveState",
]
