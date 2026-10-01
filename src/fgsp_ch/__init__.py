"""F-GSP-CH Phase 0 numerical baseline."""

from .equations.generalized_ch import GeneralizedCH, GeneralizedCHParameters
from .solvers.semi_implicit import SemiImplicitCHSolver

__all__ = ["GeneralizedCH", "GeneralizedCHParameters", "SemiImplicitCHSolver"]
__version__ = "0.1.0"

