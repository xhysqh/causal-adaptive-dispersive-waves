"""Discrete metric utilities used in Phase 0."""

from .metric import a_inner, a_norm, discrete_energy, discrete_inner

__all__ = ["a_inner", "a_norm", "discrete_energy", "discrete_inner"]
"""Discrete geometry submodules use explicit imports to avoid solver cycles."""
