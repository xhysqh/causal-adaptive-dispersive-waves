"""Deterministic action contract for the CMAME-M2.4 learned controller."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from fgsp_ch.models.cmame_m24_causal_router import ROUTES


@dataclass(frozen=True, slots=True)
class RepresentationAction:
    route: str
    space_level: int
    time_multiplier: float
    patch_operation: int = 0  # -1 contract, 0 keep, +1 expand

    def __post_init__(self) -> None:
        if self.route not in ROUTES:
            raise ValueError(f"unregistered representation route {self.route}")
        if self.space_level not in (0, 1, 2) or self.time_multiplier <= 0.0:
            raise ValueError("invalid M2.4 space/time action")
        if self.patch_operation not in (-1, 0, 1):
            raise ValueError("patch operation must be -1, 0 or 1")

    def features(self) -> np.ndarray:
        one_hot = [float(self.route == name) for name in ROUTES]
        return np.asarray([
            *one_hot, self.space_level / 2.0,
            np.log2(self.time_multiplier), float(self.patch_operation),
            float(self.patch_operation > 0), float(self.patch_operation < 0),
        ], dtype=np.float64)


@dataclass(frozen=True, slots=True)
class ActionPrediction:
    action: RepresentationAction
    upper_log_error: np.ndarray
    predicted_cost: float
    predicted_budget_fraction: float


@dataclass(frozen=True, slots=True)
class CausalDecision:
    action: RepresentationAction
    fallback: bool
    admissible_actions: int
    maximum_error_ratio: float


def route_is_eligible(route: str, *, field_ratio: float, atomic_ratio: float) -> bool:
    """Hard representation support; labels and family names are not inputs."""
    if route == "smooth":
        return atomic_ratio <= 1.0e-8
    if route == "particle":
        return field_ratio <= 1.0e-8 and atomic_ratio > 0.0
    if route == "hybrid_amr":
        return field_ratio > 1.0e-8 and atomic_ratio > 0.0
    return False


def choose_causal_action(
    predictions: Iterable[ActionPrediction], *, remaining_budget_fraction: float,
    field_ratio: float, atomic_ratio: float, switching_penalty: float = 0.05,
    previous_route: str | None = None, horizon_index: int = -1,
) -> CausalDecision:
    """Choose minimum work only after the requested horizon gate passes.

    A terminal H1 tolerance is not an additive per-step quantity.  Therefore
    the gate evaluates the certificate matching the macro-step that will
    actually be committed, instead of summing or taking a maximum over
    unrelated horizons.
    """
    if not 0.0 < remaining_budget_fraction <= 1.0:
        raise ValueError("remaining budget fraction must lie in (0,1]")
    candidates = []
    for row in predictions:
        if not route_is_eligible(row.action.route, field_ratio=field_ratio, atomic_ratio=atomic_ratio):
            continue
        upper = np.asarray(row.upper_log_error, dtype=np.float64)
        maximum = float(np.exp(upper[horizon_index]))
        if maximum <= remaining_budget_fraction:
            switch = switching_penalty if previous_route not in (None, row.action.route) else 0.0
            candidates.append((row.predicted_cost + switch, maximum, row))
    if candidates:
        _, maximum, selected = min(candidates, key=lambda item: (item[0], item[1]))
        return CausalDecision(selected.action, False, len(candidates), maximum)

    # Deterministic accuracy-first fallback.  It never increases dt or
    # contracts a patch and is independent of a learned threshold.
    route = "particle" if field_ratio <= 1.0e-8 and atomic_ratio > 0.0 else (
        "smooth" if atomic_ratio <= 1.0e-8 else "hybrid_amr"
    )
    return CausalDecision(RepresentationAction(route, 2, 0.5, 1), True, 0, np.inf)
