"""Strict zero-shot KdV adapter for the frozen mCH--BO causal mechanism.

There is deliberately no trainable KdV adapter.  Every candidate is evaluated
through both frozen donor adapters and the larger conformal risk is used.  If
either donor regards the feature row as out of support, authority is withheld.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
import torch

from fgsp_ch.discretization.grids import PeriodicGrid
from fgsp_ch.nonlocal_waves.adaptivity.prospective_causal_operator import (
    ACTIONS,
    EQUATIONS,
    FEATURE_ORDER,
    ActionEnvelope,
    FrozenProspectiveMechanism,
    analytical_safe_actions,
    prospective_features,
)
from fgsp_ch.nonlocal_waves.operators.kdv_fourier import KdVFourierOperator


@dataclass(frozen=True, slots=True)
class KdVResolutionIndicators:
    tail_ratio: float
    aliasing_defect: float
    unresolved_increment: float
    dispersive_stiffness: float
    nonlinear_stiffness: float
    localized_sharpness: float


def project_kdv_spectrum(
    values: NDArray[np.floating], source: PeriodicGrid, target_points: int,
) -> tuple[NDArray[np.floating], PeriodicGrid]:
    if target_points < 16 or target_points % 2:
        raise ValueError("target KdV grid must be even and contain at least 16 points")
    data = np.asarray(values, dtype=np.float64)
    source.validate_state(data)
    target = PeriodicGrid(int(target_points), source.length, source.x_min)
    source_hat = KdVFourierOperator(source).spectrum(data)
    target_hat = np.zeros(target.points // 2 + 1, dtype=np.complex128)
    copied = min(source.points // 2, target.points // 2)
    target_hat[:copied] = (target.points / source.points) * source_hat[:copied]
    target_hat[-1] = 0.0
    return KdVFourierOperator(target).physical(target_hat), target


def kdv_resolution_indicators(
    values: NDArray[np.floating], grid: PeriodicGrid, dt: float,
    *, tail_fraction: float = 0.65,
) -> KdVResolutionIndicators:
    operator = KdVFourierOperator(grid)
    spectrum = operator.spectrum(values)
    weights = 1 + operator.wave_numbers**2
    cutoff = max(1, int(tail_fraction * (len(spectrum) - 1)))
    total = float(np.sum(weights * np.abs(spectrum)**2))
    tail = float(np.sum(weights[cutoff:] * np.abs(spectrum[cutoff:])**2))
    derivative = operator.derivative(values)
    amplitude = max(float(np.max(np.abs(values))), 1e-14)
    # Match the F1 localized-sharpness semantics: concentration of a local
    # indicator, not its dimensional magnitude.  This remains in [0,1].
    derivative_indicator = np.abs(derivative)
    localized = float(
        np.max(derivative_indicator)
        / max(np.linalg.norm(derivative_indicator), np.finfo(float).eps)
    )
    kmax = float(np.max(operator.wave_numbers))
    return KdVResolutionIndicators(
        tail_ratio=float(np.sqrt(tail / max(total, np.finfo(float).eps))),
        aliasing_defect=operator.aliasing_defect(values),
        unresolved_increment=operator.unresolved_nonlinear_increment(values, dt),
        # The linear term is integrated exactly, but this remains an informative
        # cross-equation feature rather than a hard CFL restriction.
        dispersive_stiffness=float(abs(dt) * kmax**3),
        nonlinear_stiffness=float(abs(dt) * 6 * amplitude * kmax),
        localized_sharpness=localized,
    )


def kdv_prospective_features(
    indicators: KdVResolutionIndicators, *, temporal: float,
    migration: float, invariant_drift: float, invariant_limit: float,
    local_budget: float, remaining_budget_fraction: float,
    remaining_time_fraction: float, relative_work: float,
    previous_rejected: bool, dt_factor: float, resolution_factor: float,
) -> NDArray[np.float64]:
    return prospective_features(
        temporal=temporal,
        spatial=max(indicators.tail_ratio, indicators.unresolved_increment),
        representation=indicators.aliasing_defect,
        migration=migration,
        invariant_drift=invariant_drift,
        invariant_limit=invariant_limit,
        # F1's BO branch defined stiffness as spectral tail divided by the
        # aliasing floor.  Reuse that registered dimensionless semantics;
        # physical KdV phases remain explicit analytical diagnostics.
        stiffness=indicators.tail_ratio / max(indicators.aliasing_defect, 1e-12),
        local_budget=local_budget,
        remaining_budget_fraction=remaining_budget_fraction,
        remaining_time_fraction=remaining_time_fraction,
        relative_work=relative_work,
        previous_rejected=previous_rejected,
        localized_sharpness=indicators.localized_sharpness,
        nonlocal_tail=indicators.tail_ratio,
        dt_factor=dt_factor,
        resolution_factor=resolution_factor,
        probe_requested=resolution_factor > 1,
        representation_change=resolution_factor != 1,
    )


@dataclass(frozen=True, slots=True)
class KdVConsensusDecision:
    action_id: str | None
    abstained: bool
    upper_effectivity: float
    safe_action_count: int
    support_distances: dict[str, float]
    donor_upper_effectivity: dict[str, float]


class FrozenKdVConsensusMechanism:
    """Use the intersection of both original adapters without fitting KdV."""

    feature_kind = "legacy"

    def __init__(self, frozen: FrozenProspectiveMechanism):
        self.frozen = frozen

    @classmethod
    def load(cls, root, manifest: dict, device="cpu"):
        return cls(FrozenProspectiveMechanism.load(root, manifest, device))

    @torch.no_grad()
    def choose(
        self, feature_rows: NDArray[np.floating], actions: tuple[ActionEnvelope, ...],
        *, reserve_fraction: float,
    ) -> KdVConsensusDecision:
        safe = analytical_safe_actions(actions, reserve_fraction=reserve_fraction)
        rows = np.asarray(feature_rows, dtype=np.float64)
        if rows.shape != (len(actions), len(FEATURE_ORDER)):
            raise ValueError("one KdV feature row is required per action")
        if not safe:
            return KdVConsensusDecision(None, True, np.inf, 0, {}, {})
        safe_indices = [index for index, action in enumerate(actions) if action in safe]
        distances: dict[str, float] = {}
        for donor in EQUATIONS:
            support = self.frozen.manifest["support"][donor]
            mean = np.asarray(support["mean"], dtype=np.float64)
            scale = np.maximum(np.asarray(support["scale"], dtype=np.float64), 1e-8)
            distances[donor] = float(np.max(np.sqrt(np.mean(
                ((rows[safe_indices] - mean) / scale)**2, axis=1
            ))))
        if any(
            distances[donor] > float(self.frozen.manifest["support"][donor]["radius"])
            for donor in EQUATIONS
        ):
            return KdVConsensusDecision(None, True, np.inf, len(safe), distances, {})

        x = torch.as_tensor(
            rows[safe_indices], dtype=torch.float32, device=self.frozen.device
        )
        donor_logs, donor_costs = {}, {}
        for donor_index, donor in enumerate(EQUATIONS):
            equation_index = torch.full(
                (len(safe_indices),), donor_index, dtype=torch.long,
                device=self.frozen.device,
            )
            outputs = [model(x, equation_index) for model in self.frozen.models]
            upper = torch.stack([row.upper_log_effectivity for row in outputs]).cpu().numpy()
            cost = torch.stack([row.log_cost for row in outputs]).cpu().numpy()
            donor_logs[donor] = (
                upper.mean(axis=0) + upper.std(axis=0)
                + float(self.frozen.manifest["margins"][donor])
                + float(self.frozen.manifest["margins"]["global"])
            )
            donor_costs[donor] = cost.mean(axis=0)
        consensus_log = np.maximum.reduce(list(donor_logs.values()))
        consensus_cost = np.maximum.reduce(list(donor_costs.values()))
        effectivity = np.exp(np.clip(consensus_log, -32, 16))
        admitted = [
            (local, action_index)
            for local, action_index in enumerate(safe_indices)
            if effectivity[local] * actions[action_index].deterministic_error
            <= (1 - reserve_fraction) * actions[action_index].remaining_budget
        ]
        donor_upper = {
            donor: float(np.max(np.exp(np.clip(logs, -32, 16))))
            for donor, logs in donor_logs.items()
        }
        if not admitted:
            return KdVConsensusDecision(
                None, True, float(np.max(effectivity)), len(safe), distances, donor_upper
            )
        selected_local, selected_index = min(admitted, key=lambda pair: (
            actions[pair[1]].work * float(np.exp(np.clip(consensus_cost[pair[0]], -4, 4))),
            actions[pair[1]].action_id,
        ))
        return KdVConsensusDecision(
            actions[selected_index].action_id, False,
            float(effectivity[selected_local]), len(safe), distances, donor_upper,
        )
