"""Stage-4 safety-shielded use of the frozen V2 predictive-risk ensemble.

The learned model has no authority to waive numerical admissibility.  It can
only retain a safe hold, rank already-safe corrective candidates, or request a
verification.  Coarsening is deliberately outside the learned action set: it
requires an independent projection/invariant verification.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np
import torch

from fgsp_ch.nonlocal_waves.adaptivity.mechanism_coordinates import (
    CAUSAL_HISTORY_FEATURE_ORDER, MECHANISM_FEATURE_ORDER,
    V2_CAUSAL_FEATURE_ORDER, knn_support_score,
)
from fgsp_ch.nonlocal_waves.adaptivity.v2_predictive_risk_operator import V2PredictiveRiskOperator


LEARNED_ACTIONS = ("shrink_time", "hold", "refine_space")
COARSEN_ACTION = "coarsen_space"


@dataclass(frozen=True, slots=True)
class V2Candidate:
    """A pre-built candidate with deterministic conditions evaluated first."""

    action_id: str
    features: np.ndarray
    deterministic_error: float
    remaining_budget: float
    work: float
    invariant_passed: bool
    projection_passed: bool
    local_budget: float = 1.0
    spectral_tail: float = 0.0
    localized_sharpness: float = 0.0
    history_velocity_norm: float = 0.0
    history_acceleration_norm: float = 0.0
    # A coarsening candidate must have this independent verification.  It is
    # never inferred by the V2 learned model because stage-2 labels did not
    # contain a coarsening action.
    external_verification_passed: bool = False
    verified_future_risk: float | None = None

    def __post_init__(self) -> None:
        row = np.asarray(self.features, dtype=np.float64)
        values = (self.deterministic_error, self.remaining_budget, self.work,
                  self.local_budget, self.spectral_tail, self.localized_sharpness,
                  self.history_velocity_norm, self.history_acceleration_norm)
        if self.action_id not in (*LEARNED_ACTIONS, COARSEN_ACTION):
            raise ValueError("unknown V2 candidate action")
        if row.shape != (len(V2_CAUSAL_FEATURE_ORDER),) or not np.all(np.isfinite(row)):
            raise ValueError("invalid V2 candidate feature row")
        if not np.all(np.isfinite(values)) or min(values) < 0 or self.work <= 0:
            raise ValueError("invalid V2 deterministic candidate envelope")
        if self.verified_future_risk is not None and (
            not np.isfinite(self.verified_future_risk) or self.verified_future_risk < 0
        ):
            raise ValueError("invalid independently verified coarsening risk")


@dataclass(frozen=True, slots=True)
class V2RiskAssessment:
    upper_risk: float
    ensemble_spread: float
    support_distance: float
    in_support: bool
    probe_probability: float


class FrozenV2PredictiveRiskEnsemble:
    """Frozen, equation-neutral risk envelope; never advances a PDE state."""

    def __init__(self, models: Sequence[V2PredictiveRiskOperator], manifest: dict, *, device="cpu") -> None:
        if len(models) != 3:
            raise ValueError("V2 stage 4 requires exactly three frozen members")
        if tuple(manifest["feature_order"]) != V2_CAUSAL_FEATURE_ORDER:
            raise ValueError("V2 frozen manifest feature contract mismatch")
        self.models = tuple(model.eval() for model in models)
        self.manifest = manifest
        self.device = torch.device(device)

    @classmethod
    def load(cls, root, manifest: dict, *, device="cpu") -> "FrozenV2PredictiveRiskEnsemble":
        root = Path(root)
        target = torch.device(device)
        models = []
        for name in manifest["checkpoints"]:
            payload = torch.load(root / name, map_location=target, weights_only=False)
            model = V2PredictiveRiskOperator(
                hidden=int(payload["hidden"]),
                input_width=int(payload.get("input_width", len(V2_CAUSAL_FEATURE_ORDER))),
            ).to(target)
            model.load_state_dict(payload["model"])
            models.append(model)
        registered = dict(manifest)
        registered["_root"] = str(root)
        return cls(models, registered, device=target)

    @torch.no_grad()
    def assess(self, feature_rows) -> tuple[V2RiskAssessment, ...]:
        rows = np.asarray(feature_rows, dtype=np.float64)
        if rows.ndim != 2 or rows.shape[1] != len(V2_CAUSAL_FEATURE_ORDER):
            raise ValueError("V2 assessment feature width mismatch")
        active = self.manifest.get("active_feature_indices")
        if active is not None:
            rows = rows[:, np.asarray(active, dtype=int)]
        elif self.manifest.get("input_ablation", "full") == "no_history":
            rows = rows.copy()
            start = len(MECHANISM_FEATURE_ORDER)
            stop = start + len(CAUSAL_HISTORY_FEATURE_ORDER)
            rows[:, start:stop] = 0.0
        center = np.asarray(self.manifest["robust_center"], dtype=np.float64)
        scale = np.asarray(self.manifest["robust_scale"], dtype=np.float64)
        anchors_path = Path(self.manifest["_root"]) / self.manifest["support_anchors"]
        anchors = np.load(anchors_path)["anchors"]
        distance = knn_support_score(
            (rows - center) / scale, anchors,
            neighbors=int(self.manifest["support_neighbors"]),
        )
        tensor = torch.as_tensor(rows, dtype=torch.float32, device=self.device)
        output = [model(tensor) for model in self.models]
        upper = torch.stack([item.upper_log_risk for item in output]).cpu().numpy()
        probe = torch.stack([item.probe_logit for item in output]).cpu().numpy()
        log_upper = (
            upper.mean(axis=0)
            + float(self.manifest["ensemble_std_multiplier"]) * upper.std(axis=0)
            + float(self.manifest["conformal_log_risk_margin"])
        )
        return tuple(V2RiskAssessment(
            upper_risk=float(np.exp(np.clip(log_upper[index], -32.0, 32.0))),
            ensemble_spread=float(upper[:, index].std()),
            support_distance=float(distance[index]),
            in_support=bool(distance[index] <= float(self.manifest["support_radius"])),
            probe_probability=float(1.0 / (1.0 + np.exp(-np.clip(probe[:, index].mean(), -32.0, 32.0)))),
        ) for index in range(len(rows)))


@dataclass(frozen=True, slots=True)
class V2ControllerConfig:
    reserve_fraction: float = 0.10
    hold_risk_threshold: float = 0.65
    corrective_risk_threshold: float = 1.0
    coarse_risk_threshold: float = 0.15
    cooldown_prefixes: int = 4
    stable_prefixes_for_coarsening: int = 5
    uncertainty_probe_threshold: float = 0.75
    fallback_hold_local_budget_fraction: float = 0.50
    coarse_tail_limit: float = 0.02
    coarse_localized_sharpness_limit: float = 0.95
    coarse_history_velocity_limit: float = 6.0
    coarse_history_acceleration_limit: float = 7.0

    def __post_init__(self) -> None:
        if not 0 <= self.reserve_fraction < 1:
            raise ValueError("V2 reserve fraction must be in [0, 1)")
        if not 0 < self.coarse_risk_threshold < self.hold_risk_threshold <= self.corrective_risk_threshold:
            raise ValueError("V2 risk thresholds must be ordered")
        if self.cooldown_prefixes < 0 or self.stable_prefixes_for_coarsening < 1:
            raise ValueError("V2 hysteresis lengths are invalid")
        if self.uncertainty_probe_threshold < 0:
            raise ValueError("V2 uncertainty probe threshold is invalid")
        positive = (self.fallback_hold_local_budget_fraction, self.coarse_tail_limit,
                    self.coarse_localized_sharpness_limit, self.coarse_history_velocity_limit,
                    self.coarse_history_acceleration_limit)
        if any(value <= 0 or not np.isfinite(value) for value in positive):
            raise ValueError("V2 physical stability thresholds must be positive")


@dataclass(slots=True)
class V2ControllerState:
    cooldown_remaining: int = 0
    stable_hold_prefixes: int = 0


@dataclass(frozen=True, slots=True)
class V2ControllerDecision:
    action_id: str | None
    state: str
    learned_authority: bool
    probe_required: bool
    upper_risk: float | None
    support_distance: float | None
    cooldown_remaining: int
    stable_hold_prefixes: int
    considered_actions: tuple[str, ...]


def _deterministically_admissible(candidate: V2Candidate, reserve_fraction: float) -> bool:
    return bool(
        candidate.invariant_passed and candidate.projection_passed
        and candidate.deterministic_error
        <= (1.0 - reserve_fraction) * candidate.remaining_budget
    )


class V2SafetyShieldedController:
    """Lazy controller whose learned branch can only make choices stricter."""

    def __init__(self, ensemble: FrozenV2PredictiveRiskEnsemble, config: V2ControllerConfig) -> None:
        self.ensemble = ensemble
        self.config = config
        self.state = V2ControllerState()

    def _physically_stable_hold(self, candidate: V2Candidate) -> bool:
        return bool(
            candidate.deterministic_error <= self.config.coarse_risk_threshold * candidate.local_budget
            and candidate.spectral_tail <= self.config.coarse_tail_limit
            and candidate.localized_sharpness <= self.config.coarse_localized_sharpness_limit
            and candidate.history_velocity_norm <= self.config.coarse_history_velocity_limit
            and candidate.history_acceleration_norm <= self.config.coarse_history_acceleration_limit
        )

    def _commit_state(self, action_id: str, *, physically_stable_hold: bool) -> None:
        if action_id in ("refine_space", COARSEN_ACTION):
            self.state.cooldown_remaining = self.config.cooldown_prefixes
            self.state.stable_hold_prefixes = 0
            return
        if self.state.cooldown_remaining:
            self.state.cooldown_remaining -= 1
        if action_id == "hold" and physically_stable_hold:
            self.state.stable_hold_prefixes += 1
        else:
            self.state.stable_hold_prefixes = 0

    def choose(self, build: Callable[[tuple[str, ...]], Iterable[V2Candidate]]) -> V2ControllerDecision:
        """Build hold first, expand only when hold cannot be safely retained."""
        initial = tuple(build(("hold",)))
        if len(initial) != 1 or initial[0].action_id != "hold":
            raise ValueError("V2 controller requires exactly one hold candidate first")
        hold = initial[0]
        considered = [hold]
        hold_safe = _deterministically_admissible(hold, self.config.reserve_fraction)
        hold_assessment = self.ensemble.assess(np.stack([hold.features]))[0] if hold_safe else None
        low_risk_hold = self._physically_stable_hold(hold)
        coarsen_window = bool(
            low_risk_hold and self.state.cooldown_remaining == 0
            and self.state.stable_hold_prefixes + 1 >= self.config.stable_prefixes_for_coarsening
        )
        if hold_safe and hold_assessment is not None and hold_assessment.in_support and (
            hold_assessment.upper_risk <= self.config.hold_risk_threshold
        ) and not coarsen_window:
            self._commit_state("hold", physically_stable_hold=low_risk_hold)
            probe = bool(hold_assessment.ensemble_spread >= self.config.uncertainty_probe_threshold)
            return V2ControllerDecision(
                "hold", "hold_retained", True, probe, hold_assessment.upper_risk,
                hold_assessment.support_distance, self.state.cooldown_remaining,
                self.state.stable_hold_prefixes, ("hold",),
            )
        alternatives = tuple(build(("shrink_time", "refine_space", COARSEN_ACTION)))
        by_action = {candidate.action_id: candidate for candidate in (*initial, *alternatives)}
        considered = [by_action[action] for action in ("hold", "shrink_time", "refine_space", COARSEN_ACTION) if action in by_action]
        learned = [candidate for candidate in considered if candidate.action_id in LEARNED_ACTIONS
                   and _deterministically_admissible(candidate, self.config.reserve_fraction)]
        assessments = self.ensemble.assess(np.stack([candidate.features for candidate in learned])) if learned else ()
        admitted = [
            (candidate, assessment) for candidate, assessment in zip(learned, assessments)
            if assessment.in_support and assessment.upper_risk <= self.config.corrective_risk_threshold
        ]
        coarse = by_action.get(COARSEN_ACTION)
        coarse_allowed = bool(
            coarse is not None and coarsen_window
            and _deterministically_admissible(coarse, self.config.reserve_fraction)
            and coarse.external_verification_passed
            and coarse.verified_future_risk is not None
            and coarse.verified_future_risk <= self.config.coarse_risk_threshold
        )
        if coarse_allowed:
            self._commit_state(COARSEN_ACTION, physically_stable_hold=False)
            return V2ControllerDecision(
                COARSEN_ACTION, "externally_verified_coarsening", False, False,
                coarse.verified_future_risk, None, self.state.cooldown_remaining,
                self.state.stable_hold_prefixes, tuple(item.action_id for item in considered),
            )
        if hold_safe and hold_assessment is not None and not hold_assessment.in_support and (
            hold.deterministic_error <= self.config.fallback_hold_local_budget_fraction * hold.local_budget
        ) and (self.state.cooldown_remaining > 0 or self.state.stable_hold_prefixes > 0):
            self._commit_state("hold", physically_stable_hold=low_risk_hold)
            return V2ControllerDecision(
                "hold", "verified_hold_fallback", False, True, hold_assessment.upper_risk,
                hold_assessment.support_distance, self.state.cooldown_remaining,
                self.state.stable_hold_prefixes, tuple(item.action_id for item in considered),
            )
        if admitted:
            chosen, assessment = min(admitted, key=lambda pair: (
                pair[0].work * (1.0 + pair[1].upper_risk), pair[1].upper_risk, pair[0].action_id,
            ))
            self._commit_state(chosen.action_id, physically_stable_hold=low_risk_hold)
            probe = bool(
                assessment.ensemble_spread >= self.config.uncertainty_probe_threshold
                or not hold_assessment or not hold_assessment.in_support
            )
            return V2ControllerDecision(
                chosen.action_id, "learned_corrective_candidate", True, probe,
                assessment.upper_risk, assessment.support_distance,
                self.state.cooldown_remaining, self.state.stable_hold_prefixes,
                tuple(item.action_id for item in considered),
            )
        fallback = [candidate for candidate in considered if _deterministically_admissible(candidate, self.config.reserve_fraction)
                    and candidate.action_id != COARSEN_ACTION]
        if fallback:
            # Once a deterministic candidate is already comfortably inside
            # the local error allocation, differences far below that scale do
            # not justify extra PDE work.  Prefer the least expensive such
            # candidate; retain the original error-first ordering only when no
            # fallback candidate reaches the prescribed local-budget margin.
            budget_sufficient = [
                candidate for candidate in fallback
                if candidate.deterministic_error
                <= self.config.fallback_hold_local_budget_fraction * candidate.local_budget
            ]
            if budget_sufficient:
                chosen = min(
                    budget_sufficient,
                    key=lambda candidate: (
                        candidate.work, candidate.deterministic_error,
                        candidate.action_id,
                    ),
                )
            else:
                chosen = min(
                    fallback,
                    key=lambda candidate: (
                        candidate.deterministic_error, candidate.work,
                        candidate.action_id,
                    ),
                )
            self._commit_state(chosen.action_id, physically_stable_hold=low_risk_hold)
            return V2ControllerDecision(
                chosen.action_id, "deterministic_fallback", False, True,
                hold_assessment.upper_risk if hold_assessment else None,
                hold_assessment.support_distance if hold_assessment else None,
                self.state.cooldown_remaining, self.state.stable_hold_prefixes,
                tuple(item.action_id for item in considered),
            )
        return V2ControllerDecision(
            None, "no_deterministically_admissible_candidate", False, True,
            hold_assessment.upper_risk if hold_assessment else None,
            hold_assessment.support_distance if hold_assessment else None,
            self.state.cooldown_remaining, self.state.stable_hold_prefixes,
            tuple(item.action_id for item in considered),
        )


def v2_controller_config(mapping) -> V2ControllerConfig:
    """Build the immutable stage-4 controller configuration from YAML data."""
    return V2ControllerConfig(
        reserve_fraction=float(mapping["reserve_fraction"]),
        hold_risk_threshold=float(mapping["hold_risk_threshold"]),
        corrective_risk_threshold=float(mapping["corrective_risk_threshold"]),
        coarse_risk_threshold=float(mapping["coarse_risk_threshold"]),
        cooldown_prefixes=int(mapping["cooldown_prefixes"]),
        stable_prefixes_for_coarsening=int(mapping["stable_prefixes_for_coarsening"]),
        uncertainty_probe_threshold=float(mapping["uncertainty_probe_threshold"]),
        fallback_hold_local_budget_fraction=float(mapping.get("fallback_hold_local_budget_fraction", 0.50)),
        coarse_tail_limit=float(mapping.get("coarse_tail_limit", 0.02)),
        coarse_localized_sharpness_limit=float(mapping.get("coarse_localized_sharpness_limit", 0.95)),
        coarse_history_velocity_limit=float(mapping.get("coarse_history_velocity_limit", 6.0)),
        coarse_history_acceleration_limit=float(mapping.get("coarse_history_acceleration_limit", 7.0)),
    )
