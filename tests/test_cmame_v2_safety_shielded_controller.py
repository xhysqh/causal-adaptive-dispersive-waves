import numpy as np
import torch

from fgsp_ch.nonlocal_waves.adaptivity.mechanism_coordinates import V2_CAUSAL_FEATURE_ORDER
from fgsp_ch.nonlocal_waves.adaptivity.v2_predictive_risk_operator import V2PredictiveRiskOperator
from fgsp_ch.nonlocal_waves.adaptivity.v2_safety_shielded_controller import (
    COARSEN_ACTION, FrozenV2PredictiveRiskEnsemble, V2Candidate,
    V2ControllerConfig, V2SafetyShieldedController,
)


def _ensemble():
    width = len(V2_CAUSAL_FEATURE_ORDER)
    models = [V2PredictiveRiskOperator(hidden=8).eval() for _ in range(3)]
    manifest = {
        "feature_order": V2_CAUSAL_FEATURE_ORDER, "robust_center": [0.0] * width,
        "robust_scale": [1.0] * width, "_root": ".", "support_anchors": "unused.npz",
        "support_neighbors": 1, "support_radius": 1e6,
        "ensemble_std_multiplier": 1.0, "conformal_log_risk_margin": 0.0,
    }
    ensemble = FrozenV2PredictiveRiskEnsemble(models, manifest)
    # Avoid filesystem use in unit tests while preserving the frozen inference contract.
    ensemble.assess = lambda rows: tuple(type("A", (), {
        "upper_risk": 0.1, "ensemble_spread": 0.0, "support_distance": 0.0,
        "in_support": True, "probe_probability": 0.0,
    })() for _ in rows)
    return ensemble


def _candidate(action, *, safe=True, verified=False):
    return V2Candidate(
        action, np.zeros(len(V2_CAUSAL_FEATURE_ORDER)), 1e-4 if safe else 2.0,
        1.0, 1.0, safe, safe,
        external_verification_passed=verified,
        verified_future_risk=0.1 if verified else None,
    )


def test_safe_in_support_hold_is_retained_without_expanding_candidates():
    controller = V2SafetyShieldedController(_ensemble(), V2ControllerConfig())
    calls = []
    def build(actions):
        calls.append(actions)
        return tuple(_candidate(action) for action in actions)
    decision = controller.choose(build)
    assert decision.action_id == "hold" and decision.learned_authority
    assert calls == [("hold",)]


def test_unsafe_hold_expands_and_never_selects_unverified_coarsening():
    controller = V2SafetyShieldedController(_ensemble(), V2ControllerConfig())
    def build(actions):
        return tuple(_candidate(action, safe=action != "hold") for action in actions)
    decision = controller.choose(build)
    assert decision.action_id in {"shrink_time", "refine_space"}
    assert decision.action_id != COARSEN_ACTION


def test_coarsening_requires_external_verification_after_hysteresis():
    controller = V2SafetyShieldedController(_ensemble(), V2ControllerConfig(cooldown_prefixes=0, stable_prefixes_for_coarsening=1))
    controller.state.stable_hold_prefixes = 1
    controller.ensemble.assess = lambda rows: tuple(type("A", (), {
        "upper_risk": 0.1, "ensemble_spread": 0.0, "support_distance": 0.0,
        "in_support": True, "probe_probability": 0.0,
    })() for _ in rows)
    def build(actions):
        return tuple(_candidate(action, verified=action == COARSEN_ACTION) for action in actions)
    decision = controller.choose(build)
    assert decision.action_id == COARSEN_ACTION
    assert not decision.learned_authority


def test_verified_hold_precedes_learned_refinement_outside_support_during_cooldown():
    controller = V2SafetyShieldedController(_ensemble(), V2ControllerConfig())
    controller.state.cooldown_remaining = 2
    controller.ensemble.assess = lambda rows: tuple(type("A", (), {
        "upper_risk": 0.2, "ensemble_spread": 0.0, "support_distance": 1e9,
        "in_support": False, "probe_probability": 0.0,
    })() for _ in rows)
    hold = V2Candidate(
        "hold", np.zeros(len(V2_CAUSAL_FEATURE_ORDER)), 1e-4, 1.0, 1.0, True, True,
        local_budget=1e-3, localized_sharpness=0.1,
    )
    refine = V2Candidate(
        "refine_space", np.zeros(len(V2_CAUSAL_FEATURE_ORDER)), 1e-6, 1.0, 2.0, True, True,
    )
    shrink = V2Candidate(
        "shrink_time", np.zeros(len(V2_CAUSAL_FEATURE_ORDER)), 1e-6, 1.0, 2.0, True, True,
    )
    def build(actions):
        lookup = {"hold": hold, "refine_space": refine, "shrink_time": shrink}
        return tuple(lookup[action] for action in actions if action in lookup)
    decision = controller.choose(build)
    assert decision.action_id == "hold"
    assert decision.state == "verified_hold_fallback"
    assert not decision.learned_authority and decision.probe_required


def test_deterministic_fallback_prefers_work_after_local_budget_is_satisfied():
    controller = V2SafetyShieldedController(_ensemble(), V2ControllerConfig())
    controller.ensemble.assess = lambda rows: tuple(type("A", (), {
        "upper_risk": 2.0, "ensemble_spread": 0.0, "support_distance": 0.0,
        "in_support": True, "probe_probability": 0.0,
    })() for _ in rows)
    candidates = {
        "hold": V2Candidate(
            "hold", np.zeros(len(V2_CAUSAL_FEATURE_ORDER)), 4e-4,
            1.0, 1.0, True, True, local_budget=1e-3,
        ),
        "shrink_time": V2Candidate(
            "shrink_time", np.zeros(len(V2_CAUSAL_FEATURE_ORDER)), 1e-6,
            1.0, 2.0, True, True, local_budget=1e-3,
        ),
        "refine_space": V2Candidate(
            "refine_space", np.zeros(len(V2_CAUSAL_FEATURE_ORDER)), 1e-7,
            1.0, 3.0, True, True, local_budget=1e-3,
        ),
    }
    decision = controller.choose(
        lambda actions: tuple(candidates[action] for action in actions if action in candidates)
    )
    assert decision.action_id == "hold"
    assert decision.state == "deterministic_fallback"
    assert not decision.learned_authority and decision.probe_required
