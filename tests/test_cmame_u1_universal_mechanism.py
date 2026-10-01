from pathlib import Path

import numpy as np
import torch

from fgsp_ch.nonlocal_waves.adaptivity.mechanism_coordinates import (
    CAUSAL_CONTROL_FEATURE_ORDER,
    CAUSAL_HISTORY_FEATURE_ORDER,
    MECHANISM_FEATURE_ORDER,
    RobustMechanismScaler,
    CausalHistoryState,
    causal_control_coordinates,
    coordinates_for_equation,
    knn_support_score,
    mechanism_history_coordinates,
    trajectory_conformal_radius,
    v2_causal_coordinates,
    V2_CAUSAL_FEATURE_ORDER,
)
from fgsp_ch.nonlocal_waves.adaptivity.prospective_causal_operator import (
    ACTIONS,
    FEATURE_ORDER,
    ActionEnvelope,
)
from fgsp_ch.nonlocal_waves.adaptivity.trajectory_error_ledger import (
    TrajectoryErrorLedger,
)
from fgsp_ch.nonlocal_waves.adaptivity.universal_causal_operator import (
    FrozenUniversalCausalMechanism,
    UniversalCausalBudgetOperator,
)
from fgsp_ch.nonlocal_waves.adaptivity.v2_predictive_risk_operator import (
    V2PredictiveRiskOperator,
    predictive_risk_loss,
)
from fgsp_ch.utils.config import load_yaml


def test_mechanism_coordinates_are_dimensionless_masked_and_equation_id_free():
    legacy = np.zeros(len(FEATURE_ORDER))
    legacy[10], legacy[11] = 1.7, -3.0
    mch = coordinates_for_equation(legacy, "modified_camassa_holm")
    bo = coordinates_for_equation(legacy, "benjamin_ono")
    kdv = coordinates_for_equation(legacy, "kdv")
    assert mch.shape == bo.shape == kdv.shape == (len(MECHANISM_FEATURE_ORDER),)
    assert mch[10] == 1.0 and mch[11] == 0.0
    assert tuple(mch[-3:]) == (1.0, 0.0, 1.0)
    assert tuple(bo[-3:]) == tuple(kdv[-3:]) == (1.0, 1.0, 1.0)
    assert all("equation" not in name for name in MECHANISM_FEATURE_ORDER)


def test_robust_scaler_knn_support_and_group_conformal_are_finite():
    rng = np.random.default_rng(8)
    rows = rng.normal(size=(30, len(MECHANISM_FEATURE_ORDER)))
    rows[:, 0] = 1.0
    scaler = RobustMechanismScaler.fit(rows, minimum_scale=0.02)
    transformed = scaler.transform(rows)
    scores = knn_support_score(transformed[:6], transformed, neighbors=4)
    radius = trajectory_conformal_radius(
        scores, np.array(["a", "a", "b", "b", "c", "c"]), coverage=0.8
    )
    assert np.min(scaler.scale) >= 0.02
    assert np.all(np.isfinite(scores)) and np.all(scores >= 0)
    assert np.isfinite(radius) and radius >= 0


def test_causal_history_is_cold_started_nonanticipative_and_dimensionally_fixed():
    legacy = np.zeros(len(FEATURE_ORDER))
    legacy[1], legacy[2], legacy[5], legacy[10], legacy[11], legacy[4] = (
        -2.0, -3.0, 0.7, 0.2, 0.4, -5.0
    )
    base = coordinates_for_equation(legacy, "benjamin_ono")
    state = CausalHistoryState(rho=0.7)
    first = state.observe(mechanism_history_coordinates(base))
    assert np.allclose(first.deviation, 0.0)
    assert np.allclose(first.velocity, 0.0)
    assert np.allclose(first.acceleration, 0.0)
    updated = base.copy(); updated[1] += 0.6
    second = state.observe(mechanism_history_coordinates(updated))
    assert np.isclose(second.velocity[0], 0.6)
    assert np.isclose(second.acceleration[0], 0.0)
    assert np.isclose(second.deviation[0], 0.7 * 0.6)
    control = causal_control_coordinates(
        remaining_budget_fraction=0.8, remaining_time_fraction=0.5,
        resolution_fraction=0.25, current_dt=0.01, cooldown_fraction=0.0,
    )
    row = v2_causal_coordinates(updated, second, control)
    assert row.shape == (len(V2_CAUSAL_FEATURE_ORDER),)
    assert len(CAUSAL_HISTORY_FEATURE_ORDER) == 18
    assert len(CAUSAL_CONTROL_FEATURE_ORDER) == 5
    assert np.all(np.isfinite(row))


def test_v2_predictive_risk_upper_envelope_and_loss_are_finite():
    torch.manual_seed(7)
    model = V2PredictiveRiskOperator(hidden=8)
    features = torch.zeros((4, len(V2_CAUSAL_FEATURE_ORDER)))
    output = model(features)
    assert output.upper_log_risk.shape == (4,)
    assert torch.all(output.upper_log_risk >= output.mean_log_risk)
    loss = predictive_risk_loss(
        output, torch.zeros(4), torch.zeros(4), torch.tensor([0.0, 1.0, 0.0, 0.0]),
        upper_quantile=0.95, underestimate_weight=3.0,
        cost_weight=0.2, probe_weight=0.05, probe_positive_weight=50.0,
    )
    assert torch.isfinite(loss)


def test_universal_operator_has_one_path_and_returns_ordered_upper_bound():
    model = UniversalCausalBudgetOperator(hidden=12).eval()
    rows = torch.randn(5, len(MECHANISM_FEATURE_ORDER))
    output = model(rows)
    assert output.upper_log_effectivity.shape == (5,)
    assert torch.all(output.upper_log_effectivity >= output.median_log_effectivity)
    source = Path(
        "src/fgsp_ch/nonlocal_waves/adaptivity/universal_causal_operator.py"
    ).read_text()
    assert "equation_adapter" not in source and "equation_id" not in source


def test_trajectory_ledger_never_spends_more_than_its_registered_budget():
    ledger = TrajectoryErrorLedger(tolerance=1.0, final_time=1.0, reserve_fraction=0.1)
    assert ledger.admissible(0.2)
    ledger.commit(0.2)
    ledger.commit(0.3)
    assert np.isclose(ledger.remaining, 0.5)
    assert not ledger.admissible(0.6)


def test_frozen_universal_mechanism_abstains_outside_frozen_support():
    models = [UniversalCausalBudgetOperator(hidden=8).eval() for _ in range(3)]
    width = len(MECHANISM_FEATURE_ORDER)
    manifest = {
        "robust_center": [0.0] * width,
        "robust_scale": [1.0] * width,
        "support_neighbors": 2,
        "support_radius": 0.1,
        "conformal_margin": 0.0,
    }
    frozen = FrozenUniversalCausalMechanism(
        models, manifest, np.zeros((8, width)), device="cpu"
    )
    actions = tuple(
        ActionEnvelope(action, 1e-6, 1e-2, 0.0, 1e-3, 1.0)
        for action in ACTIONS
    )
    result = frozen.choose(np.full((3, width), 4.0), actions, reserve_fraction=0.1)
    assert result.abstained and result.action_id is None
    assert result.support_distances["universal"] > manifest["support_radius"]


def test_u1_and_blind_kdv_protocols_are_disjoint_and_precommitted():
    root = Path(__file__).resolve().parents[1]
    u1 = load_yaml(root / "configs/experiment/cmame_u1_universal_mechanism.yaml")
    kdv = load_yaml(root / "configs/experiment/cmame_kdv_u1_zero_shot.yaml")
    audit = (root / "scripts/run_cmame_kdv_z0_zero_shot_audit.py").read_text()
    training = (root / "scripts/train_cmame_u1_universal_operator.py").read_text()
    assert kdv["mechanism"]["kind"] == "universal"
    assert kdv["seeds"]["external_base"] > max(
        int(spec["seed"]) + 600000 for spec in u1["splits"].values()
    )
    assert "kdv_rows_used\": False" in training
    assert audit.index("online_decisions.csv") < audit.index("reference_cache = {}")
    assert "SAFE_BUT_UTILITY_NOT_ESTABLISHED" in audit
    assert "nontrivial_action_intervention" in audit
    dataset_builder = (root / "scripts/build_cmame_u1_universal_dataset.py").read_text()
    assert '"training_equations"' in dataset_builder
    assert '"equation_rows"' in dataset_builder
