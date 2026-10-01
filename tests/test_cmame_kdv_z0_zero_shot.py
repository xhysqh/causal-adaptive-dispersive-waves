from pathlib import Path

import numpy as np
import torch

from fgsp_ch.discretization.grids import PeriodicGrid
from fgsp_ch.nonlocal_waves.adaptivity.kdv_zero_shot import (
    FrozenKdVConsensusMechanism,
    kdv_resolution_indicators,
    project_kdv_spectrum,
)
from fgsp_ch.nonlocal_waves.adaptivity.prospective_causal_operator import (
    ACTIONS,
    FEATURE_ORDER,
    ActionEnvelope,
    FrozenProspectiveMechanism,
    SharedProspectiveBudgetOperator,
)
from fgsp_ch.nonlocal_waves.evaluation.kdv_metrics import kdv_invariants
from fgsp_ch.nonlocal_waves.initial_conditions.kdv_families import kdv_initial_condition
from fgsp_ch.nonlocal_waves.operators.kdv_fourier import KdVFourierOperator
from fgsp_ch.nonlocal_waves.solvers.kdv_reference import KdVIFRK4Solver, solve_kdv_dop853
from fgsp_ch.utils.config import load_yaml


def test_kdv_fourier_flux_and_ifrk_step_are_finite_and_mass_conservative():
    grid = PeriodicGrid(64, 20.0, -10.0)
    values = kdv_initial_condition("single_soliton", grid, seed=3)
    operator = KdVFourierOperator(grid)
    rhs = operator.rhs_spectrum(operator.spectrum(values))
    assert rhs.shape == (33,)
    assert rhs[0] == 0 and np.all(np.isfinite(rhs))
    initial = kdv_invariants(values, grid)
    advanced = KdVIFRK4Solver(grid).step(values, 2e-4)
    final = kdv_invariants(advanced, grid)
    assert np.all(np.isfinite(advanced))
    assert abs(final.mass - initial.mass) < 1e-12


def test_kdv_interaction_picture_agrees_with_independent_dop853_reference():
    grid = PeriodicGrid(32, 20.0, -10.0)
    values = kdv_initial_condition("single_soliton", grid, seed=7)
    operator = KdVFourierOperator(grid)
    values = operator.physical(operator.spectrum(values))
    final_time = 4e-4
    ifrk = KdVIFRK4Solver(grid).step(values, final_time)
    reference = solve_kdv_dop853(
        values, grid, evaluation_times=[0.0, final_time],
        rtol=1e-11, atol=1e-13, maximum_step=final_time / 4,
    ).final_state
    assert np.max(np.abs(ifrk - reference)) < 2e-10


def test_kdv_projection_roundtrip_and_online_indicators():
    grid = PeriodicGrid(64, 20.0, -10.0)
    values = kdv_initial_condition("two_soliton_collision", grid, seed=5)
    fine, fine_grid = project_kdv_spectrum(values, grid, 128)
    restored, _ = project_kdv_spectrum(fine, fine_grid, 64)
    # The unpaired even-grid Nyquist coefficient is intentionally removed by
    # every dealiased operator and is therefore outside the migration contract.
    filtered = KdVFourierOperator(grid).physical(KdVFourierOperator(grid).spectrum(values))
    assert np.max(np.abs(restored - filtered)) < 1e-12
    indicators = kdv_resolution_indicators(values, grid, 1e-3)
    assert all(np.isfinite(value) and value >= 0 for value in (
        indicators.tail_ratio, indicators.aliasing_defect,
        indicators.unresolved_increment, indicators.dispersive_stiffness,
        indicators.nonlinear_stiffness, indicators.localized_sharpness,
    ))


def test_strict_dual_adapter_consensus_abstains_outside_either_support():
    models = [SharedProspectiveBudgetOperator(hidden=8).eval() for _ in range(3)]
    support = {
        equation: {"mean": [0.0] * len(FEATURE_ORDER),
                   "scale": [1.0] * len(FEATURE_ORDER), "radius": 0.1}
        for equation in ("modified_camassa_holm", "benjamin_ono")
    }
    manifest = {
        "support": support,
        "margins": {"modified_camassa_holm": 0.0, "benjamin_ono": 0.0, "global": 0.0},
    }
    frozen = FrozenProspectiveMechanism(models, manifest, torch.device("cpu"))
    mechanism = FrozenKdVConsensusMechanism(frozen)
    actions = tuple(ActionEnvelope(action, 1e-6, 1e-2, 0, 1e-3, 1.0)
                    for action in ACTIONS)
    result = mechanism.choose(
        np.full((3, len(FEATURE_ORDER)), 10.0), actions, reserve_fraction=0.1
    )
    assert result.abstained and result.action_id is None
    assert set(result.support_distances) == {"modified_camassa_holm", "benjamin_ono"}


def test_kdv_protocol_is_zero_shot_fixed_horizon_and_claim_separating():
    root = Path(__file__).resolve().parents[1]
    source = (root / "scripts/run_cmame_kdv_z0_zero_shot_audit.py").read_text()
    adapter = (root / "src/fgsp_ch/nonlocal_waves/adaptivity/kdv_zero_shot.py").read_text()
    cfg = load_yaml(root / "configs/experiment/cmame_kdv_z0_zero_shot.yaml")
    assert "optimizer" not in source.lower() and "optimizer" not in adapter.lower()
    assert "reference_cache" in source and "online_decisions.csv" in source
    assert source.index("online_decisions.csv") < source.index("reference_cache = {}")
    assert "np.maximum.reduce(list(donor_logs.values()))" in adapter
    assert "SAFE_BUT_UTILITY_NOT_ESTABLISHED" in source
    assert cfg["seeds"]["external_base"] >= 90000000
    assert cfg["full"]["decisions"] >= 32
    assert set(cfg["full"]["tolerances"]) == {0.01, 0.003}
