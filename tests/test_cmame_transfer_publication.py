from pathlib import Path

import numpy as np
import pytest

from fgsp_ch.visualization.transfer_publication import (
    TransferTrajectory, load_kdv, online_local_difficulty, plot_kdv_transfer,
    load_f2,
)


def test_f2_loader_matches_actions_times_and_resolution(tmp_path):
    x = np.arange(8, dtype=float)
    field = np.zeros((3, 8))
    path = tmp_path / "case.npz"
    np.savez(path, x=x, bo_time=[0., .1, .2], bo_field=field,
             bo_reference=field, bo_actions=["hold", "refine_space"])
    ledger = tmp_path / "ledger.csv"
    header = "equation,prefix,time,action,points\n"
    rows = "benjamin_ono,0,.1,hold,32\nbenjamin_ono,1,.2,refine_space,64\n"
    ledger.write_text(header + rows)
    result = load_f2(path, ledger, equation="bo", initial_points=32)
    assert list(result.active_points) == [32, 32, 64]
    assert list(result.actions) == ["initial", "hold", "refine_space"]
    ledger.write_text(header + rows.replace(".2,refine", ".3,refine"))
    with pytest.raises(ValueError, match="time mismatch"):
        load_f2(path, ledger, equation="bo", initial_points=32)


def test_transfer_trajectory_requires_aligned_online_evidence():
    x = np.linspace(0, 1, 8); time = np.linspace(0, 1, 4)
    field = np.zeros((4, 8))
    with pytest.raises(ValueError, match="online evidence"):
        TransferTrajectory("kdv", x, time, field, field, np.array(["hold"]),
                           np.ones(4), "precommitted")


def test_kdv_loader_rejects_reference_visibility_violation(tmp_path: Path):
    path = tmp_path / "bad.npz"
    x = np.linspace(0, 1, 8); time = np.linspace(0, 1, 4)
    reference = np.zeros((4, 8)); adaptive = np.ones((4, 8))
    np.savez(path, x=x, times=time, reference=reference, adaptive=adaptive,
             absolute_error=np.ones_like(reference), actions=np.full(4, "hold"),
             active_points=np.full(4, 8), selection_rule=np.asarray("fixed"),
             reference_visibility=np.asarray("online"))
    with pytest.raises(ValueError, match="posthoc-only"):
        load_kdv(path)


def test_transfer_entry_uses_post_comparison_figure_numbers():
    source = Path("scripts/plot_cmame_transfer_figures.py").read_text()
    assert "figure_13_kdv_causal_transfer" in source
    assert "figure_14_ilw_causal_transfer" in source
    assert "figure_15_power_law_causal_transfer" in source
    assert "local_diagnostics_exclude_reference_features" in source
    assert "local_sites_are_diagnostics_not_local_refinement_claims" in source
    assert 'raise SystemExit(0 if acceptance["status"] == "PASS" else 2)' in source


def test_local_online_diagnostic_is_finite_and_reference_free():
    x = np.linspace(0.0, 2.0 * np.pi, 64, endpoint=False)
    time = np.linspace(0.0, 1.0, 7)
    field = np.stack([np.sin(x - value) + 0.15 * np.sin(9.0 * x + value) for value in time])
    diagnostic = online_local_difficulty(field, float(x[1] - x[0]))
    assert diagnostic.shape == field.shape
    assert np.isfinite(diagnostic).all()
    assert np.allclose(diagnostic.max(axis=1), 1.0)


def test_high_difficulty_display_region_uses_only_online_eta():
    diagnostic = np.asarray([[1.0, 0.7, 0.5, 0.1], [0.2, 1.0, 0.6, 0.0]])
    high = diagnostic >= 0.60 * diagnostic.max(axis=1, keepdims=True)
    assert high.tolist() == [[True, True, False, False], [False, True, True, False]]


def test_final_figure_uses_minimal_authority_ledger_without_decision_plane():
    source = Path("src/fgsp_ch/visualization/transfer_publication.py").read_text()
    assert "_draw_decision_plane" not in source
    assert "causal decision time" not in source
    assert "posthoc validation:\\n" not in source
    assert "early-time zoom" not in source  # The inset title is only its time range.
    assert "hold_marker_stride': 6" in source
    assert r'difficult prefix $\tau^\star$' not in source
    assert "_decision_snapshot_indices" not in source
    assert "ax3d.text2D" not in source


def test_transfer_manifold_uses_frozen_adaptive_fields_and_saved_actions(tmp_path: Path):
    x = np.linspace(0.0, 2.0 * np.pi, 48, endpoint=False)
    time = np.linspace(0.0, 0.64, 65)
    adaptive = np.stack([np.sin(x - value) + 0.08 * np.sin(8.0 * x + value) for value in time])
    reference = adaptive + 1.0e-4 * np.cos(x)[None, :]
    trajectory = TransferTrajectory(
        "kdv", x, time, reference, adaptive,
        np.asarray(["initial", "refine_space"] + ["hold"] * 63),
        np.asarray([32] + [64] * 64), "precommitted geometry-only case",
    )
    target = tmp_path / "transfer_manifold"
    plot_kdv_transfer(trajectory, target)
    assert target.with_suffix(".png").is_file()
    assert target.with_suffix(".pdf").is_file()


def test_kdv_contract_stops_export_on_incorrect_ledger(tmp_path: Path):
    from fgsp_ch.visualization.transfer_publication import _assert_kdv_figure_contract
    x = np.linspace(0, 2 * np.pi, 32, endpoint=False)
    time = np.linspace(0, .64, 65)
    field = np.tile(np.sin(x), (65, 1))
    actions = np.asarray(["initial", "refine_space"] + ["hold"] * 63)
    n = np.asarray([32] + [64] * 64)
    case = TransferTrajectory("kdv", x, time, field, field, actions, n, "fixed")
    _assert_kdv_figure_contract(case)
    n[1] = 128
    with pytest.raises(AssertionError):
        plot_kdv_transfer(case, tmp_path / "invalid")
    assert not (tmp_path / "invalid.pdf").exists()
