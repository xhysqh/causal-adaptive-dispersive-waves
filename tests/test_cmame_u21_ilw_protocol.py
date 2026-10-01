from pathlib import Path

from fgsp_ch.utils.config import load_yaml


def test_ilw_p0_and_u21_are_fresh_frozen_protocols():
    root = Path(__file__).resolve().parents[1]
    p0 = load_yaml(root / "configs/experiment/cmame_ilw_p0_reference.yaml")
    u21 = load_yaml(root / "configs/experiment/cmame_u21_ilw_blind_bridge.yaml")
    source = (root / "scripts/run_cmame_u21_ilw_blind_audit.py").read_text()
    assert p0["next_phase_on_pass"] == "CMAME-U2.1_ILW_blind_bridge_audit"
    assert u21["seeds"]["external_base"] >= 141000000
    assert u21["full"]["depths"] == [0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 20.0]
    assert source.index("online_decisions.csv") < source.rindex("solve_ilw_dop853(")
    assert "raw_proposal" in source
    assert "optimizer" not in source.lower()
    assert '"equation_id":' not in source
    assert '"depth": depth' in source
    assert '"reference_visibility": "posthoc_only"' in source
    assert "manifest_immutable" in source
    assert "zero_unsafe_release" in source
    assert "adaptive_action_intervention" in source


def test_u21_does_not_relax_full_acceptance_in_development_block():
    root = Path(__file__).resolve().parents[1]
    config = load_yaml(root / "configs/experiment/cmame_u21_ilw_blind_bridge.yaml")
    assert config["acceptance"]["minimum_tolerance_coverage"] >= 0.95
    assert config["acceptance"]["minimum_depth_coverage"] >= 0.90
    assert config["acceptance"]["minimum_probe_reduction_fraction"] >= 0.50
    assert config["acceptance"]["maximum_median_error_ratio_to_always_probe"] <= 1.05
