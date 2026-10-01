from pathlib import Path

import numpy as np
import pytest

from fgsp_ch.visualization.pinn_comparison_publication import (
    ADVANTAGE_EPSILON,
    PINNComparisonTrajectory,
    _regularized_advantage,
    _six_registered_indices,
    comparison_metrics,
    plot_first_three_comparison_figures,
)


def sample() -> PINNComparisonTrajectory:
    x = np.linspace(0.0, 2.0 * np.pi, 32, endpoint=False)
    time = np.linspace(0.0, 1.0, 61)
    reference = np.stack([0.4 + 0.2 * np.cos(x - 0.3 * t) for t in time])
    baseline = reference + time[:, None] * 0.02 * np.sin(2 * x)[None, :]
    ours = reference + time[:, None] * 0.004 * np.sin(3 * x)[None, :]
    return PINNComparisonTrajectory(x, time, reference, baseline, ours)


def test_comparison_contract_and_metrics() -> None:
    trajectory = sample()
    metrics = comparison_metrics(trajectory)
    assert set(metrics) == {
        "pinn_l2", "pinn_h1", "pinn_linf", "ours_l2", "ours_h1", "ours_linf"
    }
    assert all(value.shape == trajectory.time.shape for value in metrics.values())
    assert np.allclose(metrics["pinn_l2"][0], 0.0)
    assert metrics["ours_l2"][-1] < metrics["pinn_l2"][-1]


def test_misaligned_field_is_rejected() -> None:
    trajectory = sample()
    with pytest.raises(ValueError, match="baseline_pinn"):
        PINNComparisonTrajectory(
            trajectory.x, trajectory.time, trajectory.reference,
            trajectory.baseline_pinn[:, :-1], trajectory.causal_adaptive,
        )


def test_first_three_figures_export(tmp_path: Path) -> None:
    stems = plot_first_three_comparison_figures(sample(), tmp_path, start_figure=10)
    assert len(stems) == 3
    assert [stem.name for stem in stems] == [
        "figure_10_full_field_comparison",
        "figure_11_six_prefix_errors",
        "figure_12_error_propagation",
    ]
    for stem in stems:
        assert stem.with_suffix(".pdf").is_file()
        assert stem.with_suffix(".png").is_file()


def test_six_registered_profiles_are_ordered_and_error_independent() -> None:
    x = np.linspace(0.0, 1.0, 16)
    time = np.linspace(0.0, 1.0, 101)
    field = np.cos(x)[None, :] * (1.0 + time[:, None])
    trajectory = PINNComparisonTrajectory(x, time, field, field + .1, field + .01)
    selected = _six_registered_indices(trajectory)
    assert len(selected) == len(set(selected)) == 6
    assert selected[-1] == len(time) - 1
    assert all(right > left for left, right in zip(selected, selected[1:]))


def test_advantage_uses_explicit_positive_regularizer() -> None:
    trajectory = sample()
    advantage = _regularized_advantage(trajectory)
    assert ADVANTAGE_EPSILON == 1.0e-12
    assert advantage.shape == trajectory.reference.shape
    assert np.isfinite(advantage).all()


def test_plot_only_entry_separates_mch_and_bo_numbering() -> None:
    source = Path("scripts/plot_cmame_figures_7_to_12.py").read_text()
    assert 'output / "mch"' in source
    assert 'output / "bo"' in source
    assert 'config["figure_numbers"]["mch"]' in source
    assert 'config["figure_numbers"]["bo"]' in source
