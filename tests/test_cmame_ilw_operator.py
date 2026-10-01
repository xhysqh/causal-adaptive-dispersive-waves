from pathlib import Path

import numpy as np

from fgsp_ch.discretization.grids import PeriodicGrid
from fgsp_ch.nonlocal_waves.adaptivity.ilw_unknown import ilw_indicators
from fgsp_ch.nonlocal_waves.evaluation.ilw_metrics import relative_ilw_h1
from fgsp_ch.nonlocal_waves.operators.bo_fourier import BOFourierOperator
from fgsp_ch.nonlocal_waves.operators.ilw_fourier import (
    ILWFourierOperator,
    ilw_dispersion_multiplier,
    project_ilw_spectrum,
)
from fgsp_ch.nonlocal_waves.solvers.ilw_reference import (
    ILWIFRK54Solver,
    solve_ilw_dop853,
)


def _state(grid):
    return 0.2 + 0.15 * np.cos(2 * np.pi * grid.x / grid.length) + 0.04 * np.sin(
        6 * np.pi * grid.x / grid.length
    )


def test_ilw_multiplier_is_stable_in_shallow_water():
    k = np.asarray([0.0, 0.25, 1.0, 4.0])
    depth = 1e-6
    multiplier = ilw_dispersion_multiplier(k, depth)
    expected = depth * k**2 / 3.0
    np.testing.assert_allclose(multiplier, expected, rtol=2e-11, atol=1e-20)


def test_ilw_deep_water_symbol_converges_to_bo():
    grid = PeriodicGrid(64, 20.0, -10.0)
    ilw = ILWFourierOperator(grid, 1e7)
    bo = BOFourierOperator(grid)
    np.testing.assert_allclose(ilw.linear_symbol[1:-1], bo.linear_symbol[1:-1], rtol=4e-7)


def test_rescaled_ilw_shallow_symbol_converges_to_kdv():
    grid = PeriodicGrid(64, 20.0, -10.0)
    depth = 1e-5
    ilw = ILWFourierOperator(grid, depth)
    normalized = (3.0 / depth) * ilw.linear_symbol
    expected = 1j * ilw.wave_numbers**3
    np.testing.assert_allclose(normalized[1:-1], expected[1:-1], rtol=3e-8)


def test_ilw_embedded_step_matches_independent_dop853():
    grid = PeriodicGrid(64, 20.0, -10.0)
    values, dt = _state(grid), 2e-4
    embedded = ILWIFRK54Solver(grid, 0.8).step(values, dt)
    reference = solve_ilw_dop853(
        values, grid, 0.8, evaluation_times=[0.0, dt],
        rtol=1e-12, atol=1e-14, maximum_step=dt/4,
    ).states[-1]
    assert relative_ilw_h1(embedded, reference, grid) < 2e-10


def test_ilw_symbol_geometry_and_projection_are_finite():
    grid = PeriodicGrid(32, 20.0, -10.0)
    values = _state(grid)
    indicators = ilw_indicators(values, grid, 0.7, 1e-3)
    vector = indicators.symbol.mechanism_vector()
    assert vector.shape == (5,)
    assert np.all(np.isfinite(vector))
    assert indicators.symbol.effective_order_low >= indicators.symbol.effective_order_high
    fine, fine_grid = project_ilw_spectrum(values, grid, 128, 0.7)
    restored, _ = project_ilw_spectrum(fine, fine_grid, 32, 0.7)
    np.testing.assert_allclose(restored, values, rtol=2e-14, atol=2e-14)


def test_ilw_protocol_files_are_equation_name_free_at_model_boundary():
    root = Path(__file__).resolve().parents[1]
    source = (root / "src/fgsp_ch/nonlocal_waves/adaptivity/ilw_unknown.py").read_text()
    assert "equation_id" not in source
    assert "raw_proposal" not in source
    assert "effective_order_active" in source


def test_ilw_blind_audit_uses_local_budget_and_complete_work_accounting():
    root = Path(__file__).resolve().parents[1]
    audit = (root / "scripts/run_cmame_u21_ilw_blind_audit.py").read_text()
    assert "ledger.local_budget(prefix * macro_dt, macro_dt)" in audit
    assert '"candidate_solver_work"' in audit
    assert '"pde_online_work"' in audit
    assert '"full_symbol_geometry_is_posthoc_only": True' in audit
    assert "representative_trajectories.npz" in audit
    assert "reference_floor_audit.csv" in audit
