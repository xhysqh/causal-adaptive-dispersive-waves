"""Frozen method/scenario registry for the CMAME numerical baseline.

P0 contains no trainable model and no learned risk gate.  Its only purpose is
to bind each registered solution sector to an independent reference, baseline
methods and paper-facing metrics before the learned discretisation is built.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class MethodSpec:
    name: str
    representation: str
    sectors: tuple[str, ...]
    role: str
    conservative: bool
    adaptive: bool
    trainable: bool = False


@dataclass(frozen=True, slots=True)
class ScenarioSpec:
    name: str
    sector: str
    reference: str
    baselines: tuple[str, ...]
    required_metrics: tuple[str, ...]


METHODS = (
    MethodSpec("projected_heun", "eulerian", ("smooth",), "baseline", True, False),
    MethodSpec("invariant_midpoint", "eulerian", ("smooth",), "independent_baseline", True, False),
    MethodSpec("fourier_dop853", "eulerian_spectral", ("smooth",), "reference", False, False),
    MethodSpec("particle_dop853", "periodic_measure", ("peakon",), "reference", True, False),
    MethodSpec("uniform_hybrid", "eulerian_particle", ("mixed",), "reference", True, False),
    MethodSpec("hybrid_amr", "eulerian_particle_amr", ("mixed",), "baseline", True, True),
)

SCENARIOS = (
    ScenarioSpec("smooth_wave", "smooth", "fourier_dop853", ("projected_heun", "invariant_midpoint"), ("relative_l2", "relative_h1", "mass_drift", "h1_drift", "weak_residual")),
    ScenarioSpec("smooth_random", "smooth", "fourier_dop853", ("projected_heun",), ("relative_l2", "relative_h1", "mass_drift", "h1_drift", "weak_residual")),
    ScenarioSpec("single_peakon", "peakon", "particle_dop853", (), ("position_disagreement", "h1_drift", "weak_residual")),
    ScenarioSpec("separated_multipeakon", "peakon", "particle_dop853", (), ("position_disagreement", "h1_drift", "weak_residual", "minimum_separation")),
    ScenarioSpec("clustered_multipeakon", "peakon", "particle_dop853", (), ("position_disagreement", "h1_drift", "weak_residual", "minimum_separation")),
    ScenarioSpec("mixed_field_peakon", "mixed", "uniform_hybrid", ("hybrid_amr",), ("relative_h1", "mass_drift", "h1_drift", "active_fraction")),
)


def validate_registry() -> dict[str, bool]:
    methods = {row.name: row for row in METHODS}
    names = [row.name for row in SCENARIOS]
    references_exist = all(row.reference in methods for row in SCENARIOS)
    baselines_exist = all(name in methods for row in SCENARIOS for name in row.baselines)
    sector_compatible = all(
        row.sector in methods[row.reference].sectors
        and all(row.sector in methods[name].sectors for name in row.baselines)
        for row in SCENARIOS
    )
    return {
        "method_names_unique": len(methods) == len(METHODS),
        "scenario_names_unique": len(names) == len(set(names)),
        "references_exist": references_exist,
        "baselines_exist": baselines_exist,
        "sector_compatible": sector_compatible,
        "no_trainable_method_in_p0": not any(row.trainable for row in METHODS),
        "smooth_peakon_mixed_covered": {row.sector for row in SCENARIOS} == {"smooth", "peakon", "mixed"},
    }


def registry_metadata() -> dict:
    checks = validate_registry()
    return {
        "version": "CMAME-P0-v1",
        "frozen_scope": "one-dimensional periodic pre-collision modified Camassa-Holm",
        "methods": [asdict(row) for row in METHODS],
        "scenarios": [asdict(row) for row in SCENARIOS],
        "checks": checks,
        "status": "PASS" if all(checks.values()) else "STOP",
    }
