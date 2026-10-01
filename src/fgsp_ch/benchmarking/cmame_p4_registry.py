"""Pre-registered CMAME-P4 scenario and method matrix."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import product


@dataclass(frozen=True, slots=True)
class P4Method:
    name: str
    role: str
    adaptive_space: bool
    adaptive_time: bool
    p1: bool
    p22: bool
    ensemble: bool


@dataclass(frozen=True, slots=True)
class P4Case:
    case_id: str
    family: str
    points: int
    alpha: float
    initial_dt: float
    seed: int
    final_time: float


METHODS = (
    P4Method("p3_full", "proposed", True, True, True, True, True),
    P4Method("uniform_coarse_p22", "learned_baseline", False, False, True, True, True),
    P4Method("uniform_fine_p22", "learned_baseline", False, False, True, True, True),
    P4Method("classical_coarse", "classical_baseline", False, False, False, False, False),
    P4Method("fixed_space", "forced_ablation", False, True, True, True, True),
    P4Method("fixed_time", "forced_ablation", True, False, True, True, True),
    P4Method("single_controller", "forced_ablation", True, True, True, True, False),
    P4Method("no_p1", "forced_ablation", True, True, False, True, True),
    P4Method("no_p22", "forced_ablation", True, True, True, False, True),
    P4Method("zero_neural_flux", "forced_ablation", True, True, False, False, True),
)


def build_cases(matrix: dict, *, length: float = 2.0 * 3.141592653589793) -> list[P4Case]:
    rows: list[P4Case] = []
    families = list(map(str, matrix["families"]))
    alphas = list(map(float, matrix["alphas"]))
    dts = list(map(float, matrix["initial_dts"]))
    for family, points, alpha, dt, replicate in product(
        families, map(int, matrix["grids"]), alphas, dts,
        range(int(matrix["replicates"])),
    ):
        # A physical trajectory keeps the same seed across grids.  This is
        # required for legitimate matched-accuracy and convergence pairings.
        physical_index = (
            families.index(family) * len(alphas) * len(dts)
            + alphas.index(alpha) * len(dts) + dts.index(dt)
        )
        seed = int(matrix["seed_base"]) + 1000 * physical_index + replicate
        case_id = f"{family}:n{points}:a{alpha:g}:dt{dt:g}:s{seed}"
        rows.append(P4Case(
            case_id, family, points, alpha, dt, seed,
            float(matrix["horizon_steps"]) * dt,
        ))
    return rows


def registry_metadata(matrix: dict) -> dict:
    cases = build_cases(matrix)
    seed_sets: dict[tuple[str, float, float], set[int]] = {}
    for row in cases:
        seed_sets.setdefault((row.family, row.alpha, row.initial_dt), set()).add(row.seed)
    checks = {
        "method_names_unique": len({row.name for row in METHODS}) == len(METHODS),
        "proposed_method_unique": sum(row.role == "proposed" for row in METHODS) == 1,
        "classical_baseline_present": any(row.role == "classical_baseline" for row in METHODS),
        "uniform_fine_baseline_present": any(row.name == "uniform_fine_p22" for row in METHODS),
        "all_required_ablations_present": {
            "fixed_space", "fixed_time", "single_controller", "no_p1", "no_p22", "zero_neural_flux"
        }.issubset({row.name for row in METHODS}),
        "matrix_nonempty": bool(cases),
        "fresh_seed_namespace": all(row.seed >= int(matrix["seed_base"]) for row in cases),
        "grid_pairing_uses_common_seeds": all(
            len(seeds) == int(matrix["replicates"]) for seeds in seed_sets.values()
        ),
    }
    return {
        "version": "CMAME-P4-v1", "status": "PASS" if all(checks.values()) else "STOP",
        "checks": checks, "methods": [asdict(row) for row in METHODS],
        "cases": [asdict(row) for row in cases],
        "scope": "Frozen-model benchmark registry; no training, threshold selection or model replacement.",
    }
