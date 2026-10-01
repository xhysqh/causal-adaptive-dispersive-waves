"""Frozen, paper-facing protocol for the CMAME-M2 benchmark.

M2.0 does not execute a solver.  It turns the numerical claims into immutable
case, method, reference and metric contracts before any comparison is run.
This prevents a failed method or a favourable reference from changing the
test population after results have been observed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Any, Iterable


@dataclass(frozen=True, slots=True)
class MethodSpec:
    name: str
    category: str
    representation: str
    families: tuple[str, ...]
    implementation: str
    trainable: bool
    adaptive_space: bool
    adaptive_time: bool
    frozen: bool


@dataclass(frozen=True, slots=True)
class FamilySpec:
    name: str
    sector: str
    reference: str
    required_methods: tuple[str, ...]
    required_metrics: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class BenchmarkCase:
    case_id: str
    group_id: str
    family: str
    sector: str
    alpha: float
    tolerance: float
    final_time: float
    seed: int
    reference: str


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def content_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest().upper()


def _method(row: dict[str, Any]) -> MethodSpec:
    return MethodSpec(
        name=str(row["name"]),
        category=str(row["category"]),
        representation=str(row["representation"]),
        families=tuple(map(str, row["families"])),
        implementation=str(row["implementation"]),
        trainable=bool(row["trainable"]),
        adaptive_space=bool(row["adaptive_space"]),
        adaptive_time=bool(row["adaptive_time"]),
        frozen=bool(row["frozen"]),
    )


def _family(row: dict[str, Any]) -> FamilySpec:
    return FamilySpec(
        name=str(row["name"]),
        sector=str(row["sector"]),
        reference=str(row["reference"]),
        required_methods=tuple(map(str, row["required_methods"])),
        required_metrics=tuple(map(str, row["required_metrics"])),
    )


def build_cases(config: dict[str, Any], mode: str) -> list[BenchmarkCase]:
    if mode not in config["design"]:
        raise ValueError(f"unknown M2 mode {mode!r}")
    design = config["design"][mode]
    families = [_family(row) for row in config["families"]]
    seeds_per_cell = int(design["seeds_per_cell"])
    if seeds_per_cell < 1:
        raise ValueError("seeds_per_cell must be positive")
    cases: list[BenchmarkCase] = []
    seed_base = int(design["seed_base"])
    for family_index, family in enumerate(families):
        for alpha_index, alpha in enumerate(map(float, design["alphas"])):
            for tolerance_index, tolerance in enumerate(map(float, design["tolerances"])):
                for horizon_index, final_time in enumerate(map(float, design["final_times"])):
                    for replicate in range(seeds_per_cell):
                        # A trajectory group is a paired initial condition.
                        # Tolerance and horizon must never alter its seed.
                        seed = seed_base + 100000 * family_index + 10000 * alpha_index + replicate
                        group_id = f"{family.name}:a{alpha:g}:seed{seed}"
                        case_id = (
                            f"{group_id}:tol{tolerance:g}:T{final_time:g}"
                        )
                        cases.append(
                            BenchmarkCase(
                                case_id=case_id,
                                group_id=group_id,
                                family=family.name,
                                sector=family.sector,
                                alpha=alpha,
                                tolerance=tolerance,
                                final_time=final_time,
                                seed=seed,
                                reference=family.reference,
                            )
                        )
    if len({row.case_id for row in cases}) != len(cases):
        raise ValueError("M2 case ids are not unique")
    group_seeds = {row.group_id: row.seed for row in cases}
    if any(row.seed != group_seeds[row.group_id] for row in cases):
        raise ValueError("M2 paired trajectory group does not share one seed")
    if len(set(group_seeds.values())) != len(group_seeds):
        raise ValueError("M2 seeds must be unique across trajectory groups")
    return cases


def validate_protocol(config: dict[str, Any], cases: Iterable[BenchmarkCase]) -> dict[str, bool]:
    methods = [_method(row) for row in config["methods"]]
    families = [_family(row) for row in config["families"]]
    method_map = {row.name: row for row in methods}
    case_rows = list(cases)
    reference_names = {row.reference for row in families}
    registered_metrics = set(map(str, config["metrics"]["registered"]))
    m13 = method_map.get("proposed_m13")
    return {
        "pure_mch_equation_frozen": (
            str(config["equation"]["name"]) == "modified_camassa_holm"
            and float(config["equation"]["gamma"]) == 0.0
        ),
        "method_names_unique": len(method_map) == len(methods),
        "family_names_unique": len({row.name for row in families}) == len(families),
        "references_registered": reference_names <= set(method_map),
        "references_independent_and_audit_only": all(
            method_map[name].category == "reference"
            and not method_map[name].trainable
            and method_map[name].implementation != "proposed"
            for name in reference_names
        ),
        "required_methods_registered": all(
            name in method_map for row in families for name in row.required_methods
        ),
        "required_methods_support_family": all(
            row.name in method_map[name].families
            for row in families
            for name in row.required_methods
        ),
        "required_metrics_registered": all(
            set(row.required_metrics) <= registered_metrics for row in families
        ),
        "proposed_solver_frozen": bool(m13 and m13.frozen and m13.implementation == "proposed"),
        "trajectory_ids_unique": len({row.case_id for row in case_rows}) == len(case_rows),
        "paired_group_seed_consistent": all(
            len({row.seed for row in case_rows if row.group_id == group}) == 1
            for group in {row.group_id for row in case_rows}
        ),
        "group_seeds_unique": len({row.seed for row in case_rows}) == len({row.group_id for row in case_rows}),
        "all_families_exercised": {row.family for row in case_rows} == {row.name for row in families},
        "paired_tolerances_present": len({row.tolerance for row in case_rows}) >= 2,
        "multiple_horizons_present": len({row.final_time for row in case_rows}) >= 2,
        "reference_excluded_from_online_controller": config["reference_contract"]["visibility"] == "posthoc_audit_only",
        "group_level_statistics_frozen": config["statistics"]["resampling_unit"] == "trajectory_group",
    }


def protocol_metadata(config: dict[str, Any], mode: str) -> dict[str, Any]:
    methods = [_method(row) for row in config["methods"]]
    families = [_family(row) for row in config["families"]]
    cases = build_cases(config, mode)
    checks = validate_protocol(config, cases)
    payload = {
        "version": "CMAME-M2.0-v1",
        "mode": mode,
        "equation": config["equation"],
        "methods": [asdict(row) for row in methods],
        "families": [asdict(row) for row in families],
        "metrics": config["metrics"],
        "statistics": config["statistics"],
        "reference_contract": config["reference_contract"],
        "cases": [asdict(row) for row in cases],
        "checks": checks,
        "status": "PASS" if all(checks.values()) else "STOP",
    }
    payload["protocol_sha256"] = content_sha256(payload)
    return payload
