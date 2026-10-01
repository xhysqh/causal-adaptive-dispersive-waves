"""Pre-registered CMAME-E3 routing and paper-statistics protocol."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict
from typing import Iterable

import numpy as np

from fgsp_ch.benchmarking.cmame_p4_registry import P4Case, build_cases


PRIMARY_METHODS = (
    "e3_full",
    "classical_atlas",
    "classical_uniform_coarse",
    "classical_uniform_fine",
    "classical_residual_amr",
    "uniform_fine_neural",
)
CONVENTIONAL_METHODS = frozenset(PRIMARY_METHODS[1:])


def representation_route(initial, *, relative_tolerance: float = 1.0e-12) -> str:
    """Select a solver atlas from the represented state, never a family label."""
    positions=np.asarray(initial.positions)
    regular=np.asarray(initial.regular_momentum,dtype=float)
    if positions.size==0:
        return "uniform_eulerian"
    regular_scale=float(np.linalg.norm(regular))
    amplitude_scale=float(np.linalg.norm(np.asarray(initial.amplitudes,dtype=float)))
    if regular_scale<=relative_tolerance*max(amplitude_scale,1.0):
        return "particle"
    return "hybrid_amr"


def cohort_key(case:P4Case,route_name:str)->tuple[int,str,float]:
    """Cohorts share tensor shape, representation route and final time."""
    return int(case.points),str(route_name),float(case.final_time)


def registry_metadata(matrix:dict)->dict:
    cases=build_cases(matrix)
    identities=[case.case_id for case in cases]
    physical=defaultdict(set)
    for case in cases:
        physical[(case.family,case.alpha,case.initial_dt,case.seed)].add(case.points)
    expected_grids=set(map(int,matrix["grids"]))
    checks={
        "case_ids_unique":len(identities)==len(set(identities)),
        "matrix_nonempty":bool(cases),
        "all_physical_trajectories_cross_grid_paired":all(rows==expected_grids for rows in physical.values()),
        "primary_methods_unique":len(PRIMARY_METHODS)==len(set(PRIMARY_METHODS)),
        "proposed_method_unique":sum(name=="e3_full" for name in PRIMARY_METHODS)==1,
        "strong_baselines_present":{
            "classical_atlas","classical_uniform_fine","classical_residual_amr",
            "uniform_fine_neural"
        }.issubset(PRIMARY_METHODS),
    }
    return {
        "version":"CMAME-E3-v1",
        "status":"PASS" if all(checks.values()) else "STOP",
        "checks":checks,
        "methods":list(PRIMARY_METHODS),
        "cases":[asdict(row) for row in cases],
        "scope":"Frozen paper registry; family labels generate cases but are forbidden from representation routing.",
    }


def matched_accuracy(rows:Iterable[dict],*,relative_tolerance:float=0.05)->list[dict]:
    rows=list(rows); records=[]
    for proposal in (row for row in rows if row["method"]=="e3_full"):
        candidates=[row for row in rows if (
            row["method"] in CONVENTIONAL_METHODS
            and row["family"]==proposal["family"]
            and row["alpha"]==proposal["alpha"]
            and row["initial_dt"]==proposal["initial_dt"]
            and row["seed"]==proposal["seed"]
            and row["final_time"]==proposal["final_time"]
            and row["relative_h1"]<=(1.0+relative_tolerance)*proposal["relative_h1"]
        )]
        if not candidates:continue
        baseline=min(candidates,key=lambda row:row["runtime_seconds"])
        records.append({
            "case_id":proposal["case_id"],"family":proposal["family"],
            "alpha":proposal["alpha"],"initial_dt":proposal["initial_dt"],
            "seed":proposal["seed"],"final_time":proposal["final_time"],
            "e3_points":proposal["points"],"baseline_method":baseline["method"],
            "baseline_points":baseline["points"],"e3_h1":proposal["relative_h1"],
            "baseline_h1":baseline["relative_h1"],
            "speedup":baseline["runtime_seconds"]/max(proposal["runtime_seconds"],np.finfo(float).eps),
        })
    return records


def accuracy_dominance(rows:Iterable[dict],*,relative_tolerance:float=0.05)->list[dict]:
    """Return proposed rows more accurate than every registered baseline.

    These rows resolve accuracy coverage, but they are deliberately excluded
    from matched-speed statistics because no equal-accuracy comparator exists.
    """
    rows=list(rows);records=[]
    for proposal in (row for row in rows if row["method"]=="e3_full"):
        candidates=[row for row in rows if (
            row["method"] in CONVENTIONAL_METHODS
            and row["family"]==proposal["family"]
            and row["alpha"]==proposal["alpha"]
            and row["initial_dt"]==proposal["initial_dt"]
            and row["seed"]==proposal["seed"]
            and row["final_time"]==proposal["final_time"]
        )]
        if candidates and all(
            row["relative_h1"]>(1.0+relative_tolerance)*proposal["relative_h1"]
            for row in candidates
        ):
            nearest=min(candidates,key=lambda row:row["relative_h1"])
            records.append({
                "case_id":proposal["case_id"],"family":proposal["family"],
                "alpha":proposal["alpha"],"initial_dt":proposal["initial_dt"],
                "seed":proposal["seed"],"final_time":proposal["final_time"],
                "e3_points":proposal["points"],"e3_h1":proposal["relative_h1"],
                "nearest_baseline_method":nearest["method"],
                "nearest_baseline_h1":nearest["relative_h1"],
                "accuracy_gain":nearest["relative_h1"]/max(proposal["relative_h1"],np.finfo(float).eps),
            })
    return records


def grouped_bootstrap_median_interval(records:list[dict],*,seed:int=20260824,draws:int=2000)->list[float]:
    """Bootstrap physical trajectories, retaining their cross-grid dependence."""
    if not records:return [0.0,0.0]
    grouped=defaultdict(list)
    for row in records:
        grouped[(row["family"],row["alpha"],row["initial_dt"],row["seed"],row["final_time"])].append(float(row["speedup"]))
    groups=list(grouped.values())
    if len(groups)==1:
        value=float(np.median(groups[0]));return [value,value]
    rng=np.random.default_rng(seed);medians=[]
    for _ in range(int(draws)):
        selected=rng.integers(0,len(groups),size=len(groups))
        values=[value for index in selected for value in groups[int(index)]]
        medians.append(float(np.median(values)))
    return [float(np.quantile(medians,.025)),float(np.quantile(medians,.975))]


def pareto_front(method_summary:dict)->list[str]:
    names=[]
    for name,row in method_summary.items():
        error=float(row["median_h1"]);runtime=float(row["median_seconds"])
        dominated=any(
            other!=name
            and float(candidate["median_h1"])<=error
            and float(candidate["median_seconds"])<=runtime
            and (float(candidate["median_h1"])<error or float(candidate["median_seconds"])<runtime)
            for other,candidate in method_summary.items()
        )
        if not dominated:names.append(name)
    return sorted(names)
