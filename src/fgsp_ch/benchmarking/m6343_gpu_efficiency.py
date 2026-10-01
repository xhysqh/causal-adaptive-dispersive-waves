from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Callable

import numpy as np
import torch

from fgsp_ch.discretization.helmholtz import HelmholtzOperator
from fgsp_ch.geometry.metric import a_norm
from fgsp_ch.solvers.mch_hybrid_amr import (
    HierarchicalPeriodicGreenSolver,
    HybridAMRState,
)
from fgsp_ch.solvers.multipeakon import reconstruct_multipeakon


@dataclass(frozen=True, slots=True)
class TimingProtocol:
    warmup: int = 20
    repeats: int = 30

    def __post_init__(self) -> None:
        if self.warmup < 0 or self.repeats < 3:
            raise ValueError("warmup must be nonnegative and repeats at least three.")


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def benchmark_callable(
    operation: Callable[[], object],
    *,
    device: torch.device,
    protocol: TimingProtocol,
    work_items: int = 1,
) -> dict[str, float | int]:
    """Time an already-constructed operation; model/data setup is excluded."""
    if work_items < 1:
        raise ValueError("work_items must be positive.")
    for _ in range(protocol.warmup):
        operation()
    synchronize(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    samples = []
    for _ in range(protocol.repeats):
        synchronize(device)
        started = perf_counter()
        operation()
        synchronize(device)
        samples.append(perf_counter() - started)
    values = np.asarray(samples, dtype=np.float64)
    median = float(np.median(values))
    return {
        "warmup": protocol.warmup,
        "repeats": protocol.repeats,
        "work_items": work_items,
        "median_seconds": median,
        "p95_seconds": float(np.quantile(values, 0.95)),
        "items_per_second": float(work_items / max(median, np.finfo(float).eps)),
        "peak_device_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda" else 0
        ),
    }


def fine_total(state: HybridAMRState, green: HierarchicalPeriodicGreenSolver) -> np.ndarray:
    values = green.solve(state.composite_field)
    if state.positions.size:
        values += reconstruct_multipeakon(
            green.fine_grid, state.positions, state.amplitudes
        )
    return np.asarray(values)


def relative_state_h1(
    left: HybridAMRState,
    right: HybridAMRState,
    green: HierarchicalPeriodicGreenSolver,
) -> float:
    left_field, right_field = fine_total(left, green), fine_total(right, green)
    operator = HelmholtzOperator(green.fine_grid)
    return float(a_norm(left_field - right_field, operator) / max(
        a_norm(right_field, operator), np.finfo(float).eps
    ))


def percentile_breakdown(rows: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    """Aggregate optional solver phase timings without hiding fallback cost."""
    keys = sorted({key for row in rows for key in row})
    return {
        key: {
            "median_seconds": float(np.median([row.get(key, 0.0) for row in rows])),
            "p95_seconds": float(np.quantile([row.get(key, 0.0) for row in rows], 0.95)),
        }
        for key in keys
    }
