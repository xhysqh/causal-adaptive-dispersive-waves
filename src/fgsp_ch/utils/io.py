from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from fgsp_ch.simulation import Trajectory


def save_trajectory(
    path: str | Path,
    trajectory: Trajectory,
    *,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Save a trajectory and JSON metadata in one compressed NPZ artifact."""
    encoded_metadata = json.dumps(
        metadata or {}, ensure_ascii=False, sort_keys=True
    )
    np.savez_compressed(
        Path(path),
        times=trajectory.times,
        states=trajectory.states,
        momenta=trajectory.momenta,
        energies=trajectory.energies,
        projection_scales=trajectory.projection_scales,
        metadata=np.asarray(encoded_metadata),
    )


def load_trajectory(
    path: str | Path,
) -> tuple[Trajectory, dict[str, Any]]:
    """Load a trajectory saved by :func:`save_trajectory`."""
    with np.load(Path(path), allow_pickle=False) as archive:
        trajectory = Trajectory(
            times=archive["times"],
            states=archive["states"],
            momenta=archive["momenta"],
            energies=archive["energies"],
            projection_scales=archive["projection_scales"],
        )
        metadata = json.loads(str(archive["metadata"].item()))
    return trajectory, metadata

