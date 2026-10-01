from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Any

import numpy as np


_ARRAY_KEYS = (
    "features",
    "parameters",
    "atom_positions",
    "atom_amplitudes",
    "atom_mask",
    "cluster_centers",
    "cluster_signs",
    "cluster_moments",
    "cluster_mask",
    "position_target",
    "cluster_whitened_target",
    "cluster_inverse_sqrt",
    "field_momentum_target",
    "momentum_defect",
)


def load_mch_training_samples(source: str | Path) -> list[dict[str, Any]]:
    """Load the audited M3 shards needed by M6.2 joint physical training."""
    root = Path(source)
    entries = json.loads((root / "dataset_manifest.json").read_text(encoding="utf-8"))
    samples: list[dict[str, Any]] = []
    for entry in entries:
        shard = root / entry["file"]
        actual_hash = hashlib.sha256(shard.read_bytes()).hexdigest().upper()
        expected_hash = str(entry.get("sha256", "")).upper()
        if not expected_hash or actual_hash != expected_hash:
            raise RuntimeError(
                f"M3 shard integrity failure for {entry['file']}: "
                f"expected {expected_hash or 'missing'}, got {actual_hash}."
            )
        with np.load(shard, allow_pickle=False) as archive:
            missing = [key for key in _ARRAY_KEYS if key not in archive]
            if missing:
                raise KeyError(f"M3 shard {entry['file']} lacks {missing}.")
            for index in range(int(entry["samples"])):
                sample = {
                    "split": entry["split"],
                    "group": entry["trajectory_group"],
                    "atlas": int(archive["atlas_code"][index]),
                    "state_scale": float(archive["state_scale"][index]),
                    "current_index": int(archive["current_index"][index]),
                    "cluster_position_identifiable": bool(
                        archive["cluster_position_identifiable"][index]
                    ),
                }
                for key in _ARRAY_KEYS:
                    target_key = "field_target" if key == "field_momentum_target" else key
                    values = archive[key][index]
                    sample[target_key] = values.astype(bool if key.endswith("mask") else np.float32)
                samples.append(sample)
    return samples
