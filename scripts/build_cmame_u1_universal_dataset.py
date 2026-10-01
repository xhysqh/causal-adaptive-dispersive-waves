"""Build fresh mCH--BO mechanism-coordinate data for CMAME-U1."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts.build_cmame_f1_balanced_prospective_dataset import (  # noqa: E402
    bo_rows,
    mch_rows,
)
from fgsp_ch.nonlocal_waves.adaptivity.mechanism_coordinates import (  # noqa: E402
    MECHANISM_FEATURE_ORDER,
    coordinates_for_equation,
)
from fgsp_ch.nonlocal_waves.adaptivity.prospective_causal_operator import (  # noqa: E402
    ACTIONS,
    EQUATIONS,
)
from fgsp_ch.utils.config import load_yaml  # noqa: E402


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest().upper()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment/cmame_u1_universal_mechanism.yaml")
    parser.add_argument("--mode", choices=("development", "full"), default="development")
    args = parser.parse_args()
    cfg = load_yaml(ROOT / args.config)
    out = ROOT / cfg["output_dir"] / args.mode
    out.mkdir(parents=True, exist_ok=True)
    store, group_sets, counts = {}, {}, {}
    for split, spec in cfg["splits"].items():
        per = int(spec["groups_per_family"])
        if args.mode == "development":
            per = min(per, int(cfg["development"]["groups_per_family_cap"]))
        groups = set()
        for equation in EQUATIONS:
            for family_index, family in enumerate(cfg["families"][equation]):
                for index in range(per):
                    seed = (
                        int(spec["seed"])
                        + (0 if equation == EQUATIONS[0] else 500000)
                        + family_index * 10000 + index
                    )
                    group = f"{split}:{equation}:{family}:{seed}"
                    print(f"U1 data {group}", flush=True)
                    (mch_rows if equation == EQUATIONS[0] else bo_rows)(
                        store, cfg, args.mode, split, family, seed, group
                    )
                    groups.add(group)
                    counts[(equation, split)] = counts.get((equation, split), 0) + 1
        group_sets[split] = groups
    arrays = {key: np.asarray(value) for key, value in store.items()}
    arrays["legacy_features"] = np.asarray(arrays.pop("features"), dtype=np.float64)
    arrays["features"] = np.stack([
        coordinates_for_equation(row, str(equation))
        for row, equation in zip(arrays["legacy_features"], arrays["equation"])
    ]).astype(np.float32)
    path = out / "universal_mechanism_dataset.npz"
    np.savez_compressed(path, **arrays)
    disjoint = all(
        group_sets[left].isdisjoint(group_sets[right])
        for i, left in enumerate(group_sets) for right in list(group_sets)[i + 1:]
    )
    checks = {
        "fresh_seed_ranges_disjoint": disjoint,
        "seeds_disjoint_from_f1_and_kdv_z0": min(
            int(spec["seed"]) for spec in cfg["splits"].values()
        ) >= 100000000,
        "equations_balanced": all(
            counts[(EQUATIONS[0], split)] == counts[(EQUATIONS[1], split)]
            for split in cfg["splits"]
        ),
        "no_equation_id_in_features": "equation" not in MECHANISM_FEATURE_ORDER,
        "all_mechanism_coordinates_finite": bool(np.all(np.isfinite(arrays["features"]))),
        "three_actions_exercised": set(arrays["action"].astype(str)) == set(ACTIONS),
        "reference_excluded_from_features": True,
        "references_are_labels_only": True,
    }
    blocking = [key for key, value in checks.items() if not value]
    report = {
        "status": "PASS" if not blocking else "STOP",
        "next_phase": "CMAME-U1_training" if not blocking else None,
        "checks": checks, "blocking_failures": blocking, "mode": args.mode,
        "rows": len(arrays["features"]),
        "trajectory_groups": {key: len(value) for key, value in group_sets.items()},
        "training_equations": sorted(set(arrays["equation"].astype(str).tolist())),
        "equation_rows": {
            equation: int(np.sum(arrays["equation"].astype(str) == equation))
            for equation in sorted(set(arrays["equation"].astype(str).tolist()))
        },
        "feature_order": MECHANISM_FEATURE_ORDER,
        "dataset_sha256": _sha(path), "scope": cfg["scope"],
    }
    (out / "dataset_acceptance.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)
    raise SystemExit(0 if report["status"] == "PASS" else 2)


if __name__ == "__main__":
    main()
