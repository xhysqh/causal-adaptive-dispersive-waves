from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
FORBIDDEN_SUFFIXES = {".npy", ".npz", ".pt", ".pth", ".ckpt"}
SKIP_PARTS = {".git", ".venv", "venv", "results", "output", "artifacts"}
WINDOWS_ABSOLUTE = re.compile(r"(?i)(?:[a-z]:\\(?:users|documents and settings)\\)")


def iter_source_files():
    for path in ROOT.rglob("*"):
        if path.is_file() and not any(part.lower() in SKIP_PARTS for part in path.parts):
            yield path


def nested_strings(value: Any):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from nested_strings(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from nested_strings(item)


def main() -> int:
    failures: list[str] = []

    distributed_binary_data = [
        path.relative_to(ROOT).as_posix()
        for path in iter_source_files()
        if path.suffix.lower() in FORBIDDEN_SUFFIXES
    ]
    if distributed_binary_data:
        failures.append("source-only policy violated: " + ", ".join(distributed_binary_data))

    for path in iter_source_files():
        if path.suffix.lower() not in {".py", ".yaml", ".yml", ".json", ".md", ".toml", ".txt"}:
            continue
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if WINDOWS_ABSOLUTE.search(content):
            failures.append(f"machine-specific user path in {path.relative_to(ROOT)}")

    missing_configs: set[str] = set()
    for path in (ROOT / "configs").rglob("*.yaml"):
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        for value in nested_strings(payload):
            normalized = value.replace("\\", "/")
            if normalized.startswith("configs/") and normalized.endswith((".yaml", ".yml")):
                if not (ROOT / normalized).is_file():
                    missing_configs.add(normalized)
    if missing_configs:
        failures.append("missing referenced configurations: " + ", ".join(sorted(missing_configs)))

    try:
        from fgsp_ch.nonlocal_waves.adaptivity.mechanism_coordinates import (
            CAUSAL_CONTROL_FEATURE_ORDER,
            CAUSAL_HISTORY_FEATURE_ORDER,
            MECHANISM_FEATURE_ORDER,
            V2_CAUSAL_FEATURE_ORDER,
        )

        widths = (
            len(MECHANISM_FEATURE_ORDER),
            len(CAUSAL_HISTORY_FEATURE_ORDER),
            len(CAUSAL_CONTROL_FEATURE_ORDER),
            len(V2_CAUSAL_FEATURE_ORDER),
        )
        if widths != (19, 18, 5, 42):
            failures.append(f"feature contract is {widths}, expected (19, 18, 5, 42)")
    except Exception as exc:  # pragma: no cover - diagnostic path
        failures.append(f"package import failed: {exc}")

    if failures:
        print("RELEASE CHECK: FAIL")
        for failure in failures:
            print(f"- {failure}")
        return 1

    print("RELEASE CHECK: PASS")
    print("- source-only policy: satisfied")
    print("- machine-specific user paths: none")
    print("- referenced configuration files: present")
    print("- estimator feature contract: 19 + 18 + 5 = 42")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
