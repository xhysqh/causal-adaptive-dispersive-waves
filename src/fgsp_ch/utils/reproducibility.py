from __future__ import annotations

import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
import scipy
import yaml


def set_seed(seed: int) -> None:
    np.random.seed(seed)


def environment_metadata() -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    metadata = {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "matplotlib": matplotlib.__version__,
        "pandas": pd.__version__,
        "pyyaml": yaml.__version__,
        "processor": platform.processor(),
        "machine": platform.machine(),
        "cpu_count": __import__("os").cpu_count(),
        "compute_backend": "cpu",
        "gpu_memory_bytes": 0,
        "git_commit": commit,
    }
    try:
        import torch

        metadata.update(
            {
                "torch": torch.__version__,
                "torch_cuda_available": torch.cuda.is_available(),
                "torch_cuda_version": torch.version.cuda,
            }
        )
    except ImportError:
        metadata["torch"] = None
    return metadata


def write_environment(path: str | Path) -> None:
    Path(path).write_text(
        json.dumps(environment_metadata(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
