"""Frozen evidence registry for the CMAME-U3 multi-equation benchmark."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


@dataclass(frozen=True, slots=True)
class FrozenEvidenceSource:
    source_id: str
    equations: tuple[str, ...]
    acceptance: Path
    rows: Path
    proposed_arm: str
    always_probe_arm: str | None = None
    analytical_arm: str | None = None

    def validate(self) -> dict:
        missing = [path for path in (self.acceptance, self.rows) if not path.exists()]
        if missing:
            raise FileNotFoundError(
                f"U3 source {self.source_id} is incomplete: {missing}"
            )
        report = json.loads(self.acceptance.read_text(encoding="utf-8"))
        if report.get("status") != "PASS":
            raise RuntimeError(f"U3 source {self.source_id} did not pass")
        return report

    def frozen_record(self) -> dict:
        report = self.validate()
        return {
            "source_id": self.source_id,
            "equations": list(self.equations),
            "acceptance": self.acceptance.as_posix(),
            "rows": self.rows.as_posix(),
            "acceptance_sha256": file_sha256(self.acceptance),
            "rows_sha256": file_sha256(self.rows),
            "source_status": report["status"],
            "proposed_arm": self.proposed_arm,
            "always_probe_arm": self.always_probe_arm,
            "analytical_arm": self.analytical_arm,
        }


def build_registry(root: Path, mapping: dict) -> tuple[FrozenEvidenceSource, ...]:
    result = []
    for source_id, row in mapping.items():
        result.append(FrozenEvidenceSource(
            source_id=str(source_id),
            equations=tuple(map(str, row["equations"])),
            acceptance=root / row["acceptance"],
            rows=root / row["rows"],
            proposed_arm=str(row["proposed_arm"]),
            always_probe_arm=(
                str(row["always_probe_arm"])
                if row.get("always_probe_arm") is not None else None
            ),
            analytical_arm=(
                str(row["analytical_arm"])
                if row.get("analytical_arm") is not None else None
            ),
        ))
    identifiers = [row.source_id for row in result]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("U3 evidence source identifiers must be unique")
    equations = [equation for row in result for equation in row.equations]
    if len(set(equations)) != len(equations):
        raise ValueError("each U3 equation must have exactly one primary source")
    return tuple(result)
