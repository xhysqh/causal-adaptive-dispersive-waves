from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class DeploymentSupportCertificate:
    """Finite-sample support certificate for deployment candidate features."""

    mean: np.ndarray
    scale: np.ndarray
    centers: dict[str, np.ndarray]
    radii: dict[str, np.ndarray]
    radius_fraction: float

    @staticmethod
    def cell_key(family: str, points: int) -> str:
        return f"{family}:{int(points)}"

    def support_ratio(
        self, features: np.ndarray, families: np.ndarray, points: np.ndarray,
    ) -> np.ndarray:
        rows = np.asarray(features, dtype=np.float64)
        normalized = (rows - self.mean) / self.scale
        result = np.full(rows.shape[0], np.inf, dtype=np.float64)
        for index, (row, family, grid) in enumerate(
            zip(normalized, families, points, strict=True)
        ):
            key = self.cell_key(str(family), int(grid))
            centers = self.centers.get(key)
            radii = self.radii.get(key)
            if centers is None or radii is None or not len(centers):
                continue
            distance = np.linalg.norm(centers - row[None, :], axis=1)
            result[index] = float(np.min(distance / np.maximum(radii, 1.0e-12)))
        return result

    def contains(self, features: np.ndarray, family: str, points: int) -> bool:
        ratio = self.support_ratio(
            np.asarray(features, dtype=np.float64)[None, :],
            np.asarray([family]), np.asarray([points]),
        )
        return bool(ratio[0] <= 1.0)

    def to_dict(self) -> dict:
        return {
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
            "centers": {key: value.tolist() for key, value in self.centers.items()},
            "radii": {key: value.tolist() for key, value in self.radii.items()},
            "radius_fraction": self.radius_fraction,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "DeploymentSupportCertificate":
        return cls(
            mean=np.asarray(payload["mean"], dtype=np.float64),
            scale=np.asarray(payload["scale"], dtype=np.float64),
            centers={
                key: np.asarray(value, dtype=np.float64)
                for key, value in payload["centers"].items()
            },
            radii={
                key: np.asarray(value, dtype=np.float64)
                for key, value in payload["radii"].items()
            },
            radius_fraction=float(payload["radius_fraction"]),
        )


def build_support_certificate(
    features: np.ndarray,
    safe: np.ndarray,
    families: np.ndarray,
    points: np.ndarray,
    *,
    radius_fraction: float,
) -> DeploymentSupportCertificate:
    if not 0.0 < radius_fraction < 1.0:
        raise ValueError("radius_fraction must lie strictly between zero and one.")
    rows = np.asarray(features, dtype=np.float64)
    safe = np.asarray(safe, dtype=bool)
    mean = rows.mean(axis=0)
    scale = rows.std(axis=0)
    scale = np.where(scale > 1.0e-8, scale, 1.0)
    normalized = (rows - mean) / scale
    centers: dict[str, np.ndarray] = {}
    radii: dict[str, np.ndarray] = {}
    cells = sorted({
        DeploymentSupportCertificate.cell_key(str(family), int(grid))
        for family, grid in zip(families, points, strict=True)
    })
    for key in cells:
        family, grid_text = key.rsplit(":", 1)
        cell = (families == family) & (points == int(grid_text))
        safe_centers = normalized[cell & safe]
        unsafe_centers = normalized[cell & ~safe]
        if not len(safe_centers) or not len(unsafe_centers):
            continue
        distance = np.linalg.norm(
            safe_centers[:, None, :] - unsafe_centers[None, :, :], axis=2,
        )
        centers[key] = safe_centers
        radii[key] = radius_fraction * np.min(distance, axis=1)
    return DeploymentSupportCertificate(mean, scale, centers, radii, radius_fraction)
