from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GramSupportCertificate:
    mean: np.ndarray
    whitener: np.ndarray
    radius: float
    condition_number: float
    samples: int

    def distance(self, embedding: np.ndarray) -> np.ndarray:
        values = np.asarray(embedding, dtype=np.float64)
        return np.linalg.norm((values - self.mean) @ self.whitener.T, axis=1)

    def support_ratio(self, embedding: np.ndarray) -> np.ndarray:
        return self.distance(embedding) / max(self.radius, np.finfo(float).eps)

    def to_dict(self) -> dict:
        return {
            "mean": self.mean.tolist(),
            "whitener": self.whitener.tolist(),
            "radius": self.radius,
            "condition_number": self.condition_number,
            "samples": self.samples,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "GramSupportCertificate":
        return cls(
            mean=np.asarray(payload["mean"], dtype=np.float64),
            whitener=np.asarray(payload["whitener"], dtype=np.float64),
            radius=float(payload["radius"]),
            condition_number=float(payload["condition_number"]),
            samples=int(payload["samples"]),
        )


def fit_gram_support(
    embedding: np.ndarray,
    *,
    radius_quantile: float = 0.99,
    shrinkage: float = 0.05,
    eigenvalue_floor: float = 1e-6,
) -> GramSupportCertificate:
    values = np.asarray(embedding, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < values.shape[1] + 2:
        raise ValueError("insufficient embedding rows for Gram whitening")
    if not np.all(np.isfinite(values)):
        raise ValueError("embedding must be finite")
    if not 0.5 < radius_quantile < 1.0 or not 0.0 <= shrinkage < 1.0:
        raise ValueError("invalid support hyperparameters")
    mean = values.mean(axis=0)
    centered = values - mean
    covariance = centered.T @ centered / max(values.shape[0] - 1, 1)
    scale = float(np.trace(covariance) / covariance.shape[0])
    regularized = (1.0 - shrinkage) * covariance + shrinkage * scale * np.eye(
        covariance.shape[0]
    )
    eigenvalues, eigenvectors = np.linalg.eigh(regularized)
    floor = max(eigenvalue_floor, eigenvalue_floor * max(scale, 1.0))
    eigenvalues = np.maximum(eigenvalues, floor)
    whitener = (eigenvectors * (1.0 / np.sqrt(eigenvalues))) @ eigenvectors.T
    distance = np.linalg.norm(centered @ whitener.T, axis=1)
    radius = float(np.quantile(distance, radius_quantile, method="higher"))
    return GramSupportCertificate(
        mean=mean,
        whitener=whitener,
        radius=max(radius, np.finfo(float).eps),
        condition_number=float(eigenvalues.max() / eigenvalues.min()),
        samples=int(values.shape[0]),
    )
