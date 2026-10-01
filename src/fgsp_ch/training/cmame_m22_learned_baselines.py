"""Training utilities for the independent CMAME-M2.2 learned baselines."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from fgsp_ch.models.cmame_m22_learned_baselines import (
    CoordinatePINN,
    FNO1dBaseline,
    periodic_context,
    periodic_derivative,
)


@dataclass(frozen=True, slots=True)
class LearnedArrays:
    initial: np.ndarray
    target: np.ndarray
    alpha: np.ndarray
    time: np.ndarray
    family: np.ndarray
    seed: np.ndarray

    def validate(self) -> None:
        rows = self.initial.shape[0]
        if self.initial.ndim != 2 or self.target.shape != self.initial.shape:
            raise ValueError("learned profiles must be aligned [samples,points]")
        if any(np.asarray(value).shape != (rows,) for value in (self.alpha, self.time, self.family, self.seed)):
            raise ValueError("learned metadata must align with profiles")
        if not np.all(np.isfinite(self.initial)) or not np.all(np.isfinite(self.target)):
            raise FloatingPointError("learned dataset contains nonfinite profiles")


def _coordinate_prediction(
    model: CoordinatePINN,
    initial: torch.Tensor,
    time: torch.Tensor,
    alpha: torch.Tensor,
    *,
    final_time_scale: float,
) -> torch.Tensor:
    batch, points = initial.shape
    angle = 2.0 * torch.pi * torch.arange(points, device=initial.device, dtype=initial.dtype) / points
    coordinate = angle[None, :].expand(batch, -1)
    tau = (time / final_time_scale)[:, None].expand_as(initial)
    scaled_alpha = alpha[:, None].expand_as(initial)
    context = periodic_context(initial, model.modes)[:, None, :].expand(-1, points, -1)
    return model(coordinate, tau, scaled_alpha, initial, context)


def predict_profiles(
    model: torch.nn.Module,
    method: str,
    initial: torch.Tensor,
    time: torch.Tensor,
    alpha: torch.Tensor,
    *,
    final_time_scale: float,
) -> torch.Tensor:
    if method == "wang_yan_pinn":
        return _coordinate_prediction(model, initial, time, alpha, final_time_scale=final_time_scale)
    if method == "fno":
        return model(initial, time / final_time_scale, alpha)
    raise ValueError(f"unknown learned baseline {method}")


def _smooth_physics_loss(
    model: CoordinatePINN,
    initial: torch.Tensor,
    time: torch.Tensor,
    alpha: torch.Tensor,
    smooth: torch.Tensor,
    *,
    length: float,
    final_time_scale: float,
) -> torch.Tensor:
    """Finite-time strong residual, applied only away from peakon measures."""
    if not bool(torch.any(smooth)):
        return initial.new_zeros(())
    u0, t, a = initial[smooth], time[smooth], alpha[smooth]
    delta = torch.minimum(0.05 * torch.ones_like(t), (0.05 * final_time_scale / t.clamp_min(1e-8)))
    t0 = torch.clamp(t * (1.0 - delta), min=0.0)
    t1 = torch.clamp(t * (1.0 + delta), max=final_time_scale)
    left = _coordinate_prediction(model, u0, t0, a, final_time_scale=final_time_scale)
    right = _coordinate_prediction(model, u0, t1, a, final_time_scale=final_time_scale)
    middle = _coordinate_prediction(model, u0, t, a, final_time_scale=final_time_scale)
    m_left = left - periodic_derivative(left, 2, length)
    m_right = right - periodic_derivative(right, 2, length)
    m = middle - periodic_derivative(middle, 2, length)
    ux = periodic_derivative(middle, 1, length)
    flux = a[:, None] * (middle.square() - ux.square()) * m
    residual = (m_right - m_left) / (t1 - t0).clamp_min(1e-8)[:, None]
    residual = residual + periodic_derivative(flux, 1, length)
    scale = m.square().mean(dim=-1).sqrt().clamp_min(1e-5)
    return torch.mean((residual / scale[:, None]).square())


def train_baseline(
    method: str,
    arrays: LearnedArrays,
    validation: LearnedArrays,
    *,
    seed: int,
    device: torch.device,
    epochs: int,
    learning_rate: float,
    batch_size: int,
    final_time_scale: float,
    length: float,
    physics_weight: float,
    model_options: dict,
) -> tuple[torch.nn.Module, dict[str, float]]:
    arrays.validate(); validation.validate()
    torch.manual_seed(int(seed))
    if method == "wang_yan_pinn":
        model: torch.nn.Module = CoordinatePINN(**model_options)
    elif method == "fno":
        model = FNO1dBaseline(**model_options)
    else:
        raise ValueError(f"unknown learned baseline {method}")
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1.0e-5)
    initial = torch.as_tensor(arrays.initial, dtype=torch.float32, device=device)
    target = torch.as_tensor(arrays.target, dtype=torch.float32, device=device)
    alpha = torch.as_tensor(arrays.alpha, dtype=torch.float32, device=device)
    time = torch.as_tensor(arrays.time, dtype=torch.float32, device=device)
    smooth = torch.as_tensor(arrays.family == "smooth_periodic", device=device)
    generator = torch.Generator(device="cpu").manual_seed(int(seed) + 991)
    for epoch in range(epochs):
        permutation = torch.randperm(len(initial), generator=generator)
        model.train()
        for start in range(0, len(initial), batch_size):
            index = permutation[start : start + batch_size].to(device)
            prediction = predict_profiles(
                model, method, initial[index], time[index], alpha[index],
                final_time_scale=final_time_scale,
            )
            scale = target[index].square().mean(dim=-1).sqrt().clamp_min(1e-4)
            loss = torch.mean(((prediction - target[index]) / scale[:, None]).square())
            if method == "wang_yan_pinn" and physics_weight > 0.0:
                loss = loss + physics_weight * _smooth_physics_loss(
                    model, initial[index], time[index], alpha[index], smooth[index],
                    length=length, final_time_scale=final_time_scale,
                )
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        if epoch == 0 or (epoch + 1) % max(1, epochs // 5) == 0:
            print(f"M2.2 train method={method} seed={seed} epoch={epoch+1}/{epochs} loss={loss.detach().item():.6g}", flush=True)
    model.eval()
    with torch.no_grad():
        vi = torch.as_tensor(validation.initial, dtype=torch.float32, device=device)
        vt = torch.as_tensor(validation.target, dtype=torch.float32, device=device)
        va = torch.as_tensor(validation.alpha, dtype=torch.float32, device=device)
        vtime = torch.as_tensor(validation.time, dtype=torch.float32, device=device)
        prediction = predict_profiles(model, method, vi, vtime, va, final_time_scale=final_time_scale)
        mse = torch.mean((prediction - vt).square()).item()
        persistence = torch.mean((vi - vt).square()).item()
    return model, {
        "validation_mse": float(mse),
        "persistence_mse": float(persistence),
        "validation_ratio_to_persistence": float(mse / max(persistence, 1e-30)),
    }
