"""Independent periodic coordinate-PINN baseline for Benjamin--Ono."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from fgsp_ch.models.cmame_m22_learned_baselines import CoordinatePINN, periodic_context
from fgsp_ch.training.cmame_m22_learned_baselines import predict_profiles


@dataclass(frozen=True)
class BOPINNArrays:
    initial: np.ndarray
    target: np.ndarray
    amplitude: np.ndarray
    time: np.ndarray
    family: np.ndarray
    group_id: np.ndarray

    def validate(self) -> None:
        rows = len(self.initial)
        if self.initial.ndim != 2 or self.target.shape != self.initial.shape:
            raise ValueError("BO PINN fields must align as [samples, points]")
        for value in (self.amplitude, self.time, self.family, self.group_id):
            if np.asarray(value).shape != (rows,):
                raise ValueError("BO PINN metadata must align with samples")
        if not np.isfinite(self.initial).all() or not np.isfinite(self.target).all():
            raise FloatingPointError("BO PINN fields must be finite")


class BOCoordinatePINN(nn.Module):
    """Periodic coordinate PINN with a BO-resolved Fourier coordinate map."""

    def __init__(self, *, hidden=96, layers=5, modes=12, coordinate_modes=12):
        super().__init__()
        self.modes = int(modes)
        self.coordinate_modes = int(coordinate_modes)
        width = 2 * self.coordinate_modes + 2 + 2 + 2 * self.modes
        blocks: list[nn.Module] = [nn.Linear(width, hidden), nn.Tanh()]
        for _ in range(int(layers) - 1):
            blocks.extend((nn.Linear(hidden, hidden), nn.Tanh()))
        blocks.append(nn.Linear(hidden, 1))
        self.network = nn.Sequential(*blocks)

    def forward(self, coordinate, tau, alpha, initial_value, context):
        harmonics = []
        for mode in range(1, self.coordinate_modes + 1):
            harmonics.extend((torch.sin(mode * coordinate), torch.cos(mode * coordinate)))
        features = torch.cat(
            (*[value.unsqueeze(-1) for value in harmonics], tau.unsqueeze(-1),
             alpha.unsqueeze(-1), context), dim=-1,
        )
        return initial_value + tau * self.network(features).squeeze(-1)


def _spectral_derivative(values: torch.Tensor, length: float) -> torch.Tensor:
    wave = 2.0 * torch.pi * torch.fft.fftfreq(
        values.shape[-1], d=length / values.shape[-1], device=values.device
    ).to(values.dtype)
    return torch.fft.ifft(
        1j * wave * torch.fft.fft(values, dim=-1), dim=-1
    ).real


def _abs_d(values: torch.Tensor, length: float) -> torch.Tensor:
    wave = 2.0 * torch.pi * torch.fft.fftfreq(
        values.shape[-1], d=length / values.shape[-1], device=values.device
    ).to(values.dtype)
    return torch.fft.ifft(
        torch.abs(wave) * torch.fft.fft(values, dim=-1), dim=-1
    ).real


def bo_physics_loss(
    model: BOCoordinatePINN,
    initial: torch.Tensor,
    time: torch.Tensor,
    amplitude: torch.Tensor,
    *,
    length: float,
    final_time_scale: float,
) -> torch.Tensor:
    """Centered finite-time BO residual on predicted periodic profiles."""
    half_width = torch.minimum(
        torch.full_like(time, 0.025 * final_time_scale),
        0.20 * torch.clamp(time, min=0.025 * final_time_scale),
    )
    left_time = torch.clamp(time - half_width, min=0.0)
    right_time = torch.clamp(time + half_width, max=final_time_scale)
    left = predict_profiles(
        model, "wang_yan_pinn", initial, left_time, amplitude,
        final_time_scale=final_time_scale,
    )
    middle = predict_profiles(
        model, "wang_yan_pinn", initial, time, amplitude,
        final_time_scale=final_time_scale,
    )
    right = predict_profiles(
        model, "wang_yan_pinn", initial, right_time, amplitude,
        final_time_scale=final_time_scale,
    )
    ut = (right - left) / (right_time - left_time).clamp_min(1e-8)[:, None]
    nonlinear = 0.5 * _spectral_derivative(middle.square(), length)
    dispersion = _abs_d(_spectral_derivative(middle, length), length)
    residual = ut + nonlinear - dispersion
    scale = torch.sqrt(torch.mean(ut.square() + dispersion.square(), dim=-1)).clamp_min(1e-4)
    return torch.mean((residual / scale[:, None]).square())


def train_bo_coordinate_pinn(
    train: BOPINNArrays,
    validation: BOPINNArrays,
    *,
    seed: int,
    device: torch.device,
    model_options: dict,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    physics_weight: float,
    length: float,
    final_time_scale: float,
) -> tuple[BOCoordinatePINN, dict[str, float]]:
    train.validate(); validation.validate()
    torch.manual_seed(int(seed))
    model = BOCoordinatePINN(**model_options).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    initial = torch.as_tensor(train.initial, dtype=torch.float32, device=device)
    target = torch.as_tensor(train.target, dtype=torch.float32, device=device)
    amplitude = torch.as_tensor(train.amplitude, dtype=torch.float32, device=device)
    time = torch.as_tensor(train.time, dtype=torch.float32, device=device)
    generator = torch.Generator(device="cpu").manual_seed(int(seed) + 701)
    for epoch in range(int(epochs)):
        permutation = torch.randperm(len(initial), generator=generator)
        model.train()
        for start in range(0, len(initial), int(batch_size)):
            index = permutation[start:start + int(batch_size)].to(device)
            prediction = predict_profiles(
                model, "wang_yan_pinn", initial[index], time[index], amplitude[index],
                final_time_scale=final_time_scale,
            )
            increment = target[index] - initial[index]
            floor = 0.01 * target[index].square().mean(-1).sqrt()
            scale = increment.square().mean(-1).sqrt().maximum(floor).clamp_min(1e-4)
            data_loss = torch.mean(((prediction - target[index]) / scale[:, None]).square())
            physics = bo_physics_loss(
                model, initial[index], time[index], amplitude[index], length=length,
                final_time_scale=final_time_scale,
            )
            loss = data_loss + float(physics_weight) * physics
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        if epoch == 0 or (epoch + 1) % max(1, epochs // 5) == 0:
            print(
                f"BO-PINN seed={seed} epoch={epoch+1}/{epochs} "
                f"data={data_loss.item():.6g} physics={physics.item():.6g}", flush=True,
            )
    model.eval()
    with torch.no_grad():
        vi = torch.as_tensor(validation.initial, dtype=torch.float32, device=device)
        vt = torch.as_tensor(validation.target, dtype=torch.float32, device=device)
        va = torch.as_tensor(validation.amplitude, dtype=torch.float32, device=device)
        vtime = torch.as_tensor(validation.time, dtype=torch.float32, device=device)
        prediction = predict_profiles(
            model, "wang_yan_pinn", vi, vtime, va,
            final_time_scale=final_time_scale,
        )
        validation_mse = torch.mean((prediction - vt).square()).item()
        persistence_mse = torch.mean((vi - vt).square()).item()
    return model, {
        "validation_mse": float(validation_mse),
        "persistence_mse": float(persistence_mse),
        "validation_ratio_to_persistence": float(validation_mse / max(persistence_mse, 1e-30)),
    }
