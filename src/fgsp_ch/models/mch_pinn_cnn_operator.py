from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from fgsp_ch.models.mch_atlas_network import MinimalMCHAtlasNetwork


@dataclass(frozen=True, slots=True)
class MCHOperatorOutput:
    particle_position: torch.Tensor
    particle_amplitude: torch.Tensor
    cluster_position: torch.Tensor
    cluster_moments: torch.Tensor
    field_momentum: torch.Tensor
    predicted_error_ratio: torch.Tensor
    step_factor: torch.Tensor


class MonotoneAdaptiveHead(nn.Module):
    """Positive monotone error head; it can only tighten a certified step.

    Indicators are non-negative causal quantities: embedded error, weak
    residual, ensemble disagreement, correction size, and support distance.
    Positive weights prevent training from declaring a larger defect safer.
    """

    def __init__(self, indicators: int = 5, *, order: int = 2) -> None:
        super().__init__()
        if indicators < 1 or order < 1:
            raise ValueError("Adaptive-head dimensions and order must be positive.")
        self.raw_weights = nn.Parameter(torch.zeros(indicators))
        self.bias = nn.Parameter(torch.tensor(-2.0))
        self.register_buffer("calibration_scale", torch.tensor(1.0))
        self.order = order

    def set_calibration_scale(self, value: float) -> None:
        """Apply a one-sided post-training calibration multiplier."""
        if not torch.isfinite(torch.tensor(value)) or value < 1.0:
            raise ValueError("Calibration may only increase predicted risk.")
        self.calibration_scale.fill_(float(value))

    def forward(
        self,
        indicators: torch.Tensor,
        *,
        safety_factor: float = 0.9,
        minimum_factor: float = 0.25,
        maximum_factor: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if indicators.shape[-1] != self.raw_weights.numel():
            raise ValueError("Unexpected number of adaptive indicators.")
        safe = torch.nan_to_num(indicators, nan=1e6, posinf=1e6, neginf=0.0).clamp_min(0.0)
        weights = torch.nn.functional.softplus(self.raw_weights)
        ratio = self.calibration_scale * torch.nn.functional.softplus(
            self.bias + safe @ weights
        )
        factor = safety_factor * ratio.clamp_min(1e-12).pow(-1.0 / (self.order + 1.0))
        # M6 never lets the learned controller enlarge the M5-certified step.
        return ratio, factor.clamp(minimum_factor, maximum_factor)


class MCHPINNCNNOperator(nn.Module):
    """Unified trainable mCH atlas operator and causal adaptive-error head."""

    def __init__(
        self,
        *,
        field_channels: int = 10,
        parameter_channels: int = 8,
        hidden: int = 16,
        adaptive_indicators: int = 5,
    ) -> None:
        super().__init__()
        self.atlas = MinimalMCHAtlasNetwork(
            field_channels=field_channels,
            parameter_channels=parameter_channels,
            hidden=hidden,
            conserve_amplitude=True,
        )
        self.adaptive = MonotoneAdaptiveHead(adaptive_indicators)

    def forward(
        self,
        fields: torch.Tensor,
        parameters: torch.Tensor,
        positions: torch.Tensor,
        amplitudes: torch.Tensor,
        particle_mask: torch.Tensor,
        cluster_centers: torch.Tensor,
        cluster_signs: torch.Tensor,
        cluster_moments: torch.Tensor,
        cluster_mask: torch.Tensor,
        graph_parameters: torch.Tensor,
        length: torch.Tensor,
        adaptive_indicators: torch.Tensor,
    ) -> MCHOperatorOutput:
        particle_position, particle_amplitude = self.atlas.particle(
            positions, amplitudes, particle_mask, graph_parameters, length
        )
        cluster_position, moment_change = self.atlas.cluster(
            cluster_centers, cluster_signs, cluster_moments, cluster_mask,
            graph_parameters, length,
        )
        field_momentum = self.atlas.field(fields, parameters)
        error_ratio, step_factor = self.adaptive(adaptive_indicators)
        return MCHOperatorOutput(
            particle_position, particle_amplitude, cluster_position,
            moment_change, field_momentum, error_ratio, step_factor,
        )


def differentiable_periodic_helmholtz_solve(
    momentum: torch.Tensor, *, h: float
) -> torch.Tensor:
    """Invert the frozen centered-difference Helmholtz map with a torch FFT."""
    if momentum.ndim != 2 or h <= 0.0:
        raise ValueError("momentum must be [batch, points] and h must be positive.")
    modes = torch.arange(
        momentum.shape[-1], device=momentum.device, dtype=momentum.dtype
    )
    eigenvalues = 1.0 + 4.0 * torch.sin(
        torch.pi * modes / momentum.shape[-1]
    ).square() / h**2
    transformed = torch.fft.fft(momentum, dim=-1)
    return torch.fft.ifft(transformed / eigenvalues, dim=-1).real


def compose_mch_physical_step(
    baseline: torch.Tensor,
    field_momentum_rate: torch.Tensor,
    measure_basis: torch.Tensor,
    physical_measure_rate: torch.Tensor,
    *,
    state_scale: torch.Tensor,
    measure_scale: torch.Tensor,
    dt: torch.Tensor,
    h: float,
    measure_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Differentiably compose atlas heads into a physical Eulerian candidate.

    ``physical_measure_rate`` is already mapped out of cluster-whitened
    coordinates when a cluster atlas is active.  The field component is H1
    orthogonalized against the supplied causal measure tangent basis.
    """
    if baseline.shape != field_momentum_rate.shape or baseline.ndim != 2:
        raise ValueError("baseline and field rate must have equal [batch, points] shapes.")
    if measure_basis.ndim != 3 or measure_basis.shape[:2] != baseline.shape:
        raise ValueError("measure_basis must have shape [batch, points, coordinates].")
    if physical_measure_rate.shape != (
        baseline.shape[0], measure_basis.shape[-1]
    ):
        raise ValueError("Measure coordinate shape does not match its basis.")
    if measure_mask is None:
        measure_mask = torch.ones_like(physical_measure_rate, dtype=torch.bool)
    if measure_mask.shape != physical_measure_rate.shape:
        raise ValueError("measure_mask must match physical measure coordinates.")
    batch_scale = state_scale.reshape(-1, 1)
    batch_dt = dt.reshape(-1, 1)
    field = differentiable_periodic_helmholtz_solve(
        field_momentum_rate * batch_scale * batch_dt, h=h
    )
    if measure_basis.shape[-1]:
        applied_basis = measure_basis - (
            torch.roll(measure_basis, -1, dims=1)
            - 2.0 * measure_basis
            + torch.roll(measure_basis, 1, dims=1)
        ) / h**2
        active = measure_mask.to(baseline.dtype)
        gram = h * torch.bmm(measure_basis.transpose(1, 2), applied_basis)
        gram = gram + torch.diag_embed(1.0 - active)
        field_momentum = field - (
            torch.roll(field, -1, dims=1)
            - 2.0 * field
            + torch.roll(field, 1, dims=1)
        ) / h**2
        rhs = h * torch.bmm(
            measure_basis.transpose(1, 2), field_momentum.unsqueeze(-1)
        )
        coefficients = torch.linalg.solve(gram, rhs * active.unsqueeze(-1))
        field = field - torch.bmm(measure_basis, coefficients).squeeze(-1)
        measure = torch.bmm(
            measure_basis, (physical_measure_rate * active).unsqueeze(-1)
        ).squeeze(-1) * measure_scale.reshape(-1, 1) * batch_dt
    else:
        measure = torch.zeros_like(field)
    return baseline + field + measure
