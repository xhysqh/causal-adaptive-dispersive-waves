from __future__ import annotations

import torch


def squared_periodic_h1(
    values: torch.Tensor, h: torch.Tensor
) -> torch.Tensor:
    """Per-sample squared discrete periodic H1/A norm."""
    h_column = h.reshape(-1, 1)
    derivative = (
        torch.roll(values, -1, dims=-1)
        - torch.roll(values, 1, dims=-1)
    ) / (2.0 * h_column)
    return h * torch.sum(values**2 + derivative**2, dim=-1)


def relative_h1_defect_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    h: torch.Tensor,
    *,
    epsilon: float = 1.0e-12,
) -> torch.Tensor:
    """Mean relative periodic discrete H1 defect loss."""
    return torch.mean(
        squared_periodic_h1(prediction - target, h)
        / (squared_periodic_h1(target, h) + epsilon)
    )


def apply_trust_region(
    correction: torch.Tensor,
    baseline_increment: torch.Tensor,
    h: torch.Tensor,
    *,
    rho: float,
    epsilon: float = 1.0e-12,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply the A2 A-norm trust gate and return full audit quantities."""
    r_theta = torch.sqrt(
        torch.clamp(squared_periodic_h1(correction, h), min=0.0)
    )
    r_baseline = torch.sqrt(
        torch.clamp(squared_periodic_h1(baseline_increment, h), min=0.0)
    )
    scale = torch.minimum(
        torch.ones_like(r_theta),
        rho * (r_baseline + epsilon) / (r_theta + epsilon),
    )
    return correction * scale[:, None], scale, r_baseline, r_theta


def trust_region_penalty(
    correction: torch.Tensor,
    baseline_increment: torch.Tensor,
    h: torch.Tensor,
    *,
    rho: float,
    epsilon: float = 1.0e-12,
) -> torch.Tensor:
    r_theta = torch.sqrt(
        torch.clamp(squared_periodic_h1(correction, h), min=0.0)
    )
    r_baseline = torch.sqrt(
        torch.clamp(
            squared_periodic_h1(baseline_increment, h), min=0.0
        )
    )
    return torch.mean(
        torch.relu(r_theta / (r_baseline + epsilon) - rho).pow(2)
    )
