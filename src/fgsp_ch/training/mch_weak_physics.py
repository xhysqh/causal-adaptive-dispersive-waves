from __future__ import annotations

from dataclasses import dataclass

import torch


def periodic_test_bank(
    points: int,
    *,
    modes: int = 4,
    localized_centers: int = 4,
    localized_width: float = 0.08,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Return normalized global and localized periodic weak test functions.

    Coordinates are represented on the unit torus.  The bank deliberately
    combines low Fourier modes with wrapped Gaussian windows so that a weak
    loss sees both the smooth field and localized peak/cluster defects.
    """
    if points < 5 or modes < 1 or localized_centers < 0:
        raise ValueError("Invalid periodic weak-test-bank dimensions.")
    if not 0.0 < localized_width < 0.5:
        raise ValueError("localized_width must lie in (0, 0.5).")
    x = torch.arange(points, device=device, dtype=dtype) / points
    tests = [torch.ones_like(x)]
    for mode in range(1, modes + 1):
        phase = 2.0 * torch.pi * mode * x
        tests.extend((torch.sin(phase), torch.cos(phase)))
    for index in range(localized_centers):
        center = (index + 0.5) / max(localized_centers, 1)
        distance = torch.remainder(x - center + 0.5, 1.0) - 0.5
        tests.append(torch.exp(-0.5 * (distance / localized_width) ** 2))
    bank = torch.stack(tests)
    norm = torch.sqrt(torch.mean(bank.square(), dim=-1, keepdim=True)).clamp_min(
        torch.finfo(dtype).eps
    )
    return bank / norm


def _first(values: torch.Tensor, h: float | torch.Tensor) -> torch.Tensor:
    return (torch.roll(values, -1, dims=-1) - torch.roll(values, 1, dims=-1)) / (
        2.0 * torch.as_tensor(h, dtype=values.dtype, device=values.device)
    )


def _second(values: torch.Tensor, h: float | torch.Tensor) -> torch.Tensor:
    spacing = torch.as_tensor(h, dtype=values.dtype, device=values.device)
    return (
        torch.roll(values, -1, dims=-1)
        - 2.0 * values
        + torch.roll(values, 1, dims=-1)
    ) / spacing.square()


def discrete_h1_squared(values: torch.Tensor, h: float | torch.Tensor) -> torch.Tensor:
    """Batchwise periodic discrete H1 squared norm."""
    spacing = torch.as_tensor(h, dtype=values.dtype, device=values.device)
    return spacing * torch.sum(values * (values - _second(values, spacing)), dim=-1)


def mch_space_time_weak_residual(
    previous: torch.Tensor,
    future: torch.Tensor,
    tests: torch.Tensor,
    *,
    h: float | torch.Tensor,
    dt: float | torch.Tensor,
    alpha: float | torch.Tensor = 1.0,
    gamma: float | torch.Tensor = 0.0,
) -> torch.Tensor:
    r"""Differentiable midpoint weak residual for the smooth Eulerian sector.

    It evaluates
    ``<(m1-m0)/dt,phi> - <alpha*(u^2-u_x^2)*m,phi_x>
    + <gamma*u_x,phi>``.  The loss is not used to redefine the frozen peakon
    product: particle states remain governed by the M0 weak-form contract.
    """
    if previous.shape != future.shape or previous.ndim != 2:
        raise ValueError("previous and future must have equal [batch, points] shapes.")
    if tests.ndim not in (2, 3) or tests.shape[-1] != previous.shape[-1]:
        raise ValueError("tests must have shape [tests, points] or [batch, tests, points].")
    spacing = torch.as_tensor(h, dtype=previous.dtype, device=previous.device)
    time_step = torch.as_tensor(dt, dtype=previous.dtype, device=previous.device)
    if torch.any(spacing <= 0.0) or torch.any(time_step <= 0.0):
        raise ValueError("h and dt must be positive.")
    midpoint = 0.5 * (previous + future)
    momentum_previous = previous - _second(previous, spacing)
    momentum_future = future - _second(future, spacing)
    momentum_midpoint = midpoint - _second(midpoint, spacing)
    ux = _first(midpoint, spacing)
    alpha_value = torch.as_tensor(alpha, dtype=previous.dtype, device=previous.device)
    gamma_value = torch.as_tensor(gamma, dtype=previous.dtype, device=previous.device)
    if alpha_value.ndim == 1:
        alpha_value = alpha_value[:, None]
    if gamma_value.ndim == 1:
        gamma_value = gamma_value[:, None]
    flux = alpha_value * (
        midpoint.square() - ux.square()
    ) * momentum_midpoint
    gamma_ux = gamma_value * ux
    bank = tests.unsqueeze(0) if tests.ndim == 2 else tests
    phi_x = _first(bank, spacing)
    momentum_rate = (momentum_future - momentum_previous) / time_step
    residual_density = (
        momentum_rate[:, None, :] * bank
        - flux[:, None, :] * phi_x
        + gamma_ux[:, None, :] * bank
    )
    return spacing * torch.sum(residual_density, dim=-1)


def mch_hybrid_measure_weak_residual(
    field_momentum_previous: torch.Tensor,
    field_momentum_future: torch.Tensor,
    field_flux_midpoint: torch.Tensor,
    tests: torch.Tensor,
    test_derivatives: torch.Tensor,
    atom_amplitudes_previous: torch.Tensor,
    atom_amplitudes_future: torch.Tensor,
    atom_tests_previous: torch.Tensor,
    atom_tests_future: torch.Tensor,
    atom_test_derivatives_midpoint: torch.Tensor,
    atom_flux_weights_midpoint: torch.Tensor,
    atom_mask: torch.Tensor,
    *,
    h: float | torch.Tensor,
    dt: float | torch.Tensor,
) -> torch.Tensor:
    r"""Weak residual for a disjoint continuous-plus-atomic momentum measure.

    The caller evaluates each smooth test function at particle locations.  No
    rasterization or pointwise derivative of a Dirac mass is performed.
    """
    if field_momentum_previous.shape != field_momentum_future.shape:
        raise ValueError("field momentum states must have equal shapes.")
    if field_flux_midpoint.shape != field_momentum_previous.shape:
        raise ValueError("field flux must match field momentum.")
    if tests.shape != test_derivatives.shape or tests.ndim != 2:
        raise ValueError("tests and derivatives must have shape [tests, points].")
    batch, points = field_momentum_previous.shape
    if tests.shape[-1] != points:
        raise ValueError("test grid does not match the field grid.")
    atoms = atom_amplitudes_previous.shape[-1]
    expected_atom_tests = (batch, tests.shape[0], atoms)
    if (
        atom_amplitudes_future.shape != (batch, atoms)
        or atom_mask.shape != (batch, atoms)
        or atom_flux_weights_midpoint.shape != (batch, atoms)
        or atom_tests_previous.shape != expected_atom_tests
        or atom_tests_future.shape != expected_atom_tests
        or atom_test_derivatives_midpoint.shape != expected_atom_tests
    ):
        raise ValueError("atomic weak-form tensors have inconsistent shapes.")
    spacing = torch.as_tensor(h, dtype=field_momentum_previous.dtype, device=field_momentum_previous.device)
    time_step = torch.as_tensor(dt, dtype=field_momentum_previous.dtype, device=field_momentum_previous.device)
    if torch.any(spacing <= 0.0) or torch.any(time_step <= 0.0):
        raise ValueError("h and dt must be positive.")
    field_rate = (field_momentum_future - field_momentum_previous) / time_step
    field_part = spacing * torch.sum(
        field_rate[:, None, :] * tests[None, :, :]
        - field_flux_midpoint[:, None, :] * test_derivatives[None, :, :],
        dim=-1,
    )
    active = atom_mask.to(field_momentum_previous.dtype)
    atomic_temporal = torch.sum(
        (
            atom_amplitudes_future[:, None, :] * atom_tests_future
            - atom_amplitudes_previous[:, None, :] * atom_tests_previous
        ) * active[:, None, :],
        dim=-1,
    ) / time_step
    atomic_flux = torch.sum(
        atom_flux_weights_midpoint[:, None, :]
        * atom_test_derivatives_midpoint
        * active[:, None, :],
        dim=-1,
    )
    return field_part + atomic_temporal - atomic_flux


@dataclass(frozen=True, slots=True)
class MCHPhysicsLoss:
    total: torch.Tensor
    data_h1: torch.Tensor
    weak: torch.Tensor
    mass: torch.Tensor
    energy: torch.Tensor


def mch_physics_informed_loss(
    previous: torch.Tensor,
    prediction: torch.Tensor,
    target: torch.Tensor,
    tests: torch.Tensor,
    *,
    h: float,
    dt: float,
    alpha: float = 1.0,
    gamma: float = 0.0,
    weak_weight: float = 1.0,
    conservation_weight: float = 1.0,
) -> MCHPhysicsLoss:
    """Joint H1-data, weak-PDE, mass and H1-invariant training objective."""
    if prediction.shape != target.shape:
        raise ValueError("prediction and target shapes must agree.")
    scale = discrete_h1_squared(target, h).mean().clamp_min(torch.finfo(target.dtype).eps)
    data = discrete_h1_squared(prediction - target, h).mean() / scale
    residual = mch_space_time_weak_residual(
        previous, prediction, tests, h=h, dt=dt, alpha=alpha, gamma=gamma
    )
    weak = residual.square().mean() / scale
    mass_previous = h * previous.sum(dim=-1)
    mass_prediction = h * prediction.sum(dim=-1)
    mass = (mass_prediction - mass_previous).square().mean() / scale
    energy_previous = 0.5 * discrete_h1_squared(previous, h)
    energy_prediction = 0.5 * discrete_h1_squared(prediction, h)
    energy = (energy_prediction - energy_previous).square().mean() / scale.square()
    total = data + weak_weight * weak + conservation_weight * (mass + energy)
    return MCHPhysicsLoss(total, data, weak, mass, energy)
