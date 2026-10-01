from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import torch

from fgsp_ch.discretization.grids import PeriodicGrid
from fgsp_ch.discretization.helmholtz import HelmholtzOperator
from fgsp_ch.models.mch_atlas_network import MinimalMCHAtlasNetwork
from fgsp_ch.models.mch_pinn_cnn_operator import compose_mch_physical_step
from fgsp_ch.representations.peakon_tangent_basis import build_peakon_tangent_basis
from fgsp_ch.solvers.multipeakon import periodic_peakon_kernel_derivative
from fgsp_ch.training.mch_weak_physics import mch_physics_informed_loss, periodic_test_bank
from fgsp_ch.training.mch_weak_physics import discrete_h1_squared


@dataclass(frozen=True, slots=True)
class MCHJointLoss:
    total: torch.Tensor
    supervised: torch.Tensor
    data_h1: torch.Tensor
    weak: torch.Tensor
    conservation: torch.Tensor


@dataclass(frozen=True, slots=True)
class MCHJointTrainingMetrics:
    final_loss: float
    examples_per_second: float
    peak_device_memory_bytes: int
    optimizer_steps: int
    selected_epoch: int | None = None
    selected_validation_loss: float | None = None
    selection_evaluations: int = 0


def resolve_training_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA training was requested but CUDA is unavailable.")
    return device


def _tensor(value: Any, device: torch.device, dtype: torch.dtype | None = None) -> torch.Tensor:
    return torch.as_tensor(value, device=device, dtype=dtype)


def prepare_mch_joint_batch(
    samples: list[dict[str, Any]],
    normalization: dict[str, Any],
    *,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Create one same-grid GPU batch with causal physical reconstruction data."""
    if not samples:
        raise ValueError("Cannot prepare an empty joint-training batch.")
    points = samples[0]["features"].shape[-1]
    if any(sample["features"].shape[-1] != points for sample in samples):
        raise ValueError("Joint batches must be bucketed by grid size.")
    maximum_coordinates = samples[0]["atom_mask"].size
    fields, parameters, bases, transforms, targets = [], [], [], [], []
    measure_scales, purities = [], []
    for sample in samples:
        raw_features = np.asarray(sample["features"], dtype=np.float64)
        fields.append(
            (raw_features - np.asarray(normalization["field_mean"])[:, None])
            / np.asarray(normalization["field_std"])[:, None]
        )
        parameters.append(
            (np.asarray(sample["parameters"]) - np.asarray(normalization["parameter_mean"]))
            / np.asarray(normalization["parameter_std"])
        )
        length = float(sample["parameters"][2])
        grid = PeriodicGrid(points=points, length=length)
        helmholtz = HelmholtzOperator(grid)
        basis = np.zeros((points, maximum_coordinates), dtype=np.float64)
        transform = np.zeros((maximum_coordinates, maximum_coordinates), dtype=np.float64)
        measure_scale = 1.0
        if int(sample["atlas"]) == 1:
            count = int(np.count_nonzero(sample["atom_mask"]))
            if count:
                tangent = build_peakon_tangent_basis(
                    grid,
                    sample["atom_positions"][:count],
                    sample["atom_amplitudes"][:count],
                )[:, count:]
                basis[:, :count] = tangent
                transform[:count, :count] = np.eye(count)
                measure_scale = length
        elif int(sample["atlas"]) == 2 and bool(sample["cluster_position_identifiable"]):
            count = int(np.count_nonzero(sample["cluster_mask"]))
            if count:
                derivative = periodic_peakon_kernel_derivative(
                    grid.x[:, None] - sample["cluster_centers"][None, :count], length
                )
                basis[:, :count] = -derivative * sample["cluster_moments"][None, :count, 0]
                transform[:count, :count] = sample["cluster_inverse_sqrt"][:count, :count]
                measure_scale = float(sample["state_scale"])
        bases.append(basis)
        transforms.append(transform)
        measure_scales.append(measure_scale)
        current = raw_features[1]
        baseline = current + raw_features[5]
        target = baseline + helmholtz.solve(np.asarray(sample["momentum_defect"], dtype=np.float64))
        targets.append(target)
        purities.append(
            np.sqrt(np.mean(raw_features[7] ** 2))
            / max(np.sqrt(np.mean(current**2)), np.finfo(float).eps)
        )
    stack = lambda name: np.stack([sample[name] for sample in samples])
    raw = np.stack([sample["features"] for sample in samples]).astype(np.float64)
    return {
        "features": _tensor(np.stack(fields), device, torch.float32),
        "parameters": _tensor(np.stack(parameters), device, torch.float32),
        "raw_previous": _tensor(raw[:, 0], device, torch.float64),
        "raw_current": _tensor(raw[:, 1], device, torch.float64),
        "baseline": _tensor(raw[:, 1] + raw[:, 5], device, torch.float64),
        "target": _tensor(np.stack(targets), device, torch.float64),
        "length": _tensor([sample["parameters"][2] for sample in samples], device, torch.float32),
        "h": _tensor([sample["parameters"][3] for sample in samples], device, torch.float64),
        "dt": _tensor([sample["parameters"][4] for sample in samples], device, torch.float64),
        "state_scale": _tensor([sample["state_scale"] for sample in samples], device, torch.float64),
        "measure_scale": _tensor(measure_scales, device, torch.float64),
        "measure_basis": _tensor(np.stack(bases), device, torch.float64),
        "measure_transform": _tensor(np.stack(transforms), device, torch.float64),
        "measure_purity": _tensor(purities, device, torch.float32),
        "atom_positions": _tensor(stack("atom_positions"), device, torch.float32),
        "atom_amplitudes": _tensor(stack("atom_amplitudes"), device, torch.float32),
        "atom_mask": _tensor(stack("atom_mask"), device, torch.bool),
        "cluster_centers": _tensor(stack("cluster_centers"), device, torch.float32),
        "cluster_signs": _tensor(stack("cluster_signs"), device, torch.float32),
        "cluster_moments": _tensor(stack("cluster_moments"), device, torch.float32),
        "cluster_mask": _tensor(stack("cluster_mask"), device, torch.bool),
        "atlas": _tensor([sample["atlas"] for sample in samples], device, torch.int64),
        "measure_mask": _tensor(
            np.stack([
                sample["atom_mask"] if int(sample["atlas"]) == 1
                else sample["cluster_mask"]
                if int(sample["atlas"]) == 2 and bool(sample["cluster_position_identifiable"])
                else np.zeros_like(sample["atom_mask"], dtype=bool)
                for sample in samples
            ]),
            device,
            torch.bool,
        ),
        "position_target": _tensor(stack("position_target"), device, torch.float32),
        "cluster_whitened_target": _tensor(stack("cluster_whitened_target"), device, torch.float32),
        "field_target": _tensor(stack("field_target"), device, torch.float32),
    }


def mch_joint_forward(
    model: MinimalMCHAtlasNetwork,
    batch: dict[str, torch.Tensor],
    normalization: dict[str, Any],
    *,
    use_amp: bool = False,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Network forward in fp32/AMP, physical composition in fp64."""
    amp = (
        torch.autocast(device_type="cuda", dtype=torch.float16)
        if use_amp and batch["features"].device.type == "cuda"
        else nullcontext()
    )
    with amp:
        selected = batch["features"][:, (0, 1, 5, 7), :]
        context = torch.cat(
            (torch.mean(selected, dim=-1), torch.sqrt(torch.mean(selected**2, dim=-1) + 1e-12)),
            dim=1,
        )
        graph_parameters = torch.cat((batch["parameters"], context), dim=1)
        position, _ = model.particle(
            batch["atom_positions"], batch["atom_amplitudes"], batch["atom_mask"],
            graph_parameters, batch["length"],
        )
        cluster, _ = model.cluster(
            batch["cluster_centers"], batch["cluster_signs"], batch["cluster_moments"],
            batch["cluster_mask"], graph_parameters, batch["length"],
        )
        field = model.field(batch["features"], batch["parameters"])
    pure = (batch["measure_purity"] <= 1e-6).to(position.dtype)[:, None]
    position = position * pure * float(normalization["position_scale"])
    cluster = cluster * float(normalization["cluster_whitened_scale"])
    field = field * float(normalization["field_target_scale"])
    atlas = batch["atlas"]
    coordinates = torch.where(
        (atlas == 1)[:, None], position,
        torch.where((atlas == 2)[:, None], cluster, torch.zeros_like(position)),
    ).to(torch.float64)
    physical_coordinates = torch.bmm(
        batch["measure_transform"], coordinates.unsqueeze(-1)
    ).squeeze(-1)
    # Same-grid buckets have a common h by construction.
    h = float(batch["h"][0].item())
    candidate = compose_mch_physical_step(
        batch["baseline"], field.to(torch.float64), batch["measure_basis"],
        physical_coordinates, state_scale=batch["state_scale"],
        measure_scale=batch["measure_scale"], dt=batch["dt"], h=h,
        measure_mask=batch["measure_mask"],
    )
    return candidate, {"position": position, "cluster": cluster, "field": field}


def mch_joint_physics_loss(
    model: MinimalMCHAtlasNetwork,
    batch: dict[str, torch.Tensor],
    normalization: dict[str, Any],
    *,
    weak_weight: float,
    conservation_weight: float,
    supervised_weight: float,
    use_amp: bool = False,
) -> MCHJointLoss:
    candidate, heads = mch_joint_forward(model, batch, normalization, use_amp=use_amp)
    h_values = batch["h"]
    if not torch.allclose(h_values, h_values[:1]):
        raise ValueError("Physics batches must share one grid spacing.")
    h = float(h_values[0].item())
    dt_values = batch["dt"]
    if not torch.allclose(dt_values, dt_values[:1]):
        raise ValueError("Physics batches must also be bucketed by dt.")
    dt = float(dt_values[0].item())
    # Parameters are raw in M3 parameter slots before normalization; recover
    # alpha/gamma from train-only affine statistics without label leakage.
    mean = _tensor(normalization["parameter_mean"], candidate.device, torch.float64)
    std = _tensor(normalization["parameter_std"], candidate.device, torch.float64)
    raw_parameters = batch["parameters"].to(torch.float64) * std + mean
    tests = periodic_test_bank(
        candidate.shape[-1], modes=4, localized_centers=4,
        device=candidate.device, dtype=torch.float64,
    )
    physics = mch_physics_informed_loss(
        batch["raw_current"], candidate, batch["target"], tests,
        h=h, dt=dt, alpha=raw_parameters[:, 0], gamma=raw_parameters[:, 1],
        weak_weight=weak_weight, conservation_weight=conservation_weight,
    )
    supervised = torch.mean(
        ((heads["field"] - batch["field_target"]) / float(normalization["field_target_scale"])) ** 2
    )
    separated = batch["atom_mask"] & (batch["atlas"] == 1)[:, None]
    if torch.any(separated):
        supervised = supervised + torch.mean(
            ((heads["position"][separated] - batch["position_target"][separated])
             / float(normalization["position_scale"])) ** 2
        )
    clustered = batch["cluster_mask"] & (batch["atlas"] == 2)[:, None]
    if torch.any(clustered):
        supervised = supervised + torch.mean(
            ((heads["cluster"][clustered] - batch["cluster_whitened_target"][clustered])
             / float(normalization["cluster_whitened_scale"])) ** 2
        )
    total = supervised_weight * supervised + physics.total
    return MCHJointLoss(
        total, supervised, physics.data_h1, physics.weak, physics.mass + physics.energy
    )


def mch_multistep_consistency_loss(
    candidates: torch.Tensor,
    current_states: torch.Tensor,
    targets: torch.Tensor,
    *,
    h: float,
    one_step_weight: float = 0.25,
) -> torch.Tensor:
    """Cumulative 2--K step H1 loss for ordered model-visited windows."""
    if candidates.shape != current_states.shape or candidates.shape != targets.shape:
        raise ValueError("Multistep states must have equal [batch, horizon, points] shapes.")
    if candidates.ndim != 3 or candidates.shape[1] < 2:
        raise ValueError("Multistep loss requires horizon >= 2.")
    predicted_increment = candidates - current_states
    target_increment = targets - current_states
    cumulative_error = torch.cumsum(predicted_increment - target_increment, dim=1)
    flat = cumulative_error.reshape(-1, cumulative_error.shape[-1])
    cumulative = torch.mean(discrete_h1_squared(flat, h))
    anchor = torch.mean(
        discrete_h1_squared(
            (predicted_increment - target_increment).reshape(-1, flat.shape[-1]), h
        )
    )
    scale = torch.mean(
        discrete_h1_squared(target_increment.reshape(-1, flat.shape[-1]), h)
    ).clamp_min(torch.finfo(targets.dtype).eps)
    return (cumulative + one_step_weight * anchor) / scale


def prepare_mch_rollout_windows(
    samples: list[dict[str, Any]],
    normalization: dict[str, Any],
    *,
    device: torch.device,
    horizons: tuple[int, ...] = (2, 3),
    batch_size: int = 1,
) -> list[list[dict[str, torch.Tensor]]]:
    """Build leakage-safe ordered windows within one trajectory and grid."""
    if not horizons or min(horizons) < 2 or batch_size < 1:
        raise ValueError("Rollout horizons must all be at least two.")
    groups: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for sample in samples:
        groups.setdefault(
            (str(sample["group"]), int(sample["features"].shape[-1])), []
        ).append(sample)
    raw_windows: list[list[dict[str, Any]]] = []
    for sequence in groups.values():
        sequence.sort(key=lambda item: int(item["current_index"]))
        for horizon in horizons:
            for start in range(len(sequence) - horizon + 1):
                selected = sequence[start : start + horizon]
                if all(
                    int(selected[index + 1]["current_index"])
                    == int(selected[index]["current_index"]) + 1
                    for index in range(horizon - 1)
                ):
                    raw_windows.append(selected)
    grouped: dict[tuple[Any, ...], list[list[dict[str, Any]]]] = {}
    for window in raw_windows:
        signature = (
            len(window),
            int(window[0]["features"].shape[-1]),
            tuple(round(float(sample["parameters"][3]), 12) for sample in window),
            tuple(round(float(sample["parameters"][4]), 12) for sample in window),
        )
        grouped.setdefault(signature, []).append(window)
    windows: list[list[dict[str, torch.Tensor]]] = []
    for same_shape in grouped.values():
        for start in range(0, len(same_shape), batch_size):
            chunk = same_shape[start : start + batch_size]
            windows.append([
                prepare_mch_joint_batch(
                    [window[time_index] for window in chunk],
                    normalization,
                    device=device,
                )
                for time_index in range(len(chunk[0]))
            ])
    return windows


def train_mch_joint_model(
    model: MinimalMCHAtlasNetwork,
    batches: list[dict[str, torch.Tensor]],
    normalization: dict[str, Any],
    *,
    epochs: int,
    learning_rate: float,
    weak_weight: float,
    conservation_weight: float,
    supervised_weight: float,
    gradient_clip: float = 1.0,
    use_amp: bool = False,
    rollout_windows: list[list[dict[str, torch.Tensor]]] | None = None,
    multistep_weight: float = 0.0,
    selection_objective: Callable[[MinimalMCHAtlasNetwork], float] | None = None,
    selection_interval: int = 1,
    initial_selection_score: float | None = None,
) -> MCHJointTrainingMetrics:
    """Device-resident M6.2 joint fine-tuning with safe checkpoint selection."""
    if not batches or epochs < 1 or learning_rate <= 0.0 or gradient_clip <= 0.0:
        raise ValueError("Joint training settings and batches must be positive.")
    if selection_interval < 1:
        raise ValueError("selection_interval must be positive.")
    device = next(model.parameters()).device
    if any(batch["features"].device != device for batch in batches):
        raise ValueError("Model and every joint batch must share one device.")
    amp_enabled = bool(use_amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    import time
    started = time.perf_counter()
    seen = 0
    steps = 0
    final = float("nan")
    best_state: dict[str, torch.Tensor] | None = None
    best_score: float | None = None
    best_epoch: int | None = None
    selection_evaluations = 0
    if selection_objective is not None:
        score = (
            float(initial_selection_score)
            if initial_selection_score is not None
            else float(selection_objective(model))
        )
        if not np.isfinite(score):
            raise FloatingPointError("Initial validation objective is not finite.")
        best_score = score
        best_epoch = 0
        best_state = {
            name: value.detach().cpu().clone()
            for name, value in model.state_dict().items()
        }
        selection_evaluations = 1
    model.train()
    for epoch_index in range(epochs):
        for batch in batches:
            optimizer.zero_grad(set_to_none=True)
            loss = mch_joint_physics_loss(
                model, batch, normalization,
                weak_weight=weak_weight,
                conservation_weight=conservation_weight,
                supervised_weight=supervised_weight,
                use_amp=amp_enabled,
            )
            scaler.scale(loss.total).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            scaler.step(optimizer)
            scaler.update()
            final = float(loss.total.detach().cpu())
            seen += int(batch["features"].shape[0])
            steps += 1
        if rollout_windows and multistep_weight > 0.0:
            for window in rollout_windows:
                optimizer.zero_grad(set_to_none=True)
                candidates = []
                currents = []
                targets = []
                for batch in window:
                    candidate, _ = mch_joint_forward(
                        model, batch, normalization, use_amp=amp_enabled
                    )
                    candidates.append(candidate)
                    currents.append(batch["raw_current"])
                    targets.append(batch["target"])
                multistep = mch_multistep_consistency_loss(
                    torch.stack(candidates, dim=1),
                    torch.stack(currents, dim=1),
                    torch.stack(targets, dim=1),
                    h=float(window[0]["h"][0].item()),
                )
                scaler.scale(multistep_weight * multistep).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                scaler.step(optimizer)
                scaler.update()
                final = float(multistep.detach().cpu())
                seen += len(window) * int(window[0]["features"].shape[0])
                steps += 1
        if selection_objective is not None and (
            (epoch_index + 1) % selection_interval == 0
            or epoch_index + 1 == epochs
        ):
            model.eval()
            score = float(selection_objective(model))
            selection_evaluations += 1
            if np.isfinite(score) and (best_score is None or score < best_score):
                best_score = score
                best_epoch = epoch_index + 1
                best_state = {
                    name: value.detach().cpu().clone()
                    for name, value in model.state_dict().items()
                }
            model.train()
    if best_state is not None:
        model.load_state_dict(best_state)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak = int(torch.cuda.max_memory_allocated(device))
    else:
        peak = 0
    elapsed = max(time.perf_counter() - started, 1e-12)
    model.eval()
    return MCHJointTrainingMetrics(
        final,
        seen / elapsed,
        peak,
        steps,
        selected_epoch=best_epoch,
        selected_validation_loss=best_score,
        selection_evaluations=selection_evaluations,
    )
