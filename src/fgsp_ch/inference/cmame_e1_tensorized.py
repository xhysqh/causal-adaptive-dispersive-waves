"""Tensorized, device-resident inference for the frozen CMAME P1--P3 stack.

The numerical AMR backbone remains authoritative.  This module changes only
how frozen neural fluxes and controller heads are evaluated: features cross
the host/device boundary once, ensemble members are vectorized, and the P1
models are placed on the requested device instead of being silently kept on
the CPU.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from fgsp_ch.datasets.cmame_p21_mixed_flux import interaction_features
from fgsp_ch.discretization.difference import first_derivative
from fgsp_ch.discretization.grids import PeriodicGrid
from fgsp_ch.equations.modified_ch import ModifiedCHParameters
from fgsp_ch.models.cmame_conservative_flux_operator import MCHConservativeFluxOperator
from fgsp_ch.models.cmame_p22_interface_flux import MCHConservativeInterfaceFluxOperator
from fgsp_ch.models.cmame_p3_adaptive_controller import MCHSpaceTimeAdaptiveController
from fgsp_ch.representations.cmame_p2_hybrid import CMAMEHybridState
from fgsp_ch.solvers.cmame_p3_adaptive import bounded_error_indicator, deterministic_adaptive_decision
from fgsp_ch.solvers.mch_hybrid_amr import HybridMCHState, hybrid_particle_velocity, hybrid_total_state
from fgsp_ch.utils.config import load_yaml


@dataclass(frozen=True, slots=True)
class E1Evaluation:
    features: np.ndarray
    parameters: np.ndarray
    flux_scale: float
    p1_rows: torch.Tensor
    p22_rows: torch.Tensor

    @property
    def correction_tensor(self) -> torch.Tensor:
        return self.p1_rows.mean(0) + self.flux_scale * self.p22_rows.mean(0)


class _BatchedEnsemble:
    """Evaluate every frozen member on one shared physical-state batch.

    The three members deliberately remain distinct modules.  A parameter-axis
    ``vmap`` changes GroupNorm's reduction semantics in current PyTorch
    releases, whereas stacking the member outputs is bitwise-equivalent to the
    published P1--P3 inference path.  The expensive state axis is still fully
    tensorized and crosses the PCIe boundary only once.
    """

    def __init__(self, models: Sequence[torch.nn.Module], output_fields: tuple[str, ...]) -> None:
        if not models:
            raise ValueError("an E1 ensemble cannot be empty")
        self.models = torch.nn.ModuleList(models)
        self.output_fields = output_fields

    def __call__(self, features: torch.Tensor, parameters: torch.Tensor) -> tuple[torch.Tensor, ...]:
        outputs = [model(features, parameters) for model in self.models]
        return tuple(
            torch.stack([getattr(output, name) for output in outputs], dim=0)
            for name in self.output_fields
        )


class TensorizedConservativeAdaptiveEngine:
    """Frozen P1/P2.2/P3 inference engine with one device transfer per state."""

    def __init__(self, root: Path, p3_config: dict, device: torch.device | str) -> None:
        self.root = Path(root)
        self.device = torch.device(device)
        self.p3_config = p3_config
        p22_config = load_yaml(self.root / p3_config["p22_config"])
        p1_config = load_yaml(self.root / p22_config["p1_config"])

        p1_models = []
        for seed in map(int, p1_config["training"]["seeds"]):
            model = MCHConservativeFluxOperator(
                hidden=int(p1_config["training"]["hidden"])
            ).to(self.device)
            checkpoint = torch.load(
                self.root / p22_config["p1_output_dir"] / f"seed_{seed}.pt",
                map_location=self.device, weights_only=False,
            )
            model.load_state_dict(checkpoint["model"]); model.eval(); p1_models.append(model)

        model_cfg = p22_config["model"]
        p22_models = []
        for seed in map(int, p22_config["training"]["seeds"]):
            model = MCHConservativeInterfaceFluxOperator(
                hidden=int(model_cfg["hidden"]),
                dilations=tuple(map(int, model_cfg["dilations"])),
                cluster_threshold=float(model_cfg["cluster_threshold"]),
                cluster_temperature=float(model_cfg["cluster_temperature"]),
            ).to(self.device)
            checkpoint = torch.load(
                self.root / p22_config["output_dir"] / f"seed_{seed}.pt",
                map_location=self.device, weights_only=False,
            )
            model.load_state_dict(checkpoint["model"]); model.eval(); p22_models.append(model)

        controller_models = []
        for seed in map(int, p3_config["training"]["seeds"]):
            model = MCHSpaceTimeAdaptiveController(
                hidden=int(p3_config["model"]["hidden"])
            ).to(self.device)
            checkpoint = torch.load(
                self.root / p3_config["output_dir"] / f"controller_seed_{seed}.pt",
                map_location=self.device, weights_only=False,
            )
            model.load_state_dict(checkpoint["model"]); model.eval(); controller_models.append(model)

        self.p1 = _BatchedEnsemble(p1_models, ("face_correction",))
        self.p22 = _BatchedEnsemble(p22_models, ("normalized_face_flux",))
        self.controllers = _BatchedEnsemble(
            controller_models, ("action_logits", "log_local_error", "time_multiplier")
        )
        self.host_to_device_transfers = 0
        self.device_to_host_transfers = 0
        self.state_evaluations = 0
        self.cache_hits = 0
        self._primed_key: tuple | None = None
        self._primed_evaluation: E1Evaluation | None = None
        self._last_batch_evaluations: list[E1Evaluation] = []

    def reset_counters(self) -> None:
        self.host_to_device_transfers = self.device_to_host_transfers = self.state_evaluations = 0
        self.cache_hits = 0

    @staticmethod
    def _state_key(
        state: HybridMCHState, grid: PeriodicGrid,
        parameters: ModifiedCHParameters, dt: float,
    ) -> tuple:
        """Cheap identity plus content guard for the decision/flux hand-off."""
        return (
            grid.points, float(grid.length), float(parameters.alpha),
            float(parameters.gamma), float(dt),
            np.asarray(state.positions).tobytes(),
            np.asarray(state.amplitudes).tobytes(),
            np.asarray(state.field_momentum).tobytes(),
        )

    def _prime(
        self, state: HybridMCHState, grid: PeriodicGrid,
        parameters: ModifiedCHParameters, dt: float, evaluation: E1Evaluation,
    ) -> None:
        self._primed_key = self._state_key(state, grid, parameters, dt)
        self._primed_evaluation = evaluation

    def _consume_primed(
        self, state: HybridMCHState, grid: PeriodicGrid,
        parameters: ModifiedCHParameters, dt: float,
    ) -> E1Evaluation | None:
        key = self._state_key(state, grid, parameters, dt)
        if key != self._primed_key:
            return None
        result = self._primed_evaluation
        self._primed_key = None
        self._primed_evaluation = None
        self.cache_hits += int(result is not None)
        return result

    @torch.inference_mode()
    def prime_flux(
        self, state: HybridMCHState, grid: PeriodicGrid,
        parameters: ModifiedCHParameters, dt: float,
    ) -> None:
        """Prepare the next authoritative coarse-grid flux evaluation.

        This is useful when the controller changes ``dt``: the decision was
        evaluated at the old step size, while the conservative update must use
        features at the accepted step size.
        """
        key = self._state_key(state, grid, parameters, dt)
        if key == self._primed_key:
            return
        evaluation = self.evaluate_batch([state], grid, [parameters], [dt])[0]
        self._prime(state, grid, parameters, dt, evaluation)

    @torch.inference_mode()
    def evaluate_batch(
        self, states: Sequence[HybridMCHState], grid: PeriodicGrid,
        parameters: Sequence[ModifiedCHParameters], dts: Sequence[float],
    ) -> list[E1Evaluation]:
        if not states or not (len(states) == len(parameters) == len(dts)):
            raise ValueError("states, parameters and dts must have equal non-zero lengths")
        feature_rows=[]; parameter_rows=[]; scales=[]
        for state, equation_parameters, dt in zip(states, parameters, dts, strict=True):
            mixed = CMAMEHybridState(
                np.asarray(state.field_momentum), np.asarray(state.positions), np.asarray(state.amplitudes)
            )
            features, global_parameters, _, scale = interaction_features(
                mixed, grid, equation_parameters, float(dt)
            )
            feature_rows.append(features); parameter_rows.append(global_parameters); scales.append(scale)
        x = torch.as_tensor(np.stack(feature_rows), dtype=torch.float32, device=self.device)
        p = torch.as_tensor(np.stack(parameter_rows), dtype=torch.float32, device=self.device)
        self.host_to_device_transfers += 1; self.state_evaluations += len(states)
        p1_rows = self.p1(x[:, :10], p[:, :6])[0]
        # Preserve the frozen P1 structural reduction exactly: a zero regular
        # field has no learned smooth-flux correction.
        p1_active = torch.as_tensor(
            [np.linalg.norm(state.field_momentum) > 100.0 * np.finfo(float).eps for state in states],
            dtype=p1_rows.dtype, device=self.device,
        )
        p1_rows = p1_rows * p1_active[None, :, None]
        p22_rows = self.p22(x, p)[0]
        return [
            E1Evaluation(feature_rows[i], parameter_rows[i], float(scales[i]),
                         p1_rows[:, i], p22_rows[:, i])
            for i in range(len(states))
        ]

    @torch.inference_mode()
    def __call__(
        self, state: HybridMCHState, grid: PeriodicGrid,
        parameters: ModifiedCHParameters, dt: float,
    ) -> np.ndarray:
        evaluation = self._consume_primed(state, grid, parameters, dt)
        if evaluation is None:
            evaluation = self.evaluate_batch([state], grid, [parameters], [dt])[0]
        result = evaluation.correction_tensor
        result = result - result.mean()
        self.device_to_host_transfers += 1
        return result.cpu().numpy().astype(np.float64)

    @torch.inference_mode()
    def decide_batch(
        self, states: Sequence[HybridMCHState], grid: PeriodicGrid,
        parameters: Sequence[ModifiedCHParameters], dts: Sequence[float],
    ) -> list:
        """Evaluate all frozen controllers in one physical-state batch."""
        evaluations = self.evaluate_batch(states, grid, parameters, dts)
        self._last_batch_evaluations = evaluations
        features = torch.as_tensor(
            np.stack([row.features for row in evaluations]),
            dtype=torch.float32, device=self.device,
        )
        global_parameters = torch.as_tensor(
            np.stack([row.parameters for row in evaluations]),
            dtype=torch.float32, device=self.device,
        )
        rows = torch.stack([row.p22_rows for row in evaluations], dim=1)
        mean = rows.mean(0); std = rows.std(0, correction=0)
        jump = torch.abs(mean - torch.roll(mean, 1, dims=-1))
        proxy = features[:, 5].abs() + features[:, 19] + std + jump / torch.sqrt(
            torch.mean(mean.square(), dim=-1, keepdim=True)
        ).clamp_min(1.0e-8)
        controller_features = torch.cat(
            (features, mean[:, None], std[:, None], jump[:, None], proxy[:, None]), dim=1
        )
        logits_rows, error_rows, multiplier_rows = self.controllers(
            controller_features, global_parameters
        )
        logits = logits_rows.mean(0).cpu().numpy()
        predicted_log_error = error_rows.mean(0).cpu().numpy()
        multipliers = multiplier_rows.mean(0).cpu().numpy()
        proxies = proxy.cpu().numpy()
        self.device_to_host_transfers += 1
        cfg = self.p3_config["adaptivity"]
        decisions=[]
        for index, (state, equation_parameters, dt, evaluation) in enumerate(
            zip(states, parameters, dts, evaluations, strict=True)
        ):
            indicator = bounded_error_indicator(predicted_log_error[index], proxies[index])
            total = hybrid_total_state(state, grid); slope = first_derivative(total, grid)
            speed = max(float(np.max(np.abs(equation_parameters.alpha*(total**2-slope**2)))),1.0e-8)
            velocity = hybrid_particle_velocity(state, grid, alpha=equation_parameters.alpha)
            relative_speed = float(np.ptp(velocity)) if velocity.size > 1 else max(
                float(np.max(np.abs(velocity))) if velocity.size else 0.0, 1.0e-8
            )
            decisions.append(deterministic_adaptive_decision(
                indicator, logits[index], evaluation.features[19], current_dt=float(dt),
                neural_multiplier=float(multipliers[index]), h=grid.h,
                characteristic_speed=speed,
                minimum_separation=float(evaluation.parameters[7]),
                relative_particle_speed=relative_speed, theta=float(cfg["dorfler_theta"]),
                maximum_fraction=float(cfg["maximum_refined_fraction"]),
                minimum_dt=float(self.p3_config["rollout"]["minimum_dt"]),
                maximum_dt=float(self.p3_config["rollout"]["maximum_dt"]),
            ))
        return decisions

    @torch.inference_mode()
    def decide(
        self, state: HybridMCHState, grid: PeriodicGrid,
        parameters: ModifiedCHParameters, dt: float,
    ):
        decision = self.decide_batch([state], grid, [parameters], [dt])[0]
        evaluation = self._last_batch_evaluations[0]
        # HybridMCHAMRSolver asks for the same coarse flux immediately after a
        # decision.  Reuse this evaluation once; fine-grid stages remain fresh.
        self._prime(state, grid, parameters, dt, evaluation)
        return decision
