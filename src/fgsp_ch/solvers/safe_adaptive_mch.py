from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
import torch

from fgsp_ch.atlas.mch_atlas import (
    AtlasDecision,
    AtlasKind,
    MCHAtlasConfig,
    select_atlas,
)
from fgsp_ch.datasets.mch_atlas_dataset import causal_feature_channels
from fgsp_ch.equations.modified_ch import ModifiedCH
from fgsp_ch.geometry.metric import a_norm, discrete_energy
from fgsp_ch.models.mch_atlas_network import MinimalMCHAtlasNetwork
from fgsp_ch.representations.peakon_tangent_basis import build_peakon_tangent_basis
from fgsp_ch.representations.cluster_whitening import (
    build_cluster_whitening,
    reconstruct_cluster_position_defect,
)
from fgsp_ch.safety.mch_correction_gate import (
    MCHCorrectionCertificate,
    MCHCorrectionGateConfig,
    certify_mch_correction,
)
from fgsp_ch.solvers.mch_energy_projected import MassH1ProjectedMCHSolver


M5_VARIANTS = (
    "baseline_only",
    "ungated",
    "confidence_only",
    "conservation_only",
    "no_particle",
    "no_cluster",
    "no_field",
    "full",
)


@dataclass(frozen=True, slots=True)
class MCHCorrectionProposal:
    correction: NDArray[np.floating]
    atlas: AtlasKind
    confidence: float
    ood_distance: float
    measure_purity: float
    separation_cells: float
    field_norm: float
    measure_norm: float
    ensemble_h1_std: float
    ensemble_relative_uncertainty: float


class MCHNetworkCorrector:
    """Convert dimensionless M4 heads into a physical Eulerian step defect."""

    def __init__(
        self,
        model: MinimalMCHAtlasNetwork | list[MinimalMCHAtlasNetwork],
        normalization: dict[str, Any],
        *,
        maximum_atoms: int = 8,
        maximum_clusters: int = 8,
        selected_member: int | None = None,
        device: str | torch.device = "cpu",
    ) -> None:
        requested = torch.device(device)
        if requested.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        self.device = requested
        self.models = [
            item.to(self.device).eval()
            for item in (model if isinstance(model, list) else [model])
        ]
        self.model = self.models[0]
        self.stats = {
            key: np.asarray(value, dtype=np.float32)
            if isinstance(value, list) else value
            for key, value in normalization.items()
        }
        self.maximum_atoms = maximum_atoms
        self.maximum_clusters = maximum_clusters
        if selected_member is not None and not 0 <= selected_member < len(self.models):
            raise ValueError("Selected ensemble member index is out of range.")
        self.selected_member = selected_member

    @classmethod
    def from_artifacts(
        cls,
        checkpoint: Path,
        acceptance: Path,
        *,
        hidden: int,
        maximum_atoms: int = 8,
        maximum_clusters: int = 8,
        selected_member: int | None = None,
        device: str | torch.device = "cpu",
    ) -> "MCHNetworkCorrector":
        metadata = json.loads(acceptance.read_text(encoding="utf-8"))
        if metadata.get("status") != "PASS":
            raise RuntimeError("M5 requires a passing M4 artifact.")
        model = MinimalMCHAtlasNetwork(hidden=hidden, conserve_amplitude=True)
        model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
        return cls(
            model, metadata["normalization"], maximum_atoms=maximum_atoms,
            maximum_clusters=maximum_clusters, selected_member=selected_member,
            device=device,
        )

    @classmethod
    def from_ensemble_artifacts(
        cls,
        checkpoints: list[Path],
        acceptance: Path,
        *,
        hidden: int,
        maximum_atoms: int = 8,
        maximum_clusters: int = 8,
        selected_member: int | None = None,
        device: str | torch.device = "cpu",
    ) -> "MCHNetworkCorrector":
        if len(checkpoints) < 3:
            raise ValueError("M5.1 requires at least three independently trained models.")
        metadata = json.loads(acceptance.read_text(encoding="utf-8"))
        if metadata.get("status") != "PASS":
            raise RuntimeError("M5.1 requires a passing M4 artifact.")
        models = []
        for checkpoint in checkpoints:
            model = MinimalMCHAtlasNetwork(hidden=hidden, conserve_amplitude=True)
            model.load_state_dict(
                torch.load(checkpoint, map_location="cpu", weights_only=True)
            )
            models.append(model)
        return cls(
            models, metadata["normalization"], maximum_atoms=maximum_atoms,
            maximum_clusters=maximum_clusters, selected_member=selected_member,
            device=device,
        )

    @staticmethod
    def _pad(values: NDArray[np.floating], size: int, width: int | None = None) -> NDArray[np.floating]:
        if width is None:
            result = np.zeros(size, dtype=np.float32)
            count = min(size, values.size)
            result[:count] = np.asarray(values).ravel()[:count]
        else:
            result = np.zeros((size, width), dtype=np.float32)
            count = min(size, values.shape[0])
            result[:count] = np.asarray(values)[:count]
        return result

    def propose(
        self,
        equation: ModifiedCH,
        previous: NDArray[np.floating],
        current: NDArray[np.floating],
        baseline: NDArray[np.floating],
        decision: AtlasDecision,
        dt: float,
        *,
        variant: str = "full",
    ) -> MCHCorrectionProposal:
        if variant not in M5_VARIANTS:
            raise ValueError(f"Unknown M5 variant: {variant}.")
        features = causal_feature_channels(
            previous, current, baseline, decision, equation.helmholtz
        ).astype(np.float32)
        raw_parameters = np.asarray(
            [
                equation.parameters.alpha,
                equation.parameters.gamma,
                equation.grid.length,
                equation.grid.h,
                dt,
                equation.grid.points,
                decision.decomposition.detection_confidence,
                min(decision.identifiability.separation_cells, 1.0e6),
            ],
            dtype=np.float32,
        )
        parameters = (
            raw_parameters - self.stats["parameter_mean"]
        ) / self.stats["parameter_std"]
        normalized_fields = (
            features - self.stats["field_mean"][:, None]
        ) / self.stats["field_std"][:, None]
        selected = normalized_fields[(0, 1, 5, 7), :]
        context = np.concatenate(
            (np.mean(selected, axis=1), np.sqrt(np.mean(selected**2, axis=1) + 1e-12))
        ).astype(np.float32)
        graph_parameters = np.concatenate((parameters, context)).astype(np.float32)

        atoms = min(decision.decomposition.positions.size, self.maximum_atoms)
        clusters = min(decision.clusters.centers.size, self.maximum_clusters)
        atom_mask = np.arange(self.maximum_atoms) < atoms
        cluster_mask = np.arange(self.maximum_clusters) < clusters
        moments = (
            np.stack(
                (
                    decision.clusters.zeroth,
                    decision.clusters.first,
                    decision.clusters.second,
                ),
                axis=1,
            )
            if decision.clusters.centers.size else np.empty((0, 3))
        )
        tensor = lambda value: torch.as_tensor(
            value, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        field_enabled = variant not in ("baseline_only", "no_field")
        particle_enabled = variant not in ("baseline_only", "no_particle")
        cluster_enabled = variant not in ("baseline_only", "no_cluster")
        state_scale = max(a_norm(current, equation.helmholtz), np.finfo(float).eps)
        whitening = build_cluster_whitening(decision, equation.helmholtz)
        measure_basis = np.empty((equation.grid.points, 0), dtype=np.float64)
        if decision.kind == AtlasKind.SEPARATED and atoms:
            measure_basis = build_peakon_tangent_basis(
                equation.grid,
                decision.decomposition.positions[:atoms],
                decision.decomposition.amplitudes[:atoms],
            )[:, atoms:]
        elif decision.kind == AtlasKind.CLUSTER and clusters and whitening.identifiable:
            measure_basis = whitening.basis

        def orthogonal_field(values: NDArray[np.floating]) -> NDArray[np.floating]:
            if measure_basis.shape[1] == 0:
                return np.asarray(values)
            applied_basis = np.stack(
                [
                    equation.helmholtz.apply(measure_basis[:, index])
                    for index in range(measure_basis.shape[1])
                ],
                axis=1,
            )
            gram = equation.grid.h * measure_basis.T @ applied_basis
            # A single projection can leave a small cancellation residual when
            # the field and measure components have very different scales.
            # Reproject the residual once (iterative refinement) so the
            # A_h-orthogonality gate is reproducible across BLAS backends.
            orthogonal = np.asarray(values, dtype=np.float64).copy()
            for _ in range(2):
                rhs = (
                    equation.grid.h
                    * measure_basis.T
                    @ equation.helmholtz.apply(orthogonal)
                )
                coefficients = np.linalg.solve(gram, rhs)
                orthogonal -= measure_basis @ coefficients
            return orthogonal

        member_corrections = []
        member_fields = []
        member_measures = []
        for model in self.models:
            with torch.no_grad():
                position, _amplitude = model.particle(
                    tensor(self._pad(decision.decomposition.positions, self.maximum_atoms)),
                    tensor(self._pad(decision.decomposition.amplitudes, self.maximum_atoms)),
                    tensor(atom_mask), tensor(graph_parameters),
                    torch.tensor(
                        [equation.grid.length], dtype=torch.float32, device=self.device
                    ),
                )
                cluster_white, _cluster_moment = model.cluster(
                    tensor(self._pad(decision.clusters.centers, self.maximum_clusters)),
                    tensor(self._pad(decision.clusters.signs, self.maximum_clusters)),
                    tensor(self._pad(moments, self.maximum_clusters, 3)),
                    tensor(cluster_mask), tensor(graph_parameters),
                    torch.tensor(
                        [equation.grid.length], dtype=torch.float32, device=self.device
                    ),
                )
                field_rate = model.field(
                    tensor(normalized_fields), tensor(parameters)
                )[0].cpu().numpy()
            field_component = np.zeros_like(current)
            if field_enabled:
                momentum = (
                    field_rate * float(self.stats["field_target_scale"])
                    * state_scale * dt
                )
                field_component = equation.helmholtz.solve(momentum)
                field_component = orthogonal_field(field_component)
            measure_component = np.zeros_like(current)
            if decision.kind == AtlasKind.SEPARATED and atoms and particle_enabled:
                rate = position[0, :atoms].cpu().numpy() * float(
                    self.stats["position_scale"]
                )
                measure_component = measure_basis @ (
                    rate * equation.grid.length * dt
                )
            elif (
                decision.kind == AtlasKind.CLUSTER and clusters
                and cluster_enabled and whitening.identifiable
            ):
                whitened_rate = cluster_white[0, :clusters].cpu().numpy() * float(
                    self.stats["cluster_whitened_scale"]
                )
                measure_component = reconstruct_cluster_position_defect(
                    whitened_rate, whitening, state_scale=state_scale, dt=dt
                )
            member_fields.append(np.asarray(field_component))
            member_measures.append(np.asarray(measure_component))
            member_corrections.append(np.asarray(field_component + measure_component))
        ensemble_mean = np.mean(member_corrections, axis=0)
        if self.selected_member is None:
            correction = ensemble_mean.astype(current.dtype)
            field_component = np.mean(member_fields, axis=0)
            measure_component = np.mean(member_measures, axis=0)
        else:
            correction = member_corrections[self.selected_member].astype(current.dtype)
            field_component = member_fields[self.selected_member]
            measure_component = member_measures[self.selected_member]
        ensemble_variance = np.mean([
            a_norm(member - ensemble_mean, equation.helmholtz) ** 2
            for member in member_corrections
        ])
        ensemble_std = float(np.sqrt(ensemble_variance))
        ensemble_relative = ensemble_std / (
            a_norm(ensemble_mean, equation.helmholtz) + np.finfo(float).eps
        )
        current_rms = float(np.sqrt(np.mean(features[1] ** 2)))
        field_rms = float(np.sqrt(np.mean(features[7] ** 2)))
        purity = field_rms / max(current_rms, np.finfo(np.float32).eps)
        confidence = (
            decision.decomposition.detection_confidence
            if decision.kind != AtlasKind.FIELD
            else 1.0 - decision.decomposition.detection_confidence
        )
        return MCHCorrectionProposal(
            correction, decision.kind, float(confidence),
            float(np.max(np.abs(parameters))), purity,
            float(decision.identifiability.separation_cells),
            a_norm(field_component, equation.helmholtz),
            a_norm(measure_component, equation.helmholtz),
            ensemble_std, ensemble_relative,
        )


@dataclass(frozen=True, slots=True)
class SafeAdaptiveMCHConfig:
    initial_dt: float = 0.01
    minimum_dt: float = 0.0025
    maximum_dt: float = 0.01
    relative_tolerance: float = 5.0e-3
    safety_factor: float = 0.9
    maximum_growth: float = 1.5
    minimum_shrink: float = 0.5
    maximum_retries: int = 8

    def __post_init__(self) -> None:
        if not 0 < self.minimum_dt <= self.initial_dt <= self.maximum_dt:
            raise ValueError("Adaptive time-step bounds are inconsistent.")
        if self.relative_tolerance <= 0.0 or self.maximum_retries < 1:
            raise ValueError("Adaptive tolerance and retry count must be positive.")


@dataclass(frozen=True, slots=True)
class SafeAdaptiveMCHStep:
    state: NDArray[np.floating]
    time: float
    dt: float
    next_dt: float
    error_estimate: float
    retries: int
    atlas: AtlasKind
    switched: bool
    network_accepted: bool
    certificate_reason: str
    mass_drift: float
    energy_drift: float


@dataclass(frozen=True, slots=True)
class SafeAdaptiveMCHRollout:
    times: NDArray[np.floating]
    states: NDArray[np.floating]
    steps: tuple[SafeAdaptiveMCHStep, ...]
    rejected_steps: int
    fallback_steps: int
    atlas_switches: int
    maximum_mass_drift: float
    maximum_energy_drift: float


@dataclass(slots=True)
class SafeAdaptiveMCHSolver:
    equation: ModifiedCH
    corrector: MCHNetworkCorrector
    detector: MCHAtlasConfig = field(default_factory=MCHAtlasConfig)
    gate: MCHCorrectionGateConfig = field(default_factory=MCHCorrectionGateConfig)
    adaptive: SafeAdaptiveMCHConfig = field(default_factory=SafeAdaptiveMCHConfig)
    variant: str = "full"

    def __post_init__(self) -> None:
        if self.variant not in M5_VARIANTS:
            raise ValueError(f"Unknown M5 variant: {self.variant}.")

    def _candidate(
        self,
        previous: NDArray[np.floating],
        current: NDArray[np.floating],
        dt: float,
        previous_atlas: AtlasKind | None,
    ) -> tuple[NDArray[np.floating], AtlasDecision, MCHCorrectionCertificate]:
        baseline = MassH1ProjectedMCHSolver(self.equation, dt).step(current).state
        decision = select_atlas(current, self.equation.grid, self.detector, previous=previous_atlas)
        if self.variant == "baseline_only":
            certificate = MCHCorrectionCertificate(
                baseline, np.zeros_like(baseline), False, "baseline_only", 0.0,
                0.0, 0.0, 0.0, 1.0,
            )
            return certificate.state, decision, certificate
        proposal = self.corrector.propose(
            self.equation, previous, current, baseline, decision, dt, variant=self.variant
        )
        if self.variant == "ungated":
            trial = baseline + proposal.correction
            if np.all(np.isfinite(trial)):
                certificate = MCHCorrectionCertificate(
                    trial, proposal.correction, True, "ungated", 1.0,
                    a_norm(proposal.correction, self.equation.helmholtz)
                    / (a_norm(baseline - current, self.equation.helmholtz) + 1e-12),
                    0.0, 0.0, 1.0,
                )
            else:
                certificate = MCHCorrectionCertificate(
                    baseline, np.zeros_like(baseline), False, "nonfinite_correction",
                    0.0, np.inf, 0.0, 0.0, 1.0,
                )
        else:
            half_baseline = MassH1ProjectedMCHSolver(
                self.equation, 0.5 * dt
            ).step(current).state
            fine_baseline = MassH1ProjectedMCHSolver(
                self.equation, 0.5 * dt
            ).step(half_baseline).state
            embedded_error_norm = a_norm(
                fine_baseline - baseline, self.equation.helmholtz
            )
            certificate = certify_mch_correction(
                self.equation, current, baseline, proposal.correction,
                atlas=proposal.atlas, confidence=proposal.confidence,
                ood_distance=proposal.ood_distance,
                measure_purity=proposal.measure_purity,
                separation_cells=proposal.separation_cells, config=self.gate,
                embedded_error_norm=embedded_error_norm,
                embedded_direction=fine_baseline - baseline,
                ensemble_relative_uncertainty=proposal.ensemble_relative_uncertainty,
                enforce_confidence=self.variant != "conservation_only",
                enforce_conservation=self.variant != "confidence_only",
            )
        return certificate.state, decision, certificate

    def solve(
        self,
        initial: NDArray[np.floating],
        *,
        final_time: float,
    ) -> SafeAdaptiveMCHRollout:
        self.equation.grid.validate_state(initial)
        if final_time <= 0.0:
            raise ValueError("final_time must be positive.")
        states = [np.asarray(initial).copy()]
        times = [0.0]
        records: list[SafeAdaptiveMCHStep] = []
        previous_atlas: AtlasKind | None = None
        dt = self.adaptive.initial_dt
        rejected = 0
        initial_mass = float(self.equation.grid.h * np.sum(initial))
        initial_energy = discrete_energy(initial, self.equation.helmholtz)
        while times[-1] < final_time - 1e-14:
            trial_dt = min(dt, final_time - times[-1])
            retries = 0
            while True:
                previous = states[-2] if len(states) > 1 else states[-1]
                full, full_decision, full_certificate = self._candidate(
                    previous, states[-1], trial_dt, previous_atlas
                )
                half, half_decision, half_certificate = self._candidate(
                    previous, states[-1], 0.5 * trial_dt, previous_atlas
                )
                fine, fine_decision, fine_certificate = self._candidate(
                    states[-1], half, 0.5 * trial_dt, half_decision.kind
                )
                error = a_norm(fine - full, self.equation.helmholtz) / max(
                    a_norm(fine, self.equation.helmholtz), np.finfo(float).eps
                )
                tolerance_met = error <= self.adaptive.relative_tolerance
                at_floor = trial_dt <= self.adaptive.minimum_dt * (1.0 + 1e-12)
                if tolerance_met or at_floor:
                    break
                rejected += 1
                retries += 1
                if retries >= self.adaptive.maximum_retries:
                    raise RuntimeError("Adaptive mCH step exceeded maximum retries.")
                factor = max(
                    self.adaptive.minimum_shrink,
                    self.adaptive.safety_factor
                    * (self.adaptive.relative_tolerance / max(error, 1e-15)) ** (1.0 / 3.0),
                )
                trial_dt = max(self.adaptive.minimum_dt, trial_dt * factor)
            states.append(np.asarray(fine))
            times.append(times[-1] + trial_dt)
            mass_drift = abs(float(self.equation.grid.h * np.sum(fine)) - initial_mass)
            energy_drift = abs(discrete_energy(fine, self.equation.helmholtz) - initial_energy)
            growth = self.adaptive.maximum_growth if error == 0.0 else min(
                self.adaptive.maximum_growth,
                self.adaptive.safety_factor
                * (self.adaptive.relative_tolerance / max(error, 1e-15)) ** (1.0 / 3.0),
            )
            next_dt = min(self.adaptive.maximum_dt, max(self.adaptive.minimum_dt, trial_dt * growth))
            records.append(
                SafeAdaptiveMCHStep(
                    states[-1], times[-1], trial_dt, next_dt, error, retries,
                    fine_decision.kind,
                    half_decision.switched or fine_decision.switched,
                    half_certificate.accepted or fine_certificate.accepted,
                    f"half:{half_certificate.reason}|fine:{fine_certificate.reason}",
                    mass_drift, energy_drift,
                )
            )
            previous_atlas = fine_decision.kind
            dt = next_dt
        return SafeAdaptiveMCHRollout(
            np.asarray(times), np.stack(states), tuple(records), rejected,
            sum(not step.network_accepted for step in records),
            sum(step.switched for step in records),
            max((step.mass_drift for step in records), default=0.0),
            max((step.energy_drift for step in records), default=0.0),
        )
