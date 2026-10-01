"""Non-projected compatible 2:1 local AMR for periodic hybrid mCH states.

The composite regular momentum consists of coarse cell averages plus one
zero-mass fine-detail degree of freedom per covered coarse cell.  Interface
fluxes are shared before the implicit compatible update.  The accepted state
is never projected onto invariant level sets.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from fgsp_ch.discretization.adaptive_mesh import PeriodicRefinementPatch, finite_volume_divergence
from fgsp_ch.discretization.grids import PeriodicGrid
from fgsp_ch.equations.modified_ch import ModifiedCHParameters
from fgsp_ch.representations.cmame_p2_hybrid import CMAMEHybridState, hybrid_h1_squared, hybrid_mass
from fgsp_ch.solvers.mch_compatible_invariant import CompatibleInvariantHybridMCHSolver
from fgsp_ch.solvers.mch_compatible_adaptive import invariant_preserving_transfer
from fgsp_ch.solvers.mch_hybrid_amr import HybridMCHState, hybrid_flux_and_source, hybrid_particle_velocity


@dataclass(frozen=True, slots=True)
class CompatibleCompositeState:
    coarse_momentum: NDArray[np.floating]
    detail: NDArray[np.floating]
    positions: NDArray[np.floating]
    amplitudes: NDArray[np.floating]
    patch: PeriodicRefinementPatch


@dataclass(frozen=True, slots=True)
class CompatibleLocalStepDiagnostics:
    iterations: int
    residual: float
    raw_constraint_defect: float
    corrected_constraint_defect: float
    relative_vector_correction: float
    mass_drift: float
    h1_drift: float
    maximum_reflux_rate_defect: float
    active_fraction: float


def reconstruct_fine_momentum(state: CompatibleCompositeState) -> NDArray[np.floating]:
    coarse = np.asarray(state.coarse_momentum, dtype=np.float64)
    detail = np.asarray(state.detail, dtype=np.float64)
    if detail.shape != coarse.shape or state.patch.coarse_mask.shape != coarse.shape:
        raise ValueError("composite coarse/detail/patch shapes disagree")
    if np.any(np.abs(detail[~state.patch.coarse_mask]) > 1.0e-14):
        raise ValueError("fine detail must vanish outside the patch")
    fine = np.repeat(coarse, 2)
    fine[0::2] += detail
    fine[1::2] -= detail
    return fine


def restrict_fine_momentum(values: NDArray[np.floating]) -> NDArray[np.floating]:
    fine = np.asarray(values, dtype=np.float64)
    if fine.ndim != 1 or fine.size % 2:
        raise ValueError("2:1 restriction requires an even one-dimensional array")
    return 0.5 * (fine[0::2] + fine[1::2])


def composite_from_uniform(state: HybridMCHState, patch: PeriodicRefinementPatch) -> CompatibleCompositeState:
    points = state.field_momentum.size
    if patch.coarse_mask.shape != (points,):
        raise ValueError("patch and uniform state have incompatible grids")
    return CompatibleCompositeState(
        np.asarray(state.field_momentum, dtype=np.float64).copy(), np.zeros(points),
        np.asarray(state.positions, dtype=np.float64).copy(),
        np.asarray(state.amplitudes, dtype=np.float64).copy(), patch,
    )


def composite_from_fine_uniform(
    fine_state: HybridMCHState,
    coarse_grid: PeriodicGrid,
) -> CompatibleCompositeState:
    """Encode one independently sampled fine initial state without transfer error."""
    if fine_state.field_momentum.shape != (2 * coarse_grid.points,):
        raise ValueError("fine state is not 2:1 over the requested coarse grid")
    mask = np.ones(coarse_grid.points, dtype=bool)
    patch = PeriodicRefinementPatch(mask, np.repeat(mask, 2), 1.0, 1.0)
    fine = np.asarray(fine_state.field_momentum, dtype=np.float64)
    return CompatibleCompositeState(
        restrict_fine_momentum(fine),
        0.5 * (fine[0::2] - fine[1::2]),
        fine_state.positions.copy(), fine_state.amplitudes.copy(), patch,
    )


def composite_from_uniform_invariantly(
    state: HybridMCHState,
    coarse_grid: PeriodicGrid,
    patch: PeriodicRefinementPatch,
) -> tuple[CompatibleCompositeState, float, float, float]:
    """Lift a coarse state to the 2:1 composite space without changing invariants.

    The uniform invariant-preserving transfer is performed first.  Removing
    fine details outside ``patch`` is then treated as mesh migration and is
    corrected *before* time integration.  Thus initialization obeys the same
    contract as every later patch change.
    """
    if state.field_momentum.shape != (coarse_grid.points,):
        raise ValueError("uniform state and coarse grid are incompatible")
    if patch.coarse_mask.shape != (coarse_grid.points,):
        raise ValueError("patch and coarse grid are incompatible")
    fine_grid = PeriodicGrid(2 * coarse_grid.points, coarse_grid.length, coarse_grid.x_min)
    lifted, lift_mass, lift_h1 = invariant_preserving_transfer(state, coarse_grid, fine_grid)
    full_mask = np.ones(coarse_grid.points, dtype=bool)
    full_patch = PeriodicRefinementPatch(full_mask, np.repeat(full_mask, 2), 1.0, 1.0)
    fine = np.asarray(lifted.field_momentum, dtype=np.float64)
    full = CompatibleCompositeState(
        restrict_fine_momentum(fine),
        0.5 * (fine[0::2] - fine[1::2]),
        lifted.positions.copy(), lifted.amplitudes.copy(), full_patch,
    )
    local, migration_mass, migration_h1, distance = migrate_patch_invariantly(
        full, fine_grid, patch
    )
    return local, max(lift_mass, migration_mass), max(lift_h1, migration_h1), distance


def uniform_fine_state(state: CompatibleCompositeState) -> HybridMCHState:
    return HybridMCHState(
        np.asarray(state.positions).copy(), np.asarray(state.amplitudes).copy(),
        reconstruct_fine_momentum(state),
    )


def _represented(state: CompatibleCompositeState) -> CMAMEHybridState:
    return CMAMEHybridState(reconstruct_fine_momentum(state), state.positions, state.amplitudes)


def migrate_patch_invariantly(
    state: CompatibleCompositeState, fine_grid: PeriodicGrid, target_patch: PeriodicRefinementPatch
) -> tuple[CompatibleCompositeState, float, float, float]:
    """Change patch support before a step while exactly retaining mass and H1."""
    if fine_grid.points != 2 * state.coarse_momentum.size:
        raise ValueError("fine grid is not 2:1 over the composite coarse state")
    if target_patch.coarse_mask.shape != state.coarse_momentum.shape:
        raise ValueError("target patch and composite state have incompatible shapes")
    before = _represented(state)
    target_mass = hybrid_mass(before, fine_grid)
    target_h1 = hybrid_h1_squared(before, fine_grid)
    old_full = reconstruct_fine_momentum(state)
    coarse = restrict_fine_momentum(old_full)
    raw_detail = 0.5 * (old_full[0::2] - old_full[1::2])
    raw_detail[~target_patch.coarse_mask] = 0.0
    tentative = CompatibleCompositeState(coarse, raw_detail, state.positions.copy(), state.amplitudes.copy(), target_patch)
    raw = reconstruct_fine_momentum(tentative)
    atomic_mass = hybrid_mass(CMAMEHybridState(np.zeros(fine_grid.points), state.positions, state.amplitudes), fine_grid)
    constant = (target_mass - atomic_mass) / fine_grid.length
    zero = raw - np.mean(raw)

    def candidate(scale: float) -> CMAMEHybridState:
        return CMAMEHybridState(constant + scale * zero, state.positions.copy(), state.amplitudes.copy())

    e0 = hybrid_h1_squared(candidate(0.0), fine_grid)
    ep = hybrid_h1_squared(candidate(1.0), fine_grid)
    em = hybrid_h1_squared(candidate(-1.0), fine_grid)
    a = 0.5 * (ep + em) - e0
    b = 0.5 * (ep - em)
    c = e0 - target_h1
    roots: list[float] = []
    if abs(c) <= 1e-13 * max(1.0, target_h1): roots.append(1.0)
    if abs(a) <= 1e-15:
        if abs(b) > 1e-15: roots.append(-c / b)
    else:
        disc = b * b - 4.0 * a * c
        if disc >= -1e-12 * max(1.0, b * b):
            root = np.sqrt(max(disc, 0.0)); roots += [(-b + root)/(2*a), (-b - root)/(2*a)]
    roots = [value for value in roots if np.isfinite(value)]
    if not roots:
        raise RuntimeError("patch migration has no real invariant-compatible scale")
    final_rep = candidate(min(roots, key=lambda value: abs(value - 1.0)))
    final_full = final_rep.regular_momentum
    final_coarse = restrict_fine_momentum(final_full)
    final_detail = 0.5 * (final_full[0::2] - final_full[1::2])
    final_detail[~target_patch.coarse_mask] = 0.0
    final = CompatibleCompositeState(final_coarse, final_detail, state.positions.copy(), state.amplitudes.copy(), target_patch)
    mass_defect = abs(hybrid_mass(_represented(final), fine_grid)-target_mass)/max(abs(target_mass),np.finfo(float).eps)
    h1_defect = abs(hybrid_h1_squared(_represented(final), fine_grid)-target_h1)/max(abs(target_h1),np.finfo(float).eps)
    helper = CompatibleInvariantHybridMCHSolver(fine_grid, ModifiedCHParameters(1.0,0.0), 1.0)
    old_values = helper._pack(uniform_fine_state(state))
    new_values = helper._pack(uniform_fine_state(final))
    distance = float(np.linalg.norm(new_values-old_values)/max(np.linalg.norm(old_values),np.finfo(float).eps))
    return final, mass_defect, h1_defect, distance


@dataclass(slots=True)
class CompatibleLocalAMRSolver:
    coarse_grid: PeriodicGrid
    parameters: ModifiedCHParameters
    dt: float
    nonlinear_tolerance: float = 1e-11
    maximum_iterations: int = 80
    damping: float = 1.0
    rank_tolerance: float = 1e-12

    @property
    def fine_grid(self) -> PeriodicGrid:
        return PeriodicGrid(2*self.coarse_grid.points, self.coarse_grid.length, self.coarse_grid.x_min)

    def _active(self, patch: PeriodicRefinementPatch) -> NDArray[np.integer]:
        return np.flatnonzero(patch.coarse_mask)

    def _pack(self, state: CompatibleCompositeState) -> NDArray[np.floating]:
        active = self._active(state.patch)
        return np.concatenate((state.coarse_momentum, state.detail[active], state.positions))

    def _unpack(self, values: NDArray[np.floating], template: CompatibleCompositeState, *, wrap: bool) -> CompatibleCompositeState:
        n = self.coarse_grid.points; active = self._active(template.patch); p = template.positions.size
        if values.shape != (n+active.size+p,): raise ValueError("packed composite vector has wrong shape")
        detail=np.zeros(n); detail[active]=values[n:n+active.size]
        q=values[n+active.size:].copy()
        if wrap: q=np.remainder(q-self.coarse_grid.x_min,self.coarse_grid.length)+self.coarse_grid.x_min
        return CompatibleCompositeState(values[:n].copy(),detail,q,template.amplitudes.copy(),template.patch)

    def invariants(self, values: NDArray[np.floating], template: CompatibleCompositeState) -> NDArray[np.floating]:
        rep=_represented(self._unpack(values,template,wrap=False))
        return np.asarray([hybrid_mass(rep,self.fine_grid),0.5*hybrid_h1_squared(rep,self.fine_grid)])

    def midpoint_gradients(self, values: NDArray[np.floating], template: CompatibleCompositeState) -> NDArray[np.floating]:
        state=self._unpack(values,template,wrap=False); fine_state=uniform_fine_state(state)
        helper=CompatibleInvariantHybridMCHSolver(self.fine_grid,self.parameters,self.dt)
        gradients=helper.midpoint_gradients(helper._pack(fine_state),fine_state.amplitudes)
        n=self.coarse_grid.points; active=self._active(template.patch); p=template.positions.size
        mapped=np.empty((n+active.size+p,2))
        mapped[:n]=gradients[:2*n].reshape(n,2,2).sum(axis=1)
        mapped[n:n+active.size]=gradients[:2*n].reshape(n,2,2)[active,0]-gradients[:2*n].reshape(n,2,2)[active,1]
        mapped[n+active.size:]=gradients[2*n:]
        return mapped

    def raw_rhs(self, values: NDArray[np.floating], template: CompatibleCompositeState) -> tuple[NDArray[np.floating],float]:
        state=self._unpack(values,template,wrap=False); fine_state=uniform_fine_state(state)
        coarse_state=HybridMCHState(state.positions,state.amplitudes,state.coarse_momentum)
        coarse_flux,coarse_source=hybrid_flux_and_source(coarse_state,self.coarse_grid,self.parameters)
        fine_flux,fine_source=hybrid_flux_and_source(fine_state,self.fine_grid,self.parameters)
        covered=state.patch.coarse_mask; effective=coarse_flux.copy()
        for face in range(self.coarse_grid.points):
            if covered[face] or covered[(face+1)%self.coarse_grid.points]: effective[face]=fine_flux[2*face+1]
        coarse_rate=-finite_volume_divergence(effective,self.coarse_grid.h)+coarse_source
        fine_rate=-finite_volume_divergence(fine_flux,self.fine_grid.h)+fine_source
        average=0.5*(fine_rate[0::2]+fine_rate[1::2])
        reflux_defect=float(np.max(np.abs(average[covered]-coarse_rate[covered]))) if np.any(covered) else 0.0
        detail_rate=0.5*(fine_rate[0::2]-fine_rate[1::2])
        active=self._active(state.patch)
        dq=hybrid_particle_velocity(fine_state,self.fine_grid,alpha=self.parameters.alpha)
        return np.concatenate((coarse_rate,detail_rate[active],dq)),reflux_defect

    def discrete_gradients(self, initial, final, template):
        delta=final-initial; gradients=self.midpoint_gradients(0.5*(initial+final),template)
        denominator=float(delta@delta)
        if denominator<=np.finfo(float).eps:return gradients
        defect=self.invariants(final,template)-self.invariants(initial,template)-gradients.T@delta
        return gradients+np.outer(delta,defect/denominator)

    def tangent_vector(self,raw,gradients):
        multiplier=np.linalg.pinv(gradients.T@gradients,rcond=self.rank_tolerance)@(gradients.T@raw)
        corrected=raw-gradients@multiplier
        return corrected,float(np.linalg.norm(gradients.T@raw)),float(np.linalg.norm(gradients.T@corrected)),float(np.linalg.norm(corrected-raw)/max(np.linalg.norm(raw),np.finfo(float).eps))

    def step(self,state:CompatibleCompositeState)->tuple[CompatibleCompositeState,CompatibleLocalStepDiagnostics]:
        initial=self._pack(state); initial_inv=self.invariants(initial,state)
        raw,reflux=self.raw_rhs(initial,state); guess=initial+self.dt*raw
        residual=np.inf; raw_defect=corrected_defect=correction=np.inf; max_reflux=reflux
        for iteration in range(1,self.maximum_iterations+1):
            midpoint=0.5*(initial+guess); raw,reflux=self.raw_rhs(midpoint,state); max_reflux=max(max_reflux,reflux)
            gradients=self.discrete_gradients(initial,guess,state)
            tangent,raw_defect,corrected_defect,correction=self.tangent_vector(raw,gradients)
            target=initial+self.dt*tangent; update=target-guess
            residual=float(np.linalg.norm(update)/max(1.0,np.linalg.norm(target)))
            guess+=self.damping*update
            if residual<=self.nonlinear_tolerance:break
        else:raise RuntimeError(f"compatible local AMR midpoint failed: {residual:.3e}")
        final_inv=self.invariants(guess,state); scale=np.maximum(np.abs(initial_inv),np.finfo(float).eps)
        drift=np.abs(final_inv-initial_inv)/scale
        result=self._unpack(guess,state,wrap=True)
        return result,CompatibleLocalStepDiagnostics(iteration,residual,raw_defect,corrected_defect,correction,float(drift[0]),float(drift[1]),max_reflux,state.patch.active_fraction)
