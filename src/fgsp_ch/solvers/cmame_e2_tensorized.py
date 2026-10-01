"""Batched GPU numerical backbone for the frozen CMAME adaptive solver.

The module mirrors the authoritative NumPy 2:1 AMR algorithm.  It changes the
execution layout, not the equations: periodic conservative divergence, two
fine substeps, reflux and the mass--H1 projection remain explicit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
from torch import Tensor


@dataclass(slots=True)
class TensorHybridState:
    positions: Tensor          # [B,P]
    amplitudes: Tensor         # [B,P]
    particle_mask: Tensor      # [B,P] bool
    field_momentum: Tensor     # [B,N]


@dataclass(slots=True)
class TensorHybridAMRState:
    positions: Tensor
    amplitudes: Tensor
    particle_mask: Tensor
    coarse_momentum: Tensor    # [B,N]
    fine_detail: Tensor        # [B,2N]
    patch: Tensor              # [B,N] bool


@dataclass(frozen=True, slots=True)
class TensorAMRDiagnostics:
    composite_mass_drift: Tensor
    composite_relative_h1_drift: Tensor
    maximum_fine_cfl: Tensor


Correction = Callable[[TensorHybridState, float, Tensor], Tensor]


def first_derivative(values: Tensor, h: float) -> Tensor:
    return (torch.roll(values, -1, -1) - torch.roll(values, 1, -1)) / (2.0*h)


def second_derivative(values: Tensor, h: float) -> Tensor:
    return (torch.roll(values, -1, -1)-2.0*values+torch.roll(values, 1, -1))/(h*h)


def helmholtz_apply(values: Tensor, h: float) -> Tensor:
    return values-second_derivative(values,h)


def helmholtz_solve(momentum: Tensor, h: float) -> Tensor:
    points=momentum.shape[-1]
    modes=torch.arange(points,dtype=momentum.dtype,device=momentum.device)
    eigenvalues=1.0+4.0*torch.sin(torch.pi*modes/points).square()/(h*h)
    return torch.fft.ifft(torch.fft.fft(momentum,dim=-1)/eigenvalues,dim=-1).real


def fourier_prolong(values: Tensor) -> Tensor:
    """Real Fourier resampling from N to 2N, matching scipy.signal.resample."""
    n=values.shape[-1]; m=2*n
    source=torch.fft.rfft(values,dim=-1)
    target=torch.zeros(*source.shape[:-1],m//2+1,dtype=source.dtype,device=source.device)
    target[...,:n//2+1]=source
    if n%2==0:
        target[...,n//2]*=0.5
    return torch.fft.irfft(target,n=m,dim=-1)*(m/n)


def periodic_peakon(displacement: Tensor, length: float) -> Tensor:
    wrapped=torch.remainder(displacement+0.5*length,length)-0.5*length
    return torch.cosh(0.5*length-wrapped.abs())/torch.cosh(
        torch.as_tensor(0.5*length,dtype=displacement.dtype,device=displacement.device)
    )


def periodic_peakon_derivative(displacement: Tensor, length: float) -> Tensor:
    wrapped=torch.remainder(displacement+0.5*length,length)-0.5*length
    result=-torch.sign(wrapped)*torch.sinh(0.5*length-wrapped.abs())/torch.cosh(
        torch.as_tensor(0.5*length,dtype=displacement.dtype,device=displacement.device)
    )
    return torch.where(wrapped==0.0,torch.zeros_like(result),result)


def reconstruct_particles(state: TensorHybridState, points: int, length: float) -> Tensor:
    x=torch.arange(points,dtype=state.field_momentum.dtype,device=state.field_momentum.device)*(length/points)
    kernel=periodic_peakon(x[None,:,None]-state.positions[:,None,:],length)
    weights=state.amplitudes*state.particle_mask.to(state.amplitudes.dtype)
    return torch.sum(kernel*weights[:,None,:],dim=-1)


def periodic_interpolate(values: Tensor, positions: Tensor, mask: Tensor, length: float) -> Tensor:
    points=values.shape[-1]; coordinate=torch.remainder(positions,length)/(length/points)
    left=torch.floor(coordinate).long()%points; fraction=coordinate-left.to(coordinate.dtype)
    a=torch.gather(values,1,left); b=torch.gather(values,1,(left+1)%points)
    return torch.where(mask,(1.0-fraction)*a+fraction*b,torch.zeros_like(a))


def total_state(state: TensorHybridState, length: float) -> Tensor:
    h=length/state.field_momentum.shape[-1]
    return helmholtz_solve(state.field_momentum,h)+reconstruct_particles(
        state,state.field_momentum.shape[-1],length
    )


def particle_velocity(state: TensorHybridState, alpha: Tensor, length: float) -> Tensor:
    points=state.field_momentum.shape[-1]; h=length/points
    field=helmholtz_solve(state.field_momentum,h); slope=first_derivative(field,h)
    delta=state.positions[:,:,None]-state.positions[:,None,:]
    weights=state.amplitudes*state.particle_mask.to(state.amplitudes.dtype)
    u=torch.sum(periodic_peakon(delta,length)*weights[:,None,:],dim=-1)
    ux=torch.sum(periodic_peakon_derivative(delta,length)*weights[:,None,:],dim=-1)
    u+=periodic_interpolate(field,state.positions,state.particle_mask,length)
    ux+=periodic_interpolate(slope,state.positions,state.particle_mask,length)
    self_slope=state.amplitudes*torch.tanh(torch.as_tensor(
        0.5*length,dtype=field.dtype,device=field.device
    ))
    velocity=alpha[:,None]*(u.square()-ux.square()-self_slope.square())
    return torch.where(state.particle_mask,velocity,torch.zeros_like(velocity))


def flux_and_source(state: TensorHybridState, alpha: Tensor, gamma: Tensor, length: float) -> tuple[Tensor,Tensor]:
    h=length/state.field_momentum.shape[-1]; total=total_state(state,length)
    slope=first_derivative(total,h)
    cell=alpha[:,None]*(total.square()-slope.square())*state.field_momentum
    return 0.5*(cell+torch.roll(cell,-1,-1)),-gamma[:,None]*slope


def energy(values: Tensor, h: float) -> Tensor:
    return 0.5*h*torch.sum(values*helmholtz_apply(values,h),dim=-1)


def project_mass_h1(candidate: Tensor, target_mass: Tensor, target_energy: Tensor, length: float) -> Tensor:
    h=length/candidate.shape[-1]; mean=target_mass/length
    available=(target_energy-0.5*length*mean.square()).clamp_min(0.0)
    centered=candidate-candidate.mean(-1,keepdim=True)
    centered_energy=energy(centered,h).clamp_min(torch.finfo(candidate.dtype).eps)
    scale=torch.sqrt(available/centered_energy)
    return mean[:,None]+scale[:,None]*centered


def base_fine_momentum(coarse_momentum: Tensor, length: float) -> Tensor:
    n=coarse_momentum.shape[-1]; coarse_h=length/n; fine_h=0.5*coarse_h
    return helmholtz_apply(fourier_prolong(helmholtz_solve(coarse_momentum,coarse_h)),fine_h)


def composite_total(state: TensorHybridAMRState, length: float) -> Tensor:
    n=state.coarse_momentum.shape[-1]; coarse_h=length/n; fine_h=0.5*coarse_h
    regular=fourier_prolong(helmholtz_solve(state.coarse_momentum,coarse_h))+helmholtz_solve(state.fine_detail,fine_h)
    view=TensorHybridState(state.positions,state.amplitudes,state.particle_mask,state.fine_detail)
    # Only the particle arrays and output shape are used by reconstruction.
    x=torch.arange(2*n,dtype=regular.dtype,device=regular.device)*fine_h
    kernel=periodic_peakon(x[None,:,None]-state.positions[:,None,:],length)
    weights=state.amplitudes*state.particle_mask.to(state.amplitudes.dtype)
    return regular+torch.sum(kernel*weights[:,None,:],dim=-1)


def migrate_detail(detail: Tensor, old_patch: Tensor, new_patch: Tensor, fine_h: float) -> Tensor:
    old=torch.repeat_interleave(old_patch,2,-1); new=torch.repeat_interleave(new_patch,2,-1)
    old_mass=fine_h*torch.sum(torch.where(old,detail,0.0),dim=-1)
    result=torch.where(old&new,detail,torch.zeros_like(detail))
    defect=old_mass-fine_h*torch.sum(torch.where(new,result,0.0),dim=-1)
    count=new.sum(-1).clamp_min(1)
    return torch.where(new,result+defect[:,None]/(fine_h*count[:,None]),torch.zeros_like(result))


def reflux(coarse: Tensor, fine: Tensor, coarse_flux: Tensor, fine_flux: Tensor, patch: Tensor, dt: Tensor, h: float) -> Tensor:
    result=torch.where(patch,0.5*(fine[...,0::2]+fine[...,1::2]),coarse)
    right=torch.roll(patch,-1,-1); mismatch=fine_flux[...,1::2]-coarse_flux
    left_boundary=(~patch)&right; right_boundary=patch&(~right)
    result=result+torch.where(left_boundary,-dt[:,None]*mismatch/h,0.0)
    right_delta=torch.where(right_boundary,dt[:,None]*mismatch/h,0.0)
    return result+torch.roll(right_delta,1,-1)


class TensorizedHybridAMRSolver:
    """One batched authoritative 2:1 AMR step for a same-grid cohort."""

    def __init__(self,length:float,alpha:Tensor,gamma:Tensor,dt:Tensor,correction:Correction|None=None):
        self.length=float(length); self.alpha=alpha; self.gamma=gamma; self.dt=dt; self.correction=correction

    def _flux(self,state:TensorHybridState,dt:Tensor)->tuple[Tensor,Tensor]:
        flux,source=flux_and_source(state,self.alpha,self.gamma,self.length)
        if self.correction is not None:
            flux=flux+self.correction(state,self.length/state.field_momentum.shape[-1],dt).to(flux)
        return flux,source

    @torch.inference_mode()
    def step(self,state:TensorHybridAMRState)->tuple[TensorHybridAMRState,TensorAMRDiagnostics]:
        n=state.coarse_momentum.shape[-1]; h=self.length/n; hf=0.5*h
        initial_fine=composite_total(state,self.length)
        target_composite_mass=hf*initial_fine.sum(-1); target_composite_energy=energy(initial_fine,hf)
        coarse=TensorHybridState(state.positions,state.amplitudes,state.particle_mask,state.coarse_momentum)
        initial_total=total_state(coarse,self.length)
        target_mass=h*initial_total.sum(-1); target_energy=energy(initial_total,h)
        coarse_flux,coarse_source=self._flux(coarse,self.dt)
        coarse_provisional=state.coarse_momentum-self.dt[:,None]*(coarse_flux-torch.roll(coarse_flux,1,-1))/h+self.dt[:,None]*coarse_source
        fine_m=base_fine_momentum(state.coarse_momentum,self.length)+state.fine_detail
        q=state.positions.clone(); fine_fluxes=[]; maximum_cfl=torch.zeros_like(self.dt)
        fine_mask=torch.repeat_interleave(state.patch,2,-1)
        for _ in range(2):
            fine_state=TensorHybridState(q,state.amplitudes,state.particle_mask,fine_m)
            flux,source=self._flux(fine_state,0.5*self.dt); fine_fluxes.append(flux)
            total=total_state(fine_state,self.length); slope=first_derivative(total,hf)
            speed=(self.alpha[:,None]*(total.square()-slope.square())).abs().amax(-1)
            maximum_cfl=torch.maximum(maximum_cfl,0.5*self.dt*speed/hf)
            rhs=-(flux-torch.roll(flux,1,-1))/hf+source
            fine_m=torch.where(fine_mask,fine_m+0.5*self.dt[:,None]*rhs,fine_m)
            q=torch.remainder(q+0.5*self.dt[:,None]*particle_velocity(fine_state,self.alpha,self.length),self.length)
        synchronized=reflux(coarse_provisional,fine_m,coarse_flux,0.5*(fine_fluxes[0]+fine_fluxes[1]),state.patch,self.dt,h)
        candidate=TensorHybridState(q,state.amplitudes,state.particle_mask,synchronized)
        projected=project_mass_h1(total_state(candidate,self.length),target_mass,target_energy,self.length)
        atomic=TensorHybridState(q,state.amplitudes,state.particle_mask,torch.zeros_like(synchronized))
        projected_m=helmholtz_apply(projected-reconstruct_particles(atomic,n,self.length),h)
        base=base_fine_momentum(projected_m,self.length)
        detail=torch.where(fine_mask,fine_m-base,torch.zeros_like(fine_m))
        provisional=TensorHybridAMRState(q,state.amplitudes,state.particle_mask,projected_m,detail,state.patch)
        provisional_total=composite_total(provisional,self.length)
        defect=target_composite_mass-hf*provisional_total.sum(-1)
        count=fine_mask.sum(-1).clamp_min(1)
        detail=torch.where(fine_mask,detail+defect[:,None]/(hf*count[:,None]),detail)
        result=TensorHybridAMRState(q,state.amplitudes,state.particle_mask,projected_m,detail,state.patch)
        final_total=composite_total(result,self.length)
        # A second zero-mode closure removes FFT round-off after the first
        # large cancellation without changing any non-zero Fourier mode.
        residual=target_composite_mass-hf*final_total.sum(-1)
        detail=torch.where(fine_mask,detail+residual[:,None]/(hf*count[:,None]),detail)
        result=TensorHybridAMRState(q,state.amplitudes,state.particle_mask,projected_m,detail,state.patch)
        final_total=composite_total(result,self.length)
        mass=(hf*final_total.sum(-1)-target_composite_mass).abs()
        h1=(energy(final_total,hf)-target_composite_energy).abs()/target_composite_energy.abs().clamp_min(torch.finfo(final_total.dtype).eps)
        return result,TensorAMRDiagnostics(mass,h1,maximum_cfl)


class TensorizedUniformHybridSolver:
    """One conservative batched uniform-grid step.

    This is the representation-reduced E3 route used when no AMR interface is
    present.  It deliberately shares the flux, particle velocity and
    mass--H1 projection with the hybrid solver.
    """

    def __init__(self,length:float,alpha:Tensor,gamma:Tensor,dt:Tensor,correction:Correction|None=None):
        self.length=float(length); self.alpha=alpha; self.gamma=gamma; self.dt=dt; self.correction=correction

    @torch.inference_mode()
    def step(self,state:TensorHybridState)->tuple[TensorHybridState,TensorAMRDiagnostics]:
        points=state.field_momentum.shape[-1]; h=self.length/points
        initial=total_state(state,self.length)
        target_mass=h*initial.sum(-1); target_energy=energy(initial,h)
        flux,source=flux_and_source(state,self.alpha,self.gamma,self.length)
        if self.correction is not None:
            flux=flux+self.correction(state,h,self.dt).to(flux)
        provisional=state.field_momentum-self.dt[:,None]*(flux-torch.roll(flux,1,-1))/h+self.dt[:,None]*source
        q=torch.remainder(
            state.positions+self.dt[:,None]*particle_velocity(state,self.alpha,self.length),
            self.length,
        )
        candidate=TensorHybridState(q,state.amplitudes,state.particle_mask,provisional)
        projected=project_mass_h1(total_state(candidate,self.length),target_mass,target_energy,self.length)
        atomic=TensorHybridState(q,state.amplitudes,state.particle_mask,torch.zeros_like(provisional))
        regular=helmholtz_apply(projected-reconstruct_particles(atomic,points,self.length),h)
        result=TensorHybridState(q,state.amplitudes,state.particle_mask,regular)
        final=total_state(result,self.length)
        mass=(h*final.sum(-1)-target_mass).abs()
        h1=(energy(final,h)-target_energy).abs()/target_energy.abs().clamp_min(torch.finfo(final.dtype).eps)
        slope=first_derivative(initial,h)
        speed=(self.alpha[:,None]*(initial.square()-slope.square())).abs().amax(-1)
        return result,TensorAMRDiagnostics(mass,h1,self.dt*speed/h)


@torch.inference_mode()
def particle_only_step(state:TensorHybridState,alpha:Tensor,dt:Tensor,length:float)->TensorHybridState:
    """Exact-representation mCH multipeakon position step; amplitudes stay fixed."""
    tolerance=100.0*torch.finfo(state.field_momentum.dtype).eps
    if torch.any(state.field_momentum.abs()>tolerance):
        raise ValueError("particle_only_step requires zero regular momentum")
    q=torch.remainder(
        state.positions+dt[:,None]*particle_velocity(state,alpha,float(length)),
        float(length),
    )
    return TensorHybridState(q,state.amplitudes,state.particle_mask,state.field_momentum)
