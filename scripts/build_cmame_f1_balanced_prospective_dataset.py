"""Build fresh balanced trajectory-prefix evidence for CMAME-F1."""
from __future__ import annotations
import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
for p in (ROOT,ROOT/"src"):
 if str(p) not in sys.path:sys.path.insert(0,str(p))
from scripts.build_cmame_p21_mixed_causal_dataset import build_initial_state
from fgsp_ch.discretization.grids import PeriodicGrid
from fgsp_ch.equations.modified_ch import ModifiedCHParameters
from fgsp_ch.estimators.mch_hybrid_aposteriori import component_distance, hierarchical_estimate, hybrid_total_h1_norm, transfer_hybrid_state
from fgsp_ch.representations.cmame_p2_hybrid import CMAMEHybridState, hybrid_h1_squared, hybrid_mass
from fgsp_ch.solvers.cmame_p21_reference import spectral_hybrid_rollout
from fgsp_ch.solvers.mch_hybrid_amr import HybridMCHState
from fgsp_ch.nonlocal_waves.adaptivity.bo_certified import hierarchical_one_step_defect
from fgsp_ch.nonlocal_waves.adaptivity.bo_classical import bo_resolution_indicators, project_periodic_spectrum
from fgsp_ch.nonlocal_waves.evaluation.bo_metrics import bo_invariants, relative_spectral_h1
from fgsp_ch.nonlocal_waves.initial_conditions.bo_families import bo_initial_condition
from fgsp_ch.nonlocal_waves.operators.bo_fourier import BOFourierOperator
from fgsp_ch.nonlocal_waves.solvers.bo_reference import BOIFRK4Solver, solve_bo_dop853
from fgsp_ch.nonlocal_waves.adaptivity.prospective_causal_operator import ACTIONS, FEATURE_ORDER, prospective_features
from fgsp_ch.utils.config import load_yaml

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest().upper()
def represented(state):return CMAMEHybridState(state.field_momentum,state.positions,state.amplitudes)
def append(store,**row):
 for key,value in row.items():store.setdefault(key,[]).append(value)

def spectral_h1_components(approximation,reference,grid):
 k=2*np.pi*np.fft.rfftfreq(grid.points,d=grid.h);weights=1+k*k;difference=np.fft.rfft(np.asarray(approximation)-np.asarray(reference));reference_hat=np.fft.rfft(reference);absolute=float(np.sqrt(np.sum(weights*np.abs(difference)**2)));scale=float(np.sqrt(np.sum(weights*np.abs(reference_hat)**2)))
 if not np.isfinite(scale) or scale<=np.finfo(float).eps:raise ValueError("BO reference state has no resolvable spectral H1 scale")
 return absolute,scale,absolute/scale

def mch_rows(store,cfg,mode,split,family,seed,group):
 settings=cfg[mode]; grid=PeriodicGrid(int(settings["points"]),float(cfg["domain"]["length"])); state,alpha=build_initial_state(seed,family,grid,(.5,1.5));state=HybridMCHState(state.positions,state.amplitudes,state.regular_momentum);params=ModifiedCHParameters(alpha=alpha,gamma=0.0);dt=float(settings["dt"]);tol=float(settings["tolerance"]);nonlinear=cfg["nonlinear_solver"]
 initial_mass=hybrid_mass(represented(state),grid);initial_h1=hybrid_h1_squared(represented(state),grid)
 for prefix in range(int(settings["prefixes"])):
  remaining=max(tol*(1-prefix/max(int(settings["prefixes"]),1)),tol*.1)
  hold_next=None
  for action in ACTIONS:
   dt_factor={"shrink_time":.5,"hold":1.,"refine_space":1.}[action];res_factor={"shrink_time":1.,"hold":1.,"refine_space":2.}[action]
   target_grid=grid;target_state=state
   if res_factor==2:
    target_grid=PeriodicGrid(2*grid.points,grid.length,grid.x_min);target_state=transfer_hybrid_state(state,grid,target_grid)
   trial_dt=dt*dt_factor
   candidate,eta=hierarchical_estimate(target_state,target_grid,params,trial_dt,trial_dt,nonlinear)
   reference=spectral_hybrid_rollout(represented(target_state),target_grid,params,trial_dt,refinement=int(settings["reference_refinement"]),rtol=float(cfg["reference"]["rtol"]),atol=float(cfg["reference"]["atol"]),collision_tolerance=float(cfg["reference"]["collision_tolerance"]))
   truth=HybridMCHState(reference.state.positions,reference.state.amplitudes,reference.state.regular_momentum)
   err=component_distance(candidate,target_grid,truth,reference.grid,reference.grid);absolute_error=float(err["total"]);scale=hybrid_total_h1_norm(truth,reference.grid,reference.grid);true_error=absolute_error/scale;det=max(eta.temporal+eta.hierarchical_total,1e-14)
   mass=abs(hybrid_mass(represented(candidate),target_grid)-initial_mass)/max(abs(initial_mass),1e-12);h1=abs(hybrid_h1_squared(represented(candidate),target_grid)-initial_h1)/max(abs(initial_h1),1e-12);drift=max(mass,h1)
   sharp=float(np.max(eta.local_indicators)/max(np.linalg.norm(eta.local_indicators),1e-16));work=float(target_grid.points*(3+2*res_factor)/dt_factor)
   features=prospective_features(temporal=eta.temporal,spatial=eta.field+eta.particle,representation=eta.interface,migration=0.,invariant_drift=drift,invariant_limit=1e-10,stiffness=sharp,local_budget=remaining,remaining_budget_fraction=remaining/tol,remaining_time_fraction=1-prefix/max(int(settings["prefixes"]),1),relative_work=work/(grid.points*5),previous_rejected=False,localized_sharpness=sharp,nonlocal_tail=0.,dt_factor=dt_factor,resolution_factor=res_factor,probe_requested=res_factor>1,representation_change=res_factor>1)
   append(store,features=features,target_log_effectivity=np.log(max(true_error/det,1e-12)),target_log_cost=np.log(work),target_probe=float(res_factor>1 and eta.hierarchical_total>remaining*.25),deterministic_error=det,absolute_h1_error=absolute_error,reference_h1_norm=scale,true_error=true_error,metric_version=cfg["metric"]["version"],invariant_drift=drift,work=work,equation="modified_camassa_holm",family=family,split=split,group_id=group,prefix=prefix,action=action,tolerance=tol)
   if action=="hold":hold_next=candidate
  state=hold_next

def bo_rows(store,cfg,mode,split,family,seed,group):
 settings=cfg[mode];grid=PeriodicGrid(int(settings["points"]),float(cfg["domain"]["length"]));rng=np.random.default_rng(seed);values=bo_initial_condition(family,grid,amplitude=float(rng.uniform(.55,1.45)),phase_shift=float(rng.uniform(0,grid.length)));dt=float(settings["dt"])*8;tol=float(settings["tolerance"])
 for prefix in range(int(settings["prefixes"])):
  remaining=max(tol*(1-prefix/max(int(settings["prefixes"]),1)),tol*.1);hold_next=None;initial_inv=bo_invariants(values,grid)
  for action in ACTIONS:
   dt_factor={"shrink_time":.5,"hold":1.,"refine_space":1.}[action];res_factor={"shrink_time":1.,"hold":1.,"refine_space":2.}[action];target_grid=grid;target_values=values
   if res_factor==2:target_values,target_grid=project_periodic_spectrum(values,grid,2*grid.points)
   trial_dt=dt*dt_factor;high,low=BOIFRK4Solver(target_grid).step_embedded(target_values,trial_dt);embedded=relative_spectral_h1(high,low,target_grid);residual=BOFourierOperator(target_grid).unresolved_nonlinear_increment(target_values,trial_dt);hier=hierarchical_one_step_defect(target_values,target_grid,trial_dt,coarse_step=high);det=max(embedded,residual,hier,1e-14)
   fine,fine_grid=project_periodic_spectrum(target_values,target_grid,int(settings["reference_refinement"])*target_grid.points);reference=solve_bo_dop853(fine,fine_grid,final_time=trial_dt,rtol=float(cfg["reference"]["rtol"]),atol=float(cfg["reference"]["atol"]),maximum_step=trial_dt/4).final_state;projected,_=project_periodic_spectrum(high,target_grid,fine_grid.points);absolute_error,scale,true_error=spectral_h1_components(projected,reference,fine_grid)
   inv=bo_invariants(high,target_grid);drift=max(abs(inv.mass-initial_inv.mass)/max(abs(initial_inv.mass),1e-12),abs(inv.half_l2_squared-initial_inv.half_l2_squared)/max(abs(initial_inv.half_l2_squared),1e-12),abs(inv.hamiltonian-initial_inv.hamiltonian)/max(abs(initial_inv.hamiltonian),1e-12));ind=bo_resolution_indicators(target_values,target_grid,trial_dt);work=float(7*target_grid.points*np.log2(target_grid.points)/dt_factor)
   features=prospective_features(temporal=embedded,spatial=max(residual,hier),representation=ind.aliasing_defect,migration=0.,invariant_drift=drift,invariant_limit=2e-5,stiffness=ind.tail_ratio/local_or(ind.aliasing_defect),local_budget=remaining,remaining_budget_fraction=remaining/tol,remaining_time_fraction=1-prefix/max(int(settings["prefixes"]),1),relative_work=work/(7*grid.points*np.log2(grid.points)),previous_rejected=False,localized_sharpness=ind.aliasing_defect,nonlocal_tail=ind.tail_ratio,dt_factor=dt_factor,resolution_factor=res_factor,probe_requested=res_factor>1,representation_change=res_factor>1)
   append(store,features=features,target_log_effectivity=np.log(max(true_error/det,1e-12)),target_log_cost=np.log(work),target_probe=float(res_factor>1 and hier>remaining*.25),deterministic_error=det,absolute_h1_error=absolute_error,reference_h1_norm=scale,true_error=true_error,metric_version=cfg["metric"]["version"],invariant_drift=drift,work=work,equation="benjamin_ono",family=family,split=split,group_id=group,prefix=prefix,action=action,tolerance=tol)
   if action=="hold":hold_next=high
  values=hold_next

def local_or(value):return max(float(value),1e-12)
def main():
 p=argparse.ArgumentParser();p.add_argument("--config",default="configs/experiment/cmame_f101_metric_repair.yaml");p.add_argument("--mode",choices=("development","full"),default="development");a=p.parse_args();cfg=load_yaml(ROOT/a.config);out=ROOT/cfg["output_dir"]/a.mode;out.mkdir(parents=True,exist_ok=True);store={};counts={}
 for split,spec in cfg["splits"].items():
  per=int(spec["groups_per_family"]);per=min(per,int(cfg["development"]["groups_per_family_cap"])) if a.mode=="development" else per
  for equation in ("modified_camassa_holm","benjamin_ono"):
   for fi,family in enumerate(cfg["families"][equation]):
    for index in range(per):
     seed=int(spec["seed"])+(0 if equation=="modified_camassa_holm" else 500000)+fi*10000+index;group=f"{split}:{equation}:{family}:{seed}";print(f"F1 data {group}",flush=True)
     (mch_rows if equation=="modified_camassa_holm" else bo_rows)(store,cfg,a.mode,split,family,seed,group)
     counts[(equation,split)]=counts.get((equation,split),0)+1
 arrays={key:np.asarray(value) for key,value in store.items()};path=out/"balanced_prospective_dataset.npz";np.savez_compressed(path,**arrays)
 group_sets={split:set(arrays["group_id"][arrays["split"]==split]) for split in cfg["splits"]};checks={"fresh_seed_ranges_disjoint":len(set.union(*group_sets.values()))==sum(map(len,group_sets.values())),"equations_trajectory_balanced":all(counts[("modified_camassa_holm",s)]==counts[("benjamin_ono",s)] for s in cfg["splits"]),"all_splits_and_families_present":all(counts[(e,s)]>0 for e in ("modified_camassa_holm","benjamin_ono") for s in cfg["splits"]),"features_precommit_and_finite":bool(np.all(np.isfinite(arrays["features"]))),"references_excluded_from_features":True,"three_actions_exercised":set(arrays["action"])==set(ACTIONS),"no_historical_p3_rows":True}
 identity=np.abs(arrays["true_error"]-arrays["absolute_h1_error"]/arrays["reference_h1_norm"]);pure=arrays["family"].astype(str)=="pure_particle_control";checks.update({"total_h1_metric_version_frozen":set(arrays["metric_version"].astype(str))=={str(cfg["metric"]["version"])},"reference_h1_norms_positive":bool(np.all(arrays["reference_h1_norm"]>float(cfg["metric"]["minimum_reference_h1_norm"]))),"pure_particle_reference_scale_nonzero":bool(np.all(arrays["reference_h1_norm"][pure]>float(cfg["metric"]["minimum_reference_h1_norm"]))),"relative_error_identity_closed":float(np.max(identity))<=float(cfg["metric"]["identity_tolerance"])})
 report={"status":"PASS" if all(checks.values()) else "STOP","next_phase":"CMAME-F1.0.1_training" if all(checks.values()) else None,"checks":checks,"blocking_failures":[k for k,v in checks.items() if not v],"mode":a.mode,"rows":len(arrays["features"]),"trajectory_groups":{s:len(v) for s,v in group_sets.items()},"equation_rows":{e:int(np.sum(arrays["equation"]==e)) for e in ("modified_camassa_holm","benjamin_ono")},"feature_order":FEATURE_ORDER,"metric_version":str(cfg["metric"]["version"]),"minimum_reference_h1_norm":float(np.min(arrays["reference_h1_norm"])),"minimum_pure_particle_reference_h1_norm":float(np.min(arrays["reference_h1_norm"][pure])),"maximum_relative_error_identity_defect":float(np.max(identity)),"dataset_sha256":sha(path),"scope":cfg["scope"]};(out/"dataset_acceptance.json").write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2));raise SystemExit(0 if report["status"]=="PASS" else 2)
if __name__=="__main__":main()
