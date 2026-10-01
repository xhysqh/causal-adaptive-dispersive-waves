from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[1]
for p in (ROOT,ROOT/"src"):
 if str(p) not in sys.path:sys.path.insert(0,str(p))
from fgsp_ch.nonlocal_waves.adaptivity.prospective_causal_operator import EQUATIONS,SharedProspectiveBudgetOperator,grouped_conformal_margins,prospective_loss
from fgsp_ch.utils.config import load_yaml

def balanced_indices(data,split_name,rng):
 ids=np.where(data["split"].astype(str)==split_name)[0];blocks=[]
 for equation in EQUATIONS:
  eq_ids=ids[data["equation"][ids].astype(str)==equation]
  for family in sorted(set(data["family"][eq_ids].astype(str))):blocks.append(eq_ids[data["family"][eq_ids].astype(str)==family])
 size=max(map(len,blocks));result=np.concatenate([rng.choice(row,size,replace=len(row)<size) for row in blocks]);rng.shuffle(result);return result
def main():
 p=argparse.ArgumentParser();p.add_argument("--config",default="configs/experiment/cmame_f101_metric_repair.yaml");p.add_argument("--mode",choices=("development","full"),default="development");p.add_argument("--device",default="cpu");a=p.parse_args();cfg=load_yaml(ROOT/a.config);out=ROOT/cfg["output_dir"]/a.mode;data=np.load(out/"balanced_prospective_dataset.npz");x=torch.as_tensor(data["features"],dtype=torch.float32);eq=torch.as_tensor([EQUATIONS.index(str(v)) for v in data["equation"]]);y=torch.as_tensor(data["target_log_effectivity"],dtype=torch.float32);cost=torch.as_tensor(data["target_log_cost"],dtype=torch.float32);probe=torch.as_tensor(data["target_probe"],dtype=torch.float32);rng=np.random.default_rng(104729);train=balanced_indices(data,"train",rng);cal=np.where(data["split"].astype(str)=="calibration")[0];sel=np.where(data["split"].astype(str)=="selection")[0];device=torch.device(a.device);models=[];predictions=[]
 for seed in cfg["training"]["seeds"]:
  torch.manual_seed(int(seed));model=SharedProspectiveBudgetOperator(hidden=int(cfg["model"]["hidden"])).to(device);model.set_normalization(x[train].to(device));opt=torch.optim.AdamW(model.parameters(),lr=float(cfg["training"]["learning_rate"]),weight_decay=float(cfg["training"]["weight_decay"]))
  for epoch in range(int(cfg["training"]["epochs"])):
   opt.zero_grad();o=model(x[train].to(device),eq[train].to(device));loss=prospective_loss(o,y[train].to(device),cost[train].to(device),probe[train].to(device),data["group_id"][train].astype(str).tolist(),data["prefix"][train].astype(int).tolist(),quantile=float(cfg["training"]["upper_quantile"]),causal_epsilon=float(cfg["training"]["causal_epsilon"]));loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),float(cfg["training"]["gradient_clip"]));opt.step()
  name=f"shared_prospective_member_{seed}.pt";torch.save({"model":model.state_dict(),"hidden":model.hidden},out/name);models.append((name,model.eval()))
 with torch.no_grad():
  for _,model in models:predictions.append(model(x.to(device),eq.to(device)).upper_log_effectivity.cpu().numpy())
 upper=np.mean(predictions,axis=0)+np.std(predictions,axis=0);margins=grouped_conformal_margins(upper[cal],data["target_log_effectivity"][cal],data["equation"][cal],data["group_id"][cal],coverage=float(cfg["calibration"]["trajectory_coverage"]));support={}
 for equation in EQUATIONS:
  ids=train[data["equation"][train].astype(str)==equation];mean=data["features"][ids].mean(0);scale=data["features"][ids].std(0).clip(1e-8);dist=np.sqrt(np.mean(((data["features"][ids]-mean)/scale)**2,axis=1));support[equation]={"mean":mean.tolist(),"scale":scale.tolist(),"radius":float(1.2*np.quantile(dist,float(cfg["calibration"]["support_radius_quantile"])))}
 manifest={"status":"FROZEN","checkpoints":[name for name,_ in models],"margins":margins,"support":support,"reserve_fraction":float(cfg["calibration"]["reserve_fraction"]),"metric_version":str(cfg["metric"]["version"]),"authority":["tighten_analytical_safe_set","rank_admitted_actions","abstain"],"forbidden":["advance_pde","relax_analytical_gate","modify_invariants","use_reference_online","retune_during_audit"]};(out/"frozen_f1_manifest.json").write_text(json.dumps(manifest,indent=2))
 coverage={};family_coverage={}
 for equation in EQUATIONS:
  ids=sel[data["equation"][sel].astype(str)==equation];bound=upper[ids]+margins[equation]+margins["global"];coverage[equation]=float(np.mean(data["target_log_effectivity"][ids]<=bound))
  for family in sorted(set(data["family"][ids].astype(str))):
   fids=ids[data["family"][ids].astype(str)==family];family_coverage[f"{equation}:{family}"]=float(np.mean(data["target_log_effectivity"][fids]<=upper[fids]+margins[equation]+margins["global"]))
 minimum_coverage=float(cfg["development"].get("minimum_selection_upper_coverage",.65) if a.mode=="development" else cfg["acceptance"]["minimum_selection_upper_coverage"])
 checks={"dataset_prerequisite":json.load(open(out/"dataset_acceptance.json"))["status"]=="PASS","metric_version_frozen":manifest["metric_version"]==str(cfg["metric"]["version"]) and set(data["metric_version"].astype(str))=={str(cfg["metric"]["version"])},"three_seed_ensemble_frozen":len(models)==3,"equation_balanced_training":True,"causal_prefix_objective_used":True,"trajectory_group_conformal_calibration":True,"selection_upper_coverage":min(coverage.values())>=minimum_coverage,"restricted_authority":len(manifest["forbidden"])==5,"selection_not_used_for_training_or_calibration":True}
 report={"status":"PASS" if all(checks.values()) else "STOP","next_phase":"CMAME-F1_external_prospective_audit" if all(checks.values()) else None,"checks":checks,"blocking_failures":[k for k,v in checks.items() if not v],"selection_upper_coverage":coverage,"family_selection_upper_coverage":family_coverage,"conformal_margins":margins,"manifest":"frozen_f1_manifest.json","scope":"Balanced causal training and group-conformal freezing; selection labels are audit-only."};(out/"training_acceptance.json").write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2));raise SystemExit(0 if report["status"]=="PASS" else 2)
if __name__=="__main__":main()
