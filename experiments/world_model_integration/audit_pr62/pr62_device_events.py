"""PR62 checkpoint-dependent CONDITIONAL device-event validation, NO training.

**Do not claim this audit was run from GitHub's PR61 alone.** PR61 commits
only compact per-scenario CSV + SHA hashes, not the actual best.pt weights.
An authorized agent on the GPU host must restore the trusted 6.66MB local
archive and invoke this reader against the EXISTING 12 PR59 validation
episodes. Never generate, read, or evaluate BLIND TEST. No optimization.

Five groups:
    current device UP, next UP: healthy persistence
    current UP, next DOWN: observed new failure
    current DOWN, next DOWN: continuing outage
    current DOWN, next UP: actual recovery
This specifically avoids treating a low overall device Brier, dominated
by healthy/still states, as evidence of outage/recovery forecasting skill.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import argparse
import csv
import hashlib
import json

import torch
from torch_geometric.loader import DataLoader

from experiments.world_model_integration.sealed_pr59.pr59_gpu_train import load_actual_source
from experiments.world_model_integration.sealed_pr59.pr59_dual_model import PublicMapTemporalRelationalWorld

EXPECTED_SEEDS={1810,1811,1812}
ARMS={"map_on","map_off"}
TYPES=("healthy_hold","fault_onset","continuing_down","recovery")


def sha256(p):
    h=hashlib.sha256()
    with p.open("rb") as f:
        for block in iter(lambda:f.read(1<<20),b""):
            h.update(block)
    return h.hexdigest()


def read_ckpts(root,manifest,validation_sha,device):
    rows=list(csv.DictReader(manifest.open(newline="")))
    if len(rows)!=6:
        raise AssertionError("six source checkpoint selections required")
    found={}
    for row in rows:
        arm=row["arm"]
        seed=int(row["seed"])
        if arm not in ARMS or seed not in EXPECTED_SEEDS or (arm,seed) in found:
            raise AssertionError("checkpoint manifest duplicate or unregistered model")
        matches=list(root.rglob(f"{arm}/{seed}/best.pt"))
        if len(matches)!=1:
            raise AssertionError(f"missing trusted best.pt under local checkpoint archive: {arm}/{seed}")
        path=matches[0]
        if sha256(path)!=row["best_sha256"]:
            raise AssertionError(f"checkpoint not same as PR61 manifest: {arm}/{seed}")
        # Source is explicitly trusted user-supplied GPU archive, SHA verified.
        # torch.load(weights_only=False) is unsafe for untrusted checkpoints.
        checkpoint=torch.load(path,map_location="cpu",weights_only=False)
        if (checkpoint["arm"]!=arm or checkpoint["seed"]!=seed
            or checkpoint["git_head_source"]!="e32c402ecbf67d19788c0a5c99e5282d6b5017e5"
            or validation_sha not in checkpoint["source_sha256"]["validation"].values()):
            raise AssertionError("model weights not trained/selected on approved frozen source")
        model=PublicMapTemporalRelationalWorld(
            hidden=checkpoint["hidden"],map_enabled=(arm=="map_on"),
            history_enabled=True)
        model.load_state_dict(checkpoint["model_state_dict"],strict=True)
        model=model.to(device).eval()
        found[(arm,seed)]=(model,path)
    if len(found)!=6:
        raise AssertionError("not all original selected model weights restored")
    return found


def station_stage(x,y):
    if x>=0.5 and y>=0.5:
        return "healthy_hold"
    if x>=0.5 and y<0.5:
        return "fault_onset"
    if x<0.5 and y<0.5:
        return "continuing_down"
    return "recovery"


def score_group(records):
    n=len(records)
    if not n:
        return {"n":0,"available":False}
    brier=sum((p-y)**2 for p,y,x in records)/n
    persistence=sum((round(x)-y)**2 for p,y,x in records)/n
    return {
        "n":n,
        "available":True,
        "next_device_available_positive":sum(y>=0.5 for p,y,x in records),
        "predicted_mean_next_available":sum(p for p,y,x in records)/n,
        "empirical_next_available":sum(y for p,y,x in records)/n,
        "model_brier":brier,
        "trivial_previous_device_state_brier":persistence,
        "model_minus_persistence_brier":brier-persistence
    }


def validate(args):
    out=args.output
    if out.exists():
        raise FileExistsError("never overwrite original conditional-event evidence")
    if args.device=="cuda" and not torch.cuda.is_available():
        raise RuntimeError("GPU Agent must provide a real CUDA device, or explicitly request CPU-only diagnostic")
    torch.set_num_threads(args.cpu_threads)
    device=args.device
    val=Path(args.validation)
    episodes,sha=load_actual_source([val],"validation")
    if len(episodes)!=12 or {x["family"] for x in episodes}!={"diagonal_staggered_validation"}:
        raise AssertionError("not the frozen natural 12 PR59 validation scenes")
    ckpts=read_ckpts(
        args.trusted_checkpoints,Path(args.manifest),sha256(val),device
    )
    reports=[]
    for (arm,seed),(model,path) in sorted(ckpts.items()):
        by_stage=defaultdict(list)
        by_regime=defaultdict(lambda:defaultdict(list))
        by_map=defaultdict(lambda:defaultdict(list))
        by_scenario=defaultdict(lambda:defaultdict(list))
        model.eval()
        with torch.no_grad():
            for ep in episodes:
                # Source graphs only contain public t-3:t state and the
                # actual legal t dispatch. True t+1 is a label only.
                for batch in DataLoader(ep["graphs"],batch_size=args.batch_size,shuffle=False):
                    b=batch.to(device)
                    pred,_=model(b)
                    station=((b.x[:,2]>0.5)&(b.target_mask[:,0]>0.5))
                    selected=station.nonzero(as_tuple=False).flatten()
                    for index in selected.tolist():
                        p=float(pred[index,0].item())
                        y=float(b.node_target[index,0].item())
                        x=float(b.x[index,6].item())  # current PUBLIC device_available
                        if not (0<=p<=1 and y in (0.,1.) and x in (0.,1.)):
                            raise AssertionError("invalid actual public device state/probability")
                        phase=station_stage(x,y)
                        row=(p,y,x)
                        by_stage[phase].append(row)
                        by_regime[ep["regime"]][phase].append(row)
                        # All validation scenes are in diagonal-staggered
                        # family; exact map = original immutable episode id.
                        map_id=ep["id"].split("_")[2]
                        by_map[map_id][phase].append(row)
                        by_scenario[ep["id"]][phase].append(row)
        group={kind:score_group(by_stage[kind]) for kind in TYPES}
        if not all(group[k]["n"]>0 for k in TYPES):
            raise AssertionError("frozen validation source lacks a required transition phase")
        total=[v for s in TYPES for v in by_stage[s]]
        n=len(total)
        reports.append({
            "arm":arm,"seed":seed,"sha256_selected_checkpoint":sha256(path),
            "total_station_tick_labels":n,
            "overall":score_group(total),
            "conditioned_on_TRUE_transition_phase":group,
            "by_regime":{
                regime:{kind:score_group(phase.get(kind,[])) for kind in TYPES}
                for regime,phase in by_regime.items()},
            "by_validation_map":{
                m:{kind:score_group(phase.get(kind,[])) for kind in TYPES}
                for m,phase in by_map.items()},
            "by_true_validation_scenario":{
                ident:{kind:score_group(phase.get(kind,[])) for kind in TYPES}
                for ident,phase in by_scenario.items()},
        })
        print("PR62_DEVICE_EVENT_CONDITIONAL_SELECTED_MODEL",json.dumps({
            "arm":arm,"seed":seed,
            "counts":{k:group[k]["n"] for k in TYPES},
            "overall_brier":score_group(total)["model_brier"],
            "recovery_brier":group["recovery"]["model_brier"],
            "fault_onset_brier":group["fault_onset"]["model_brier"]
        }),flush=True)
    assert len(reports)==6
    result={
        "schema":"pr62-pr61-checkpoint-trusted-true-one-step-device-event-conditional-audit-v1",
        "original_GPU_PR":"https://github.com/lieuyimo/wm/pull/61",
        "frozen_source_validation_sha256":sha256(val),
        "frozen_validation_scenarios":12,
        "BLIND_TEST_read":False,
        "BLIND_TEST_generated":False,
        "new_training_performed":False,
        "source_model_parameter_updates":0,
        "trusted_weights_only_with_PR61_sha":True,
        "conditional_groups":list(TYPES),
        "important_limit":"Grouping on TRUE next-step event is descriptive conditional error, not model-available future feature and not an online threshold. Overall Brier cannot establish rare failure/recovery forecast skill. No H20/autoregressive inference was run.",
        "results":reports
    }
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    print("PR62_CONDITIONAL_DEVICE_REPORT_WRITTEN",str(out),flush=True)


def main():
    pa=argparse.ArgumentParser()
    pa.add_argument("--validation",type=Path,required=True)
    pa.add_argument("--trusted-checkpoints",type=Path,required=True)
    pa.add_argument("--manifest",type=Path,required=True)
    pa.add_argument("--output",type=Path,required=True)
    pa.add_argument("--device",choices=("cuda","cpu"),default="cuda")
    pa.add_argument("--batch-size",type=int,default=32)
    pa.add_argument("--cpu-threads",type=int,default=2)
    args=pa.parse_args()
    validate(args)


if __name__=="__main__":
    main()
