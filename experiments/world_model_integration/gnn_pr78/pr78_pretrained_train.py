"""PR78 frozen-provenance original PR61 GNN service-event training entrypoint.

REQUIRES actual PR68 SHA-manifest admitted original PR61 6 best.pt archive.
No fallback to random weights and no use of the PR59 physically sealed BLIND
source. Reuse original PyG GNN backbone, PR73 real J2 public service records,
BCEWithLogits loss, Adam, scene independent split.

First controlled stage: freeze original 52,953 GNN parameters and train ONLY
the extra 48d public-query-conditioned task service-event head for each
map_on/off × original 3 seeds. This is a fair input/target alignment gate,
not a new algorithm or proof of world-model decision benefit. H5/H20 and
factual test stay unclaimed until separate pretrained evaluation.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import platform
import subprocess
import sys

import torch
from torch_geometric.loader import DataLoader
from experiments.world_model_integration.gnn_pr78.pr78_frozen_cache import (
    frozen_public_service_rows, cached_dev_metrics,
)

from experiments.world_model_integration.gnn_pr78.pr78_model import (
    PublicMapRelationalServiceWorld, service_event_loss, SOURCE_SCHEMA,
    original_gnn_state_sha256,
)
from experiments.world_model_integration.gnn_pr78.pr78_smoke import scene_graphs
from experiments.world_model_integration.synthesis_pr73.pr73_pilot import (
    EXPECTED_TRAIN, EXPECTED_DEV,
)
from experiments.world_model_integration.outage_pr64.pr64_source import read_protocol
from experiments.world_model_integration.recovery_pr68.pr68_checkpoint import (
    ARCHIVE_SHA256, load_original_models, sha256_file,
)
from experiments.world_model_integration.stochastic_pr66.pr66_cpu_admission import verify
from experiments.world_model_integration.stress_pr48.pr48_scenarios import PROTOCOL as PR48_PROTOCOL

SEEDS=(1810,1811,1812)
ARMS=("map_off","map_on")
MAX_EPOCHS=20
BATCH_SIZE=16
LR=1e-3


def full_original_public_j2_sources(train,dev,quality):
    if sha256_file(train)!=EXPECTED_TRAIN or sha256_file(dev)!=EXPECTED_DEV:
        raise ValueError("original immutable PR64 source admission failed")
    q=json.loads(quality.read_text())
    splits={
        "train":verify(train,q,"train"),
        "developer_validation":verify(dev,q,"validation")
    }
    protocol,old=read_protocol()
    stress=json.loads(PR48_PROTOCOL.read_text())
    result={}
    for name,obj in splits.items():
        expected=24 if name=="train" else 12
        scenes=obj["scenes"]
        if len(scenes)!=expected:
            raise AssertionError("original source split incomplete")
        examples=[]
        for scene in scenes:
            graphs,count=scene_graphs(
                scene,protocol,old,stress,max_per_phase=1000000)
            examples.append({
                "id":scene["id"],"regime":scene["case"]["regime"],
                "graphs":graphs,
                "n_risk_rows":sum(count.values()),
                "by_phase":count,
            })
            print("PR78_GNN_REAL_J2_SOURCE",json.dumps({
                "split":name,"scene":scene["id"],"risk_rows":sum(count.values()),
                "by_phase":count
            }),flush=True)
        result[name]=examples
    train_ids={s["id"] for s in result["train"]}
    dev_ids={s["id"] for s in result["developer_validation"]}
    if train_ids & dev_ids:
        raise AssertionError("original PR64 TRAIN/DEV scene identity leakage")
    n_train=sum(s["n_risk_rows"] for s in result["train"])
    n_dev=sum(s["n_risk_rows"] for s in result["developer_validation"])
    if (n_train,n_dev)!=(2593,1409):
        raise AssertionError("not the original PR73/74 J2 active service risk sets")
    return result


def eval_dev(net,scenes,device):
    net.eval()
    metrics=[]
    totals={"n":0,"squared_error_sum":0.,"events":0,"to_pickup":[0,0.,0],
            "to_dropoff":[0,0.,0]}
    with torch.inference_mode():
        for scene in scenes:
            sq=0.;n=0;ev=0
            for bb in DataLoader(scene["graphs"],batch_size=BATCH_SIZE,shuffle=False):
                batch=bb.to(device)
                _,_,logits=net(batch)
                mask=batch.service_event_mask.bool()
                proba=torch.sigmoid(logits[mask])
                label=batch.service_event_label[mask]
                err=(proba-label).square()
                n+=len(label)
                sq+=float(err.sum())
                ev+=int(label.sum())
                for phase,selector in (("to_pickup",0),("to_dropoff",1)):
                    ix=mask & (batch.service_phase[:,selector]>0.5)
                    if ix.any():
                        p=torch.sigmoid(logits[ix])
                        y=batch.service_event_label[ix]
                        totals[phase][0]+=len(y)
                        totals[phase][1]+=float(((p-y)**2).sum())
                        totals[phase][2]+=int(y.sum())
            if n!=scene["n_risk_rows"]:
                raise AssertionError("GNN DEV losses include unlabeled/nonactive rows")
            metrics.append({"scene":scene["id"],"regime":scene["regime"],
                            "n":n,"events":ev,"brier":sq/n})
            totals["n"]+=n
            totals["events"]+=ev
            totals["squared_error_sum"]+=sq
    return {
        "scene_count":len(metrics),
        "real_original_eligible_service_risks":totals["n"],
        "real_next_step_events":totals["events"],
        "risk_weighted_brier":totals["squared_error_sum"]/totals["n"],
        "scene_macro_brier":sum(x["brier"] for x in metrics)/len(metrics),
        "phase":{
            name:{"n":x[0],"events":x[2],"brier":x[1]/x[0]}
            for name in ("to_pickup","to_dropoff")
            for x in [totals[name]]
        },
        "per_original_scene":metrics,
    }


def train_one(arm,seed,original_pretrained,sources,device,output_dir):
    torch.manual_seed(seed+78000)
    model=PublicMapRelationalServiceWorld(hidden=48,map_enabled=(arm=="map_on")).to(device)
    if model.original_pr61_parameters()!=52953:
        raise AssertionError("original PR61 backbone changed")
    model.graph_world.load_state_dict(original_pretrained.state_dict(),strict=True)
    original_state={k:v.detach().cpu().clone() for k,v in model.graph_world.state_dict().items()}
    for parameter in model.graph_world.parameters():
        parameter.requires_grad_(False)
    # A frozen 52,953-param PR61 backbone has no reason to recompute the
    # unchanged graph history and 3d public map every training epoch.
    # Cache EXACT original 6d next-task representations + 48d current public
    # service query; never cache the factual next-tick outcome in x.
    model.graph_world.eval()
    train_cache=frozen_public_service_rows(
        model.graph_world,sources["train"],device,batch_size=BATCH_SIZE)
    dev_cache=frozen_public_service_rows(
        model.graph_world,sources["developer_validation"],device,batch_size=BATCH_SIZE)
    if (len(train_cache["labels"]),len(dev_cache["labels"]))!=(2593,1409):
        raise AssertionError("real frozen public-service risk set changed")
    features=train_cache["features"].to(device)
    targets=train_cache["labels"].to(device)
    head=model.service_event_head
    optim=torch.optim.Adam(head.parameters(),lr=LR)
    best=None
    history=[]
    for epoch in range(MAX_EPOCHS):
        head.train()
        gen=torch.Generator().manual_seed(780000+epoch+seed)
        perm=torch.randperm(len(targets),generator=gen).to(device)
        loss_sum=0.
        updates=0
        for ix in perm.split(BATCH_SIZE):
            optim.zero_grad(set_to_none=True)
            logits=head(features[ix]).flatten()
            loss=torch.nn.functional.binary_cross_entropy_with_logits(
                logits,targets[ix])
            if not torch.isfinite(loss):
                raise ValueError("nonfinite TRAIN-fitted conditional service risk BCE")
            loss.backward()
            optim.step()
            updates+=1
            loss_sum+=float(loss.detach())
        score=cached_dev_metrics(head,dev_cache,device)
        history.append({
            "epoch":epoch+1,"train_mean_bce":loss_sum/updates,
            "train_optimizer_steps":updates,
            "developer_risk_brier":score["risk_weighted_brier"],
            "developer_scene_macro_brier":score["scene_macro_brier"],
        })
        if best is None or score["scene_macro_brier"]<best["selection_score"]:
            best={
                "epoch":epoch+1,"selection_score":score["scene_macro_brier"],
                "score":score,
                "head":{k:v.detach().cpu().clone()
                        for k,v in head.state_dict().items()},
                "optimizer":copy.deepcopy(optim.state_dict()),
                "torch_rng":torch.get_rng_state().clone(),
            }
        print("PR78_REAL_PRETRAINED_GNN_CACHED_HEAD_TRAIN",json.dumps({
            "arm":arm,"seed":seed,"epoch":epoch+1,
            "dev_scene_macro_brier":score["scene_macro_brier"],
            "dev_risk_brier":score["risk_weighted_brier"],
            "original_GNN_frozen":True,
            "frozen_features_extracted_once":True,
        }),flush=True)
    for name,v in model.graph_world.state_dict().items():
        if not torch.equal(v.detach().cpu(),original_state[name]):
            raise AssertionError("original PR61 pretrained GNN weights altered in frozen-head stage")
    model.service_event_head.load_state_dict(best["head"],strict=True)
    check=cached_dev_metrics(model.service_event_head,dev_cache,device)
    # Re-run the original END-TO-END graph forward once for final assurance:
    # caching must be exact, not a silent replacement of PR61 predictions.
    full_graph_check=eval_dev(model,sources["developer_validation"],device)
    if abs(full_graph_check["risk_weighted_brier"]-check["risk_weighted_brier"])>1e-5:
        raise AssertionError("frozen original GNN cache and full model disagree")
    if abs(check["scene_macro_brier"]-best["selection_score"])>1e-7:
        raise AssertionError("best checkpoint selection not reproducible")
    name=f"{arm}_seed{seed}_best_PR78_public_service_event_head.pt"
    checkpoint=output_dir/name
    torch.save({
        "schema":SOURCE_SCHEMA,
        "original_PR61_archive_sha256":ARCHIVE_SHA256,
        "original_PR61_arm":arm,"original_PR61_seed":seed,
        "original_pretrained_backbone_unchanged":True,
        "original_pretrained_graph_state_sha256":original_gnn_state_sha256(original_state),
        "checkpoint_contains_original_GNN_tensors":False,
        "trained_new_public_service_head_state":best["head"],
        "optimizer_state":best["optimizer"],
        "torch_rng_state":best["torch_rng"],
        "cuda_rng_state_all":torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "best_epoch":best["epoch"],"train_history":history,
        "train_input_sha":EXPECTED_TRAIN,"dev_input_sha":EXPECTED_DEV,
        "source":"existing original PR64 keyed J2 generator, PR73 active risk set",
        "online_verified":False,
        "BLIND_TEST_read":False,
        "frozen_original_graph_forward_cache":True,
        "cached_graph_risk_samples":len(train_cache["labels"]),
        "new_head_optimizer_batch_is_service_risk_rows":True,
    },checkpoint)
    return {"arm":arm,"seed":seed,"best_epoch":best["epoch"],
            "original_PR61_backbone_frozen":True,
            "new_service_head_trained":True,
            "dev":check,
            "full_original_GNN_DEV_brier_for_cache_check":full_graph_check["risk_weighted_brier"],
            "cached_train_public_risks":len(train_cache["labels"]),
            "optimizer_step_unit":"active_service_risk_row",
            "checkpoint":name,"checkpoint_sha256":sha256_file(checkpoint),
            "history":history}


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--original-pr61-archive",type=Path,required=True)
    p.add_argument("--original-pr61-manifest",type=Path,required=True)
    p.add_argument("--original-pr59-validation",type=Path)
    p.add_argument("--train",type=Path,required=True)
    p.add_argument("--dev",type=Path,required=True)
    p.add_argument("--quality",type=Path,required=True)
    p.add_argument("--output-dir",type=Path,required=True)
    p.add_argument("--device",choices=("cpu","cuda"),default="cuda")
    args=p.parse_args()
    if args.output_dir.exists():
        raise FileExistsError("PR78 original-pretrained weights and training output cannot be overwritten")
    torch.set_num_threads(1)
    if args.device=="cuda" and not torch.cuda.is_available():
        raise RuntimeError("GPU train requested but CUDA is unavailable; no CPU substitution")
    device=torch.device(args.device)
    # Mandatory admission; do not use random fallback to claim trained GNN.
    original,admission=load_original_models(
        args.original_pr61_archive,args.original_pr61_manifest,
        args.original_pr59_validation)
    if (admission.get("archive_sha256")!=ARCHIVE_SHA256
        or admission.get("weights_loaded_from_original_pr61") is not True
        or len(original)!=6):
        raise AssertionError("PR61 original six-arm trained checkpoint admission failed")
    data=full_original_public_j2_sources(args.train,args.dev,args.quality)
    args.output_dir.mkdir(parents=True,exist_ok=False)
    (args.output_dir/"pip-freeze.txt").write_text(
        subprocess.check_output([sys.executable,"-m","pip","freeze"],text=True))
    (args.output_dir/"runtime_environment.json").write_text(
        json.dumps({
            "python":sys.version,
            "platform":platform.platform(),
            "torch":torch.__version__,
            "torch_cuda_build":torch.version.cuda,
            "cuda_available":torch.cuda.is_available(),
            "cuda_device":torch.cuda.get_device_name() if torch.cuda.is_available() else None,
            "original_PR61_archive_sha256":sha256_file(args.original_pr61_archive),
            "original_PR61_manifest_sha256":sha256_file(args.original_pr61_manifest),
            "original_train_source_sha256":sha256_file(args.train),
            "original_dev_source_sha256":sha256_file(args.dev),
            "original_quality_source_sha256":sha256_file(args.quality),
            "model_dataset_schema":SOURCE_SCHEMA,
            "frozen_graph_cache":True,
            "device_request":args.device,
            "cpu_threads":torch.get_num_threads(),
        },indent=2,default=str)+"\n")
    results=[]
    for arm in ARMS:
        for seed in SEEDS:
            result=train_one(arm,seed,original[(arm,seed)],data,device,args.output_dir)
            results.append(result)
    report={
        "schema":SOURCE_SCHEMA,
        "original_PR61_model_retrained":False,
        "original_PR61_backbone_all_six_frozen_and_SHA_admitted":True,
        "new_event_heads_only_trained":True,
        "model_architecture":"PR61 GRU/RGCN/SAGEConv + existing 48d public service query BCE MLP",
        "original_PR61_archive_sha256":ARCHIVE_SHA256,
        "original_PR64_train_sha256":EXPECTED_TRAIN,
        "original_PR64_dev_sha256":EXPECTED_DEV,
        "n_train_scenes":len(data["train"]),
        "n_developer_scenes":len(data["developer_validation"]),
        "train_service_risk_count":sum(x["n_risk_rows"] for x in data["train"]),
        "dev_service_risk_count":sum(x["n_risk_rows"] for x in data["developer_validation"]),
        "model_seed_arms":results,"BLIND_TEST_read":False,
        "online_verified":False,"scheduling_gain_claim":False,
        "history_training_policy":"original PR61 frozen 6d graph-task + public 48d session cached once, then BCE head-only optimization",
        "next_gate":"paired 12-scene Brier versus PR74 Deep Sets and PR73 TRAIN hazard, then lawful particle distribution H5/H20",
    }
    (args.output_dir/"report.json").write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n")
    print("PR78_ALL_SIX_REAL_PR61_GRAPH_SERVICE_HEADS_FINISHED",json.dumps({
        "train":report["train_service_risk_count"],
        "dev":report["dev_service_risk_count"],
        "checkpoints":len(results),"BLIND":False,
        "original_backbone_retrained":False,
    }),flush=True)


if __name__=="__main__":
    main()
