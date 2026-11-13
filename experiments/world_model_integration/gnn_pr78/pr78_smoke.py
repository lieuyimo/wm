"""PR78 real-data TRAIN/DEV graph supervision and SHORT CPU gradient smoke.

Reconstruct ONLY original PR64 TRAIN and DEV J2 factual BusinessSimulator
episodes with the existing PR73 implementation. Selection is declared by
original SCENE and PHASE, never by target event label. Use the ORIGINAL PR61
public graph/mapping and join PR73 current observable 48d service query;
next-public state is read only to label after the input is materialized.

CPU-only initialization here is RANDOM, purposefully **not a trained PR61
checkpoint**. Research-trained PR61 weight admission is separately available
via pr78_model.admit_original_pr61_backbone(original_sha_archive,...).
No model-selection, BLIND/TEST or H20 benefit may be inferred from this smoke.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import torch
from torch_geometric.loader import DataLoader

from experiments.world_model_integration.gnn_pr78.pr78_model import (
    PublicServiceGraphHistory, PublicMapRelationalServiceWorld,
    service_event_loss, SOURCE_SCHEMA,
)
from experiments.world_model_integration.synthesis_pr73.pr73_pilot import (
    EXPECTED_TRAIN, EXPECTED_DEV, balanced_scenes, original_factual_episode,
)
from experiments.world_model_integration.outage_pr64.pr64_source import read_protocol
from experiments.world_model_integration.recovery_pr68.pr68_checkpoint import sha256_file
from experiments.world_model_integration.stochastic_pr66.pr66_cpu_admission import verify
from experiments.world_model_integration.stress_pr48.pr48_scenarios import PROTOCOL as PR48_PROTOCOL

REGIMES=("congestion","multi_fault","staggered_arrivals")
PHASES=("to_pickup","to_dropoff")
SEED=7810


def scene_graphs(scene, protocol, old, stress, max_per_phase=8):
    """At most N risk rows per phase (not N per future positive outcome)."""
    observations,actions,geometry,rules,_,_,_=original_factual_episode(
        scene,protocol,old,stress,2)
    graph_history=PublicServiceGraphHistory(geometry,rules)
    selected=[]
    per_phase=Counter()
    for t in range(min(len(actions),len(observations)-1)):
        state=observations[t]
        action=actions[t]
        g,keys=graph_history.encode(state,action)
        # Here only the current public phase/mask can determine selection.
        phases_present=[
            "to_pickup" if g.service_phase[i,0]>0.5 else "to_dropoff"
            for i in torch.nonzero(g.service_risk_mask).flatten().tolist()
        ]
        if not phases_present:
            continue
        if any(per_phase[p]>=max_per_phase for p in phases_present):
            continue
        g=PublicServiceGraphHistory.attach_factual_next_service_labels(
            g,keys,observations[t+1])
        g.scene_index=torch.tensor([0],dtype=torch.long)
        selected.append(g)
        per_phase.update(phases_present)
        if all(per_phase[p]>=max_per_phase for p in PHASES):
            break
    if not selected:
        raise AssertionError("source episode has no eligible service event")
    return selected,dict(per_phase)


def prepare_factual_j2(train_source,dev_source,quality_file,max_per_phase=8):
    if (sha256_file(train_source)!=EXPECTED_TRAIN or
        sha256_file(dev_source)!=EXPECTED_DEV):
        raise ValueError("not original PR64 TRAIN and DEV source: TEST/BLIND prohibited")
    quality=json.loads(quality_file.read_text())
    original_train=verify(train_source,quality,"train")
    original_dev=verify(dev_source,quality,"validation")
    if len(original_train["scenes"])!=24 or len(original_dev["scenes"])!=12:
        raise AssertionError("original PR64 split incomplete")
    protocol,old=read_protocol()
    stress=json.loads(PR48_PROTOCOL.read_text())
    datasets={}
    for name,original in (("train",original_train),("developer_validation",original_dev)):
        graphs=[]
        traces=[]
        cohort=balanced_scenes(original["scenes"],3)
        for scene in cohort:
            xs,counts=scene_graphs(scene,protocol,old,stress,max_per_phase=max_per_phase)
            graphs.extend(xs)
            traces.append({"scene":scene["id"],"regime":scene["case"]["regime"],
                           "n_graphs":len(xs),"n_public_service_risks":sum(counts.values()),
                           "by_public_phase":counts,
                           "next_events":sum(int(d.service_event_label[d.service_event_mask].sum()) for d in xs)})
            print("PR78_TRUE_TRAIN_DEV_PUBLIC_GNN_SERVICE",json.dumps({
                "split":name,**traces[-1]}),flush=True)
        if {t["regime"] for t in traces}!=set(REGIMES):
            raise AssertionError("regime-balanced original 3 source scenes missing")
        datasets[name]={"graphs":graphs,"scenes":traces}
    train_ids={x["scene"] for x in datasets["train"]["scenes"]}
    dev_ids={x["scene"] for x in datasets["developer_validation"]["scenes"]}
    if train_ids & dev_ids:
        raise AssertionError("source scene reused for both optimization and developer scoring")
    return datasets


def evaluate_smoke(net,graphs):
    net.eval()
    sq=0.;n=0
    with torch.inference_mode():
        for b in DataLoader(graphs,batch_size=4,shuffle=False):
            _,_,logits=net(b)
            mask=b.service_event_mask.bool()
            prob=torch.sigmoid(logits[mask])
            true=b.service_event_label[mask]
            sq+=float(((prob-true)**2).sum())
            n+=len(true)
    if n<1:
        raise AssertionError("no dev service risk rows")
    return {"n":n,"brier_smoke_random_initialized":sq/n}


def smoke_one(arm,datasets):
    if arm not in ("map_on","map_off"):
        raise ValueError("unknown map-arm control")
    torch.manual_seed(SEED)
    net=PublicMapRelationalServiceWorld(hidden=48,map_enabled=(arm=="map_on"))
    optimizer=torch.optim.Adam(net.parameters(),lr=1e-3)
    before=net.service_event_head[0].weight.detach().clone()
    net.train()
    losses=[]
    updates=0
    generator=torch.Generator().manual_seed(781000)
    for batch in DataLoader(datasets["train"]["graphs"],batch_size=4,
                            shuffle=True,generator=generator):
        _,_,logits=net(batch)
        loss=service_event_loss(logits,batch)
        if not torch.isfinite(loss):
            raise ValueError("nonfinite original PR61 service event BCE gradient")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if net.graph_world.entity_world.relation_step1.weight.grad is None:
            raise AssertionError("original RGCN graph state has no event supervision gradient")
        if arm=="map_on" and net.graph_world.map_conv1.lin_l.weight.grad is None:
            raise AssertionError("original SAGEConv map path missing event gradient")
        optimizer.step()
        updates+=1
        losses.append(float(loss))
    if updates<1 or torch.equal(before,net.service_event_head[0].weight):
        raise AssertionError("new PR78 service loss did not update weights")
    dev=evaluate_smoke(net,datasets["developer_validation"]["graphs"])
    return {"arm":arm,"optimizer_updates":updates,
            "mean_train_bce_smoke":sum(losses)/len(losses),
            "developer":dev,"real_original_PR61_pretrained_weights":False,
            "random_initialized_gradient_smoke_only":True,
            "original_backbone_parameters":net.original_pr61_parameters(),
            "new_service_head_parameters":net.service_head_parameters()}


def run(train,dev,quality,max_per_phase):
    torch.set_num_threads(1)
    data=prepare_factual_j2(train,dev,quality,max_per_phase=max_per_phase)
    report={
        "schema":SOURCE_SCHEMA,
        "original_train_sha256":sha256_file(train),
        "original_dev_sha256":sha256_file(dev),
        "no_TEST_or_BLIND":True,
        "original_PR61_training_weights_loaded":False,
        "neural_model_performance_claim":False,
        "original_PR61_graph_backbone_unchanged":True,
        "source":"original J2 BusinessSimulator existing PR64 TRAIN/DEV public episodes",
        "sample_selection":"first up to N each current public active service phase per original independently selected regime scene; never target label chosen",
        "max_per_public_phase_per_scene":max_per_phase,
        "scenes":{k:v["scenes"] for k,v in data.items()},
        "arm":{
            k:smoke_one(k,data) for k in ("map_off","map_on")
        },
        "limitations":(
            "ONE epoch of random original-GNN initialization only checks "
            "physical public input, source isolation, gradient path and "
            "architecture parity. Do not compare these smoke scores with "
            "actual trained PR61/PR74 or infer H20 forecasting benefit."
        ),
    }
    if (report["arm"]["map_off"]["original_backbone_parameters"]!=52953
        or report["arm"]["map_on"]["original_backbone_parameters"]!=52953):
        raise AssertionError("original PR61 graph architecture changed")
    return report


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--train",type=Path,required=True)
    p.add_argument("--dev",type=Path,required=True)
    p.add_argument("--quality",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--max-per-public-phase",type=int,default=8)
    a=p.parse_args()
    if not 1<=a.max_per_public_phase<=16 or a.out.exists():
        raise ValueError("invalid frozen smoke cap or output destination already exists")
    report=run(a.train,a.dev,a.quality,a.max_per_public_phase)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n")
    print("PR78_REAL_J2_GNN_SERVICE_GRADIENT_SMOKE",json.dumps({
        "train_graphs":len(report["scenes"]["train"]),
        "dev_scenes":len(report["scenes"]["developer_validation"]),
        "map_off":report["arm"]["map_off"],
        "map_on":report["arm"]["map_on"],
        "real_pretrained_model":False,
        "BLIND":False,
    }),flush=True)


if __name__=="__main__":
    main()
