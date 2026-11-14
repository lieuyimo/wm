"""PR78 CPU-only admission of the FULL existing J2 graph service target dataset.

This does NOT initialize or train any graph network or read original PR61
checkpoint, BLIND/TEST, or future simulator private fields. It regenerates
all original PR64 TRAIN24 / DEV12 public risk sequences and verifies the
chronological 16d PR61 graph plus current public 48d service input and t+1
factual event targets against the PR73/74 published exact same risk counts.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import torch

from experiments.world_model_integration.gnn_pr78.pr78_pretrained_train import (
    full_original_public_j2_sources,
)
from experiments.world_model_integration.gnn_pr78.pr78_model import SOURCE_SCHEMA


def check(sources):
    result={}
    for name,scenes in sources.items():
        counters=Counter()
        if len({s["id"] for s in scenes})!=len(scenes):
            raise AssertionError("duplicate original factual scene")
        for scene in scenes:
            if not scene["graphs"]:
                raise AssertionError("missing public event source graph")
            for g in scene["graphs"]:
                if g.x.shape[1]!=16 or g.history.shape[1:]!=(4,16):
                    raise AssertionError("original PR61 graph/history changed")
                if g.service_public_query.shape!=(len(g.x),48):
                    raise AssertionError("current public 48d event source changed")
                if not torch.equal(g.service_event_mask,g.service_risk_mask):
                    raise AssertionError("new event target includes nonexistent service")
                if g.service_label_public_tick.item()!=g.service_public_tick.item()+1:
                    raise AssertionError("future event label not at exact next public tick")
                m=g.service_risk_mask
                if torch.any(g.service_public_query[~m]!=0):
                    raise AssertionError("public query accidentally attached to inactive node")
                for phase,i in (("pickup",0),("delivery",1)):
                    ix=m & (g.service_phase[:,i]>0.5)
                    if not ix.any():
                        continue
                    counters[phase+"_risks"]+=int(ix.sum())
                    counters[phase+"_events"]+=int(g.service_event_label[ix].sum())
                counters["eligible_risks"]+=int(m.sum())
                counters["true_events"]+=int(g.service_event_label[m].sum())
                counters["graphs"]+=1
        result[name]={
            "independent_scenes":len(scenes),
            "regimes":dict(Counter(s["regime"] for s in scenes)),
            "by_phase_and_event":dict(counters),
            "source_only_public_current_service":True,
        }
    if (result["train"]["by_phase_and_event"]["eligible_risks"],
        result["developer_validation"]["by_phase_and_event"]["eligible_risks"])!=(2593,1409):
        raise AssertionError("full PR73/74 real active-session risk set differs")
    if (result["train"]["by_phase_and_event"]["true_events"],
        result["developer_validation"]["by_phase_and_event"]["true_events"])!=(1106,594):
        raise AssertionError("actual event labels differ from original PR74")
    if result["train"]["by_phase_and_event"]["pickup_risks"]!=1553:
        raise AssertionError("original PR73 TRAIN pickup count changed")
    if result["developer_validation"]["by_phase_and_event"]["pickup_risks"]!=831:
        raise AssertionError("original PR73 DEV pickup count changed")
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--train",type=Path,required=True)
    p.add_argument("--dev",type=Path,required=True)
    p.add_argument("--quality",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args()
    if args.out.exists():
        raise FileExistsError("cannot overwrite original graph event source audit")
    sources=full_original_public_j2_sources(args.train,args.dev,args.quality)
    report={
        "schema":"pr78-full-real-J2-public-GNN-training-source-audit-v1",
        "model_feature_schema":SOURCE_SCHEMA,
        "original_PR61_architecture_unchanged":True,
        "original_PR61_pretrained_checkpoint_loaded":False,
        "model_training_or_Brier_claim":False,
        "BLIND_TEST_read":False,
        "full_original_source":check(sources),
    }
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n")
    print("PR78_FULL_ORIGINAL_J2_PUBLIC_GNN_SERVICE_TARGET_ADMISSION_PASS",
          json.dumps(report["full_original_source"]),flush=True)


if __name__=="__main__":
    main()
