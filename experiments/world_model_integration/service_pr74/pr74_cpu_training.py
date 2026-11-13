"""PR74: existing Deep Sets session-event architecture on original-keyed jitter data.

Reuse lab_transport.business_session_learning.BusinessSessionEventModel,
encode_session_batch, metrics, and its BCE/Adam optimization pattern. True
BusinessSimulator service jitter is the unchanged original API from PR73.

This is a CPU-only, single-seed, development-split training *pilot*. It does
NOT load or claim to continue PR61 pretrained graph weights, does not invent a
new world model architecture, and does not open BLIND/TEST. Model checkpoint
is offline experimental and must never be promoted into live dispatch.
"""
from __future__ import annotations

import argparse
import copy
from collections import defaultdict
import hashlib
import json
import random
from pathlib import Path

import torch
from torch import nn

from business_transport.state_features import build_business_features, SCHEMA_VERSION
from business_transport.session_events import QUERY_LABEL_VERSION
from business_transport.session_events import eligible_sessions
from business_transport.dynamics import WorkflowDynamics
from experiments.world_model_integration.synthesis_pr73.pr73_pilot import (
    EXPECTED_TRAIN, EXPECTED_DEV, original_factual_episode, PHASES,
)
from experiments.world_model_integration.synthesis_pr73.pr73_hazard_probe import (
    evaluate_train_dev, fit_train_only, probabilities as hazard_probabilities,
)
from experiments.world_model_integration.outage_pr64.pr64_source import read_protocol
from experiments.world_model_integration.recovery_pr68.pr68_checkpoint import sha256_file
from experiments.world_model_integration.stochastic_pr66.pr66_cpu_admission import verify
from experiments.world_model_integration.stress_pr48.pr48_scenarios import PROTOCOL as PR48_PROTOCOL
from lab_transport.business_session_learning import (
    BusinessSessionEventModel, encode_session_batch, metrics, load_session_model,
    MODEL_VERSION,
)
from lab_transport.torch_structured_adapter import ProgressAwareRule

EPOCHS = 20
BATCH_SIZE = 64
LR = 0.001
SEED = 1810
JITTER = 2


def emit_scene(scene, protocol, old, stress):
    obs, actions, geometry, rules, _, _, _ = original_factual_episode(
        scene, protocol, old, stress, JITTER)
    rule = WorkflowDynamics(ProgressAwareRule(geometry), rules)
    rows, metadata = [], []
    for t in range(min(len(actions),len(obs)-1)):
        before, future = obs[t:t+2]
        action = actions[t]
        viable = [q for q, reason in eligible_sessions(
            before, action, require_active=True) if reason is None]
        if not viable:
            continue
        payload = build_business_features(before, action, geometry, rules)
        for key in ("true_remaining_service_countdown", "future_fault_schedule",
                    "unreleased_external_jobs", "counterfactual_outcome"):
            if payload["unknown"][key]["known"]:
                raise AssertionError("private future information exposed as model input")
        actual = {s.id:s for s in future.samples}
        baseline = {s.id:s for s in rule.predict_next(before,action).samples}
        robots = {r.id:r for r in before.robots}
        for query in viable:
            sid = query["sample_id"]
            if sid not in actual or sid not in baseline:
                continue
            status = PHASES[query["phase"]]
            y = int(actual[sid].status == status)
            row = {
                "features": payload, "query": query,
                "label": y, "trajectory_group": scene["id"],
                "public_tick": before.tick,
            }
            rows.append(row)
            active = robots[query["robot_id"]].service_active_elapsed
            if not isinstance(active,int) or active < 0:
                raise AssertionError("event risk does not have public active elapsed")
            metadata.append({
                "scene":scene["id"], "phase":query["phase"],
                "station":query["station_id"], "active_elapsed": active,
                "event":y, "rule_event": int(baseline[sid].status==status),
            })
    if not rows or len(rows)!=len(metadata):
        raise AssertionError("no original eligible service-session observations")
    return rows, metadata


def build_source(train_path,dev_path,quality_path):
    if sha256_file(train_path) != EXPECTED_TRAIN or sha256_file(dev_path) != EXPECTED_DEV:
        raise ValueError("different source, BLIND forbidden")
    q = json.loads(quality_path.read_text())
    tr = verify(train_path,q,"train")
    de = verify(dev_path,q,"validation")
    if len(tr["scenes"])!=24 or len(de["scenes"])!=12:
        raise AssertionError("source must include all original PR64 development cohorts")
    protocol, old = read_protocol()
    stress = json.loads(PR48_PROTOCOL.read_text())
    datasets={}
    for name, source in (("train", tr),("developer_validation",de)):
        groups = []
        meta = []
        for scene in source["scenes"]:
            rows, risks = emit_scene(scene, protocol, old, stress)
            groups.extend(rows)
            meta.extend(risks)
            print("PR74_SOURCE_SCENE",json.dumps({
                "split":name,"scene":scene["id"],
                "events":sum(r["label"] for r in rows),
                "queries":len(rows)}),flush=True)
        datasets[name]={"rows":groups,"metadata":meta}
    if ({r["trajectory_group"] for r in datasets["train"]["rows"]} &
        {r["trajectory_group"] for r in datasets["developer_validation"]["rows"]}):
        raise AssertionError("same full episode leaked into both splits")
    return datasets


def scene_macro(devrows, predictions, metadata, comparator):
    if len(predictions)!=len(devrows) or len(metadata)!=len(devrows):
        raise ValueError("model and same public DEV rows not aligned")
    by_scene=defaultdict(list)
    for i,row in enumerate(metadata):
        by_scene[row["scene"]].append(i)
    scores={}
    for scene,ids in sorted(by_scene.items()):
        vals=[(float(predictions[i])-float(devrows[i]["label"]))**2 for i in ids]
        rule=[(metadata[i]["rule_event"]-metadata[i]["event"])**2 for i in ids]
        value={"n":len(ids),"brier":sum(vals)/len(ids),
               "rule_brier":sum(rule)/len(rule)}
        if comparator is not None:
            hazard_scores=[
                (hazard_probabilities(comparator,metadata[i])["public_elapsed_hazard"]
                 - metadata[i]["event"])**2 for i in ids
            ]
            value["train_hazard_brier"]=sum(hazard_scores)/len(hazard_scores)
        scores[scene]=value
    answer={
        "n_scenes":len(scores),
        "scene_macro_model_brier":sum(v["brier"] for v in scores.values())/len(scores),
        "scene_macro_rule_brier":sum(v["rule_brier"] for v in scores.values())/len(scores),
        "per_scene":scores,
    }
    if comparator is not None:
        answer["scene_macro_train_hazard_brier"]=(
            sum(v["train_hazard_brier"] for v in scores.values())/len(scores))
        rng=random.Random(740001)
        def delta(target, seed_offset=0):
            diffs=[v["brier"]-v[target] for v in scores.values()]
            boots=sorted(
                sum(diffs[rng.randrange(len(diffs))] for _ in diffs)/len(diffs)
                for _ in range(2000)
            )
            return {"mean":sum(diffs)/len(diffs),
                    "CI95":[boots[49],boots[1949]],
                    "scenes_neural_better":sum(x<0 for x in diffs)}
        answer["paired_neural_minus_rule"]=delta("rule_brier")
        answer["paired_neural_minus_train_hazard"]=delta("train_hazard_brier")
        answer["bootstrap_independence_unit"]="original DEV scene, not service-row"
    return answer


def train_and_score(dataset,train_sha,dev_sha,output_dir):
    output_dir.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1)
    torch.manual_seed(SEED)
    random.seed(SEED)
    rows=dataset["train"]["rows"]
    dev=dataset["developer_validation"]["rows"]
    train_meta=dataset["train"]["metadata"]
    dev_meta=dataset["developer_validation"]["metadata"]
    if len(rows)<1000 or len(dev)<500:
        raise AssertionError("source quality lacks event-positive/negative support")
    if set(r["label"] for r in rows)!={0,1} or set(r["label"] for r in dev)!={0,1}:
        raise AssertionError("event labels must contain both outcomes in each split")
    x,m,q=encode_session_batch(rows)
    vx,vm,vq=encode_session_batch(dev)
    if q.shape!=(len(rows),48) or vq.shape!=(len(dev),48):
        raise AssertionError("original v2 public query contract mismatch")
    labels=torch.tensor([r["label"] for r in rows],dtype=torch.float32)
    vy=torch.tensor([r["label"] for r in dev],dtype=torch.float32)
    model=BusinessSessionEventModel(query_width=48).cpu()
    optimizer=torch.optim.Adam(model.parameters(),lr=LR)
    epochs=[]
    best=None
    for epoch in range(1,EPOCHS+1):
        model.train()
        index=torch.randperm(len(rows),generator=torch.Generator().manual_seed(74000+epoch))
        losses=[]
        for subset in index.split(BATCH_SIZE):
            optimizer.zero_grad(set_to_none=True)
            output=model(x[subset],m[subset],q[subset])
            loss=nn.functional.binary_cross_entropy_with_logits(output,labels[subset])
            if not torch.isfinite(loss):
                raise ValueError("nonfinite original Deep Sets service loss")
            loss.backward()
            optimizer.step()
            losses.append((float(loss),len(subset)))
        model.eval()
        with torch.inference_mode():
            pred=torch.sigmoid(model(vx,vm,vq))
            val=float(((pred-vy)**2).mean())
        epochs.append({"epoch":epoch,"train_bce":sum(loss*n for loss,n in losses)/len(rows),
                       "developer_validation_brier":val,
                       "real_optimizer_updates":len(losses)})
        if best is None or val<best["brier"]:
            best={"epoch":epoch,"brier":val,
                  "weights":{k:v.detach().clone() for k,v in model.state_dict().items()},
                  "optimizer":copy.deepcopy(optimizer.state_dict()),
                  "torch_rng":torch.get_rng_state()}
    model.load_state_dict(best["weights"])
    model.eval()
    with torch.inference_mode():
        probs=torch.sigmoid(model(vx,vm,vq)).cpu().tolist()
    original_metrics=metrics(dev,probs)
    per_phase={}
    for phase in PHASES:
        ids=[i for i,r in enumerate(dev_meta) if r["phase"]==phase]
        per_phase[phase]=metrics([dev[i] for i in ids],[probs[i] for i in ids])
    hazard=evaluate_train_dev(train_meta,dev_meta)
    paired=scene_macro(dev,probs,dev_meta,fit_train_only(train_meta))
    checkpoint={
        "metadata":{
            "source":"PR73 same existing simulator hashed jitter=2 only",
            "architecture":"original lab_transport.business_session_learning.BusinessSessionEventModel 48d",
            "model_version": MODEL_VERSION,
            "schema_version": SCHEMA_VERSION,
            "label_version": QUERY_LABEL_VERSION,
            "train_source_sha256":train_sha,"dev_source_sha256":dev_sha,
            "train_scenes":24,"developer_validation_scenes":12,
            "seed":SEED,"epochs":EPOCHS,"batch":BATCH_SIZE,"lr":LR,
            "best_epoch_developer_selected":best["epoch"],
            "online_verified":False,"BLIND_TEST_read":False,
            "not_PR61_continued_training":True,
        },
        "model_state":best["weights"],
        "optimizer_state":best["optimizer"],
        "torch_rng":best["torch_rng"],
    }
    ckpt=output_dir/"existing_deepsets_public_service_cpu_best.pt"
    torch.save(checkpoint,ckpt)
    admitted, admitted_meta = load_session_model(ckpt,experimental=True)
    if (admitted_meta["online_verified"] or
        admitted.target[0].in_features != 48):
        raise AssertionError("original SessionRuleDynamics offline experimental v2 load mismatch")
    report={
        "schema":"pr74-existing-DeepSets-v2-CPU-development-smoke-v1",
        "old_PR61_checkpoint_loaded":False,
        "new_architecture_invented":False,
        "existing_model_class":"lab_transport.business_session_learning.BusinessSessionEventModel",
        "feature_encoding":"existing build_business_features + encode_session_batch(48d)",
        "source_sha256":{"train":train_sha,"developer_validation":dev_sha},
        "sample_splits":{
            "train_risk_rows":len(rows),"train_events":int(labels.sum()),
            "dev_risk_rows":len(dev),"dev_events":int(vy.sum()),
            "distinct_train_scenes":len({r["trajectory_group"] for r in rows}),
            "distinct_dev_scenes":len({r["trajectory_group"] for r in dev}),
        },
        "optimization":{"seed":SEED,"epochs":EPOCHS,"batch_size":BATCH_SIZE,
                        "learning_rate":LR,"best_epoch":best["epoch"],
                        "history":epochs},
        "neural_developer_metrics":original_metrics,
        "neural_developer_per_phase":per_phase,
        "neural_scene_macro":paired,
        "original_rule_and_train_hazard_controls":hazard,
        "saved_checkpoint_sha256":hashlib.sha256(ckpt.read_bytes()).hexdigest(),
        "no_gpu":True,"BLIND_TEST_read":False,
        "offline_only":True,
        "limitations":"Developer-val checkpoint selection only; no held-out TEST, H20 or online decision benefit.",
    }
    (output_dir/"result.json").write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n")
    print("PR74_EXISTING_DEEPSETS_CPU_MODEL_DEVELOPMENT_RESULT",json.dumps({
        "metrics":original_metrics,
        "per_phase":{k:{"n":v["count"],"brier":v["brier"]} for k,v in per_phase.items()},
        "baseline":{k:v["brier"] for k,v in hazard["method"].items()},
        "best_epoch":best["epoch"],
        "scene_paired":paired,
        "train_scenes":24,"dev_scenes":12,
        "checkpoint_sha256":report["saved_checkpoint_sha256"],
        "new_model_architecture":False,"BLIND":False,
    }),flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--train",type=Path,required=True)
    p.add_argument("--dev",type=Path,required=True)
    p.add_argument("--quality",type=Path,required=True)
    p.add_argument("--output-dir",type=Path,required=True)
    a=p.parse_args()
    if a.output_dir.exists():
        raise FileExistsError("original model evidence directory must not be overwritten")
    data=build_source(a.train,a.dev,a.quality)
    train_and_score(data,sha256_file(a.train),sha256_file(a.dev),a.output_dir)


if __name__=="__main__":
    main()
