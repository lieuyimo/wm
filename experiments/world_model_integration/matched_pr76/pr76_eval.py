"""PR76 actual J2 factual-world H5/H20 conditional prediction benchmark.

Everything from the original PR73/74 input and original PR64 development
scenes; 3 pretrained/control arms on FROZEN actual public factual actions
and FROZEN actual next-device boundary, and target the actual next public
observation. No learning, oracle stage2 injection, hidden service time,
or post-hoc switching to own-action rollout.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random

import torch

from experiments.world_model_integration.synthesis_pr73.pr73_pilot import (
    EXPECTED_TRAIN,EXPECTED_DEV,original_factual_episode,
)
from experiments.world_model_integration.service_pr74.pr74_cpu_training import (
    build_source,
)
from experiments.world_model_integration.outage_pr64.pr64_source import (
    read_protocol,
)
from experiments.world_model_integration.recovery_pr68.pr68_checkpoint import sha256_file
from experiments.world_model_integration.stochastic_pr66.pr66_cpu_admission import verify
from experiments.world_model_integration.stress_pr48.pr48_scenarios import PROTOCOL as PR48_PROTOCOL
from experiments.world_model_integration.matched_pr76.pr76_factual_contract import (
    ARMS,HORIZONS,PARTICLE_SEEDS,ANCHORS,load_exact_existing_model,
    factual_conditioned_rollout,first_active_anchor,frozen_teacher_forced,
)


def summarize_conditioned(runs,horizon,anchor):
    subset=[r for r in runs if r["horizon"]==horizon and r["anchor_name"]==anchor]
    expected=12*len(PARTICLE_SEEDS)
    by_arm={}
    for arm in ARMS:
        items=[r for r in subset if r["arm"]==arm]
        if len(items)!=expected:
            raise AssertionError(f"missing true-world {arm}/{horizon}/{anchor} episodes")
        good=[r for r in items if r["full_horizon"]]
        fails=Counter(r["failure_kind"] for r in items if not r["full_horizon"])
        lengths=[r["legal_steps"] for r in items]
        by_arm[arm]={
            "attempts":len(items),"full_horizon":len(good),
            "mean_lawful_steps":sum(lengths)/len(lengths),
            "failure_reasons":dict(fails),
            "model_query_calls":sum(r["offline_model_service_queries"] for r in items),
            "full_horizon_terminal_status_error_mean":(
                sum(r["errors"][-1]["task_status_error_rate"] for r in good)/len(good)
                if good else None),
            "full_horizon_terminal_completed_absolute_error_mean":(
                sum(r["errors"][-1]["completed_abs_error"] for r in good)/len(good)
                if good else None),
        }
    matched_by_key=defaultdict(dict)
    for r in subset:
        key=(r["scene"],r["seed"])
        if r["arm"] in matched_by_key[key]:
            raise AssertionError("duplicate matched factual world per arm")
        matched_by_key[key][r["arm"]]=r
    if len(matched_by_key)!=expected or any(set(g)!=set(ARMS) for g in matched_by_key.values()):
        raise AssertionError("unbalanced actual actions/exogenous source")
    fully_matched={key:group for key,group in matched_by_key.items()
                   if all(r["full_horizon"] for r in group.values())}
    # Include EVERY scenario/particle in the shared-prefix analysis: no post
    # hoc deletion of all trajectories where one arm loses factual action
    # executability. Cannot treat a short prefix as H20 accuracy.
    prefixes=[]
    for key,group in sorted(matched_by_key.items()):
        depth=min(g["legal_steps"] for g in group.values())
        prefixes.append({
            "scene":key[0],"seed":key[1],"depth":depth,
            "status_error":{
                arm: (sum(e["task_status_error_rate"] for e in group[arm]["errors"][:depth])/depth
                      if depth else None)
                for arm in ARMS
            },
        })
    common_depths=[x["depth"] for x in prefixes]
    result={
        "attempted_matched_scene_seed_triples":len(matched_by_key),
        "fully_action_device_matched_triples":len(fully_matched),
        "shared_prefix_zero_steps":sum(d==0 for d in common_depths),
        "shared_prefix_mean_legal_steps":sum(common_depths)/len(common_depths),
        "arm":by_arm,
        "all_samples_scored_against_factual_public_t_plus_one":True,
        "no_imputation_after_unexecutable_factual_action":True,
        "no_conclusions_from_unbalanced_complete_arm_subsets":True,
        "per_scene_seed_shared_prefix":prefixes,
    }
    if fully_matched:
        # Same actual source actions, same factual device boundary, same
        # matched anchors/particles. Per-scene grouping is the confidence unit.
        per_scene=defaultdict(list)
        for (scene,seed),group in sorted(fully_matched.items()):
            per_scene[scene].append({
                arm:group[arm]["errors"][-1]["task_status_error_rate"]
                for arm in ARMS
            })
        result["full_horizon_matched_terminal"]={
            "n_independent_scenes":len(per_scene),
            "per_scene":{
                scene:{arm:sum(r[arm] for r in rows)/len(rows) for arm in ARMS}
                for scene,rows in per_scene.items()
            },
            "all_pairs_usable":len(fully_matched)==len(matched_by_key),
        }
        if len(per_scene)==12:
            rng=random.Random(76000+horizon+(0 if anchor=="initial_t0" else 31))
            paired={}
            for comp in ("rule","train_public_hazard"):
                diffs=[
                    (sum(z["existing_deepsets"]-z[comp] for z in rows)/len(rows))
                    for scene,rows in sorted(per_scene.items())
                ]
                boots=sorted(
                    sum(diffs[rng.randrange(len(diffs))] for _ in diffs)/len(diffs)
                    for _ in range(2000))
                paired[f"neural_minus_{comp}"]={
                    "scene_macro_delta":sum(diffs)/len(diffs),
                    "scene_bootstrap_95CI":[boots[49],boots[1949]],
                    "neural_lower_error_in_scenes":sum(v<0 for v in diffs),
                }
            result["full_horizon_matched_terminal"]["scene_bootstrap"]=paired
        else:
            result["full_horizon_matched_terminal"]["scene_bootstrap"]={
                "available":False,"reason":"missing independent factual DEV scenes under all-arm action consistency"
            }
    else:
        result["full_horizon_matched_terminal"]={
            "available":False,"reason":"no all-arm action-matched full horizon"
        }
    return result


def run_all(original_train,original_dev,original_quality,checkpoint):
    if sha256_file(original_train)!=EXPECTED_TRAIN or sha256_file(original_dev)!=EXPECTED_DEV:
        raise ValueError("original PR64 TRAIN/DEV SHA mismatch")
    neural_model,pretrained_meta=load_exact_existing_model(checkpoint)
    dataset=build_source(original_train,original_dev,original_quality)
    train_rows=dataset["train"]["rows"]
    train_meta=dataset["train"]["metadata"]
    dev_rows=dataset["developer_validation"]["rows"]
    dev_meta=dataset["developer_validation"]["metadata"]
    if len(train_rows)!=2593 or len(dev_rows)!=1409:
        raise AssertionError("original PR74 J2 event dataset has unexpectedly changed")
    teacher=frozen_teacher_forced(dev_rows,dev_meta,neural_model,train_meta)
    original_quality_obj=json.loads(original_quality.read_text())
    source=verify(original_dev,original_quality_obj,"validation")
    if len(source["scenes"])!=12:
        raise AssertionError("missing original J2 developer scenes")
    protocol,old=read_protocol()
    stress=json.loads(PR48_PROTOCOL.read_text())
    runs=[]
    torch.set_num_threads(1)
    for scene in source["scenes"]:
        obs,acts,geometry,rules,truth,starts,durations=original_factual_episode(
            scene,protocol,old,stress,2)
        anchors={"initial_t0":0,
                 "first_active_service":first_active_anchor(obs,acts,max(HORIZONS))}
        for anchor_name,anchor_tick in anchors.items():
            if anchor_tick is None:
                raise AssertionError("source has no eligible first active J2 H20 anchor")
            for h in HORIZONS:
                for seed in PARTICLE_SEEDS:
                    for arm in ARMS:
                        record=factual_conditioned_rollout(
                            obs,acts,geometry,rules,checkpoint,train_meta,
                            arm,seed,anchor_tick,h)
                        record.update({
                            "scene":scene["id"],
                            "anchor_name":anchor_name,
                            "original_seed":scene["case"]["seed"],
                            "truth_origin":"PR64 DEV factual BusinessSimulator existing J2 service variation",
                        })
                        runs.append(record)
                        print("PR76_TRUE_MATCHED_ACTION_DEVICE_H",json.dumps({
                            "scene":scene["id"],"anchor":anchor_name,
                            "arm":arm,"h":h,"seed":seed,
                            "legal_steps":record["legal_steps"],
                            "failure":record["failure_kind"],
                            "terminal_task_error":record["errors"][-1]["task_status_error_rate"]
                                if record["errors"] else None,
                            "service_queries":record["offline_model_service_queries"],
                        }),flush=True)
    if len(runs)!=12*len(ANCHORS)*len(HORIZONS)*len(PARTICLE_SEEDS)*len(ARMS):
        raise AssertionError("missing required original action/device-matched experiments")
    comparison={
        anchor:{str(h):summarize_conditioned(runs,h,anchor) for h in HORIZONS}
        for anchor in ANCHORS
    }
    return {
        "schema":"PR76_original_J2_factual_ACTION_EXOGENOUS_MATCHED_WORLD_H5_H20_v1",
        "original_PR74_weights_sha256":sha256_file(checkpoint),
        "original_PR64_train_sha256":sha256_file(original_train),
        "original_PR64_dev_sha256":sha256_file(original_dev),
        "original_train_scenes":24,
        "original_developer_scenes":12,
        "conditioned_actions":"actual original CP-SAT actions at t for every model, verified existence; failure tracked",
        "conditioned_exogenous_device":"actual next-device public boundary at t+1 for all arms, oracle-conditioned",
        "conditioned_arrivals":"actual stage1 only when publicly visible at t; NEVER actual future stage2",
        "factual_future_service_duration_visible":False,
        "factual_future_task_state_model_input":False,
        "factual_future_device_as_exogenous_evaluation_boundary":True,
        "original_check_business_transition_kept":True,
        "separate_ground_truth_reference":"original J2 BusinessSimulator public observation t+1",
        "PR61_checkpoint_continued":False,
        "GPU_used":False,"new_training":False,
        "BLIND_TEST_read":False,
        "one_step_service_event_predictions":teacher,
        "n_rollouts":len(runs),
        "runs":runs,
        "action_device_matched_H5_H20":comparison,
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--train",type=Path,required=True)
    p.add_argument("--dev",type=Path,required=True)
    p.add_argument("--quality",type=Path,required=True)
    p.add_argument("--checkpoint",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args()
    if args.out.exists():
        raise FileExistsError("do not overwrite original matched factual prediction results")
    report=run_all(args.train,args.dev,args.quality,args.checkpoint)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n")
    print("PR76_FACTUAL_ACTION_EXOGENOUS_MATCHED_RESULT",json.dumps({
        "true_factual_rollouts":report["n_rollouts"],
        "one_step":report["one_step_service_event_predictions"]["methods"],
        "coverage":{
            a:{h:{arm:v["arm"][arm]["full_horizon"] for arm in ARMS}
               for h,v in report["action_device_matched_H5_H20"][a].items()}
            for a in ANCHORS
        },
        "matched":{
            a:{h:v["fully_action_device_matched_triples"]
               for h,v in report["action_device_matched_H5_H20"][a].items()}
            for a in ANCHORS
        },
        "BLIND":False,"training":False,
    }),flush=True)


if __name__=="__main__":
    main()
