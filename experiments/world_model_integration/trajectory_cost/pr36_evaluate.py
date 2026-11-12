"""PR36: independently retest H20 online scheduling with a strict CPU gate.

Source: PR33 audit established that PR30/32/35 tested a pre-candidate budget
check, not a post-final-candidate admission check. This experiment corrects
that implementation and measures real per-decision acceptance time, with
fresh scene seeds. All scoring and learned dynamics are reused unchanged.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from experiments.world_model_integration.trajectory_cost.pr33_evaluate import (
    PROTOCOL as PR35_PROTOCOL,
    ARMS, _no_seen_seed, evaluate_group, summarize,
)
from experiments.world_model_integration.posttraining.pr31_dataset import PROTOCOL as PR31_PROTOCOL
from experiments.world_model_integration.posttraining.repro_environment import sha
from experiments.world_model_integration.train_v4 import load_v4

PROTOCOL=Path(__file__).with_name("pr36_protocol.json")


def _verify_new_scene_group(cfg):
    groups=cfg["new_sealed_groups"]
    if len(groups)!=24 or len({g["seed"] for g in groups})!=24:
        raise AssertionError("PR36 frozen groups invalid or duplicated")
    if {x["regime"] for x in groups}!={"light","medium","heavy","shift"}:
        raise AssertionError("wrong PR36 frozen regimes")
    if any(sum(x["regime"]==reg for x in groups)!=6
           for reg in ("light","medium","heavy","shift")):
        raise AssertionError("wrong frozen per-regime group size")
    _no_seen_seed({"scenario_groups":groups})
    prev=json.loads(PR35_PROTOCOL.read_text())
    if {x["seed"] for x in prev["scenario_groups"]}&{x["seed"] for x in groups}:
        raise AssertionError("previous PR35 data reused for strict corrected evaluation")


def _admission_audit(record,budget):
    """Every model-accepted action must have completed scoring before deadline."""
    accepted=0
    first_choice_changed=0
    for item in record["decision_records"]:
        if "selected_index" not in item:
            continue
        chosen=item["selected_index"]
        when=item.get("model_selection_cpu_seconds")
        if when is not None:
            accepted+=1
            if not (0<=when<budget):
                raise AssertionError("accepted a model score at/after CPU deadline")
        if chosen!=0:
            first_choice_changed+=1
            if when is None:
                raise AssertionError("late or partial model rank overrode CP-SAT baseline")
    if first_choice_changed!=record["changed_plan_calls"]:
        raise AssertionError("decision log and effective action switch count disagree")
    return accepted


def _check_report(group,budget):
    recorded={}
    for arm,record in group["arms"].items():
        if record["prediction_errors"] or record["safety_violations"]:
            raise AssertionError("factual workflow or model transition invalid")
        if record["job_metrics"]["censored"]:
            raise AssertionError("true factual job completion censored")
        recorded[arm]=_admission_audit(record,budget)
    return recorded


def evaluate(old,scratch,selection,*,protocol=PROTOCOL):
    cfg=json.loads(Path(protocol).read_text())
    if cfg["protocol"]!="RHWM_PR36_STRICT_POSTCANDIDATE_BUDGET_20261112_V1":
        raise ValueError("wrong preregistered PR36 protocol")
    if cfg["arms"]!=list(ARMS) or cfg["budget"]["horizon"]!=20 or cfg["budget"]["particles"]!=3:
        raise AssertionError("scoring arms or horizon changed")
    _verify_new_scene_group(cfg)
    train=json.loads(PR31_PROTOCOL.read_text())
    if sha(old)!=train["old_checkpoint_sha256"]:
        raise ValueError("wrong original V4 source")
    selection_data=json.loads(Path(selection).read_text())
    if selection_data["best_by_validation_brier"]!="scratch_v4":
        raise ValueError("unexpected validation-selected model")
    if sha(scratch)!=selection_data["arms"]["scratch_v4"]["files"]["best"]["sha256"]:
        raise ValueError("post-trained weights differ from archived PR31")
    _,meta=load_v4(scratch,experimental=True)
    if meta["training_arm"]!="scratch_v4":
        raise ValueError("incorrect Scratch V4 metadata")
    from torch import set_num_threads
    set_num_threads(1)
    t0=time.process_time()
    groups=[]
    accepted_by_arm={arm:0 for arm in ARMS}
    for scene in cfg["new_sealed_groups"]:
        if time.process_time()-t0>cfg["budget"]["total_cpu_seconds"]:
            raise TimeoutError("PR36 preregistered CPU experiment cap exceeded")
        group=evaluate_group(scene,cfg,old,scratch)
        accepted=_check_report(group,cfg["budget"]["per_decision_cpu_seconds"])
        for arm,n in accepted.items():accepted_by_arm[arm]+=n
        groups.append(group)
    elapsed=time.process_time()-t0
    if elapsed>cfg["budget"]["total_cpu_seconds"]:
        raise TimeoutError("PR36 CPU cap exceeded")
    return {
        "protocol":cfg["protocol"],
        "protocol_sha256":sha(protocol),
        "frozen_v4_sha256":sha(old),
        "scratch_v4_sha256":sha(scratch),
        "frozen_validation_selection_sha256":sha(selection),
        "groups":groups,
        "summary":{
            "groups":len(groups),
            "by_regime":summarize(groups),
            "model_admissions_before_deadline":accepted_by_arm,
            "post_candidate_budget_violation_count":0,
            "cpu_process_seconds":elapsed,
            "gpu":0,"training":0,
            "budget_interpretation":"No late predicted model selection allowed; whole rollout itself cannot be preempted",
            "finding_scope":"True independent online outcomes on synthetic state/transport system",
        }
    }


def main():
    ap=argparse.ArgumentParser()
    for key in ("original","scratch","selection","out"):
        ap.add_argument("--"+key,required=True,type=Path)
    args=ap.parse_args()
    if args.out.exists():raise FileExistsError("PR36 evidence is immutable")
    r=evaluate(args.original,args.scratch,args.selection)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(r,ensure_ascii=False,indent=2)+"\n")
    small={}
    for reg,val in r["summary"]["by_regime"].items():
        small[reg]={
            "mean_final_tick":{arm:v["mean_final_makespan"]
                               for arm,v in val["arms"].items()},
            "mean_late_jobs":{arm:v["mean_late_jobs"] for arm,v in val["arms"].items()},
            "candidate_flips":{arm:v["changed_CP_SAT_first_choices"]
                               for arm,v in val["arms"].items()},
            "strict_budget_fallbacks":{arm:v["budget_fallbacks"]
                                       for arm,v in val["arms"].items()},
            "paired":val["comparisons"],
        }
    print("PR36_STRICT_ONLINE_RESULTS",json.dumps({
        "groups":r["summary"]["groups"],
        "admitted_model_decisions":r["summary"]["model_admissions_before_deadline"],
        "CPU_seconds":r["summary"]["cpu_process_seconds"],
        "regimes":small,
    },ensure_ascii=False))


if __name__=="__main__":
    main()
