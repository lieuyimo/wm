"""PR35: true online strict decision-acceptance budget validation.

Fixes the PR33-audited defect: old PR30 policy could apply an H20
reranking even if final candidate rollout crossed 0.5 CPU seconds.
No new scoring method, neural architecture, simulator or training.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from experiments.world_model_integration.online_horizon.pr30_run import one
from experiments.world_model_integration.posttraining.pr32_online_run import (
    _no_seen_scenarios, PROTOCOL as PR32_PROTOCOL
)
from experiments.world_model_integration.posttraining.repro_environment import sha
from experiments.world_model_integration.train_v4 import load_v4

PROTOCOL=Path(__file__).with_name("pr35_protocol.json")
ARMS=("cp_sat","rule_h20","original_v4_h20","scratch_v4_h20")


def _new_cohort(proto):
    if proto["schema"]!="RHWM_PR35_STRICT_POST_CANDIDATE_H20_BUDGET_20261112_V1":
        raise ValueError("changed PR35 research protocol")
    if proto["arms"]!=list(ARMS) or proto["decision_cpu_budget_seconds"]!=0.5:
        raise ValueError("policy or CPU budget changed")
    if len(proto["groups"])!=24 or {x["regime"] for x in proto["groups"]}!={"light","medium","heavy","shift"}:
        raise AssertionError("incorrect number of independent scene groups")
    if any(sum(x["regime"]==reg for x in proto["groups"])!=6
           for reg in ("light","medium","heavy","shift")):
        raise AssertionError("six groups per independent regime required")
    _no_seen_scenarios(proto)
    previous=json.loads(PR32_PROTOCOL.read_text())
    if {x["seed"] for x in proto["groups"]}&{x["seed"] for x in previous["groups"]}:
        raise AssertionError("previous online heldout scene reused")


def _running_config(proto):
    return {
        "max_ticks":proto["max_ticks"],
        "candidate_limit":proto["candidate_limit"],
        "particles":proto["particles"],
        "decision_cpu_budget_seconds":proto["decision_cpu_budget_seconds"],
    }


def _audit_budget(result,cap):
    accepted=[r for r in result["decision_records"]
              if "selected_index" in r and r["selected_index"]!=0]
    illegal=[r for r in accepted if r["process_cpu_seconds"]>=cap]
    if illegal:
        raise AssertionError("world-model rearrangement accepted after CPU cap")
    if any("decision_acceptance_within_budget" not in x
           for x in result["decision_records"] if "selected_index" in x):
        raise AssertionError("patched strict-budget gate not in use")
    if any(not x["decision_acceptance_within_budget"]
           for x in result["decision_records"] if "selected_index" in x):
        raise AssertionError("recorded model assignment violated budget")
    actual_overrun=sum(r["process_cpu_seconds"]>=cap
                       for r in result["decision_records"] if "selected_index" in r)
    # A single model rollout cannot be preempted. These overrun actions MUST
    # execute the original legal CP-SAT proposal and be acknowledged.
    if actual_overrun>result["decision_budget_exceeded"]:
        raise AssertionError("budget overrun was not counted")
    return {
        "accepted_model_changes":len(accepted),
        "model_changes_after_cpu_cap":len(illegal),
        "actual_over_budget_decisions":actual_overrun,
        "budget_exceeded_records":result["decision_budget_exceeded"],
        "fallbacks":result["fallbacks"]
    }


def _mean(xs):
    return sum(xs)/len(xs) if xs else None


def _summary(groups):
    out={}
    for regime in ("light","medium","heavy","shift"):
        subset=[x for x in groups if x["regime"]==regime]
        arm={}
        paired={}
        for name in ARMS:
            vals=[x["arms"][name] for x in subset]
            arm[name]={
                "groups":len(vals),
                "mean_makespan":_mean([v["makespan_ticks"] for v in vals]),
                "mean_late_jobs":_mean([v["job_metrics"]["late_jobs"] for v in vals]),
                "mean_resource_wait_ticks":_mean([v["resource_wait_ticks"] for v in vals]),
                "completed":sum(not v["job_metrics"]["censored"] for v in vals),
                "mean_full_cpu_seconds":_mean([v["full_episode_cpu_seconds"] for v in vals]),
                "actual_over_budget_decisions":sum(v["strict_budget_audit"]["actual_over_budget_decisions"] for v in vals),
                "model_changes_after_cpu_cap":sum(v["strict_budget_audit"]["model_changes_after_cpu_cap"] for v in vals),
                "model_changes_accepted":sum(v["strict_budget_audit"]["accepted_model_changes"] for v in vals),
                "budget_fallbacks":sum(v["decision_budget_exceeded"] for v in vals),
                "inference_errors":sum(v["prediction_errors"] for v in vals),
                "safety_violations":sum(v["safety_violations"] for v in vals),
            }
        base=[x["arms"]["cp_sat"]["makespan_ticks"] for x in subset]
        original=[x["arms"]["original_v4_h20"]["makespan_ticks"] for x in subset]
        for name in ARMS[1:]:
            delta=[x["arms"][name]["makespan_ticks"]-b for x,b in zip(subset,base)]
            delta_original=[x["arms"][name]["makespan_ticks"]-o for x,o in zip(subset,original)]
            paired[name]={
                "delta_vs_cp_sat_mean":_mean(delta),
                "vs_cp_sat_wins":sum(x<0 for x in delta),
                "vs_cp_sat_ties":sum(x==0 for x in delta),
                "vs_cp_sat_losses":sum(x>0 for x in delta),
                "delta_vs_original_v4_mean":_mean(delta_original),
                "vs_original_wins":sum(x<0 for x in delta_original),
                "vs_original_ties":sum(x==0 for x in delta_original),
                "vs_original_losses":sum(x>0 for x in delta_original),
            }
        out[regime]={"groups":len(subset),"arms":arm,"paired":paired}
    return out


def run(original,scratch,selection_path,*,protocol=PROTOCOL):
    proto=json.loads(Path(protocol).read_text())
    _new_cohort(proto)
    selection=json.loads(Path(selection_path).read_text())
    if selection["best_by_validation_brier"]!="scratch_v4":
        raise ValueError("attempted to select another model based on test")
    if sha(original)!=proto["original_v4_sha256"]:
        raise ValueError("source original V4 weights changed")
    if sha(scratch)!=selection["arms"]["scratch_v4"]["files"]["best"]["sha256"]:
        raise ValueError("PR31 trained weights changed")
    _,new_meta=load_v4(scratch,experimental=True)
    if new_meta["training_arm"]!="scratch_v4":
        raise ValueError("selected source model identity changed")
    import torch
    torch.set_num_threads(1)
    t=time.process_time()
    groups=[]
    for scene in proto["groups"]:
        if time.process_time()-t>proto["resource_budget"]["cpu_process_seconds"]:
            raise TimeoutError("predeclared CPU resource limit reached")
        outputs={}
        for name in ARMS:
            role="v4_h20" if name in ("original_v4_h20","scratch_v4_h20") else name
            weights=scratch if name=="scratch_v4_h20" else original
            result=one(scene,_running_config(proto),role,weights)
            result["arm"]=name
            result["strict_budget_audit"]=_audit_budget(result,proto["decision_cpu_budget_seconds"])
            if result["safety_violations"] or result["prediction_errors"] or result["job_metrics"]["censored"]:
                raise AssertionError("unsafe, errored or censored online factual outcome")
            outputs[name]=result
        groups.append({"group":scene["id"],"seed":scene["seed"],"regime":scene["regime"],"arms":outputs})
    cpu=time.process_time()-t
    if cpu>proto["resource_budget"]["cpu_process_seconds"]:
        raise TimeoutError("predeclared CPU resource limit exceeded")
    return {
        "schema":proto["schema"],
        "protocol_sha256":sha(protocol),
        "original_v4_sha256":sha(original),
        "scratch_v4_sha256":sha(scratch),
        "validation_selection_sha256":sha(selection_path),
        "groups":groups,
        "summary":{
            "independent_groups":len(groups),
            "by_regime":_summary(groups),
            "total_cpu_process_seconds":cpu,
            "gpu":0,"new_training":0,
            "accepted_over_budget_nondefault_actions":sum(
                v["strict_budget_audit"]["model_changes_after_cpu_cap"]
                for x in groups for v in x["arms"].values()),
            "scope":"CPU decision ACCEPTANCE gate verified. Nonpreemptible model forward/planning can still take longer than 0.5s.",
        }
    }


def main():
    ap=argparse.ArgumentParser()
    for name in ("original","scratch","selection","out"):
        ap.add_argument("--"+name,required=True,type=Path)
    a=ap.parse_args()
    if a.out.exists():raise FileExistsError("evidence is write-once")
    result=run(a.original,a.scratch,a.selection)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n")
    compact={}
    for key,reg in result["summary"]["by_regime"].items():
        compact[key]={
            "arms":{a:{k:v[k] for k in (
                "mean_makespan","mean_late_jobs","mean_full_cpu_seconds",
                "actual_over_budget_decisions","model_changes_accepted",
                "model_changes_after_cpu_cap","budget_fallbacks")}
                for a,v in reg["arms"].items()},
            "paired":reg["paired"]
        }
    print("PR35_STRICT_CPU_H20_RESULT",json.dumps({
        "groups":result["summary"]["independent_groups"],
        "cpu":result["summary"]["total_cpu_process_seconds"],
        "accepted_over_budget":result["summary"]["accepted_over_budget_nondefault_actions"],
        "by_regime":compact},ensure_ascii=False))


if __name__=="__main__":main()
