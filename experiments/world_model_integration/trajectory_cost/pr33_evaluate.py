"""PR33: transfer existing trajectory-level business cost to frozen RHWM MPC.

Research reference: Chua et al., PETS, NeurIPS 2018: particle trajectories
scored by accumulated expected reward/cost. This uses pre-existing PR27
_trace_cost and lab_transport.decision_value.trace_score cost definitions.
It does not reproduce PETS CEM or the ensemble training algorithm.

Only the score changes: same frozen models, CP-SAT candidate generator,
3-particle H20 rollout, original legal workflow engine and 0.5s cap.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from experiments.world_model_integration.online_horizon.pr30_run import (
    OnlineHorizonPolicy,one as run_original_online,
)
from experiments.world_model_integration.decision_value.pr27_evaluate import _trace_cost
from experiments.world_model_integration.posttraining.pr31_dataset import PROTOCOL as TRAIN_PROTOCOL
from experiments.world_model_integration.posttraining.pr32_online_run import (
    PROTOCOL as PR32_PROTOCOL,
    _no_seen_scenarios as no_prior_pr32,
)
from experiments.world_model_integration.posttraining.repro_environment import sha
from experiments.world_model_integration.train_v4 import load_v4

PROTOCOL=Path(__file__).with_name("pr33_protocol.json")
ARMS=(
    "cp_sat","rule_progress_h20","rule_cost_h20",
    "v4_progress_h20","v4_cost_h20","scratch_cost_h20",
)


class TrajectoryCostPolicy(OnlineHorizonPolicy):
    """Existing PR27 MPC running cost + terminal distance, no new weights."""

    def _score_batch(self,batch,initial):
        # Use already-vetted cost evaluator, averaged over each model particle.
        # Main cost is integrated unfinished tasks and deadline overrun per
        # tick; terminal distance breaks ties lexicographically.
        score=_trace_cost(initial,batch,self.geometry)
        if len(score)!=2:raise AssertionError("existing trajectory cost changed")
        return tuple(float(x) for x in score)

    def _choose_scored_index(self,values):
        # Fixed lexicographic cost priority and stable CP-SAT-first tie break.
        return min(range(len(values)),key=lambda i:(values[i],i))


def _no_seen_seed(proto):
    if len(proto["scenario_groups"])!=24:raise AssertionError("frozen new scene count changed")
    # Original stage-wise seed checks through PR29-32.
    no_prior_pr32({"groups":proto["scenario_groups"]})
    prev=json.loads(PR32_PROTOCOL.read_text())
    old={x["seed"] for x in prev["groups"]}
    new={x["seed"] for x in proto["scenario_groups"]}
    if len(new)!=24 or new&old:raise AssertionError("PR33 online test group already seen in PR32")


def _settings(proto):
    return {
        "max_ticks":240,
        "candidate_limit":proto["budget"]["candidate_limit"],
        "particles":proto["budget"]["particles"],
        "decision_cpu_budget_seconds":proto["budget"]["per_decision_cpu_seconds"],
    }


def evaluate_group(scene,proto,old,scratch):
    records={}
    for arm in ARMS:
        if arm=="cp_sat":
            name="cp_sat"; checkpoint=old; method=OnlineHorizonPolicy
        elif arm.startswith("rule"):
            name="rule_h20";checkpoint=old
            method=TrajectoryCostPolicy if arm=="rule_cost_h20" else OnlineHorizonPolicy
        else:
            name="v4_h20"
            checkpoint=scratch if arm=="scratch_cost_h20" else old
            method=TrajectoryCostPolicy if arm.endswith("cost_h20") else OnlineHorizonPolicy
        record=run_original_online(scene,_settings(proto),name,checkpoint,policy_class=method)
        if record["prediction_errors"] or record["safety_violations"]:
            raise AssertionError("prediction/transition invalid")
        if record["job_metrics"]["censored"]:
            raise AssertionError("predeclared online experiment censored")
        record["arm"]=arm
        record["method"]="existing PR27 trajectory running+terminal cost" if method is TrajectoryCostPolicy else "original PR30 terminal progress"
        records[arm]=record
    return {"id":scene["id"],"regime":scene["regime"],"seed":scene["seed"],"arms":records}


def _mean(items):
    return sum(items)/len(items) if items else None


def summarize(groups):
    out={}
    for regime in ("light","medium","heavy","shift"):
        subset=[x for x in groups if x["regime"]==regime]
        if len(subset)!=6:raise AssertionError("precommitted group count changed")
        arms={}
        comparisons={}
        for arm in ARMS:
            vals=[x["arms"][arm] for x in subset]
            arms[arm]={
                "mean_final_makespan":_mean([x["makespan_ticks"] for x in vals]),
                "mean_late_jobs":_mean([x["job_metrics"]["late_jobs"] for x in vals]),
                "mean_resource_wait_ticks":_mean([x["resource_wait_ticks"] for x in vals]),
                "mean_full_cpu_seconds_including_model_loading":_mean([x["full_episode_cpu_seconds"] for x in vals]),
                "changed_CP_SAT_first_choices":sum(x["changed_plan_calls"] for x in vals),
                "model_predict_calls":sum(x["model_prediction_calls"] for x in vals),
                "budget_fallbacks":sum(x["decision_budget_exceeded"] for x in vals),
                "all_completed":sum(not x["job_metrics"]["censored"] for x in vals),
                "safety_violations":sum(x["safety_violations"] for x in vals),
            }
        for compare,control in (("v4_cost_h20","v4_progress_h20"),
                                 ("v4_cost_h20","rule_cost_h20"),
                                 ("scratch_cost_h20","v4_cost_h20"),
                                 ("scratch_cost_h20","v4_progress_h20"),
                                 ("v4_cost_h20","cp_sat")):
            delta=[x["arms"][compare]["makespan_ticks"]-x["arms"][control]["makespan_ticks"] for x in subset]
            comparisons[compare+"__vs__"+control]={
                "mean_delta_makespan":_mean(delta),
                "wins":sum(v<0 for v in delta),
                "ties":sum(v==0 for v in delta),
                "losses":sum(v>0 for v in delta),
                "mean_delta_late_jobs":_mean([
                    x["arms"][compare]["job_metrics"]["late_jobs"]-
                    x["arms"][control]["job_metrics"]["late_jobs"] for x in subset]),
            }
        out[regime]={"groups":len(subset),"arms":arms,"comparisons":comparisons}
    return out


def evaluate(old,scratch,selection_path,*,protocol=PROTOCOL):
    cfg=json.loads(Path(protocol).read_text())
    if (cfg["id"]!="RHWM_PR33_TRAJECTORY_COST_REFERENCE_TRANSFER_20261112_V1" or
        cfg["arms"]!=list(ARMS) or cfg["budget"]["horizon"]!=20 or
        cfg["budget"]["particles"]!=3):
        raise ValueError("frozen PR33 reference experiment changed")
    _no_seen_seed(cfg)
    from experiments.world_model_integration.posttraining.pr31_dataset import PROTOCOL as P31
    p31=json.loads(P31.read_text())
    old_expected=p31["old_checkpoint_sha256"]
    if sha(old)!=old_expected:raise ValueError("original V4 checksum wrong")
    selection=json.loads(Path(selection_path).read_text())
    if selection["best_by_validation_brier"]!="scratch_v4":
        raise ValueError("PR31 selected a different model on validation")
    if sha(scratch)!=selection["arms"]["scratch_v4"]["files"]["best"]["sha256"]:
        raise ValueError("PR31 scratch checkpoint differs from preserved original")
    _,meta=load_v4(scratch,experimental=True)
    if meta["training_arm"]!="scratch_v4":
        raise ValueError("source training metadata mismatch")
    from torch import set_num_threads
    set_num_threads(1)
    started=time.process_time()
    groups=[]
    for scene in cfg["scenario_groups"]:
        if time.process_time()-started>cfg["budget"]["total_cpu_seconds"]:
            raise TimeoutError("PR33 model-predictive experiment exceeded CPU cap")
        groups.append(evaluate_group(scene,cfg,old,scratch))
    elapsed=time.process_time()-started
    if elapsed>cfg["budget"]["total_cpu_seconds"]:
        raise TimeoutError("PR33 model-predictive experiment exceeded CPU cap")
    return {
        "protocol":cfg["id"],
        "protocol_sha256":sha(protocol),
        "original_v4_sha256":sha(old),
        "scratch_v4_sha256":sha(scratch),
        "pr31_validation_selection_sha256":sha(selection_path),
        "groups":groups,
        "summary":{
            "independent_scene_groups":len(groups),
            "by_regime":summarize(groups),
            "cpu_process_seconds":elapsed,
            "gpu":0,"training":0,
            "evidence":"True online BusinessSimulator with original CP-SAT; only score differs across model-matched arms",
            "limitations":"Current simulated workflow and workload only; uses fixed existing deadline weights, not automatic optimal control"
        }
    }


def main():
    parser=argparse.ArgumentParser()
    for name in ("original","scratch","selection","out"):
        parser.add_argument("--"+name,required=True,type=Path)
    args=parser.parse_args()
    if args.out.exists():raise FileExistsError("research results are write-once")
    r=evaluate(args.original,args.scratch,args.selection)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(r,indent=2,ensure_ascii=False)+"\n")
    tiny={}
    for reg,z in r["summary"]["by_regime"].items():
        tiny[reg]={
            "finish":{k:v["mean_final_makespan"] for k,v in z["arms"].items()},
            "late":{k:v["mean_late_jobs"] for k,v in z["arms"].items()},
            "cpu":{k:v["mean_full_cpu_seconds_including_model_loading"] for k,v in z["arms"].items()},
            "flips":{k:v["changed_CP_SAT_first_choices"] for k,v in z["arms"].items()},
            "fallbacks":{k:v["budget_fallbacks"] for k,v in z["arms"].items()},
            "paired":z["comparisons"],
        }
    print("PR33_TRAJECTORY_COST_LIVE_RESULTS",json.dumps({
        "groups":len(r["groups"]),"cpu_seconds":r["summary"]["cpu_process_seconds"],
        "regimes":tiny},ensure_ascii=False))


if __name__=="__main__":
    main()
