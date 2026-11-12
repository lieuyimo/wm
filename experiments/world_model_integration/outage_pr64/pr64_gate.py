"""PR64 original-world source admission + preregistered discrete-time hazard shift.

Fits ONLY simple smoothed state/age/time risk-set hazards on independent
TRAIN source, scores complete shifted-time developer VALIDATION sets, and
reports paired scene-block uncertainty. No model checkpoints/gradient,
GPU, H20 rollout, or original PR59 unopened horizontal-shelf BLIND TEST.

PR63's tick=6/12 dedicated bucket function is intentionally NOT used:
that would import special information from the old fixed-tick 7/13 source.
A uniform 4-tick public-clock bucket is frozen in this script.
"""
from __future__ import annotations
from collections import Counter,defaultdict
from pathlib import Path
import argparse
import hashlib
import json
import math
import random

from experiments.world_model_integration.audit_pr63.pr63_risk_set import (
    extract_records,fit_hazards,age_h4_bucket,score_rows
)
from experiments.world_model_integration.outage_pr64.pr64_source import (
    PROTOCOL,read_protocol,onset_start
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def clock4(r):
    return str(min(64,int(r["tick"])//4))


def exact_age4(r):
    return str(min(64,(max(1,int(r["age_exact"]))-1)//4))


def _p(record,table,params):
    risk=record["risk_set"]
    if table=="constant":
        hazard=params["constant"][risk]["hazard"]
    else:
        key={"age_h4":lambda r:age_h4_bucket(r["age_h4"]),
             "clock4":clock4,"exact_age4":exact_age4}[table](record)
        hazard=params[table][risk].get(key,{}).get(
            "hazard",params["constant"][risk]["hazard"])
    return 1.-hazard if risk=="up" else hazard


def fit(train):
    return {
        "constant":fit_hazards(train,None,alpha=0.),
        "age_h4":fit_hazards(train,lambda r:age_h4_bucket(r["age_h4"]),alpha=1.),
        "clock4":fit_hazards(train,clock4,alpha=1.),
        "exact_age4":fit_hazards(train,exact_age4,alpha=1.),
        "fit_only_on_train":True,
        "uniform_4_tick_clock_buckets_not_tuned_to_prior_pr63_fixed_6_12":True,
    }


def risk_set_score(records,params):
    return {method:score_rows([
        (_p(r,method,params),r["target"],r) for r in records
    ]) for method in ("constant","age_h4","clock4","exact_age4")}


def scene_bootstrap(records,params,seed=64064,replicates=2000):
    by_scene=defaultdict(list)
    for r in records:
        by_scene[r["scene"]].append(r)
    scenes=sorted(by_scene)
    if len(scenes)!=12:
        raise AssertionError("PR64 12 independent developer validation episodes required")
    methods=("constant","age_h4","clock4","exact_age4")
    per_scene={m:{} for m in methods}
    for name,scene_rows in by_scene.items():
        for m in methods:
            v=[(_p(r,m,params)-r["target"])**2 for r in scene_rows]
            per_scene[m][name]=sum(v)/len(v)
    rng=random.Random(seed)
    sample=[]
    for _ in range(replicates):
        selected=[rng.choice(scenes) for _ in scenes]
        base=sum(per_scene["constant"][x] for x in selected)/len(selected)
        sample.append({m:sum(per_scene[m][x] for x in selected)/len(selected)-base
                       for m in methods if m!="constant"})
    output={}
    for m in methods:
        overall=sum(per_scene[m].values())/len(scenes)
        obj={"n_independent_scenes":len(scenes),"scene_macro_brier":overall}
        if m!="constant":
            diffs=sorted(row[m] for row in sample)
            obj.update({
                "paired_scene_macro_delta_vs_train_constant":overall-sum(per_scene["constant"].values())/len(scenes),
                "bootstrap_scenes_2000_ci95":[diffs[49],diffs[1950]],
                "Brier_better_than_constant_in_scenes":sum(per_scene[m][s]<per_scene["constant"][s] for s in scenes)
            })
        output[m]=obj
    return output


def audit(train_path,val_path,out):
    if out.exists():raise FileExistsError("immutable independent quality assessment exists")
    p,old=read_protocol()
    original={}
    for split,path in (("train",train_path),("validation",val_path)):
        q=json.loads(path.read_text())
        if (q.get("schema")!="pr64-shifted-private-failure-event-original-business-public-source-v1"
            or q.get("split")!=split or q.get("BLIND_TEST_RUN") is not False
            or q["protocol_sha256"]!=sha(PROTOCOL)):
            raise AssertionError("not original preregistered source or TEST opened")
        cases=q["scenes"]
        if len(cases)!=p["splits"][split]["count"]:
            raise AssertionError("some independent source scenarios missing")
        planned={x["id"]:x for x in p["cohort"] if x["split"]==split}
        if {x["id"] for x in cases}!=set(planned):
            raise AssertionError("real source scene identifiers differ from protocol")
        observed=[]
        expected_count=0
        recovery_count=0
        clock_support=set(p["splits"][split]["allowed_private_start_ticks"])
        observed_maps=set()
        for scene in cases:
            case=scene["case"]
            if case!=planned[scene["id"]] or case["split"]!=split:
                raise AssertionError("private source spec changed")
            if scene["private_schedule_NOT_in_model_graph"] is not True:
                raise AssertionError("hidden schedule leakage claimed")
            rows=scene["raw_actual_public_world_transitions"]
            if (not rows or len(rows)!=scene["observed_ticks"]
                or rows[-1]["target"]["next_public_tick"]!=scene["observed_ticks"]
                or scene["public_map"]["schema"]!="pr56-public-grid-cells-v1"):
                raise AssertionError("incomplete original factual source")
            for i,row in enumerate(rows):
                if (row["input"]["public_tick"]!=i
                    or row["input"]["feature_last_observed_tick"]!=i
                    or row["target"]["next_public_tick"]!=i+1
                    or row["input"]["public_map"]["static_map_digest"]!=scene["public_map"]["static_map_digest"]):
                    raise AssertionError("public observation with future leakage or broken chronology")
                if "private" in json.dumps(row["input"]).lower() or "outage" in json.dumps(row["input"]).lower():
                    raise AssertionError("future private fault schedule unexpectedly in public graph")
            observed_maps.add(scene["public_map"]["static_map_digest"])
            if len(scene["public_observed_fault_onset_next_ticks"])!=(2 if case["regime"]=="multi_fault" else 1):
                raise AssertionError("missing original true fault onset")
            starts=scene["public_observed_fault_onset_next_ticks"]
            if not all(start in clock_support for start in starts):
                raise AssertionError("shifted actual source starts not in fixed TRAIN/VAL support")
            observed.extend(starts)
            expected_count+=len(starts)
            recovery_count+=len(scene["public_observed_fault_recovery_next_ticks"])
        original[split]={
            "source_sha256":sha(path),"source_case_count":len(cases),
            "source_distinct_map_count":len(observed_maps),
            "sampled_real_onset_count":expected_count,
            "real_recovery_count":recovery_count,
            "observed_distinct_start_ticks":sorted(set(observed)),
            "actual_onset_tick_counts":dict(sorted(Counter(observed).items())),
            "completed":sum(not s["right_censored_final_makespan"] for s in cases),
            "right_censored":sum(s["right_censored_final_makespan"] for s in cases),
        }
    r=original
    gates=p["preregistered_safety_and_information_gates"]
    checks={
        "all_true_simulation_safety_and_public_t_to_next_chronology":True,
        "24_independent_true_TRAIN_worlds":r["train"]["source_case_count"]==24,
        "12_independent_true_shifted_dev_VAL_worlds":r["validation"]["source_case_count"]==12,
        "no_original_BLIND_read_or_generated":p["splits"]["BLIND_TEST"]["read"] is False and p["splits"]["BLIND_TEST"]["generate"] is False,
        "distinct_true_device_onset_events_TRAIN":r["train"]["sampled_real_onset_count"]>=gates["train_actual_onset_count_at_least"],
        "distinct_true_device_onset_events_VAL":r["validation"]["sampled_real_onset_count"]>=gates["validation_actual_onset_count_at_least"],
        "distinct_true_recovery_events_TRAIN":r["train"]["real_recovery_count"]>=gates["train_actual_recovery_count_at_least"],
        "distinct_true_recovery_events_VAL":r["validation"]["real_recovery_count"]>=gates["validation_actual_recovery_count_at_least"],
        "distinct_true_start_times_TRAIN":len(r["train"]["observed_distinct_start_ticks"])>=gates["distinct_actual_private_start_ticks_train_at_least"],
        "distinct_true_start_times_VAL":len(r["validation"]["observed_distinct_start_ticks"])>=gates["distinct_actual_private_start_ticks_validation_at_least"],
        "actual_start_tick_distributions_have_no_overlap":not set(r["train"]["observed_distinct_start_ticks"])&set(r["validation"]["observed_distinct_start_ticks"]),
        "all_original_public_train_and_val_maps_unchanged":r["train"]["source_distinct_map_count"]==6 and r["validation"]["source_distinct_map_count"]==2,
        "new_neural_training_or_GPU":p["resource"]["new_model_training"]==0 and p["resource"]["GPU"]==0
    }
    failures=[k for k,v in checks.items() if not v]
    tr=json.loads(train_path.read_text());v=json.loads(val_path.read_text())
    train_records,_=extract_records(tr,"train")
    validation_records,_=extract_records(v,"validation")
    parameters=fit(train_records)
    quality=("PASS_TRUE_SHIFTED_OUTAGE_DATA_TO_REVIEW_MODEL_DESIGN" if not failures
             else "NO_GO_SHIFTED_OUTAGE_SOURCE_INFORMATION")
    results={
        "schema":"pr64-private-randomized-onset-risk-set-prospective-data-quality-v1",
        "source_quality_verdict":quality,
        "checks":checks,"failures":failures,
        "unmodified_original_BusinessSimulator":True,
        "new_PRIVATE_event_time_simulation_used":True,
        "new_NEURAL_training_performed":False,
        "new_GPU_hours":0,
        "original_pr59_unopened_TEST_generated":False,
        "original_pr59_unopened_TEST_read":False,
        "not_an_unseen_business_or_cross_robot_form_test":True,
        "parent_pr63":"95ef4c3bee3824f382166efd0c3d62ce568b06f5",
        "protocol_sha256":sha(PROTOCOL),
        "source_counts":original,
        "train_risk_set":{"n":len(train_records),
                          "onsets":sum(x["phase"]=="fault_onset" for x in train_records),
                          "recoveries":sum(x["phase"]=="recovery" for x in train_records)},
        "validation_risk_set":{"n":len(validation_records),
                               "onsets":sum(x["phase"]=="fault_onset" for x in validation_records),
                               "recoveries":sum(x["phase"]=="recovery" for x in validation_records)},
        "train_fitted_hazard_params":parameters,
        "dev_val_full_risk_set_metrics":risk_set_score(validation_records,parameters),
        "paired_independent_scene_macro_bootstrap":scene_bootstrap(validation_records,parameters),
        "limits":"Randomized private time shift per case tests synthetic generator shift; train vs development VAL differ by map-family as well as onset time support, so effects are not a purely isolated causal time factor. Exact age and clock are diagnostic observed-public metadata, not original PR61 model inputs. No H20 rollout or policy benefit tested.",
        "decision_boundary":"A positive source gate permits review of a stochastic exogenous hazard module, NOT claimed robust onset prediction, neural model improvement, H20 or online dispatch."
    }
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(results,indent=2,ensure_ascii=False)+"\n")
    print("PR64_RANDOM_START_CPU_QUALITY",json.dumps({
        "verdict":quality,"failures":failures,
        "source":original,
        "tick_weighted_Brier":{m:z["overall"]["brier"] for m,z in results["dev_val_full_risk_set_metrics"].items()},
        "scene_macro_brier":{m:z["scene_macro_brier"] for m,z in results["paired_independent_scene_macro_bootstrap"].items()}
    },ensure_ascii=False),flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--train",type=Path,required=True)
    p.add_argument("--validation",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    audit(a.train,a.validation,a.out)


if __name__=="__main__":
    main()
