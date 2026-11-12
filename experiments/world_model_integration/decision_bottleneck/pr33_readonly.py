"""Read-only PR33 decision-bottleneck diagnosis of immutable PR32 factual episodes.

Only consumes PR32's checksum-verified full online results. Does not run a
simulator, load models, mutate scores, choose weights, or train anything.

We compare arm decisions only on a common prefix of ALREADY EXECUTED actions
at matching ticks and matching first CP-SAT proposals. Once actions diverge,
subsequent paths differ and comparing raw model scores is no longer a
same-state comparison. Common-prefix discrepancies indicate where weight
changes first affected an online action, not causal isolation of one action.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
import argparse
import json
import math
import random

PROTO="RHWM_PR32_FROZEN_POSTTRAINED_H20_ONLINE_20261112_V1"
ORIGINAL="original_v4_h20"
SCORERS=("rule_h20","warmstart_v4_h20","scratch_v4_h20")
ARMS=("cp_sat","rule_h20",ORIGINAL,"warmstart_v4_h20","scratch_v4_h20")


def _mean(values):
    return sum(values)/len(values) if values else None


def _ci(values,seed=330112,iterations=4000):
    """Paired bootstrap at the independent scenario (not tick) level."""
    if not values:return None
    rng=random.Random(seed)
    n=len(values)
    samples=sorted(sum(values[rng.randrange(n)] for _ in range(n))/n
                   for _ in range(iterations))
    return [samples[int(0.025*iterations)],
            samples[int(0.975*iterations)]]


def _records(entry):
    """Extract one decision per tick; drop any accompanying error log entries."""
    records=[r for r in entry["decision_records"]
             if "selected_index" in r]
    if len(records)!=entry["policy_calls"]:
        raise AssertionError("factual policy-calls and decision records disagree")
    ticks=[r["tick"] for r in records]
    if ticks!=sorted(set(ticks)):
        raise AssertionError("decision ticks are not strictly increasing")
    if sum(r["selected_index"]!=0 for r in records)!=entry["changed_plan_calls"]:
        raise AssertionError("logged decisions do not match arm choice count")
    if sum(r["legal_plans"]>1 for r in records)!=entry["multiple_candidates"]:
        raise AssertionError("logged multi-option calls do not match total")
    return records


def _first_common_prefix_divergence(baseline,variant):
    """Only use the initial same-action and same-tick prefix.

    The first divergence can be evaluated on the shared public trajectory;
    afterwards the two simulations can expose different current states.
    """
    b=_records(baseline)
    v=_records(variant)
    until=min(len(b),len(v))
    for i in range(until):
        x,y=b[i],v[i]
        if x["tick"]!=y["tick"] or x["first_proposal"]!=y["first_proposal"] or x["legal_plans"]!=y["legal_plans"]:
            return {"status":"unmatched_decision_context",
                    "compared_common_steps":i,
                    "at_original_tick":x["tick"],
                    "at_variant_tick":y["tick"]}
        if x["selected"]!=y["selected"]:
            spread0=x.get("predicted_completed",[])
            spread1=y.get("predicted_completed",[])
            return {
                "status":"first_changed_action_on_common_prefix",
                "compared_common_steps":i,
                "tick":x["tick"],"candidate_count":x["legal_plans"],
                "original_selected_index":x["selected_index"],
                "variant_selected_index":y["selected_index"],
                "original_prediction_spread":max(spread0)-min(spread0) if spread0 else None,
                "variant_prediction_spread":max(spread1)-min(spread1) if spread1 else None,
                "original_decision_cpu_seconds":x["process_cpu_seconds"],
                "variant_decision_cpu_seconds":y["process_cpu_seconds"],
                "original_score":spread0,
                "variant_score":spread1,
            }
    if len(b)!=len(v):
        return {"status":"different_record_lengths_after_common_prefix",
                "compared_common_steps":until,
                "original_decisions":len(b),"variant_decisions":len(v)}
    if baseline["factual_event_sha256"]!=variant["factual_event_sha256"]:
        return {"status":"same_decisions_but_different_events_invariant",
                "compared_common_steps":until}
    return {"status":"same_actions_same_factual_events",
            "compared_common_steps":until}


def _per_policy(entry):
    d=_records(entry)
    meaningful=[r for r in d if len(r.get("predicted_completed",[]))>=2]
    spread=[max(r["predicted_completed"])-min(r["predicted_completed"]) for r in meaningful]
    near_zero=sum(abs(x)<1e-9 for x in spread)
    return {
        "calls":entry["policy_calls"],
        "multiple_candidate_calls":entry["multiple_candidates"],
        "flips":entry["changed_plan_calls"],
        "budget_fallbacks":entry["decision_budget_exceeded"],
        "all_fallbacks":entry["fallbacks"],
        "prediction_errors":entry["prediction_errors"],
        "model_init_cpu_seconds":entry["model_init_cpu_seconds"],
        "full_episode_cpu_seconds":entry["full_episode_cpu_seconds"],
        "forecast_ran_for_multi_candidate_call":len(meaningful),
        "forecast_indifferent_candidate_calls":near_zero,
        "mean_nontrivial_candidate_spread":_mean([x for x in spread if abs(x)>1e-9]),
        "factual_makespan_ticks":entry["makespan_ticks"],
        "factual_late_jobs":entry["job_metrics"]["late_jobs"],
    }


def analyze(source):
    if source["protocol"]!=PROTO or source["summary"]["total_independent_groups"]!=24:
        raise AssertionError("different read-only PR32 factual source")
    groups=source["groups"]
    report=[]
    for g in groups:
        if set(g["arms"])!=set(ARMS):
            raise AssertionError("missing preregistered 5-arm cohort")
        arms=g["arms"]
        for entry in arms.values():
            if entry["safety_violations"] or entry["prediction_errors"] or entry["job_metrics"]["censored"]:
                raise AssertionError("not a complete safe factual episode")
        comparisons={}
        for arm in SCORERS:
            comparisons[arm]={
                "common_prefix":_first_common_prefix_divergence(arms[ORIGINAL],arms[arm]),
                "delta_makespan_vs_original":arms[arm]["makespan_ticks"]-arms[ORIGINAL]["makespan_ticks"],
                "delta_late_jobs_vs_original":arms[arm]["job_metrics"]["late_jobs"]-arms[ORIGINAL]["job_metrics"]["late_jobs"],
                "factual_event_hash_equal":arms[arm]["factual_event_sha256"]==arms[ORIGINAL]["factual_event_sha256"],
            }
        report.append({
            "group":g["group"],"regime":g["regime"],"seed":g["seed"],
            "models":{arm:_per_policy(arms[arm]) for arm in ARMS},
            "comparisons_vs_original_v4":comparisons,
        })
    byregime={}
    for regime in sorted({x["regime"] for x in report}):
        subs=[x for x in report if x["regime"]==regime]
        compared={}
        for arm in SCORERS:
            deltas=[x["comparisons_vs_original_v4"][arm]["delta_makespan_vs_original"] for x in subs]
            contexts=Counter(x["comparisons_vs_original_v4"][arm]["common_prefix"]["status"] for x in subs)
            compared[arm]={
                "paired_groups":len(subs),
                "delta_makespan_mean_ticks":_mean(deltas),
                "delta_makespan_bootstrap_percentile_ci95":_ci(deltas),
                "vs_original_wins":sum(x<0 for x in deltas),
                "vs_original_ties":sum(x==0 for x in deltas),
                "vs_original_losses":sum(x>0 for x in deltas),
                "common_prefix_statuses":dict(contexts),
                "original_and_variant_final_events_identical":sum(
                    x["comparisons_vs_original_v4"][arm]["factual_event_hash_equal"] for x in subs),
            }
        costs={}
        for arm in ARMS:
            vals=[x["models"][arm] for x in subs]
            costs[arm]={
                "mean_full_episode_cpu_seconds":_mean([v["full_episode_cpu_seconds"] for v in vals]),
                "mean_model_init_cpu_seconds":_mean([v["model_init_cpu_seconds"] for v in vals]),
                "total_budget_fallbacks":sum(v["budget_fallbacks"] for v in vals),
                "total_choice_flips":sum(v["flips"] for v in vals),
                "scored_multi_calls":sum(v["forecast_ran_for_multi_candidate_call"] for v in vals),
                "equal_value_multi_calls":sum(v["forecast_indifferent_candidate_calls"] for v in vals),
            }
        byregime[regime]={"groups":len(subs),"model_comparisons":compared,"costs":costs}
    return {
        "schema":"RHWM_PR33_PR32_ARCHIVE_ONLY_DECISION_BOTTLENECK_V1",
        "source_protocol":source["protocol"],
        "source_run":"https://github.com/lieuyimo/wm/actions/runs/43144645020",
        "source_artifact":"13911157726",
        "groups":report,
        "by_regime":byregime,
        "computational_changes":{"new_simulator_runs":0,"new_model_inferences":0,
                                 "new_training":0,"gpu":0},
        "limitations":"The first different action on a common executed prefix is a same-state decision contrast; all subsequent executed states may differ. No proof of one-step causal mechanism or effective calibration of final makespan is claimed."
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--input",required=True,type=Path)
    p.add_argument("--out",required=True,type=Path)
    a=p.parse_args()
    if a.out.exists():raise FileExistsError("readonly output immutable")
    r=analyze(json.loads(a.input.read_text()))
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(r,ensure_ascii=False,indent=2)+"\n")
    print("PR33_READONLY_BOTTLENECK_RESULT",json.dumps({
        regime:{
            "groups":z["groups"],
            "comparisons":z["model_comparisons"],
            "costs":z["costs"],
        } for regime,z in r["by_regime"].items()},ensure_ascii=False))


if __name__=="__main__":main()
