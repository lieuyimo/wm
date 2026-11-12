"""PR33 actual online MPC scoring transfer: existing trajectory cost vs progress.

No new learning algorithm, model or scoring coefficients. Both objectives
consume identical, bounded H20/P3 RHWM world-model predictions and original
BusinessCPSAT legal candidates. Cumulative costs reuse the already-implemented
PR27 PETS-style trajectory cost adapter from decision_value.trace_score.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

from business_transport.simulation import BusinessSimulator,resource_audit
from experiments.business_constraints.run import lifecycle_audit
from experiments.world_model_integration.decision_value.pr27_evaluate import _trace_cost
from experiments.world_model_integration.dynamic_identifiability.pr28_experiment import build_business
from experiments.world_model_integration.online_horizon.pr30_run import (
    OnlineHorizonPolicy,_unique_plans,_facts,
)
from experiments.world_model_integration.posttraining.pr31_dataset import PROTOCOL as PR31
from experiments.world_model_integration.posttraining.pr32_online_run import (
    PROTOCOL as PR32,_no_seen_scenarios as no_older_scenarios,
)
from experiments.world_model_integration.posttraining.repro_environment import sha
from lab_transport.decision_value import candidates_for
from lab_transport.multisite_experiment import discrete_safety_violations

PROTOCOL=Path(__file__).with_name("pr33_protocol.json")
ARMS=("cp_sat","rule_progress","rule_cost",
      "original_progress","original_cost","scratch_progress","scratch_cost")


def no_seen_scenarios(proto):
    no_older_scenarios(proto)
    previous=json.loads(PR32.read_text())
    old={g["seed"] for g in previous["groups"]}
    if old&{g["seed"] for g in proto["groups"]}:
        raise AssertionError("PR32 independently evaluated groups re-used")


class ExistingMPCCostPolicy(OnlineHorizonPolicy):
    """Source-backed accumulated business cost, no custom penalty training."""

    def __init__(self,original_arm,score_kind,geometry,rules,protocol,*,checkpoint,seed):
        super().__init__(original_arm,geometry,rules,protocol,checkpoint=checkpoint,seed=seed)
        if score_kind not in ("progress","trace"):
            raise ValueError("unsupported scoring source")
        self.score_kind=score_kind
        self.horizon=20
        self.max_candidates_evaluated=0

    def __call__(self,raw,legal_candidates):
        start=time.process_time()
        obs=self.observer.observe(raw)
        plans=_unique_plans(self.solver,obs,legal_candidates)
        selected=plans[0]
        self.n_calls+=1
        if len(plans)>1:self.n_multi+=1
        values=[]
        candidate_scores=[]
        if self.model is not None and len(plans)>1:
            seed=1810+self.seed*17+obs.tick*101
            continuation=lambda state:_unique_plans(
                self.forecast_solver,state,candidates_for(state))[0]
            for index,plan in enumerate(plans):
                if time.process_time()-start>=self.protocol["decision_cpu_budget_seconds"]:
                    self.n_budget_exceeded+=1
                    values=[]
                    break
                try:
                    batch=self.model.rollout(
                        obs,plan,continuation_policy=continuation,
                        continuation_policy_name="CP_SAT_predicted_state",
                        horizon_ticks=self.horizon,
                        particles=self.protocol["particles"],
                        sampling_seed=seed)
                    if self.score_kind=="progress":
                        completed=float(batch.distributions[-1].expected_completed_samples)
                        # Model prediction and ranking unchanged from PR30;
                        # sign reversal turns maximization into minimization.
                        value=(-completed,0.,index)
                        candidate_scores.append({"expected_completed":completed})
                    else:
                        # Actual source code lives in PR27; that function
                        # implements legacy decision_value.trace_score's
                        # cumulative unfinished/lateness and final distance,
                        # expected across the RHWM particle forecasts.
                        cumulative,terminal=_trace_cost(obs,batch,self.geometry)
                        value=(float(cumulative),float(terminal),index)
                        candidate_scores.append({
                            "cumulative_unfinished_late":cumulative,
                            "terminal_remaining_route":terminal})
                    values.append(value)
                    self.n_model_queries+=1
                    self.n_model_steps+=self.horizon*self.protocol["particles"]
                except (RuntimeError,ValueError,AssertionError,TypeError,KeyError) as exc:
                    self.n_errors+=1
                    self.decisions.append({"tick":obs.tick,"error":str(exc)[:250]})
                    values=[]
                    break
                # Hard gate after scoring too, not only before starting
                # the next candidate; never use partial over-budget scores.
                if time.process_time()-start>=self.protocol["decision_cpu_budget_seconds"]:
                    self.n_budget_exceeded+=1
                    values=[]
                    break
            if len(values)==len(plans):
                best_index=min(range(len(values)),key=lambda i:values[i])
                selected=plans[best_index]
            else:self.n_fallback+=1
        self.n_flip+=int(selected!=plans[0])
        elapsed=time.process_time()-start
        self.cpu_decision_seconds+=elapsed
        self.max_candidates_evaluated=max(self.max_candidates_evaluated,len(values))
        self.decisions.append({
            "tick":obs.tick,"legal_plans":len(plans),
            "selected_index":plans.index(selected),
            "completed_full_candidate_evaluation":len(values)==len(plans)
                if self.model is not None and len(plans)>1 else None,
            "score_kind":self.score_kind,
            "scores":candidate_scores if len(values)==len(plans) else [],
            "cpu_seconds":elapsed,
        })
        return selected


def one_online(spec,protocol,arm,original,scratch):
    business,geo=build_business(spec,max_ticks=protocol["max_ticks"])
    model_source=(
        "cp_sat" if arm=="cp_sat" else
        "rule_h20" if arm.startswith("rule_") else
        "v4_h20"
    )
    source=scratch if arm.startswith("scratch_") else original
    objective="trace" if arm.endswith("_cost") else "progress"
    t0=time.process_time()
    policy=ExistingMPCCostPolicy(
        model_source,objective,geo,business.rules,protocol,
        checkpoint=source,seed=spec["seed"])
    load_cpu=time.process_time()-t0
    t1=time.process_time()
    actual=BusinessSimulator(
        business,"world",spec["seed"],collect_metrics=True).run(
            dispatch_policy=policy,on_observation=policy.observe)
    episode_cpu=time.process_time()-t1
    capacity=resource_audit(actual,business.rules)
    motion=discrete_safety_violations(actual)
    lifecycle=lifecycle_audit(actual["history"])
    if capacity["violations"] or motion or lifecycle:
        raise AssertionError("source MPC policy violated a real simulator constraint")
    if policy.n_errors:
        raise AssertionError("world-model source threw errors and fell back; report missing")
    return {
        "arm":arm,"seed":spec["seed"],"regime":spec["regime"],
        "status":actual["summary"]["status"],
        "makespan_ticks":actual["summary"]["ticks"],
        "job_metrics":_facts(actual,business),
        "resource_wait_ticks":capacity["capacity_wait_ticks"],
        "blocked_moves":actual["summary"]["blocked_move_attempts"],
        "model_load_cpu_seconds":load_cpu,
        "whole_episode_cpu_seconds":episode_cpu+load_cpu,
        "decision_cpu_seconds":policy.cpu_decision_seconds,
        "policy_calls":policy.n_calls,"multi_candidate_decisions":policy.n_multi,
        "model_queries":policy.n_model_queries,
        "model_steps":policy.n_model_steps,
        "changed_first_proposal":policy.n_flip,
        "budget_misses":policy.n_budget_exceeded,
        "fallbacks":policy.n_fallback,
        "safety_violations":0,"prediction_errors":policy.n_errors,
        "event_hash":hashlib.sha256(json.dumps(actual["events"],sort_keys=True).encode()).hexdigest(),
        "decision_trace":policy.decisions,
    }


def mean(items):
    return sum(items)/len(items) if items else None


def summarize(groups):
    by={}
    for regime in sorted(set(g["regime"] for g in groups)):
        cohort=[g for g in groups if g["regime"]==regime]
        armdata={}
        for arm in ARMS:
            values=[g["arms"][arm] for g in cohort]
            armdata[arm]={
                "groups":len(values),
                "mean_makespan":mean([x["makespan_ticks"] for x in values]),
                "mean_late_jobs":mean([x["job_metrics"]["late_jobs"] for x in values]),
                "mean_wait_ticks":mean([x["resource_wait_ticks"] for x in values]),
                "completed_groups":sum(not x["job_metrics"]["censored"] for x in values),
                "changed_proposals":sum(x["changed_first_proposal"] for x in values),
                "model_queries":sum(x["model_queries"] for x in values),
                "model_steps":sum(x["model_steps"] for x in values),
                "budget_misses":sum(x["budget_misses"] for x in values),
                "fallbacks":sum(x["fallbacks"] for x in values),
                "safety_violations":sum(x["safety_violations"] for x in values),
                "mean_full_cpu_seconds":mean([x["whole_episode_cpu_seconds"] for x in values]),
            }
        paired={}
        for src in ("rule","original","scratch"):
            old=src+"_progress"
            new=src+"_cost"
            delta=[x["arms"][new]["makespan_ticks"]-x["arms"][old]["makespan_ticks"]
                   for x in cohort]
            vs_cp=[x["arms"][new]["makespan_ticks"]-x["arms"]["cp_sat"]["makespan_ticks"]
                   for x in cohort]
            paired[src]={
                "cost_minus_progress_makespan_mean":mean(delta),
                "cost_wins":sum(d<0 for d in delta),
                "ties":sum(d==0 for d in delta),
                "cost_losses":sum(d>0 for d in delta),
                "cost_minus_cp_sat_makespan_mean":mean(vs_cp),
                "cost_vs_cp_sat_wins":sum(d<0 for d in vs_cp),
                "cost_vs_cp_sat_ties":sum(d==0 for d in vs_cp),
                "cost_vs_cp_sat_losses":sum(d>0 for d in vs_cp),
            }
        by[regime]={"groups":len(cohort),"arms":armdata,"cost_vs_progress":paired}
    return by


def run(original,scratch,selection,*,protocol=PROTOCOL):
    proto=json.loads(Path(protocol).read_text())
    if proto["protocol"]!="RHWM_PR33_SOURCE_BACKED_MPC_COST_20261112_V1" or proto["arms"]!=list(ARMS):
        raise ValueError("PR33 scoring arms or frozen protocol changed")
    if len(proto["groups"])!=24:raise AssertionError("frozen scenes changed")
    no_seen_scenarios(proto)
    sel=json.loads(Path(selection).read_text())
    if sel["best_by_validation_brier"]!="scratch_v4":
        raise ValueError("independent validation-selected model changed")
    if sha(original)!=proto["weights"]["original_v4_sha256"] or (
        sha(scratch)!=sel["arms"]["scratch_v4"]["files"]["best"]["sha256"]):
        raise ValueError("frozen model weights do not match original archive")
    from torch import set_num_threads
    set_num_threads(1)
    cfg={
        "max_ticks":proto["evaluation"]["max_ticks"],
        "candidate_limit":proto["evaluation"]["candidate_limit"],
        "particles":proto["evaluation"]["particles"],
        "decision_cpu_budget_seconds":proto["evaluation"]["score_budget_sec_per_online_decision"],
    }
    started=time.process_time()
    groups=[]
    for spec in proto["groups"]:
        if time.process_time()-started>proto["evaluation"]["whole_cpu_seconds"]:
            raise TimeoutError("PR33 CPU cap exceeded before next scenario")
        groups.append({
            "group":spec["id"],"seed":spec["seed"],"regime":spec["regime"],
            "arms":{arm:one_online(spec,cfg,arm,original,scratch) for arm in ARMS},
        })
    elapsed=time.process_time()-started
    if elapsed>proto["evaluation"]["whole_cpu_seconds"]:
        raise TimeoutError("PR33 complete test exceeded CPU budget")
    summary={
        "independent_groups":len(groups),
        "by_regime":summarize(groups),
        "cpu_process_seconds":elapsed,"gpu":0,"new_training":0,
        "source":"PETS-inspired receding horizon expected costs via existing trace_score and PR27 adapter",
        "scope":"Real online H20 policy comparison on new scenes, not trained or tuned by PR32 evaluation",
    }
    return {
        "protocol":proto["protocol"],
        "protocol_sha256":sha(protocol),
        "original_sha256":sha(original),"scratch_sha256":sha(scratch),
        "validation_selection_sha256":sha(selection),
        "groups":groups,"summary":summary,
    }


def main():
    p=argparse.ArgumentParser()
    for name in ("original","scratch","selection","out"):
        p.add_argument("--"+name,type=Path,required=True)
    args=p.parse_args()
    if args.out.exists():raise FileExistsError("experiment evidence write once")
    r=run(args.original,args.scratch,args.selection)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(r,ensure_ascii=False,indent=2)+"\n")
    compact={reg:{
        "means":{arm:v["mean_makespan"] for arm,v in rr["arms"].items()},
        "paired":rr["cost_vs_progress"],
        "budget":{arm:v["budget_misses"] for arm,v in rr["arms"].items()},
        "cpu":{arm:v["mean_full_cpu_seconds"] for arm,v in rr["arms"].items()}
    } for reg,rr in r["summary"]["by_regime"].items()}
    print("PR33_EXISTING_MPC_COST_ONLINE",json.dumps({
        "groups":r["summary"]["independent_groups"],
        "cpu":r["summary"]["cpu_process_seconds"],
        "regimes":compact},ensure_ascii=False))


if __name__=="__main__":
    main()
