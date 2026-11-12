"""ETM: literature-grounded, budget-conscious event-triggered horizon selection.

This is an engineering adaptation of Wang, Ren and Zheng (IEEE TAC 2023)
variable-horizon event-triggered MPC; NOT a reproduction of their robust
controller or guarantees. Does not train or modify any world model. Reuses
frozen V4 and existing PR30 live full-episode simulation/CP-SAT baselines.

Stage 1: H5 P1 screening all lawful CP-SAT candidates.
Stage 2: H20 P3 escalation ONLY when H5 candidates tied and >=3 incomplete
business jobs, based exclusively on the current public observation.
If time budget insufficient for a complete ranking, use complete H5 screening
or original CP-SAT first proposal. Count every overrun, including atomic
rollout overrun; never claim a hard real-time deadline without interrupts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

from business_transport.simulation import BusinessSimulator,resource_audit
from experiments.business_constraints.run import lifecycle_audit
from experiments.world_model_integration.dynamic_identifiability.pr28_experiment import build_business
from experiments.world_model_integration.horizon_generalization.pr29_evaluate import _scenario_seed_set
from experiments.world_model_integration.online_horizon.pr30_run import (
    OnlineHorizonPolicy, one, _unique_plans, _facts,
)
from experiments.world_model_integration.system_world_model.evaluate_horizons import file_sha
from lab_transport.decision_value import candidates_for
from lab_transport.multisite_experiment import discrete_safety_violations

PROTOCOL=Path(__file__).with_name("etm_protocol.json")
ARMS=("cp_sat","rule_h20","v4_h5","v4_h20","v4_adaptive")


def _unfinished_jobs(public_state):
    return len({s.id.split("::stage")[0] for s in public_state.samples
                if s.status!="completed"})


def _best(vals):
    if not vals:return 0
    top=max(vals)
    return next(i for i,x in enumerate(vals) if top-x<=1e-9)


class AdaptiveHorizonPolicy(OnlineHorizonPolicy):
    def __init__(self,geometry,rules,protocol,*,checkpoint,seed):
        settings={
            "candidate_limit":protocol["candidate_limit"],
            "particles":protocol["adaptation"]["escalation_particles"],
            "decision_cpu_budget_seconds":
                 protocol["adaptation"]["max_cpu_per_decision_seconds"],
        }
        super().__init__("v4_h20",geometry,rules,settings,checkpoint=checkpoint,seed=seed)
        self.arm="v4_adaptive"
        self.adaptation=protocol["adaptation"]
        self.screened_decisions=0
        self.escalated_decisions=0
        self.escalated_completed=0
        self.escalated_incomplete=0
        self.screen_only_decisions=0
        self.fallback_to_screen=0
        self.decision_cpu_overrun=0

    def _predict(self,obs,plans,horizon,particles,seed,start):
        values=[]
        for plan in plans:
            if time.process_time()-start>=self.adaptation["max_cpu_per_decision_seconds"]:
                self.n_budget_exceeded+=1
                return None
            continuation=lambda state:_unique_plans(
                self.forecast_solver,state,candidates_for(state))[0]
            try:
                prediction=self.model.rollout(
                    obs,plan,continuation_policy=continuation,
                    continuation_policy_name="CP_SAT_public_prediction",
                    horizon_ticks=horizon,particles=particles,
                    sampling_seed=seed)
                values.append(float(prediction.distributions[-1].expected_completed_samples))
                self.n_model_queries+=1
                self.n_model_steps+=horizon*particles
                if time.process_time()-start>self.adaptation["max_cpu_per_decision_seconds"]:
                    # A synchronous model call has already overrun. We cannot
                    # interrupt it; record the precise fact and abort ranking.
                    self.n_budget_exceeded+=1
                    return None
            except (RuntimeError,ValueError,AssertionError) as exc:
                self.n_errors+=1
                self.decisions.append({"tick":obs.tick,"model_error":str(exc)[:220]})
                return None
        return values

    def __call__(self,raw,legal_candidates):
        started=time.process_time()
        obs=self.observer.observe(raw)
        plans=_unique_plans(self.solver,obs,legal_candidates)
        self.n_calls+=1
        if len(plans)>1:self.n_multi+=1
        selected=plans[0]
        stage="cp_sat"
        screen=None
        extended=None
        trigger=False
        if len(plans)>1:
            seed=1810+self.seed*17+obs.tick*101
            self.screened_decisions+=1
            screen=self._predict(
                obs,plans,self.adaptation["screen_horizon"],
                self.adaptation["screen_particles"],seed,started)
            if screen is None:
                self.n_fallback+=1
                stage="screen_incomplete"
            else:
                selected=plans[_best(screen)]
                stage="h5_screen"
                unfinished=_unfinished_jobs(obs)
                tied=max(screen)-min(screen)<=1e-9
                trigger=unfinished>=3 and tied
                if trigger:
                    self.escalated_decisions+=1
                    extended=self._predict(
                        obs,plans,self.adaptation["escalation_horizon"],
                        self.adaptation["escalation_particles"],seed,started)
                    if extended is None:
                        self.escalated_incomplete+=1
                        self.fallback_to_screen+=1
                        self.n_fallback+=1
                        stage="h20_incomplete_using_h5"
                    else:
                        self.escalated_completed+=1
                        selected=plans[_best(extended)]
                        stage="h20_escalated"
                else:
                    self.screen_only_decisions+=1
        self.n_flip+=int(selected!=plans[0])
        elapsed=time.process_time()-started
        self.cpu_decision_seconds+=elapsed
        self.decision_cpu_overrun+=int(elapsed>self.adaptation["max_cpu_per_decision_seconds"])
        self.decisions.append({
            "tick":obs.tick,
            "legal_plans":len(plans),
            "unfinished_jobs":_unfinished_jobs(obs),
            "selected_index":plans.index(selected),
            "stage":stage,"trigger":trigger,
            "h5_prediction":screen,"h20_prediction":extended,
            "process_cpu_seconds":elapsed,
        })
        return selected


def adaptive_one(spec,proto,checkpoint):
    business,geometry=build_business(spec,max_ticks=proto["max_ticks"])
    t0=time.process_time()
    policy=AdaptiveHorizonPolicy(
        geometry,business.rules,proto,checkpoint=checkpoint,seed=spec["seed"])
    init_seconds=time.process_time()-t0
    start=time.process_time()
    factual=BusinessSimulator(
        business,"world",spec["seed"],collect_metrics=True).run(
        dispatch_policy=policy,on_observation=policy.observe)
    execution_seconds=time.process_time()-start
    audit=resource_audit(factual,business.rules)
    motion=discrete_safety_violations(factual)
    lifecycle=lifecycle_audit(factual["history"])
    if audit["violations"] or motion or lifecycle:
        raise AssertionError("real dynamic transport safety/workflow violation")
    return {
        "arm":"v4_adaptive","seed":spec["seed"],"regime":spec["regime"],
        "status":factual["summary"]["status"],
        "makespan_ticks":factual["summary"]["ticks"],
        "job_metrics":_facts(factual,business),
        "resource_wait_ticks":audit["capacity_wait_ticks"],
        "blocked_moves":factual["summary"]["blocked_move_attempts"],
        "policy_calls":policy.n_calls,"multiple_candidates":policy.n_multi,
        "changed_plan_calls":policy.n_flip,
        "model_prediction_calls":policy.n_model_queries,
        "model_prediction_steps":policy.n_model_steps,
        "decision_budget_exceeded":policy.n_budget_exceeded,
        "fallbacks":policy.n_fallback,"prediction_errors":policy.n_errors,
        "model_init_cpu_seconds":init_seconds,
        "full_episode_cpu_seconds":init_seconds+execution_seconds,
        "policy_decision_cpu_seconds":policy.cpu_decision_seconds,
        "solver_fallbacks":policy.solver.fallbacks,
        "safety_violations":0,
        "screened_decisions":policy.screened_decisions,
        "escalated_decisions":policy.escalated_decisions,
        "escalated_completed":policy.escalated_completed,
        "escalated_incomplete":policy.escalated_incomplete,
        "screen_only_decisions":policy.screen_only_decisions,
        "fallback_to_screen":policy.fallback_to_screen,
        "actual_cpu_budget_overrun_decisions":policy.decision_cpu_overrun,
        "factual_event_sha256":hashlib.sha256(json.dumps(
            factual["events"],sort_keys=True).encode()).hexdigest(),
        "decision_records":policy.decisions,
    }


def _mean(vals):
    return sum(vals)/len(vals) if vals else None


def summarize(groups):
    result={}
    for regime in sorted({s["regime"] for s in groups}):
        items=[s for s in groups if s["regime"]==regime]
        arm_results={}
        for arm in ARMS:
            vals=[s["arms"][arm] for s in items]
            arm_results[arm]={
                "groups":len(vals),
                "mean_makespan":_mean([v["makespan_ticks"] for v in vals]),
                "mean_late_jobs":_mean([v["job_metrics"]["late_jobs"] for v in vals]),
                "mean_resource_wait_ticks":_mean([v["resource_wait_ticks"] for v in vals]),
                "completed":sum(not v["job_metrics"]["censored"] for v in vals),
                "mean_full_episode_cpu_seconds":_mean([v["full_episode_cpu_seconds"] for v in vals]),
                "model_steps":sum(v["model_prediction_steps"] for v in vals),
                "model_calls":sum(v["model_prediction_calls"] for v in vals),
                "changed_choices":sum(v["changed_plan_calls"] for v in vals),
                "budget_exceeded":sum(v["decision_budget_exceeded"] for v in vals),
                "fallbacks":sum(v["fallbacks"] for v in vals),
                "prediction_errors":sum(v["prediction_errors"] for v in vals),
                "safety_violations":sum(v["safety_violations"] for v in vals),
            }
            if arm=="v4_adaptive":
                arm_results[arm]["screened_decisions"]=sum(v["screened_decisions"] for v in vals)
                arm_results[arm]["escalated_decisions"]=sum(v["escalated_decisions"] for v in vals)
                arm_results[arm]["escalated_completed"]=sum(v["escalated_completed"] for v in vals)
                arm_results[arm]["escalated_incomplete"]=sum(v["escalated_incomplete"] for v in vals)
                arm_results[arm]["cpu_budget_overrun_decisions"]=sum(
                    v["actual_cpu_budget_overrun_decisions"] for v in vals)
        comparisons={}
        for arm in ARMS[1:]:
            pairs={}
            for base in ("cp_sat","v4_h20"):
                delta=[s["arms"][arm]["makespan_ticks"]-s["arms"][base]["makespan_ticks"] for s in items]
                pairs[base]={
                    "mean_makespan_delta":_mean(delta),
                    "wins":sum(v<0 for v in delta),
                    "ties":sum(v==0 for v in delta),
                    "losses":sum(v>0 for v in delta),
                }
            comparisons[arm]=pairs
        result[regime]={"groups":len(items),"arms":arm_results,"paired":comparisons}
    return result


def validate_new_groups(cfg):
    if (cfg["schema"]!="RHWM_ETM_ADAPTIVE_HORIZON_MPC_20261112_V1"
            or cfg["arms"]!=list(ARMS) or len(cfg["groups"])!=24):
        raise ValueError("precommitted scenes or arms changed")
    previous=json.loads((PROTOCOL.parents[1]/"posttraining"/"pr32_online_protocol.json").read_text())
    seen={s["seed"] for s in previous["groups"]}
    if len({s["seed"] for s in cfg["groups"]})!=24 or seen&{s["seed"] for s in cfg["groups"]}:
        raise AssertionError("PR32 seed leakage")
    from experiments.world_model_integration.posttraining.pr32_online_run import _no_seen_scenarios
    _no_seen_scenarios({"groups":cfg["groups"]})
    _scenario_seed_set({"scene_groups":cfg["groups"]})
    if cfg["adaptation"]["screen_horizon"]!=5 or cfg["adaptation"]["escalation_horizon"]!=20:
        raise AssertionError("H5/H20 protocol changed")
    if cfg["adaptation"]["screen_particles"]!=1 or cfg["adaptation"]["escalation_particles"]!=3:
        raise AssertionError("P1/P3 protocol changed")


def evaluate(checkpoint,*,protocol=PROTOCOL):
    cfg=json.loads(Path(protocol).read_text())
    validate_new_groups(cfg)
    if file_sha(checkpoint)!=cfg["original_checkpoint_sha256"]:
        raise ValueError("V4 source model changed")
    from torch import set_num_threads
    set_num_threads(1)
    start=time.process_time()
    groups=[]
    baseline_config={
        "max_ticks":cfg["max_ticks"],
        "candidate_limit":cfg["candidate_limit"],
        "particles":3,
        "decision_cpu_budget_seconds":cfg["adaptation"]["max_cpu_per_decision_seconds"],
    }
    for spec in cfg["groups"]:
        if time.process_time()-start>cfg["resource"]["CPU_seconds"]:
            raise TimeoutError("CPU budget exceeded before next independent scene")
        arms={}
        for arm in ARMS:
            if arm=="v4_adaptive":
                arms[arm]=adaptive_one(spec,cfg,checkpoint)
            else:
                arms[arm]=one(spec,baseline_config,arm,checkpoint)
        groups.append({"id":spec["id"],"regime":spec["regime"],
                       "seed":spec["seed"],"arms":arms})
    cpu=time.process_time()-start
    if cpu>cfg["resource"]["CPU_seconds"]:raise TimeoutError("CPU budget exceeded")
    return {
        "protocol":cfg["schema"],
        "protocol_sha256":file_sha(protocol),
        "v4_checkpoint_sha256":file_sha(checkpoint),
        "groups":groups,
        "summary":{
            "groups":len(groups),"by_regime":summarize(groups),
            "cpu_process_seconds":cpu,
            "gpu":0,"new_training":0,
            "method":"State-conditional event-triggered V4 H5 P1 -> optional H20 P3; Wang/Ren/Zheng 2023 conceptual method transfer, no stability claims",
            "limits":"Synthetic benchmarks; CPU process time cannot interrupt atomic rollouts; no causal field validation",
        },
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--checkpoint",required=True,type=Path)
    p.add_argument("--out",required=True,type=Path)
    a=p.parse_args()
    if a.out.exists():raise FileExistsError("write-once evidence")
    result=evaluate(a.checkpoint)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n")
    small={}
    for regime,s in result["summary"]["by_regime"].items():
        small[regime]={
            "arms":{arm:{k:v[k] for k in (
                "mean_makespan","mean_late_jobs","mean_full_episode_cpu_seconds",
                "budget_exceeded","model_steps","changed_choices")}
                for arm,v in s["arms"].items()},
            "adaptive_vs_fixed_h20":s["paired"]["v4_adaptive"]["v4_h20"],
        }
    print("ETM_ADAPTIVE_HORIZON_RESULT",json.dumps({
        "groups":result["summary"]["groups"],
        "cpu":result["summary"]["cpu_process_seconds"],
        "regimes":small},ensure_ascii=False))


if __name__=="__main__":
    main()
