"""PR76 action/exogenous-matched factual-world controls and frozen service risks.

Source is the exact original PR64 TRAIN/DEV BusinessSimulator records replayed
with PR73 existing keyed station jitter=2, not a newly invented generator.

No DEV fitting, GPU, PR59 BLIND, original PR61 checkpoint substitution, or
unobserved future job/device input in the world predictor. The actual t+1
public state is used ONLY as a target, while the ACTUAL next public station
device is applied as explicitly oracle-conditioned exogenous boundary.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import math
import random

import torch

from business_transport.session_events import eligible_sessions
from business_transport.session_dynamics import SessionRuleDynamics
from business_transport.world_model.kaplan_meier import SurvivalEventRuleDynamics
from business_transport.dynamics import WorkflowDynamics
from lab_transport.business_session_learning import (
    encode_session_batch, load_session_model,
)
from lab_transport.torch_structured_adapter import ProgressAwareRule
from experiments.world_model_integration.synthesis_pr73.pr73_hazard_probe import (
    fit_train_only, probabilities as hazard_probabilities,
)
from experiments.world_model_integration.stochastic_pr66.pr66_rule_rollout import (
    lawful_rule_then_event,
)
from experiments.world_model_integration.audit_pr71.pr71_public_arrivals import (
    apply_current_public_stage1,
)

HORIZONS=(5,20)
ARMS=("rule","existing_deepsets","train_public_hazard")
PARTICLE_SEEDS=(1810,1811,1812)
ANCHORS=("initial_t0","first_active_service")
SOURCE_PR74_CHECKPOINT_SHA="a08b257c7f8b8a6191b954e9adb117ef8076f3ff93d5373c347d4735e1e45f02"


class ExistingTrainPublicHazard:
    """Thin adapter for PR73 Laplace-smoothed TRAIN-only empirical risk set.

    Query logic is literally the previously published PR73 `probabilities`,
    reused by the existing SurvivalEventRuleDynamics legal state adapter.
    This is NOT an invented survival training algorithm.
    """
    def __init__(self, original_train_rows):
        self.params=fit_train_only(original_train_rows)
        self.calls=0

    def query_probability(self, observation, query, active_elapsed, fallback_mean):
        if active_elapsed is None:
            raise ValueError("missing effective current public service age")
        self.calls+=1
        return hazard_probabilities(self.params,{
            "station":query["station_id"],
            "phase":query["phase"],
            "active_elapsed":active_elapsed,
            # PR73 diagnostic returns a separate original_rule slot; the
            # survival adapter reads ONLY the TRAIN-fitted public hazard.
            # A constant placeholder cannot affect the selected probability.
            "rule_event":0,
        })["public_elapsed_hazard"]


def load_exact_existing_model(path):
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    if digest!=SOURCE_PR74_CHECKPOINT_SHA:
        raise ValueError("original PR74 actual-trained checkpoint hash mismatch")
    model,meta=load_session_model(path,experimental=True)
    if (meta.get("online_verified") is not False or
        meta.get("not_PR61_continued_training") is not True or
        model.target[0].in_features!=48):
        raise AssertionError("unapproved PR74 original model metadata")
    model.eval()
    return model,meta


def make_dynamics(arm,geometry,rules,checkpoint,train_rows,seed):
    if arm=="rule":
        return WorkflowDynamics(ProgressAwareRule(geometry),rules),None
    if arm=="existing_deepsets":
        core=SessionRuleDynamics(geometry,rules,checkpoint,
                                 experimental=True,seed=seed,
                                 feature_version="v2")
        if core.metadata.get("online_verified") is not False:
            raise AssertionError("online checkpoint unexpectedly admitted")
        return WorkflowDynamics(core,rules),core
    if arm=="train_public_hazard":
        core=SurvivalEventRuleDynamics(
            geometry,rules,ExistingTrainPublicHazard(train_rows),seed=seed)
        return WorkflowDynamics(core,rules),core
    raise ValueError("unknown frozen matched-world arm")


def strict_state_error(predicted, factual):
    """Symmetric public task identity/state error; don't ignore missing stage2.

    Both missing and premature extra stages are counted against the union.
    No comparison ever reads private remaining service ticks.
    """
    if predicted.tick!=factual.tick:
        raise AssertionError("predicted and factual public tick mismatch")
    ps={s.id:s for s in predicted.samples}
    fs={s.id:s for s in factual.samples}
    if len(ps)!=len(predicted.samples) or len(fs)!=len(factual.samples):
        raise AssertionError("duplicate task in matched source")
    keys=set(ps)|set(fs)
    mismatch=sum(
        sid not in ps or sid not in fs or ps[sid].status!=fs[sid].status
        for sid in keys
    )
    pr={r.id:r for r in predicted.robots}
    fr={r.id:r for r in factual.robots}
    if set(pr)!=set(fr):
        raise AssertionError("factual robot fleet mismatched")
    completion_pred=sum(s.status=="completed" for s in predicted.samples)
    completion_true=sum(s.status=="completed" for s in factual.samples)
    active_pred=sum(bool(s.service_in_progress) for s in predicted.stations)
    active_true=sum(bool(s.service_in_progress) for s in factual.stations)
    return {
        "tick":predicted.tick,
        "visible_task_union":len(keys),
        "task_status_errors":mismatch,
        "task_status_error_rate":mismatch/len(keys) if keys else 0.,
        "missing_factual_tasks":len(set(fs)-set(ps)),
        "premature_extra_tasks":len(set(ps)-set(fs)),
        "completed_abs_error":abs(completion_pred-completion_true),
        "completed_signed_error":completion_pred-completion_true,
        "active_service_abs_error":abs(active_pred-active_true),
        "robot_position_error_count":sum(pr[r].position!=fr[r].position for r in fr),
        "robot_phase_error_count":sum(pr[r].phase!=fr[r].phase for r in fr),
    }


def first_active_anchor(observations,actions,max_horizon=20):
    # Precommitted earliest active service among factual current observations.
    # No inspection of prediction errors or completion labels at selection.
    for t in range(min(len(actions),len(observations)-max_horizon)):
        if any(reason is None for _,reason in eligible_sessions(
                observations[t],actions[t],require_active=True)):
            return t
    return None


def factual_conditioned_rollout(observations,actions,geometry,rules,
                                checkpoint,train_rows,arm,seed,anchor,horizon):
    if arm not in ARMS or horizon not in HORIZONS:
        raise ValueError("unregistered world comparison requested")
    if anchor<0 or anchor+horizon>=len(observations) or anchor+horizon>len(actions):
        return {"arm":arm,"horizon":horizon,"seed":seed,"anchor_tick":anchor,
                "legal_steps":0,"full_horizon":False,"failure_kind":"missing_factual_window",
                "errors":[],"conditioned_device_boundaries":0}
    dynamics,core=make_dynamics(arm,geometry,rules,checkpoint,train_rows,seed)
    state=observations[anchor]
    errors=[]
    public_arrival_count=0
    conditioned_device_boundaries=0
    actions_replayed=0
    failure=None
    for offset in range(horizon):
        t=anchor+offset
        factual_current=observations[t]
        factual_next=observations[t+1]
        factual_actions=tuple(actions[t])
        try:
            state,new=apply_current_public_stage1(
                state,factual_current,observations[t-1] if t else None,rules)
            public_arrival_count+=len(new)
            visible={s.id for s in state.samples}
            missing=sorted({a.sample_id for a in factual_actions}-visible)
            if missing:
                failure={
                    "kind":"factual_action_not_executable_in_predicted_world",
                    "offset":offset,"tick":state.tick,"missing_sample_ids":missing,
                }
                break
            actual_device={s.id:bool(s.device_available)
                           for s in factual_next.stations}
            if len(actual_device)!=len(state.stations) or {
                s.id for s in state.stations
            }!=set(actual_device):
                raise ValueError("factual exogenous station ID mismatch")
            # Station device t+1 here comes from factual public observation.
            # This is the PREDECLARED oracle-conditioned evaluation boundary,
            # not a future input to the learned service head.
            nxt=lawful_rule_then_event(
                state,factual_actions,next_device=actual_device,
                geometry=geometry,rules=rules,world_dynamics=dynamics)
            conditioned_device_boundaries+=1
            actions_replayed+=1
            errors.append(strict_state_error(nxt,factual_next))
            state=nxt
        except (ValueError,AssertionError,KeyError,TypeError) as exc:
            failure={"kind":"original_lawful_transition_rejected",
                     "offset":offset,"tick":state.tick,
                     "exception_type":type(exc).__name__,"message":str(exc)[:400]}
            break
    return {
        "arm":arm,"horizon":horizon,"seed":seed,"anchor_tick":anchor,
        "legal_steps":len(errors),"full_horizon":len(errors)==horizon,
        "failure_kind":None if failure is None else failure["kind"],
        "failure":failure,"errors":errors,
        "factual_actions_replayed":actions_replayed,
        "current_public_stage1_arrivals":public_arrival_count,
        "conditioned_device_boundaries":conditioned_device_boundaries,
        "offline_model_service_queries":core.query_calls if core is not None else 0,
        "factual_future_actions_are_conditioning":True,
        "next_device_is_factual_oracle_exogenous_condition":True,
        "stage2_future_truth_injected":False,
        "original_transition_checker_used":True,
    }


def frozen_teacher_forced(dev_rows,dev_meta,pretrained,train_rows):
    if len(dev_rows)!=len(dev_meta):
        raise AssertionError("same DEV event records and neural inputs not aligned")
    x,m,q=encode_session_batch(dev_rows)
    with torch.inference_mode():
        neural=torch.sigmoid(pretrained(x,m,q)).cpu().tolist()
    control=fit_train_only(train_rows)
    probabilities=[]
    per_scene=defaultdict(list)
    for r,meta,p in zip(dev_rows,dev_meta,neural):
        yy=int(r["label"])
        if yy!=meta["event"]:
            raise AssertionError("target label not an actual public service transition")
        h=hazard_probabilities(control,meta)["public_elapsed_hazard"]
        vals={"existing_deepsets":p,"train_public_hazard":h,
              "rule":float(meta["rule_event"])}
        item={
            "scene":meta["scene"],"phase":meta["phase"],"truth":yy,
            "actual_public_tick":r["public_tick"],
            "brier":{k:(v-yy)**2 for k,v in vals.items()},
            "probabilities":vals,
        }
        probabilities.append(item)
        per_scene[meta["scene"]].append(item)
    if len(per_scene)!=12:
        raise AssertionError("original 12 independently reproduced DEV worlds missing")
    summary={}
    for arm in ARMS:
        v=[(r["brier"][arm]) for r in probabilities]
        per_phase={}
        for phase in ("to_pickup","to_dropoff"):
            part=[r["brier"][arm] for r in probabilities if r["phase"]==phase]
            if not part:
                raise AssertionError("no phase support in original developer source")
            per_phase[phase]={"n":len(part),"brier":sum(part)/len(part)}
        summary[arm]={"n":len(v),"brier":sum(v)/len(v),
                      "by_phase":per_phase}
    rng=random.Random(76001)
    paired={}
    for arm in ("rule","train_public_hazard"):
        diffs=[
            sum(r["brier"]["existing_deepsets"]-r["brier"][arm]
                for r in rs)/len(rs)
            for name,rs in sorted(per_scene.items())
        ]
        boots=sorted(
            sum(diffs[rng.randrange(len(diffs))] for _ in diffs)/len(diffs)
            for _ in range(2000))
        paired[f"neural_minus_{arm}"]={
            "scene_macro_diff":sum(diffs)/len(diffs),
            "CI95":[boots[49],boots[1949]],
            "scenes_neural_better":sum(d<0 for d in diffs),
        }
    return {
        "n":len(dev_rows),"event_count":sum(r["label"] for r in dev_rows),
        "scenes":len(per_scene),"methods":summary,
        "scene_paired_bootstrap":paired,
        "model_weights_source":"PR74 exact original SHA verified; no re-training",
        "source_label":"factual t+1 public transition",
    }
