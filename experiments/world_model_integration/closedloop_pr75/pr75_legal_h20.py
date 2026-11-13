"""PR75: actual trained PR74 Deep Sets v2 service events -> lawful H5/H20 worlds.

Original PR74 CPU checkpoint is mandatory and SHA-checked. Original Business
Simulator J2 public current-tick task arrivals are EXOGENOUS conditioning, not
a generative arrival predictor. Train-fitted PR66 device hazard supplies the
next station status; no future factual device information or factual actions.

Both arms use the SAME nearest action rule on their OWN predicted state and
common exogenous device random numbers. No claim of factual counterfactual
H20 accuracy or online CP-SAT decision gains from divergent policies.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import random
from pathlib import Path

import torch

from business_transport.dynamics import WorkflowDynamics
from business_transport.session_dynamics import SessionRuleDynamics
from experiments.world_model_integration.audit_pr71.pr71_public_arrivals import (
    apply_current_public_stage1,
)
from experiments.world_model_integration.audit_pr63.pr63_risk_set import extract_records
from experiments.world_model_integration.outage_pr64.pr64_gate import fit
from experiments.world_model_integration.outage_pr64.pr64_source import read_protocol
from experiments.world_model_integration.recovery_pr68.pr68_checkpoint import sha256_file
from experiments.world_model_integration.synthesis_pr73.pr73_pilot import (
    EXPECTED_TRAIN, EXPECTED_DEV, original_factual_episode,
)
from experiments.world_model_integration.stochastic_pr66.pr66_cpu_admission import verify
from experiments.world_model_integration.stochastic_pr66.pr66_rule_rollout import (
    PublicHazardRuleRollout, lawful_rule_then_event,
)
from experiments.world_model_integration.stress_pr48.pr48_scenarios import PROTOCOL as PR48_PROTOCOL
from lab_transport.decision_value import candidates_for, nearest_batch

EXPECTED_CHECKPOINT_SHA256 = "a08b257c7f8b8a6191b954e9adb117ef8076f3ff93d5373c347d4735e1e45f02"
ARMS = ("rule", "existing_deepsets")
SEEDS = (1810, 1811, 1812)
HORIZONS = (5, 20)


def summarize(state):
    return {
        "completed_stages": sum(s.status == "completed" for s in state.samples),
        "waiting_stages": sum(s.status == "waiting" for s in state.samples),
        "in_transit_stages": sum(s.status == "in_transit" for s in state.samples),
        "visible_stage1": sum(getattr(s, "stage_index", None) == 0 for s in state.samples),
        "visible_stage2": sum(getattr(s, "stage_index", None) == 1 for s in state.samples),
        "active_services": sum(s.service_in_progress for s in state.stations),
        "busy_ports": sum(int(s.resource_occupied) for s in state.stations),
    }


def run_one(observations, geometry, rules, device_train_hazard, checkpoint,
            arm, seed, horizon):
    if horizon not in HORIZONS or len(observations) < horizon:
        raise ValueError("insufficient original future public arrival stream")
    dynamics = None
    session = None
    if arm == "existing_deepsets":
        session = SessionRuleDynamics(
            geometry, rules, checkpoint, experimental=True, seed=seed)
        if (session.metadata.get("online_verified") is not False
            or session.query_input_schema != "business-service-query-input-v2"):
            raise AssertionError("PR74 checkpoint unexpectedly production-admitted or wrong schema")
        # Exact original legal WorkflowDynamics used for all business
        # successor stages, capacity and service progress.
        dynamics = WorkflowDynamics(session, rules)
    elif arm != "rule":
        raise ValueError("unknown frozen arm")
    hazard_rollout = PublicHazardRuleRollout(
        geometry, rules, device_train_hazard, "constant")
    state = observations[0]
    rng = random.Random(seed)
    arrivals = []
    sampled = 0
    phase_events = {"to_pickup": 0, "to_dropoff": 0}
    exogenous_changes = 0
    records = []
    for t in range(horizon):
        state, new = apply_current_public_stage1(
            state, observations[t], observations[t - 1] if t else None, rules)
        arrivals.extend({"tick": state.tick, "sample_id": sid} for sid in new)
        actions = tuple(nearest_batch(state, candidates_for(state), geometry))
        device_probs = hazard_rollout._probabilities(state, None)
        next_device = {sid: rng.random() < p for sid, p in device_probs.items()}
        exogenous_changes += sum(
            bool(station.device_available) != next_device[station.id]
            for station in state.stations
        )
        nxt = lawful_rule_then_event(
            state, actions, next_device=next_device, geometry=geometry,
            rules=rules, world_dynamics=dynamics)
        if session is not None:
            for event in session.last_service_events:
                if event["sampled_finished"]:
                    sampled += 1
                    phase_events[event["query"]["phase"]] += 1
        if nxt.tick != state.tick + 1:
            raise AssertionError("lawful world clock did not progress")
        records.append({"tick": nxt.tick, **summarize(nxt)})
        state = nxt
    return {
        "arm": arm,
        "seed": seed,
        "horizon": horizon,
        "full_horizon": len(records) == horizon,
        "all_business_states_legal": True,
        "public_current_arrival_conditioned": True,
        "future_factual_device_used": False,
        "factual_future_actions_used": False,
        "new_stage1_arrivals": arrivals,
        "external_device_changes": exogenous_changes,
        "service_query_calls": session.query_calls if session is not None else 0,
        "service_forward_calls": session.service_calls if session is not None else 0,
        "neural_sampled_finish": sampled,
        "neural_sampled_by_phase": phase_events,
        "terminal": summarize(state),
        "trace": records,
    }


def scene_comparison(runs, horizon):
    pairs = defaultdict(dict)
    for item in runs:
        if item["horizon"] != horizon:
            continue
        key = item["scene"], item["seed"]
        if item["arm"] in pairs[key]:
            raise AssertionError("duplicate arm/scene/seed")
        pairs[key][item["arm"]] = item
    if len(pairs) != 12 * len(SEEDS):
        raise AssertionError("missing original matched scenario/model runs")
    sums = defaultdict(list)
    chosen = 0
    for (scene, seed), x in pairs.items():
        if set(x) != set(ARMS):
            raise AssertionError("arms did not have common source state")
        left, right = x["rule"], x["existing_deepsets"]
        if (len(left["new_stage1_arrivals"]) != len(right["new_stage1_arrivals"])
            or left["external_device_changes"] < 0 or right["external_device_changes"] < 0):
            raise AssertionError("unbalanced public arrival context")
        diff = right["terminal"]["completed_stages"] - left["terminal"]["completed_stages"]
        sums[scene].append(diff)
        chosen += diff != 0
    values = [sum(v)/len(v) for v in sums.values()]
    rg = random.Random(75100+horizon)
    boots=sorted(
        sum(values[rg.randrange(len(values))] for _ in values)/len(values)
        for _ in range(2000)
    )
    return {
        "matched_scenario_seed_pairs": len(pairs),
        "n_independent_scenes": len(values),
        "paired_completed_stage_delta_neural_minus_rule": sum(values)/len(values),
        "scene_bootstrap_95CI": [boots[49], boots[1949]],
        "n_matched_pairs_terminal_completed_different": chosen,
        "semantics": (
            "descriptive difference between lawful predicted-state nearest-policy"
            " trajectories, NOT factual H20 accuracy or scheduling improvement"
        ),
    }


def run(train_path,dev_path,quality_path,checkpoint_path):
    if (sha256_file(train_path) != EXPECTED_TRAIN
        or sha256_file(dev_path) != EXPECTED_DEV):
        raise ValueError("original PR64 train/developer source SHA mismatch")
    if sha256_file(checkpoint_path) != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError("original PR74 actually-trained CPU model checkpoint SHA mismatch")
    quality = json.loads(quality_path.read_text())
    train = verify(train_path, quality, "train")
    dev = verify(dev_path, quality, "validation")
    if len(train["scenes"]) != 24 or len(dev["scenes"]) != 12:
        raise AssertionError("original source cohort incomplete")
    train_risks, _ = extract_records(train, "train")
    hazard = fit(train_risks)
    protocol, old = read_protocol()
    stress = json.loads(PR48_PROTOCOL.read_text())
    runs=[]
    torch.set_num_threads(1)
    for scene in dev["scenes"]:
        observations, actions, geometry, rules, factual, starts, durations = original_factual_episode(
            scene, protocol, old, stress, 2)
        if len(observations) < max(HORIZONS):
            raise AssertionError("original real J2 factual episode does not cover H20 public arrivals")
        for horizon in HORIZONS:
            for seed in SEEDS:
                common_seed = 750000 + seed + 100*horizon
                for arm in ARMS:
                    result = run_one(
                        observations, geometry, rules, hazard, checkpoint_path,
                        arm, common_seed, horizon)
                    result["original_train_seed"] = seed
                    result["scene"] = scene["id"]
                    runs.append(result)
                    print("PR75_ACTUAL_TRAINED_SERVICE_H20_LAWFUL_RUN",json.dumps({
                        "scene":scene["id"],"arm":arm,"horizon":horizon,"seed":seed,
                        "terminal":result["terminal"],
                        "sampled_events":result["neural_sampled_finish"],
                        "stage1_arrivals":len(result["new_stage1_arrivals"])}),flush=True)
    if len(runs) != 12 * 2 * len(SEEDS) * len(ARMS):
        raise AssertionError("unexpected number of pretrained model H5/H20 runs")
    for x in runs:
        if not x["full_horizon"] or not x["all_business_states_legal"]:
            raise AssertionError("predicted source violates original legal business transition")
    report = {
        "schema":"pr75-original-pr74-pretrained-service-deepsets-legal-H5-H20-v1",
        "old_PR61_weights_used":False,
        "original_PR74_pretrained_checkpoint_sha256":EXPECTED_CHECKPOINT_SHA256,
        "original_PR64_train_sha256":EXPECTED_TRAIN,
        "original_PR64_dev_sha256":EXPECTED_DEV,
        "train_only_device_hazard":True,
        "public_arrival_conditioning":"only actual current-tick stage1 arrivals, never future stage2",
        "factual_actions_used":False,
        "original_BLIND_opened":False,"GPU_training":False,"new_model_training":False,
        "scenes":12,
        "runs":runs,
        "comparison":{str(h):scene_comparison(runs,h) for h in HORIZONS},
    }
    return report


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--train",type=Path,required=True)
    p.add_argument("--dev",type=Path,required=True)
    p.add_argument("--quality",type=Path,required=True)
    p.add_argument("--checkpoint",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    if a.out.exists():
        raise FileExistsError("PR75 trained-model legal rollout evidence is immutable")
    result=run(a.train,a.dev,a.quality,a.checkpoint)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n")
    print("PR75_ORIGINAL_TRAINED_SERVICE_DEEPSETS_LAWFUL_H20_RESULT",json.dumps({
        "runs":len(result["runs"]),
        "completed_delta":result["comparison"]["20"],
        "H5":result["comparison"]["5"],
        "model_queries":sum(x["service_query_calls"] for x in result["runs"]),
        "model_sampled":sum(x["neural_sampled_finish"] for x in result["runs"]),
        "BLIND":False,"training":False,
    }),flush=True)


if __name__=="__main__":
    main()
