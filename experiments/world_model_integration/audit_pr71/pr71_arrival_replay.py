"""PR71 ORIGINAL PR64 PUBLIC-arrival replay, Rule-only CPU experiments.

A. Factual-action, observed-device Rule H20 from all 12 developer-validation
   scenes with *only newly public current-tick stage1* emitted at that tick.
B. Rule rollout with fresh nearest legal actions and TRAIN-fitted hazard,
   conditioned on newly PUBLIC stage1 arrivals at each tick. This is
   **observed arrival stream conditioning**, NOT autonomous arrival prediction.

This script runs without the inaccessible local-only PR61 checkpoints.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from business_transport.dynamics import WorkflowDynamics
from business_transport.workflow_transition import check_business_transition
from lab_transport.torch_structured_adapter import ProgressAwareRule
from lab_transport.decision_value import candidates_for, nearest_batch
from experiments.world_model_integration.audit_pr63.pr63_risk_set import extract_records
from experiments.world_model_integration.lifecycle_pr70.pr70_lifecycle_eval import factual_trace
from experiments.world_model_integration.outage_pr64.pr64_gate import fit
from experiments.world_model_integration.outage_pr64.pr64_source import read_protocol
from experiments.world_model_integration.stochastic_pr66.pr66_cpu_admission import verify
from experiments.world_model_integration.stochastic_pr66.pr66_rule_rollout import (
    PublicHazardRuleRollout, apply_next_device_boundary, lawful_rule_then_event,
)
from experiments.world_model_integration.stress_pr48.pr48_scenarios import PROTOCOL as PR48_PROTOCOL
from experiments.world_model_integration.recovery_pr68.pr68_checkpoint import sha256_file
from experiments.world_model_integration.audit_pr71.pr71_public_arrivals import (
    apply_current_public_stage1,
)


def factual_action_rule_h20(observations, actions, geometry, rules):
    if len(observations) < 21 or len(actions) < 20:
        raise AssertionError("PR64 episode shorter than frozen H20")
    current = observations[0]
    dynamic = WorkflowDynamics(ProgressAwareRule(geometry), rules)
    arrivals = []
    first_complete_actual = next(
        (o.tick for o in observations if any(s.status == "completed" for s in o.samples)),
        None)
    for step in range(20):
        current, ids = apply_current_public_stage1(
            current, observations[step],
            observations[step - 1] if step else None, rules)
        arrivals.extend({"tick": current.tick, "sample_id": sid} for sid in ids)
        action = tuple(actions[step])
        next_rule = dynamic.predict_next(current, action)
        # Before *observing* factual device_{t+1}, verify a fully legal state
        # transition on the current public device and business constraints.
        check_business_transition(current, next_rule, geometry, rules, action)
        next_device = {
            s.id: bool(next(s2.device_available for s2 in observations[step + 1].stations
                            if s2.id == s.id))
            for s in next_rule.stations
        }
        current = apply_next_device_boundary(next_rule, next_device, rules)
        if current.tick != observations[step + 1].tick:
            raise AssertionError("post-boundary world clock differs")
    return {
        "full_h20": True, "legal_steps": 20,
        "arrivals": arrivals, "new_public_stage1_count": len(arrivals),
        "terminal_completed": sum(s.status == "completed" for s in current.samples),
        "actual_first_final_complete_tick": first_complete_actual,
        "factual_actions_conditioned": True,
        "factual_devices_conditioned": True,
        "factual_successors_never_injected": True,
    }


def nearest_rule_with_public_arrival_stream(observations, geometry, rules, hazard,
                                            horizon, seed):
    """Every arm can later reuse this exact arrival boundary and policy.

    This is *not* a generative arrival model; factual current-tick stage1
    public events are provided as exogenous conditioning at every step.
    Factual next device is NOT used here: TRAIN-fitted hazard samples it.
    """
    if len(observations) < horizon:
        raise ValueError("factual public arrival feed shorter than horizon")
    rollout = PublicHazardRuleRollout(geometry, rules, hazard, "constant")
    rand = random.Random(seed)
    state = observations[0]
    arrivals = []
    last_events = 0
    for step in range(horizon):
        state, ids = apply_current_public_stage1(
            state, observations[step],
            observations[step - 1] if step else None, rules)
        arrivals.extend({"tick": state.tick, "sample_id": sid} for sid in ids)
        action = tuple(nearest_batch(state, candidates_for(state), geometry))
        risks = rollout._probabilities(state, None)
        changed = {
            sid: rand.random() < p for sid, p in risks.items()
        }
        last_events += sum(
            station.device_available != changed[station.id]
            for station in state.stations)
        state = lawful_rule_then_event(
            state, action, next_device=changed, geometry=geometry, rules=rules)
    return {
        "full_horizon": True, "horizon": horizon,
        "legal_steps": horizon, "arrivals": arrivals,
        "new_public_stage1_count": len(arrivals),
        "terminal_completed": sum(s.status == "completed" for s in state.samples),
        "terminal_waiting": sum(s.status == "waiting" for s in state.samples),
        "terminal_stage1_visible": sum(getattr(s, "stage_index", None) == 0
                                       for s in state.samples),
        "device_changes": last_events,
        "factual_actions_used": False,
        "future_device_oracle_used": False,
        "public_current_arrival_stream_conditioning": True,
    }


def run(args):
    if args.output.exists():
        raise FileExistsError("never overwrite PR71 source-verified CPU evidence")
    quality = json.loads(args.pr64_quality.read_text())
    train = verify(args.pr64_train, quality, "train")
    dev = verify(args.pr64_dev, quality, "validation")
    train_rows, _ = extract_records(train, "train")
    hazard = fit(train_rows)   # public device hazard fitted TRAIN only
    protocol, old = read_protocol()
    stress = json.loads(PR48_PROTOCOL.read_text())
    result = []
    conditioned = []
    for scene in dev["scenes"][:12]:
        obs, actions, geometry, rules, meta = factual_trace(
            scene, protocol, old, stress)
        row = factual_action_rule_h20(obs, actions, geometry, rules)
        result.append({"scene": scene["id"], **row})
        for seed in (1810, 1811, 1812):
            for h in (5, 20):
                value = nearest_rule_with_public_arrival_stream(
                    obs, geometry, rules, hazard, h,
                    700000 + seed + h * 100)
                conditioned.append({
                    "scene": scene["id"], "seed": seed, **value
                })
        print("PR71_ORIGINAL_REAL_SCENE_RULE_H20", json.dumps({
            "scene": scene["id"], "stage1_arrivals": len(row["arrivals"]),
            "full": row["full_h20"]}), flush=True)
    if len(result) != 12 or not all(x["full_h20"] for x in result):
        raise AssertionError("original PR70 repaired Rule H20 regression did not reproduce")
    if len(conditioned) != 72 or not all(x["full_horizon"] for x in conditioned):
        raise AssertionError("PR71 legal public-arrival-conditioned rollout incomplete")
    doc = {
        "schema": "pr71-public-current-arrival-rule-reference-v1",
        "source": "immutable original PR64 TRAIN (hazard) / developer VAL (traces)",
        "source_dev_sha256": sha256_file(args.pr64_dev),
        "source_train_sha256": sha256_file(args.pr64_train),
        "PR61_checkpoint_needed": False,
        "new_training": False,
        "BLIND_TEST_read": False,
        "model_evaluated": False,
        "policy": "original nearest legal rule and train-fitted original PR66 hazard",
        "observed_stage1_arrivals": "only public actual observation AT current tick; separate oracle arrival input",
        "future_stage2_copied": False,
        "factual_action_rule": result,
        "nearest_rule_with_public_arrivals": conditioned,
        "outcome": {
            "rule_factual_conditioned_H20_legal": sum(x["full_h20"] for x in result),
            "rule_arrival_conditioned_legal_rollouts": len(conditioned),
            "distinct_developer_scenes": len(result),
            "factual_conditioned_stage1_arrival_injections": sum(
                x["new_public_stage1_count"] for x in result),
            "nearest_conditioned_H20_completed_mean": sum(
                x["terminal_completed"] for x in conditioned if x["horizon"] == 20) / 36,
            "nearest_conditioned_H20_visible_stage1_mean": sum(
                x["terminal_stage1_visible"] for x in conditioned if x["horizon"] == 20) / 36,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n")
    print("PR71_PUBLIC_ARRIVAL_CONDITIONED_RULE_CPU_PASS",
          json.dumps(doc["outcome"]), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pr64-train", type=Path, required=True)
    p.add_argument("--pr64-dev", type=Path, required=True)
    p.add_argument("--pr64-quality", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
