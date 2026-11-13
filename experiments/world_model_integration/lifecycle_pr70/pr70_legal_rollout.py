"""PR70 fully legal predicted-state H5/H20 rollout on factual initial states.

No factual actions or future states are used for action selection.  Each arm
starts from the same actual developer-validation public state, then chooses
nearest legal actions on its own predicted world, with common exogenous seeds
across Rule/map_off/map_on.  This is a legal-policy rollout comparison, not a
factual H20 accuracy evaluation.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import torch

from lab_transport.decision_value import candidates_for, nearest_batch
from experiments.world_model_integration.audit_pr63.pr63_risk_set import extract_records
from experiments.world_model_integration.lifecycle_pr70.pr70_lifecycle_eval import factual_trace
from experiments.world_model_integration.neural_pr67.pr67_neural_world import NeuralTaskServiceRule
from experiments.world_model_integration.outage_pr64.pr64_gate import fit
from experiments.world_model_integration.outage_pr64.pr64_source import read_protocol
from experiments.world_model_integration.recovery_pr68.pr68_checkpoint import (
    load_original_models, sha256_file,
)
from experiments.world_model_integration.stochastic_pr66.pr66_cpu_admission import verify
from experiments.world_model_integration.stochastic_pr66.pr66_rule_rollout import (
    PublicHazardRuleRollout,
)
from experiments.world_model_integration.stress_pr48.pr48_scenarios import PROTOCOL as PR48_PROTOCOL


def summarize_trace(trace, events):
    last = trace[-1]
    return {
        "legal_steps": len(trace),
        "full_horizon": True,
        "terminal_completed_visible_stages": sum(s.status == "completed" for s in last.samples),
        "terminal_waiting_visible_stages": sum(s.status == "waiting" for s in last.samples),
        "terminal_busy_ports": sum(int(s.resource_occupied) for s in last.stations),
        "terminal_active_services": sum(s.service_in_progress for s in last.stations),
        "device_changes": sum(len(e.changed_stations) for e in events),
        "all_states_legal": True,
    }


def bootstrap_scene_paired(scene_values, b=2000, seed=7001):
    ids = sorted(scene_values)
    rng = random.Random(seed)
    vals = [scene_values[s] for s in ids]
    boots = []
    for _ in range(b):
        sample = [vals[rng.randrange(len(vals))] for _ in vals]
        boots.append(sum(sample) / len(sample))
    boots.sort()
    return {
        "n_scenes": len(ids),
        "mean": sum(vals) / len(vals),
        "ci95_low": boots[int(0.025 * (b - 1))],
        "ci95_high": boots[int(0.975 * (b - 1))],
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--archive", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--original-validation", type=Path, required=True)
    p.add_argument("--pr64-train", type=Path, required=True)
    p.add_argument("--pr64-dev", type=Path, required=True)
    p.add_argument("--pr64-quality", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--max-scenes", type=int, default=12)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite legal rollout evidence")
    torch.set_num_threads(1)
    torch.manual_seed(7001)
    quality = json.loads(args.pr64_quality.read_text())
    train = verify(args.pr64_train, quality, "train")
    dev = verify(args.pr64_dev, quality, "validation")
    train_records, _ = extract_records(train, "train")
    hazard = fit(train_records)  # TRAIN only; never DEV
    protocol, old = read_protocol()
    stress = json.loads(PR48_PROTOCOL.read_text())
    models, provenance = load_original_models(
        args.archive, args.manifest, args.original_validation)
    runs = []
    replays = []
    for scene in dev["scenes"][:args.max_scenes]:
        observations, actions, geometry, rules, meta = factual_trace(
            scene, protocol, old, stress)
        initial = observations[0]
        replays.append({**meta, "initial_tick": initial.tick})
        for horizon in (5, 20):
            for seed in (1810, 1811, 1812):
                common = 700000 + seed + horizon * 100
                for arm in ("rule", "map_off", "map_on"):
                    dynamic = None
                    if arm != "rule":
                        dynamic = NeuralTaskServiceRule(
                            geometry, rules, models[(arm, seed)], seed=common)
                        dynamic.weight_origin = "original_PR61_verified_best_pt"
                    rollout = PublicHazardRuleRollout(
                        geometry, rules, hazard, "constant", world_dynamics=dynamic)
                    planner = lambda state, _: nearest_batch(
                        state, candidates_for(state), geometry)
                    trace, events = rollout.rollout(
                        initial, planner, horizon=horizon, seed=common)
                    if len(trace) != horizon:
                        raise AssertionError("legal rollout did not complete horizon")
                    summary = summarize_trace(trace, events)
                    summary["full_horizon"] = True
                    summary.update({
                        "scene": scene["id"], "arm": arm,
                        "seed": seed, "horizon": horizon,
                        "common_exogenous_rng_seed": common,
                        "neural_forward_calls": 0 if dynamic is None else dynamic.forward_calls,
                        "neural_active_service_queries": 0 if dynamic is None else dynamic.service_query_calls,
                    })
                    runs.append(summary)
                    print("PR70_LEGAL_RUN", json.dumps({
                        "scene": scene["id"], "arm": arm, "seed": seed,
                        "horizon": horizon, "completed": summary["terminal_completed_visible_stages"],
                        "legal": summary["all_states_legal"],
                    }), flush=True)
    aggregate = {}
    scene_values = defaultdict(lambda: defaultdict(list))
    for horizon in (5, 20):
        for arm in ("rule", "map_off", "map_on"):
            sub = [r for r in runs if r["horizon"] == horizon and r["arm"] == arm]
            aggregate[f"{arm}_{horizon}"] = {
                "n": len(sub),
                "full_horizon": sum(r["full_horizon"] for r in sub),
                "terminal_completed_mean": sum(r["terminal_completed_visible_stages"] for r in sub) / len(sub),
                "terminal_completed_min": min(r["terminal_completed_visible_stages"] for r in sub),
                "terminal_completed_max": max(r["terminal_completed_visible_stages"] for r in sub),
                "terminal_waiting_mean": sum(r["terminal_waiting_visible_stages"] for r in sub) / len(sub),
                "terminal_busy_mean": sum(r["terminal_busy_ports"] for r in sub) / len(sub),
                "terminal_active_mean": sum(r["terminal_active_services"] for r in sub) / len(sub),
                "device_changes_mean": sum(r["device_changes"] for r in sub) / len(sub),
                "all_legal": all(r["all_states_legal"] for r in sub),
            }
            for r in sub:
                scene_values[(horizon, arm)][r["scene"]].append(r["terminal_completed_visible_stages"])
    # Scene-level paired: average seeds per scene, then bootstrap scenes.
    paired = {}
    for horizon in (5, 20):
        for arm in ("map_off", "map_on"):
            rule_scene = {s: sum(v) / len(v)
                          for s, v in scene_values[(horizon, "rule")].items()}
            arm_scene = {s: sum(v) / len(v)
                         for s, v in scene_values[(horizon, arm)].items()}
            diff = {s: arm_scene[s] - rule_scene[s] for s in rule_scene}
            paired[f"{arm}_{horizon}"] = {
                "rule_completed": bootstrap_scene_paired(rule_scene),
                "arm_completed": bootstrap_scene_paired(arm_scene),
                "arm_minus_rule": bootstrap_scene_paired(diff),
            }
    result = {
        "schema": "pr70-fully-legal-predicted-state-rollout-v1",
        "source": "PR64 developer validation; factual initial state only",
        "source_sha256": sha256_file(args.pr64_dev),
        "pr64_train_sha256": sha256_file(args.pr64_train),
        "original_validation_sha256": sha256_file(args.original_validation),
        "archive_sha256": provenance["archive_sha256"],
        "conditioning": {
            "initial_state": "factual public state at t=0",
            "actions": "nearest legal batch on each arm's own predicted state",
            "exogenous": "TRAIN-fitted PR66 hazard with common random seeds across arms",
            "factual_actions_or_future_states_used": False,
        },
        "runs": runs,
        "aggregate": aggregate,
        "scene_paired": paired,
        "actual_GPU_training_performed": False,
        "BLIND_TEST_read": False,
        "BLIND_TEST_generated": False,
        "all_inference_on_CPU": True,
        "limitations": [
            "legal-policy rollout description; no factual H20 ground-truth accuracy",
            "terminal completed stages are descriptive, not a scheduling benefit",
            "PR64 developer validation is not a new independent test",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print("PR70_LEGAL_REPORT", json.dumps({
        "runs": len(runs), "aggregate": aggregate,
        "BLIND": False, "GPU_training": False,
    }), flush=True)


if __name__ == "__main__":
    main()
