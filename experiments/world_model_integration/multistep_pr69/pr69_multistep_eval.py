"""PR69 conditional multi-step prediction evaluation with existing PR61 weights.

No training.  Replays the original PR64 developer-validation scenes with their
factual CP-SAT action sequence and public device observations, then recurses the
PR61 model from t=0 for H5/H20.  At each step the model receives the factual
action and the factual next device boundary (oracle exogenous conditioning), so
this is a *conditional multi-step prediction* test, not an autonomous policy or
ground-truth online scheduling result.  Rule and persistence controls use the
same conditioning.  BLIND/TEST is never opened.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import replace
from pathlib import Path

import torch

from business_transport.dynamics import WorkflowDynamics
from business_transport.observation import BusinessObserver
from business_transport.simulation import BusinessSimulator, resource_audit
from lab_transport.torch_structured_adapter import ProgressAwareRule
from experiments.world_model_integration.decision_learning_pr51.pr51_labels import TrueCPsatBranch
from experiments.world_model_integration.neural_pr67.pr67_neural_world import NeuralTaskServiceRule
from experiments.world_model_integration.outage_pr64.pr64_source import (
    read_protocol, private_outage_shift,
)
from experiments.world_model_integration.recovery_pr68.pr68_checkpoint import (
    load_original_models, sha256_file,
)
from experiments.world_model_integration.stochastic_pr66.pr66_cpu_admission import verify
from experiments.world_model_integration.stochastic_pr66.pr66_rule_rollout import (
    apply_next_device_boundary,
)
from experiments.world_model_integration.stress_pr48.pr48_scenarios import PROTOCOL as PR48_PROTOCOL
from experiments.world_model_integration.topology_pr56.pr56_public_map import (
    build_topology_business,
)

HORIZONS = (5, 20)
ARMS = ("rule", "map_off", "map_on")


def factual_trace(scene, protocol, old, stress):
    case = scene["case"]
    business, geometry = build_topology_business(case, stress, old)
    allowed = protocol["splits"][case["split"]]["allowed_private_start_ticks"]
    business, starts, durations = private_outage_shift(business, case, allowed)
    observations = []
    actions = []
    observer = BusinessObserver()

    def on_obs(value):
        observations.append(observer.observe(value))

    def on_transition(tr):
        actions.append(tuple(tr.assignments))

    dispatcher = TrueCPsatBranch(geometry, business.rules, case["seed"],
                                 {"counterfactual_max_candidates": 4})
    fact = BusinessSimulator(business, "world", case["seed"],
                             collect_metrics=True).run(
        dispatch_policy=dispatcher, on_transition=on_transition,
        on_observation=on_obs)
    if (not observations or not actions or len(observations) != len(actions) + 1
            or resource_audit(fact, business.rules)["violations"]):
        raise AssertionError("factual source replay is incomplete or unsafe")
    return observations, actions, geometry, business.rules, {
        "scene": scene["id"], "regime": case["regime"],
        "starts": starts, "durations": durations,
        "ticks": fact["summary"]["ticks"],
        "status": fact["summary"]["status"],
    }


def score_state(predicted, actual):
    actual_samples = {s.id: s for s in actual.samples}
    predicted_samples = {s.id: s for s in predicted.samples}
    status_ok = 0
    missing = 0
    for sid, sample in actual_samples.items():
        other = predicted_samples.get(sid)
        if other is None:
            missing += 1
            continue
        if other.status == sample.status:
            status_ok += 1
    actual_robots = {r.id: r for r in actual.robots}
    predicted_robots = {r.id: r for r in predicted.robots}
    robot_pos = 0
    robot_phase = 0
    for rid, robot in actual_robots.items():
        other = predicted_robots.get(rid)
        if other is None:
            continue
        robot_pos += int(other.position == robot.position)
        robot_phase += int(other.phase == robot.phase)
    completed_pred = sum(s.status == "completed" for s in predicted.samples)
    completed_actual = sum(s.status == "completed" for s in actual.samples)
    return {
        "samples": len(actual_samples),
        "status_ok": status_ok,
        "status_accuracy": status_ok / len(actual_samples) if actual_samples else 1.0,
        "missing_samples": missing,
        "robots": len(actual_robots),
        "robot_position_ok": robot_pos,
        "robot_position_accuracy": robot_pos / len(actual_robots) if actual_robots else 1.0,
        "robot_phase_ok": robot_phase,
        "robot_phase_accuracy": robot_phase / len(actual_robots) if actual_robots else 1.0,
        "completed_predicted": completed_pred,
        "completed_actual": completed_actual,
        "completed_abs_error": abs(completed_pred - completed_actual),
        "tick": predicted.tick,
    }


def run_one(observations, actions, geometry, rules, arm, model, horizon, seed):
    if len(actions) < horizon or len(observations) < horizon + 1:
        return {"arm": arm, "horizon": horizon, "legal_steps": 0,
                "full_horizon": False, "error": "source shorter than horizon",
                "steps": []}
    current = observations[0]
    if arm == "rule":
        dynamic = WorkflowDynamics(ProgressAwareRule(geometry), rules)
    else:
        dynamic = NeuralTaskServiceRule(geometry, rules, model, seed=seed)
        dynamic.weight_origin = "original_PR61_verified_best_pt"
    steps = []
    legal_steps = 0
    for step in range(horizon):
        action = actions[step]
        action_ids = {a.sample_id for a in action}
        current_ids = {s.id for s in current.samples}
        if not action_ids.issubset(current_ids):
            return {
                "arm": arm, "horizon": horizon, "legal_steps": legal_steps,
                "full_horizon": False, "error": "action_references_missing_sample",
                "missing_action_samples": sorted(action_ids - current_ids),
                "steps": steps,
            }
        try:
            core = dynamic.predict_next(current, action)
            next_devices = {
                s.id: bool(next(x.device_available for x in observations[step + 1].stations
                                if x.id == s.id))
                for s in core.stations
            }
            if len(next_devices) != len(core.stations):
                raise AssertionError("actual device observation lacks station")
            predicted = apply_next_device_boundary(core, next_devices, rules)
        except Exception as exc:  # noqa: BLE001 - record the exact evaluation failure
            return {
                "arm": arm, "horizon": horizon, "legal_steps": legal_steps,
                "full_horizon": False, "error": type(exc).__name__,
                "error_message": str(exc), "steps": steps,
            }
        actual = observations[step + 1]
        steps.append(score_state(predicted, actual))
        current = predicted
        legal_steps += 1
    return {
        "arm": arm, "horizon": horizon, "legal_steps": legal_steps,
        "full_horizon": legal_steps == horizon, "error": None, "steps": steps,
    }


def aggregate(runs):
    out = {}
    for arm in ARMS:
        out[arm] = {}
        for horizon in HORIZONS:
            sub = [r for r in runs if r["arm"] == arm and r["horizon"] == horizon]
            if not sub:
                continue
            legal = [r for r in sub if r["full_horizon"]]
            prefix = [r for r in sub if r["legal_steps"] > 0]
            def avg(key):
                vals = [s[key] for r in prefix for s in r["steps"]]
                return sum(vals) / len(vals) if vals else None
            stepwise = {}
            for step in range(horizon):
                vals = [r["steps"][step] for r in prefix if len(r["steps"]) > step]
                stepwise[str(step)] = {
                    "n": len(vals),
                    "status_accuracy": (sum(v["status_accuracy"] for v in vals) / len(vals)
                                        if vals else None),
                    "robot_position_accuracy": (
                        sum(v["robot_position_accuracy"] for v in vals) / len(vals)
                        if vals else None),
                    "completed_abs_error": (
                        sum(v["completed_abs_error"] for v in vals) / len(vals)
                        if vals else None),
                }
            out[arm][str(horizon)] = {
                "scenes": len(sub),
                "full_horizon_scenes": len(legal),
                "full_horizon_rate": len(legal) / len(sub),
                "mean_legal_steps": sum(r["legal_steps"] for r in sub) / len(sub),
                "status_accuracy": avg("status_accuracy"),
                "robot_position_accuracy": avg("robot_position_accuracy"),
                "robot_phase_accuracy": avg("robot_phase_accuracy"),
                "completed_abs_error": avg("completed_abs_error"),
                "missing_samples": sum(v["missing_samples"] for r in prefix for v in r["steps"]),
                "errors": sorted({r["error"] for r in sub if r["error"]}),
                "stepwise": stepwise,
            }
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--archive", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--original-validation", type=Path, required=True)
    p.add_argument("--pr64-dev", type=Path, required=True)
    p.add_argument("--pr64-quality", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--max-scenes", type=int, default=12)
    p.add_argument("--seeds", type=int, nargs="+", default=[1810, 1811, 1812])
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite multi-step evidence")
    torch.set_num_threads(1)
    torch.manual_seed(6900)
    quality = json.loads(args.pr64_quality.read_text())
    source = verify(args.pr64_dev, quality, "validation")
    scenes = source["scenes"][:args.max_scenes]
    protocol, old = read_protocol()
    stress = json.loads(PR48_PROTOCOL.read_text())
    models, provenance = load_original_models(
        args.archive, args.manifest, args.original_validation)
    runs = []
    replays = []
    for scene in scenes:
        observations, actions, geometry, rules, meta = factual_trace(
            scene, protocol, old, stress)
        replays.append(meta)
        for horizon in HORIZONS:
            for seed in args.seeds:
                for arm in ARMS:
                    if arm == "rule":
                        model = None
                        run_seed = seed
                    else:
                        model = models[(arm, seed)]
                        run_seed = 690000 + seed + horizon * 100
                    run = run_one(observations, actions, geometry, rules,
                                  arm, model, horizon, run_seed)
                    run.update({"scene": scene["id"], "seed": seed})
                    runs.append(run)
                    print("PR69_MULTISTEP_RUN", json.dumps({
                        "scene": scene["id"], "arm": arm, "seed": seed,
                        "horizon": horizon, "legal_steps": run["legal_steps"],
                        "full": run["full_horizon"], "error": run["error"],
                    }), flush=True)
    report = {
        "schema": "pr69-factual-action-exogenous-conditioned-multistep-v1",
        "source": "PR64 developer validation (not new independent test)",
        "source_sha256": sha256_file(args.pr64_dev),
        "quality_sha256": sha256_file(args.pr64_quality),
        "original_validation_sha256": sha256_file(args.original_validation),
        "archive_sha256": provenance["archive_sha256"],
        "conditioning": {
            "actions": "original factual CP-SAT assignments at each step",
            "devices": "original factual next public device state at each step",
            "policy": "no action selection by the neural model",
            "note": "conditional multi-step prediction, not autonomous policy or factual H20 forecast",
        },
        "scenes": len(scenes),
        "seeds": args.seeds,
        "horizons": list(HORIZONS),
        "replays": replays,
        "runs": runs,
        "aggregate": aggregate(runs),
        "teacher_forced_weights": provenance,
        "actual_GPU_training_performed": False,
        "BLIND_TEST_read": False,
        "BLIND_TEST_generated": False,
        "all_inference_on_CPU": True,
        "limitations": [
            "factual actions and factual next device states are supplied at every step",
            "only start at t=0 of each scene; no later-window history seeding",
            "PR64 developer validation is not a new independent test",
            "model may stop when a factual action references a sample its predicted state lacks",
            "no claim about autonomous policy, factual H20 accuracy or scheduling benefit",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print("PR69_MULTISTEP_REPORT", json.dumps({
        "scenes": len(scenes), "runs": len(runs),
        "full_horizon": {arm: {h: report["aggregate"][arm][str(h)]["full_horizon_scenes"]
                               for h in HORIZONS} for arm in ARMS},
        "BLIND": False, "GPU_training": False,
    }), flush=True)


if __name__ == "__main__":
    main()
