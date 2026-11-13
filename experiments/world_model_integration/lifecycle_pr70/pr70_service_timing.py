"""PR70 learned service-event timing audit on factual public active sessions.

No training.  For every factual developer-validation tick, use the actual public
observation and actual CP-SAT action; query the PR61 task head at the same
active-service risk set used by PR67, then compare its next-service probability
with the factual next status.  The rule transition is evaluated on the same
rows as a deterministic baseline.  No threshold fitting, no future states as
model input, no BLIND/TEST.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import torch

from business_transport.dynamics import WorkflowDynamics
from business_transport.session_events import eligible_sessions
from lab_transport.torch_structured_adapter import ProgressAwareRule
from experiments.world_model_integration.lifecycle_pr70.pr70_lifecycle_eval import (
    factual_trace,
)
from experiments.world_model_integration.neural_pr67.pr67_neural_world import (
    NeuralTaskServiceRule,
)
from experiments.world_model_integration.outage_pr64.pr64_source import read_protocol
from experiments.world_model_integration.recovery_pr68.pr68_checkpoint import (
    load_original_models, sha256_file,
)
from experiments.world_model_integration.stochastic_pr66.pr66_cpu_admission import verify
from experiments.world_model_integration.stress_pr48.pr48_scenarios import PROTOCOL as PR48_PROTOCOL

CLIP = 1e-6


def logloss(p, y):
    p = min(1.0 - CLIP, max(CLIP, float(p)))
    return -(y * math.log(p) + (1 - y) * math.log(1.0 - p))


def score(rows):
    if not rows:
        return {"n": 0}
    brier = sum((p - y) ** 2 for p, y in rows) / len(rows)
    loss = sum(logloss(p, y) for p, y in rows) / len(rows)
    return {
        "n": len(rows),
        "events": sum(y for _, y in rows),
        "pred_mean": sum(p for p, _ in rows) / len(rows),
        "brier": brier,
        "logloss": loss,
    }


def run_scene(scene, protocol, old, stress, models, seeds):
    observations, actions, geometry, rules, meta = factual_trace(
        scene, protocol, old, stress)
    rule = WorkflowDynamics(ProgressAwareRule(geometry), rules)
    report = {
        "scene": scene["id"], "regime": scene["case"]["regime"],
        "first_completed_observation_tick": meta["first_completed_observation_tick"],
        "models": {}, "rule": {},
    }
    per_model = {}
    for arm in ("map_off", "map_on"):
        for seed in seeds:
            dynamic = NeuralTaskServiceRule(geometry, rules, models[(arm, seed)],
                                            seed=690000 + seed)
            dynamic.weight_origin = "original_PR61_verified_best_pt"
            model_rows = []
            rule_rows = []
            first_model_complete = None
            first_model_sampled_tick = None
            for t in range(min(len(actions), len(observations) - 1)):
                action = actions[t]
                observation = observations[t]
                actual_next = observations[t + 1]
                actual_map = {s.id: s for s in actual_next.samples}
                rule_next = rule.predict_next(observation, action)
                rule_map = {s.id: s for s in rule_next.samples}
                dynamic.predict_next(observation, action)
                events = {e.sample_id: e for e in dynamic.last_neural_events}
                if first_model_sampled_tick is None and any(
                        e.sampled_complete for e in dynamic.last_neural_events):
                    first_model_sampled_tick = actual_next.tick
                for query, reason in eligible_sessions(observation, action,
                                                       require_active=True):
                    if reason is not None:
                        continue
                    event = events.get(query["sample_id"])
                    if event is None:
                        raise AssertionError("neural session query missing from active events")
                    target = "in_transit" if query["phase"] == "to_pickup" else "completed"
                    actual_sample = actual_map.get(query["sample_id"])
                    if actual_sample is None:
                        continue
                    y = int(actual_sample.status == target)
                    p = float(event.probability_complete)
                    model_rows.append((p, y))
                    if y and first_model_complete is None:
                        first_model_complete = actual_next.tick
                    rule_sample = rule_map.get(query["sample_id"])
                    if rule_sample is not None:
                        rp = 1.0 if rule_sample.status == target else 0.0
                        rule_rows.append((rp, y))
            per_model[(arm, seed)] = {
                "model": score(model_rows),
                "rule": score(rule_rows),
                "neural_forward_calls": dynamic.forward_calls,
                "neural_active_service_queries": dynamic.service_query_calls,
                "first_model_completion_tick": first_model_complete,
                "first_model_sampled_completion_tick": first_model_sampled_tick,
            }
    report["models"] = {f"{arm}_{seed}": v for (arm, seed), v in per_model.items()}
    # A clean rule-only pass for the per-scene rule baseline.
    rule_rows = []
    for t in range(min(len(actions), len(observations) - 1)):
        action = actions[t]
        observation = observations[t]
        actual_next = observations[t + 1]
        actual_map = {s.id: s for s in actual_next.samples}
        rule_next = rule.predict_next(observation, action)
        rule_map = {s.id: s for s in rule_next.samples}
        for query, reason in eligible_sessions(observation, action,
                                               require_active=True):
            if reason is not None:
                continue
            target = "in_transit" if query["phase"] == "to_pickup" else "completed"
            actual_sample = actual_map.get(query["sample_id"])
            rule_sample = rule_map.get(query["sample_id"])
            if actual_sample is None or rule_sample is None:
                continue
            y = int(actual_sample.status == target)
            rule_rows.append((1.0 if rule_sample.status == target else 0.0, y))
    report["rule"] = score(rule_rows)
    return report


def aggregate(reports):
    out = {}
    for arm in ("map_off", "map_on"):
        for seed in (1810, 1811, 1812):
            key = f"{arm}_{seed}"
            rows = [r["models"][key]["model"] for r in reports]
            rule = [r["models"][key]["rule"] for r in reports]
            out[key] = {
                "model_brier": sum(x["brier"] for x in rows) / len(rows),
                "model_logloss": sum(x["logloss"] for x in rows) / len(rows),
                "model_events": sum(x["events"] for x in rows),
                "model_n": sum(x["n"] for x in rows),
                "rule_brier": sum(x["brier"] for x in rule) / len(rule),
                "rule_events": sum(x["events"] for x in rule),
                "scenes": len(rows),
            }
    # Scene-level paired bootstrap by arm (mean over seeds inside each scene).
    arm_data = {}
    for arm in ("map_off", "map_on"):
        scene_model = []
        scene_rule = []
        for r in reports:
            ms = [r["models"][f"{arm}_{seed}"]["model"] for seed in (1810, 1811, 1812)]
            rs = [r["models"][f"{arm}_{seed}"]["rule"] for seed in (1810, 1811, 1812)]
            scene_model.append(sum(x["brier"] * x["n"] for x in ms) / sum(x["n"] for x in ms))
            scene_rule.append(sum(x["brier"] * x["n"] for x in rs) / sum(x["n"] for x in rs))
        arm_data[arm] = {"model": scene_model, "rule": scene_rule}
    rng = random.Random(7000)
    for arm, values in arm_data.items():
        diffs = [m - r for m, r in zip(values["model"], values["rule"])]
        boots = []
        for _ in range(2000):
            sample = [diffs[rng.randrange(len(diffs))] for _ in diffs]
            boots.append(sum(sample) / len(sample))
        boots.sort()
        out[f"{arm}_scene_paired"] = {
            "n_scenes": len(diffs),
            "model_minus_rule_brier": sum(diffs) / len(diffs),
            "ci95_low": boots[int(0.025 * (len(boots) - 1))],
            "ci95_high": boots[int(0.975 * (len(boots) - 1))],
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
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite service-timing evidence")
    torch.set_num_threads(1)
    torch.manual_seed(7000)
    quality = json.loads(args.pr64_quality.read_text())
    source = verify(args.pr64_dev, quality, "validation")
    scenes = source["scenes"][:args.max_scenes]
    protocol, old = read_protocol()
    stress = json.loads(PR48_PROTOCOL.read_text())
    models, provenance = load_original_models(
        args.archive, args.manifest, args.original_validation)
    reports = []
    for scene in scenes:
        report = run_scene(scene, protocol, old, stress, models, (1810, 1811, 1812))
        reports.append(report)
        print("PR70_SERVICE_SCENE", json.dumps({
            "scene": report["scene"],
            "first_completed_tick": report["first_completed_observation_tick"],
            "rule": report["rule"],
            "models": {k: v["model"] for k, v in report["models"].items()},
        }), flush=True)
    result = {
        "schema": "pr70-learned-service-event-timing-v1",
        "source": "PR64 developer validation; actual public observations/actions",
        "source_sha256": sha256_file(args.pr64_dev),
        "original_validation_sha256": sha256_file(args.original_validation),
        "archive_sha256": provenance["archive_sha256"],
        "conditioning": "actual current public observation + actual CP-SAT action; actual next status is label only",
        "no_training": True,
        "no_threshold_fitting": True,
        "BLIND_TEST_read": False,
        "BLIND_TEST_generated": False,
        "all_inference_on_CPU": True,
        "aggregate": aggregate(reports),
        "per_scene": reports,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print("PR70_SERVICE_AUDIT", json.dumps(result["aggregate"], indent=2), flush=True)


if __name__ == "__main__":
    main()
