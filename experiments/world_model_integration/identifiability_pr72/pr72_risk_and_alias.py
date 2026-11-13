"""PR72: source-verified public-information and phase-specific service risk audit.

Uses *only original factual PR64 TRAIN/DEV* on CPU and the exact original
PR54/PR61 graph encoding and public ProgressAwareRule. No trained PR61 weights.
Evaluates two separate public service targets (pickup vs delivery); measures
whether current GNN input drops public features used by the Rule; and gives a
controlled same-neural-input / different-rule-result witness.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import replace
import json
import math
from pathlib import Path

import torch

from business_transport.dynamics import WorkflowDynamics
from business_transport.session_events import eligible_sessions
from experiments.world_model_integration.lifecycle_pr70.pr70_lifecycle_eval import factual_trace
from experiments.world_model_integration.neural_pr67.pr67_neural_world import PublicGraphHistory
from experiments.world_model_integration.outage_pr64.pr64_source import read_protocol
from experiments.world_model_integration.recovery_pr68.pr68_checkpoint import sha256_file
from experiments.world_model_integration.stochastic_pr66.pr66_cpu_admission import verify
from experiments.world_model_integration.stress_pr48.pr48_scenarios import PROTOCOL as PR48_PROTOCOL
from experiments.world_model_integration.temporal_pr54.pr54_public_sequence import (
    graph_public_features,
)
from lab_transport.torch_structured_adapter import ProgressAwareRule

EXPECTED_DEV_SHA = "31a36e0940db7afe015fe35689cb629102dea45e8d61123ed89250abc01d0b86"
EXPECTED_TRAIN_SHA = "4c1157777117f82f7deb44f98e1d8255e95611dfcb41c6dfff9e14dd23aee183"
PHASE_TARGETS = {"to_pickup": "in_transit", "to_dropoff": "completed"}


def validate_source(train_path, dev_path, quality_path):
    if sha256_file(train_path) != EXPECTED_TRAIN_SHA:
        raise ValueError("original PR64 TRAIN SHA mismatch")
    if sha256_file(dev_path) != EXPECTED_DEV_SHA:
        raise ValueError("original PR64 developer validation SHA mismatch")
    quality = json.loads(Path(quality_path).read_text())
    train = verify(train_path, quality, "train")
    dev = verify(dev_path, quality, "validation")
    if len(train["scenes"]) != 24 or len(dev["scenes"]) != 12:
        raise AssertionError("unexpected source split size")
    return train, dev


def public_input_collision(before, action, query, rules, geometry,
                           previous_observation=None, previous_action=()):
    """A controlled *counterfactual* witness, not a real observed service pair.

    Only currently-public station effective duration is modified: PR61 graph
    omits it, while ProgressAwareRule consumes it. Do not call these two
    counterfactual public states observed real data or trained model results.
    """
    station_id = query["station_id"]
    robot = next(r for r in before.robots if r.id == query["robot_id"])
    if not isinstance(robot.service_active_elapsed, int):
        return None
    active = robot.service_active_elapsed
    low = float(active + 1)
    high = float(active + 7)
    if low <= 0:
        return None

    def variant(expected):
        return replace(before, stations=tuple(
            replace(s, expected_service_ticks=expected,
                    observed_effective_service_mean=expected)
            if s.id == station_id else s for s in before.stations
        ))

    a, b = variant(low), variant(high)
    graph_a, keys_a, features_a = graph_public_features(a, rules, action)
    graph_b, keys_b, features_b = graph_public_features(b, rules, action)
    if graph_a != graph_b or keys_a != keys_b or features_a != features_b:
        raise AssertionError("PR61 graph actually observes effective service duration")
    history_a = PublicGraphHistory(geometry, rules)
    history_b = PublicGraphHistory(geometry, rules)
    if previous_observation is not None:
        # Seed both exact PR61 PyG timelines using the SAME factual t-1
        # public state/action. Then compare the complete current t inputs.
        history_a.encode(previous_observation, previous_action)
        history_b.encode(previous_observation, previous_action)
    pyg_a, pyg_keys_a = history_a.encode(a, action)
    pyg_b, pyg_keys_b = history_b.encode(b, action)
    if pyg_keys_a != pyg_keys_b:
        raise AssertionError("PR61 PyG node identities differed")
    for field in ("x", "edge_index", "edge_type", "history",
                  "history_mask", "map_x", "map_edge_index",
                  "entity_to_map_index", "map_node_count"):
        if not torch.equal(getattr(pyg_a, field), getattr(pyg_b, field)):
            raise AssertionError("PR61 full original dual-graph PyG changed: " + field)
    rule = WorkflowDynamics(ProgressAwareRule(geometry), rules)
    xa = rule.predict_next(a, action)
    xb = rule.predict_next(b, action)
    aa = next(s.status for s in xa.samples if s.id == query["sample_id"])
    bb = next(s.status for s in xb.samples if s.id == query["sample_id"])
    if aa == bb:
        return None
    if aa != PHASE_TARGETS[query["phase"]]:
        raise AssertionError("low effective duration did not complete the known session")
    return {
        "phase": query["phase"],
        "public_tick": before.tick,
        "public_service_active_elapsed": active,
        "public_wall_service_elapsed": robot.service_elapsed,
        "counterfactual_effective_duration_A": low,
        "counterfactual_effective_duration_B": high,
        "PR61_full_current_graph_equal": True,
        "PR61_node_features_equal": True,
        "PR61_complete_history_map_relational_PyG_input_equal": True,
        "PR61_prior_input_seeded_at_t_minus_one": previous_observation is not None,
        "rule_next_status_A": aa,
        "rule_next_status_B": bb,
        "interpretation": (
            "Two differing currently public service estimates produce identical "
            "PR61 model graph inputs but different lawful rule next states."
        ),
        "not_a_factual_observation_pair": True,
    }


def scene_phase_risks(scene, protocol, old, stress, *, witness=False):
    observations, actions, geometry, rules, meta = factual_trace(
        scene, protocol, old, stress)
    rule = WorkflowDynamics(ProgressAwareRule(geometry), rules)
    phase = {x: {"n": 0, "events": 0, "rule_errors": 0,
                 "wall_not_equal_active": 0, "public_known_effective_mean": 0,
                 "public_mean_differs_from_nominal": 0,
                 "active_elapsed_zero": 0}
             for x in PHASE_TARGETS}
    collision = None
    # BusinessSimulator on_observation does not include a final t+1 callback
    # for every terminal action. Match PR70's exact observable risk boundary.
    for t in range(min(len(actions), len(observations) - 1)):
        action = actions[t]
        before, actual = observations[t:t + 2]
        actual_tasks = {s.id: s for s in actual.samples}
        queries = [q for q, reason in eligible_sessions(
            before, action, require_active=True) if reason is None]
        if not queries:
            continue
        predicted = rule.predict_next(before, action)
        predicted_tasks = {s.id: s for s in predicted.samples}
        stations = {s.id: s for s in before.stations}
        robots = {r.id: r for r in before.robots}
        for q in queries:
            p = q["phase"]
            target = PHASE_TARGETS[p]
            task = q["sample_id"]
            if task not in actual_tasks or task not in predicted_tasks:
                continue
            y = int(actual_tasks[task].status == target)
            y_rule = int(predicted_tasks[task].status == target)
            record = phase[p]
            record["n"] += 1
            record["events"] += y
            record["rule_errors"] += y != y_rule
            robot = robots[q["robot_id"]]
            station = stations[q["station_id"]]
            active, wall = robot.service_active_elapsed, robot.service_elapsed
            record["wall_not_equal_active"] += wall != active
            record["active_elapsed_zero"] += active == 0
            mean = station.observed_effective_service_mean
            record["public_known_effective_mean"] += mean is not None
            record["public_mean_differs_from_nominal"] += (
                mean is not None and abs(float(mean) - station.expected_service_ticks) > 1e-9
            )
            if collision is None and witness:
                collision = public_input_collision(
                    before, action, q, rules, geometry,
                    observations[t - 1] if t else None,
                    actions[t - 1] if t else ())
    return {
        "scene": scene["id"],
        "regime": scene["case"]["regime"],
        "risk_set": phase,
        "public_information_collision_witness": collision,
        "factual_trace_source": "original PR64 BusinessSimulator and CP-SAT",
    }


def raw_head_targets(scene):
    """Quantify all PR61-compatible task-next-label status supervision.

    These are same-schema PR64 factual targets; PR61 originally trained on
    PR59/60, not on these rows. Do NOT call them PR61 training labels.
    """
    status_targets = ("waiting", "assigned", "in_transit", "completed")
    transitions = Counter()
    nodes_total = 0
    for row in scene["raw_actual_public_world_transitions"]:
        nodes = row["input"]["graph"]["nodes"]
        targets = row["target"]["node_target"]
        mask = row["target"]["node_mask"]
        if not len(nodes) == len(targets) == len(mask):
            raise AssertionError("invalid PR52 public target alignment")
        for n, y, m in zip(nodes, targets, mask):
            if n["kind"] != "task" or len(y) != 6 or list(m[:4]) != [1] * 4:
                continue
            label_index = max(range(4), key=lambda i: y[i])
            if y[label_index] != 1 or sum(y[:4]) != 1:
                raise AssertionError("PR52 task head target is not one-hot")
            current, future = n["status"], status_targets[label_index]
            if current not in status_targets:
                raise AssertionError("unexpected current public task status")
            transitions[current + "->" + future] += 1
            nodes_total += 1
    return {"task_node_labels": nodes_total, "transitions": dict(transitions)}


def run(train_path, dev_path, quality_path):
    train, dev = validate_source(train_path, dev_path, quality_path)
    protocol, old = read_protocol()
    stress = json.loads(PR48_PROTOCOL.read_text())
    per_scene = []
    witness = None
    for scene in dev["scenes"]:
        record = scene_phase_risks(
            scene, protocol, old, stress, witness=witness is None)
        if witness is None:
            witness = record.pop("public_information_collision_witness")
        else:
            record.pop("public_information_collision_witness")
        per_scene.append(record)
    if witness is None:
        raise AssertionError("no controlled same-graph/different-rule service witness found")

    phase = {}
    for name in PHASE_TARGETS:
        rows = [r["risk_set"][name] for r in per_scene]
        keys = rows[0]
        total = {key: sum(r[key] for r in rows) for key in keys}
        if not total["n"]:
            raise AssertionError("no phase-specific active service risk rows")
        total.update({
            "event_rate": total["events"] / total["n"],
            "rule_brier": total["rule_errors"] / total["n"],
            "n_independent_scenes": len(per_scene),
        })
        phase[name] = total

    head_targets = {}
    for split, group in (("train", train), ("developer_validation", dev)):
        rows = [raw_head_targets(s) for s in group["scenes"]]
        counts = Counter()
        for item in rows:
            counts.update(item["transitions"])
        total = sum(counts.values())
        changes = sum(v for k,v in counts.items()
                      if k.split("->")[0] != k.split("->")[1])
        head_targets[split] = {
            "scenes": len(rows), "task_node_next_state_labels": total,
            "same_status_labels": total - changes,
            "changed_status_labels": changes,
            "changed_status_rate": changes / total if total else None,
            "next_status_pairs": dict(sorted(counts.items())),
        }
    return {
        "schema": "pr72-original-public-phase-risk-and-input-alias-v1",
        "PR64_TRAIN_SHA256": sha256_file(train_path),
        "PR64_DEV_SHA256": sha256_file(dev_path),
        "split": "original TRAIN target-schema audit and developer VAL risk-set replay",
        "train_fit_or_new_network_inference": False,
        "new_GPU_training": False,
        "BLIND_TEST_read": False,
        "phase_specific_actual_public_service_risks": phase,
        "PR61_task_head_supervision_same_schema_PR64_not_PR61_train": head_targets,
        "public_input_collision_witness": witness,
        "per_scene": per_scene,
        "interpretation": {
            "raw_PR61_task_head": "next public categorical task status softmax",
            "original_PR61_loss": "masked node MSE + 0.3 new-stage event MSE + 0.1 task-status CE; not a targeted service-completion hazard loss",
            "PR61_robot_node_features": "wall service_elapsed/20, NOT service_active_elapsed",
            "PR61_station_node_features": "device, occupancy, waiting, approaching; NOT observed_effective_service_mean",
            "rule_information": "public effective elapsed and observed effective mean, when known",
            "comparison": "Rule and neural inputs are not information matched on service completion",
            "limitation": "phase-specific trained PR61 model Brier cannot be derived from archived PR70 per-scene aggregates without original checkpoint",
        },
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pr64-train", type=Path, required=True)
    p.add_argument("--pr64-dev", type=Path, required=True)
    p.add_argument("--pr64-quality", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError("immutable PR72 output already exists")
    report = run(a.pr64_train, a.pr64_dev, a.pr64_quality)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print("PR72_REAL_PHASE_RISK_AND_INFORMATION_ALIAS_AUDIT", json.dumps({
        "phase": report["phase_specific_actual_public_service_risks"],
        "head": report["PR61_task_head_supervision_same_schema_PR64_not_PR61_train"],
        "witness": report["public_input_collision_witness"],
        "GPU": False, "BLIND": False,
    }), flush=True)


if __name__ == "__main__":
    main()
