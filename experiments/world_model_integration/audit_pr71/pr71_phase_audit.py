"""PR71: independent, NO CHECKPOINT audit of PR70 event-time claim.

Read original SHA-locked PR64 developer VALIDATION raw public transitions,
plus PR70's frozen service_timing.json. Separate pickup-to-in_transit from
delivery-to_completed. This is a read-only source/metric audit, not training.
The original PR61 trained network is NEVER instantiated by this script.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

EXPECTED_PR64_DEV_SHA = "31a36e0940db7afe015fe35689cb629102dea45e8d61123ed89250abc01d0b86"
MODEL_ARMS = ("map_off", "map_on")
MODEL_SEEDS = (1810, 1811, 1812)


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def per_scene_transitions(scene):
    """Timestamp actual next publicly observed phase transitions, no private state."""
    rows = scene["raw_actual_public_world_transitions"]
    ticks = {"pickup_to_in_transit": [], "delivery_to_completed": []}
    for row in rows:
        current_tick = row["input"]["public_tick"]
        next_tick = row["target"]["next_public_tick"]
        if next_tick != current_tick + 1:
            raise ValueError("not a chronological PR64 factual public transition")
        nodes = row["input"]["graph"]["nodes"]
        labels = row["target"]["node_target"]
        masks = row["target"]["node_mask"]
        if not len(nodes) == len(labels) == len(masks):
            raise ValueError("PR64 public nodes and target alignment differs")
        for node, y, m in zip(nodes, labels, masks):
            if node["kind"] != "task":
                continue
            if len(y) != 6 or len(m) != 6:
                raise ValueError("PR52 task head target dimensionality changed")
            # PR52 explicit one-hot target order:
            # waiting/assigned/in_transit/completed (indices 0..3).
            if not all(m[:4]):
                continue  # task disappeared; not a labeled transition
            if node["status"] == "assigned" and y[2] == 1:
                ticks["pickup_to_in_transit"].append(next_tick)
            elif node["status"] == "in_transit" and y[3] == 1:
                ticks["delivery_to_completed"].append(next_tick)
    return {
        "scene": scene["id"],
        "events": {name: len(t) for name, t in ticks.items()},
        "first_observed_tick": {
            name: min(t) if t else None for name, t in ticks.items()
        },
        "last_observed_tick": {
            name: max(t) if t else None for name, t in ticks.items()
        },
    }


def run(source_path, service_path):
    if sha256_file(source_path) != EXPECTED_PR64_DEV_SHA:
        raise ValueError("not original PR64 development validation data")
    data = json.loads(Path(source_path).read_text())
    if (data.get("split") != "validation"
        or data.get("BLIND_TEST_RUN") is not False
        or len(data["scenes"]) != 12):
        raise ValueError("wrong source split or OOD BLIND selected")
    frozen = json.loads(Path(service_path).read_text())
    if (frozen["source_sha256"] != EXPECTED_PR64_DEV_SHA
        or frozen["BLIND_TEST_read"] is not False
        or frozen["no_training"] is not True
        or len(frozen["per_scene"]) != 12):
        raise ValueError("PR70 frozen service report does not match original PR64 source")
    per_scene = [per_scene_transitions(s) for s in data["scenes"]]
    original = {s["scene"]: s for s in frozen["per_scene"]}
    if set(original) != {s["scene"] for s in per_scene}:
        raise ValueError("frozen PR70 source scene set changed")
    phase_timing = []
    for record in per_scene:
        scene = record["scene"]
        prior = original[scene]
        first_delivery = record["first_observed_tick"]["delivery_to_completed"]
        if prior["first_completed_observation_tick"] != first_delivery:
            raise AssertionError(
                "PR70 first final-completion label differs from original PR64 public labels")
        sampled = {
            f"{arm}_{seed}":
            prior["models"][f"{arm}_{seed}"]["first_model_sampled_completion_tick"]
            for arm in MODEL_ARMS for seed in MODEL_SEEDS
        }
        phase_timing.append({
            **record,
            "frozen_PR70_first_final_completed_observation_tick":
                prior["first_completed_observation_tick"],
            "frozen_PR70_first_any_phase_sampled_tick": sampled,
            "sampled_event_phase_in_PR70": "mixed_pickup_or_dropoff_UNRECORDED",
            "sampled_tick_is_not_a_final_delivery_timestamp": True,
        })
    return {
        "schema": "pr71-phase-aligned-PR64-public-event-audit-v1",
        "source": "original PR64 developer validation 12 real BusinessSimulator scenes",
        "source_sha256": sha256_file(source_path),
        "frozen_PR70_service_timing_sha256": sha256_file(service_path),
        "original_checkpoint_loaded": False,
        "training": False,
        "BLIND_TEST_read": False,
        "scenes": len(per_scene),
        "pickup_events": sum(r["events"]["pickup_to_in_transit"] for r in per_scene),
        "delivery_events": sum(r["events"]["delivery_to_completed"] for r in per_scene),
        "first_pickup_ticks": {
            s["scene"]: s["first_observed_tick"]["pickup_to_in_transit"]
            for s in per_scene
        },
        "first_delivery_ticks": {
            s["scene"]: s["first_observed_tick"]["delivery_to_completed"]
            for s in per_scene
        },
        "per_scene": phase_timing,
        "interpretation": (
            "PR70 first_model_sampled_completion_tick is the first sampled "
            "pickup OR delivery service event, but PR70 first completed observation "
            "tick is the first FINAL delivery. Comparing these without matching "
            "phase is invalid. PR70's aggregate model Brier remains a meaningful "
            "all-active-service conditional metric; phase-specific learned-head "
            "calibration cannot be recovered from its aggregated per-scene scores."
        ),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pr64-dev", required=True, type=Path)
    p.add_argument("--pr70-service", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError("do not overwrite PR71 independent evidence")
    result = run(args.pr64_dev, args.pr70_service)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print("PR71_REAL_PUBLIC_EVENT_PHASE_AUDIT", json.dumps({
        "scenes": result["scenes"],
        "pickup_events": result["pickup_events"],
        "delivery_events": result["delivery_events"],
        "first_pickup_ticks": result["first_pickup_ticks"],
        "first_delivery_ticks": result["first_delivery_ticks"],
        "no_training": True,
        "BLIND": False,
    }), flush=True)


if __name__ == "__main__":
    main()
