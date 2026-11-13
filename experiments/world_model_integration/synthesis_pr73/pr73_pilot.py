"""PR73: original BusinessSimulator service-jitter mechanism and public service query audit.

Reuse Simulator._service_duration keyed service jitter without changing its
implementation. Compare baseline jitter=0 vs existing jitter_ticks=2 on
TRAIN/DEV source case families, with deterministic scenario seeds and unchanged
private outage schedules. Do NOT read BLIND, train a model, or infer from
unavailable PR61 weights. Current public state and candidate action are the
ONLY features passed to existing business service query encoder. Factual next
public observation is a LABEL only.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from dataclasses import replace
import json
from pathlib import Path

from business_transport.observation import BusinessObserver
from business_transport.simulation import BusinessSimulator, resource_audit
from business_transport.state_features import build_business_features
from business_transport.session_events import eligible_sessions, encode_query, QUERY_WIDTH
from business_transport.dynamics import WorkflowDynamics
from experiments.world_model_integration.decision_learning_pr51.pr51_labels import TrueCPsatBranch
from experiments.world_model_integration.outage_pr64.pr64_source import (
    read_protocol, private_outage_shift,
)
from experiments.world_model_integration.recovery_pr68.pr68_checkpoint import sha256_file
from experiments.world_model_integration.stochastic_pr66.pr66_cpu_admission import verify
from experiments.world_model_integration.stress_pr48.pr48_scenarios import PROTOCOL as PR48_PROTOCOL
from experiments.world_model_integration.topology_pr56.pr56_public_map import build_topology_business
from lab_transport.torch_structured_adapter import ProgressAwareRule
from experiments.world_model_integration.synthesis_pr73.pr73_hazard_probe import evaluate_train_dev

EXPECTED_TRAIN = "4c1157777117f82f7deb44f98e1d8255e95611dfcb41c6dfff9e14dd23aee183"
EXPECTED_DEV = "31a36e0940db7afe015fe35689cb629102dea45e8d61123ed89250abc01d0b86"
PHASES = {"to_pickup": "in_transit", "to_dropoff": "completed"}


def alter_jitter(world, jitter_ticks):
    """Set ONLY existing public Station.jitter_ticks in a factual simulation.

    Do not alter service_ticks, prior_ticks, geometry, scenario release clocks
    or any public field that could disclose the private jitter draw.
    """
    if jitter_ticks not in (0, 2):
        raise ValueError("only preregistered existing API jitter=0/2 pilot")
    scenario = world.scenario
    amended = replace(scenario, stations={
        sid: replace(s, jitter_ticks=jitter_ticks)
        for sid, s in scenario.stations.items()
    })
    assert amended.duration_jitter == scenario.duration_jitter
    return replace(world, scenario=amended)


def original_factual_episode(scene, protocol, old, stress, jitter_ticks):
    case = scene["case"]
    world, geometry = build_topology_business(case, stress, old)
    allowed = protocol["splits"][case["split"]]["allowed_private_start_ticks"]
    world, starts, durations = private_outage_shift(world, case, allowed)
    assert all((s.jitter_ticks or 0) == 0 for s in world.scenario.stations.values())
    world = alter_jitter(world, jitter_ticks)
    observations = []
    actions = []
    observer = BusinessObserver()

    def on_observation(raw):
        observations.append(observer.observe(raw))

    def on_transition(tr):
        actions.append(tuple(tr.assignments))

    policy = TrueCPsatBranch(
        geometry, world.rules, case["seed"],
        {"counterfactual_max_candidates": 4})
    factual = BusinessSimulator(
        world, "world", case["seed"], collect_metrics=True
    ).run(dispatch_policy=policy, on_transition=on_transition,
          on_observation=on_observation)
    if resource_audit(factual, world.rules)["violations"]:
        raise AssertionError("existing business resource safety violated")
    if len(observations) < 3 or len(actions) < 3:
        raise AssertionError("factual public episode unexpectedly empty")
    return observations, actions, geometry, world.rules, factual, starts, durations


def public_event_rows(observations, actions, geometry, rules):
    """Return only observable conditional input summaries and next event LABEL.

    Eligibility matches PR70 exact require_active=True risk definition.
    """
    rule = WorkflowDynamics(ProgressAwareRule(geometry), rules)
    rows = []
    query_counts = Counter()
    for t in range(min(len(actions), len(observations) - 1)):
        state = observations[t]
        action = actions[t]
        next_state = observations[t + 1]
        actual = {s.id: s for s in next_state.samples}
        eligible = [q for q, reason in eligible_sessions(
            state, action, require_active=True) if reason is None]
        if not eligible:
            continue
        payload = build_business_features(state, action, geometry, rules)
        if (payload["tick"] != state.tick
            or payload["unknown"]["true_remaining_service_countdown"]["known"]
            or payload["unknown"]["future_fault_schedule"]["known"]
            or payload["unknown"]["unreleased_external_jobs"]["known"]):
            raise AssertionError("private service countdown or future event in public input")
        prediction = rule.predict_next(state, action)
        predicted = {s.id: s for s in prediction.samples}
        stations = {s.id: s for s in state.stations}
        robots = {r.id: r for r in state.robots}
        for q in eligible:
            sid = q["sample_id"]
            if sid not in actual or sid not in predicted:
                continue
            features = encode_query(payload, q)
            if len(features) != QUERY_WIDTH or not all(-10 <= float(x) <= 10 for x in features):
                raise AssertionError("original service query encoding not representable")
            station = stations[q["station_id"]]
            robot = robots[q["robot_id"]]
            y = int(actual[sid].status == PHASES[q["phase"]])
            rule_y = int(predicted[sid].status == PHASES[q["phase"]])
            active = robot.service_active_elapsed
            if not isinstance(active, int) or active < 0:
                raise AssertionError("active service risk has invalid public progress")
            mean = station.observed_effective_service_mean
            # Real same-input conditional uncertainty under varying service
            # jitter is summarized by public station, phase, elapsed. This
            # coarse diagnostic is not a fitted probabilistic model.
            rows.append({
                "phase": q["phase"], "station": station.id,
                "active_elapsed": active, "wall_elapsed": robot.service_elapsed,
                "public_effective_mean_known": mean is not None,
                "public_effective_mean": mean,
                "event": y, "rule_event": rule_y,
                "query_width": len(features),
                "public_query_48": [float(v) for v in features],
                "known_active_progress_mask": True,
                "label_next_public_tick": next_state.tick,
                "public_tick": state.tick,
            })
            query_counts[q["phase"]] += 1
    return rows


def aggregate(rows):
    per_phase = {}
    for phase in PHASES:
        subset = [r for r in rows if r["phase"] == phase]
        if not subset:
            per_phase[phase] = {"n": 0}
            continue
        per_phase[phase] = {
            "n": len(subset),
            "events": sum(r["event"] for r in subset),
            "event_rate": sum(r["event"] for r in subset) / len(subset),
            "rule_mismatches": sum(r["rule_event"] != r["event"] for r in subset),
            "rule_brier": sum((r["rule_event"]-r["event"]) ** 2 for r in subset) / len(subset),
            "wall_active_mismatch": sum(r["wall_elapsed"] != r["active_elapsed"] for r in subset),
            "public_effective_mean_known": sum(r["public_effective_mean_known"] for r in subset),
        }
    strata = defaultdict(lambda: [0, 0])
    for row in rows:
        key = (row["station"], row["phase"], row["active_elapsed"])
        strata[key][0] += 1
        strata[key][1] += row["event"]
    mixed = [
        {"station": k[0], "phase": k[1], "active_elapsed": k[2],
         "n": v[0], "events": v[1]}
        for k, v in sorted(strata.items()) if 0 < v[1] < v[0]
    ]
    return {
        "eligible": len(rows),
        "events": sum(x["event"] for x in rows),
        "per_phase": per_phase,
        "coarse_public_strata_count": len(strata),
        "coarse_strata_with_both_event_outcomes": len(mixed),
        "mixed_examples": mixed[:16],
        "all_query_widths_original_48": all(x["query_width"] == 48 for x in rows),
    }


def one(scene, protocol, old, stress, jitter):
    obs, acts, geometry, rules, factual, starts, durations = original_factual_episode(
        scene, protocol, old, stress, jitter)
    rows = public_event_rows(obs, acts, geometry, rules)
    info = aggregate(rows)
    info.update({
        "scene": scene["id"],
        "jitter": jitter,
        "seed": scene["case"]["seed"],
        "ticks": factual["summary"]["ticks"],
        "real_status": factual["summary"]["status"],
        "public_station_exogenous_outage_start_ticks": [
            e["tick"] for e in factual["events"] if e["type"] == "station_unavailable"
        ][:20],
        "used_original_keyed_service_jitter": True,
        "private_jitter_draws_NOT_part_of_public_features": True,
    })
    for r in rows:
        r["scene"] = scene["id"]
    return info, rows


def balanced_scenes(source_scenes, n):
    # Choose scene ids in each original PR64 public regime. Fixed first-k
    # per regime, not chosen by observed service-event or model performance.
    regimes = sorted({s["case"]["regime"] for s in source_scenes})
    if n % len(regimes) != 0:
        raise ValueError("balance requires equal scenes in every original regime")
    each = n // len(regimes)
    subset = []
    for regime in regimes:
        group = [s for s in source_scenes if s["case"]["regime"] == regime]
        if len(group) < each:
            raise ValueError("original split insufficient for requested regime balance")
        subset.extend(group[:each])
    if len({s["id"] for s in subset}) != n:
        raise AssertionError("duplicate original scenario selected")
    return subset


def run(train, dev, quality, ntrain, ndev):
    if sha256_file(train) != EXPECTED_TRAIN or sha256_file(dev) != EXPECTED_DEV:
        raise ValueError("not immutable original PR64 TRAIN and developer VAL sources")
    q = json.loads(quality.read_text())
    tr = verify(train, q, "train")
    va = verify(dev, q, "validation")
    protocol, old = read_protocol()
    stress = json.loads(PR48_PROTOCOL.read_text())
    results = {"train": [], "developer_validation": []}
    train_rows = {0: [], 2: []}
    dev_rows = {0: [], 2: []}
    for split, data, n, bundle in (
        ("train", tr, ntrain, train_rows),
        ("developer_validation", va, ndev, dev_rows),
    ):
        for scene in balanced_scenes(data["scenes"], n):
            for jitter in (0, 2):
                report, rows = one(scene, protocol, old, stress, jitter)
                results[split].append(report)
                bundle[jitter].extend(rows)
                print("PR73_ORIGINAL_SIMULATOR_JITTER_PILOT", json.dumps({
                    "split": split, "scene": scene["id"], "jitter": jitter,
                    "event_risks": report["eligible"], "events": report["events"],
                    "rule_errors": sum(v["rule_mismatches"] for v in report["per_phase"].values())
                }), flush=True)
    return {
        "schema": "pr73-original-BusinessSimulator-keyed-service-jitter-data-quality-pilot-v1",
        "source_train_sha256": sha256_file(train),
        "source_dev_sha256": sha256_file(dev),
        "source": "original PR64 TRAIN/DEV scenario specs, reconstructed BusinessSimulator",
        "train_cases": ntrain,
        "dev_cases": ndev,
        "mechanism": "unchanged Simulator._service_duration per-sample/phase/station keyed SHA jitter (0 vs 2 ticks)",
        "data_generator_modified": False,
        "station_nominal_and_prior_ticks_unchanged": True,
        "same_scenario_seed_and_outage_protocol_across_jitter_arms": True,
        "new_network_training_or_inference": False,
        "BLIND_TEST_read": False,
        "jitter_private_draw_exposed_to_input": False,
        "event_risk_dataset": {
            split: [
                {"scene": r["scene"], "public_tick": r["public_tick"],
                 "label_next_public_tick": r["label_next_public_tick"],
                 "phase": r["phase"], "station_for_analysis_only": r["station"],
                 "public_query_48": r["public_query_48"],
                 "next_step_event_label": r["event"]}
                for r in bundle[2]
            ]
            for split, bundle in (("train", train_rows),
                                  ("developer_validation", dev_rows))
        },
        "event_risk_dataset_semantics": "factual public-only t action query; t+1 service event is LABEL ONLY; no private jitter draw, sample ID, RNG or future outage inputs",
        "original_query_encoder": "business_transport.session_events.encode_query 48d",
        "report": results,
        "train_only_discrete_hazard_dev_controls": {
            str(j): evaluate_train_dev(train_rows[j], dev_rows[j]) for j in (0, 2)
        },
        "all_phase_hazard_fit_from_TRAIN_only": True,
        "balanced_selection": "fixed first k/each original PR64 regime, without peeking at risk outcomes",
        "aggregates": {
            split: {str(j): aggregate(bundle[j]) for j in (0, 2)}
            for split, bundle in (("train", train_rows),
                                  ("developer_validation", dev_rows))
        },
        "limitations": [
            "small diagnostic pilot using existing scenario seed families, not a new independent model test",
            "jitter draws are keyed pseudorandom, NOT a deterministic public signal",
            "coarse mixed-label strata show uncertainty, not calibration or learnable benefit",
            "no PR61 pretrained inference and no decision benefit claimed",
        ],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=Path, required=True)
    ap.add_argument("--dev", type=Path, required=True)
    ap.add_argument("--quality", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--train-scenes", type=int, default=12)
    ap.add_argument("--dev-scenes", type=int, default=4)
    a = ap.parse_args()
    if (not 3 <= a.train_scenes <= 24 or a.train_scenes % 3
        or not 3 <= a.dev_scenes <= 12 or a.dev_scenes % 3):
        raise ValueError("original three-regime cohort requires 3..24 TRAIN and 3..12 DEV in multiples of 3")
    if a.output.exists():
        raise FileExistsError("PR73 output immutable once written")
    result = run(a.train, a.dev, a.quality, a.train_scenes, a.dev_scenes)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print("PR73_JITTER_PILOT_RESULTS", json.dumps({
        "train": result["aggregates"]["train"],
        "dev": result["aggregates"]["developer_validation"],
        "train_fitted_dev_hazard_controls": result["train_only_discrete_hazard_dev_controls"],
        "BLIND": False, "model_training": False,
    }), flush=True)


if __name__ == "__main__":
    main()
