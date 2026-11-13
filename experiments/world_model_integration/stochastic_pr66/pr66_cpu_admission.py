"""Original PR64 TRAIN/DEV world replay plus lawful CPU stochastic H20 pilot.

Read SHA-verified original PR64 sources, use TRAIN-only hazard. Re-create
ONE TRAIN and ONE developer-VAL *known* BusinessSimulator/CP-SAT source world
with the exact original PR64 configuration, observe t=0 through genuine
business hooks, and compare recorded original public graph data. Then
perform legal rule-constrained stochastic H5/H20 from that public state.
Original BLIND/TEST never read or generated. No GNN weights or GPU.
"""
from __future__ import annotations

from pathlib import Path
import argparse
import hashlib
import json

from business_transport.observation import BusinessObserver
from business_transport.simulation import BusinessSimulator, resource_audit
from experiments.world_model_integration.audit_pr63.pr63_risk_set import extract_records
from experiments.world_model_integration.decision_learning_pr51.pr51_labels import TrueCPsatBranch
from experiments.world_model_integration.outage_pr64.pr64_gate import fit
from experiments.world_model_integration.outage_pr64.pr64_source import (
    read_protocol, private_outage_shift,
)
from experiments.world_model_integration.stochastic_pr66.pr66_rule_rollout import (
    PublicHazardRuleRollout, initial_public_station_history,
)
from experiments.world_model_integration.stress_pr48.pr48_scenarios import PROTOCOL as PR48_PROTOCOL
from experiments.world_model_integration.temporal_pr54.pr54_public_sequence import PublicTransitionCollector
from experiments.world_model_integration.topology_pr56.pr56_public_map import (
    augment_public_transition, build_topology_business,
)
from lab_transport.decision_value import candidates_for, nearest_batch


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify(path, quality, split):
    data = json.loads(Path(path).read_text())
    if (data.get("schema") != "pr64-shifted-private-failure-event-original-business-public-source-v1"
        or data.get("split") != split or data.get("BLIND_TEST_RUN") is not False
        or sha(path) != quality["source_counts"][split]["source_sha256"]
        or len(data["scenes"]) != {"train": 24, "validation": 12}[split]):
        raise AssertionError("original source not SHA-verified and split-locked")
    return data


def one_actual_public_world(source, split, protocol, old, stress):
    actual = source["scenes"][0]
    case = actual["case"]
    if case["split"] != split:
        raise AssertionError("incorrect known-source split")
    business, geometry = build_topology_business(case, stress, old)
    allowed = protocol["splits"][split]["allowed_private_start_ticks"]
    business, _, _ = private_outage_shift(business, case, allowed)
    collector = PublicTransitionCollector(business.rules, window=4, max_steps=260)
    observer = BusinessObserver()
    public_states = []

    def observed(value):
        # This callback sees PUBLIC current-t observations only, not private
        # outages, RNG or future labels.
        if value.tick == 0:
            public_states.append(observer.observe(value))

    dispatcher = TrueCPsatBranch(geometry, business.rules, case["seed"],
                                 {"counterfactual_max_candidates": 4})
    fact = BusinessSimulator(business, "world", case["seed"], collect_metrics=True).run(
        dispatch_policy=dispatcher, on_transition=collector, on_observation=observed)
    if not public_states or not collector.rows or resource_audit(fact, business.rules)["violations"]:
        raise AssertionError("original business source replay failed")
    first = collector.rows[0]
    augment_public_transition(first, geometry)
    original_first = actual["raw_actual_public_world_transitions"][0]
    if first["input"] != original_first["input"] or first["target"] != original_first["target"]:
        raise AssertionError("real replay is not exactly the frozen PR64 public source")
    initial = public_states[0]
    hist = initial_public_station_history(first["input"], initial)
    if not hist or initial.tick != first["input"]["public_tick"]:
        raise AssertionError("missing current public world")
    return initial, first["input"], geometry, business.rules, fact["summary"]["status"]


def execute(train, validation, quality, report):
    if Path(report).exists():
        raise FileExistsError("do not silently overwrite prior results")
    q = json.loads(Path(quality).read_text())
    if (q.get("source_quality_verdict") != "PASS_TRUE_SHIFTED_OUTAGE_DATA_TO_REVIEW_MODEL_DESIGN"
        or q.get("failures") or q.get("new_GPU_hours") != 0
        or q.get("original_pr59_unopened_TEST_generated")
        or q.get("original_pr59_unopened_TEST_read")):
        raise AssertionError("PR64 original TRAIN/DEV gate failed")
    tr = verify(train, q, "train")
    va = verify(validation, q, "validation")
    train_records, _ = extract_records(tr, "train")
    params = fit(train_records)
    protocol, old = read_protocol()
    stress = json.loads(PR48_PROTOCOL.read_text())
    cohorts = []
    for split, source in (("train", tr), ("validation", va)):
        initial, public_input, geometry, rules, factual_status = one_actual_public_world(
            source, split, protocol, old, stress)
        for method in ("constant", "age_h4"):
            rollout = PublicHazardRuleRollout(geometry, rules, params, method)
            planner = lambda state, _: nearest_batch(state, candidates_for(state), geometry)
            for horizon in (5, 20):
                kwargs = ({"initial_public_input": public_input}
                          if method == "age_h4" else {})
                trace, records = rollout.rollout(
                    initial, planner, horizon=horizon, seed=6600+horizon, **kwargs)
                replay, again = rollout.rollout(
                    initial, planner, horizon=horizon, seed=6600+horizon, **kwargs)
                if trace != replay or records != again:
                    raise AssertionError("stochastic legal rule rollout not repeatable")
                if not all(st.tick == initial.tick+i+1 for i, st in enumerate(trace)):
                    raise AssertionError("rollout skipped legal t+1 chronology")
                if len(trace) != horizon:
                    raise AssertionError("not the declared full H horizon")
                cohorts.append({
                    "source_split": split,
                    "real_source_id": source["scenes"][0]["id"],
                    "original_business_status": factual_status,
                    "method": method, "H": horizon,
                    "all_steps_rule_and_device_boundary_legal": True,
                    "stochastic_worlds": 1,
                    "event_changes": sum(len(r.changed_stations) for r in records),
                    "end_completed_stages": records[-1].completed_stages_after,
                    "end_service_in_progress": records[-1].active_services_after,
                })
    out = {
        "schema": "pr66-rule-constrained-stochastic-public-world-cpu-pilot-v1",
        "original_train_sha256": sha(train),
        "original_dev_sha256": sha(validation),
        "original_quality_sha256": sha(quality),
        "original_worlds_replayed": 2,
        "real_source_public_t0_t1_input_and_label_identical": True,
        "traces": cohorts,
        "new_neural_training": False,
        "PR61_checkpoint_loaded": False,
        "gpu_used": False,
        "original_BLIND_opened": False,
        "lawful_rule_rollout_not_learned_GNN_H20": True,
        "scientific_verdict": "PASS_CPU_RULE_EXOGENOUS_BOUNDARY_ONLY_BLOCKED_GPU_LEARNED_H20",
    }
    Path(report).parent.mkdir(parents=True, exist_ok=True)
    Path(report).write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")
    print("PR66_WORLD_RECURSION_CPU", json.dumps(out, ensure_ascii=False), flush=True)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train", required=True, type=Path)
    p.add_argument("--validation", required=True, type=Path)
    p.add_argument("--quality", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()
    execute(args.train, args.validation, args.quality, args.out)


if __name__ == "__main__":
    main()
