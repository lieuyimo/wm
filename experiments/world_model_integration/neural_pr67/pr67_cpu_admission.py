"""PR67 original-source public world structural CPU admission.

Use the immutable PR64 TRAIN/DEV source checks, replay exactly the public
t0/t1 state via original BusinessSimulator, then run actual PR61 architecture
forward() at EVERY predicted step under legal H5/H20 stochastic device
recursion. PR61 GPU checkpoints are unavailable, therefore these tests use
RANDOM-INITIALIZED PR61 weights exclusively as a wiring stress test.

Not a trained world-model accuracy test, not predictive generalization,
not a checkpoint promotion, not a decision-performance experiment.
"""
from __future__ import annotations

from pathlib import Path
import argparse
import hashlib
import json

import torch

from experiments.world_model_integration.outage_pr64.pr64_gate import fit
from experiments.world_model_integration.audit_pr63.pr63_risk_set import extract_records
from experiments.world_model_integration.stochastic_pr66.pr66_cpu_admission import (
    verify, one_actual_public_world,
)
from experiments.world_model_integration.stochastic_pr66.pr66_rule_rollout import (
    PublicHazardRuleRollout,
)
from experiments.world_model_integration.neural_pr67.pr67_neural_world import (
    NeuralTaskServiceRule, PublicGraphHistory,
)
from experiments.world_model_integration.sealed_pr59.pr59_dual_model import (
    PublicMapTemporalRelationalWorld,
)
from experiments.world_model_integration.outage_pr64.pr64_source import read_protocol
from experiments.world_model_integration.stress_pr48.pr48_scenarios import PROTOCOL as PR48_PROTOCOL
from lab_transport.decision_value import nearest_batch, candidates_for


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(train_path, dev_path, quality_path, out_path):
    if Path(out_path).exists():
        raise FileExistsError("PR67 evidence must never overwrite existing results")
    quality = json.loads(Path(quality_path).read_text())
    if (quality.get("source_quality_verdict") !=
        "PASS_TRUE_SHIFTED_OUTAGE_DATA_TO_REVIEW_MODEL_DESIGN"
        or quality.get("failures") or quality.get("original_pr59_unopened_TEST_read")
        or quality.get("original_pr59_unopened_TEST_generated")):
        raise AssertionError("invalid original PR64 public source quality")
    train = verify(train_path, quality, "train")
    dev = verify(dev_path, quality, "validation")
    train_records, _ = extract_records(train, "train")
    hazard = fit(train_records)
    protocol, old = read_protocol()
    stress = json.loads(PR48_PROTOCOL.read_text())
    torch.set_num_threads(1)
    trials = []
    for group, source in (("train", train), ("validation", dev)):
        state, public, geom, rules, completed = one_actual_public_world(
            source, group, protocol, old, stress)
        original = source["scenes"][0]["raw_actual_public_world_transitions"][0]["input"]
        if state.tick != original["public_tick"]:
            raise AssertionError("real first public observation time differs")
        # Public entity order/IDs/static map are identical. Assigned candidate
        # field is allowed to differ when evaluating a different legal action.
        from experiments.world_model_integration.temporal_pr54.pr54_public_sequence import graph_public_features
        g, keys, feats = graph_public_features(state, rules, ())
        for expected, actual in zip(original["graph"]["nodes"], g["nodes"]):
            if {k:v for k,v in expected.items() if k!="selected"} != {k:v for k,v in actual.items() if k!="selected"}:
                raise AssertionError("public world graph entity source mismatch")
        if len(g["nodes"]) != len(original["graph"]["nodes"]):
            raise AssertionError("public world graph size/order mismatch")
        for seed in (6710, 6711):
            torch.manual_seed(seed)
            network = PublicMapTemporalRelationalWorld(
                hidden=48, history_enabled=True, map_enabled=True).cpu().eval()
            dynamic = NeuralTaskServiceRule(geom, rules, network, seed=seed)
            kernel = PublicHazardRuleRollout(
                geom, rules, hazard, "constant", world_dynamics=dynamic)
            planner = lambda s, _: nearest_batch(s, candidates_for(s), geom)
            for horizon in (5, 20):
                states, steps = kernel.rollout(
                    state, planner, horizon=horizon, seed=seed*100+horizon)
                if len(states) != horizon or len(steps) != horizon:
                    raise AssertionError("did not generate full legal H")
                if dynamic.forward_calls < horizon:
                    raise AssertionError("PR61 model not evaluated on each recursive state")
                if any(not torch.isfinite(torch.tensor([z for _, z in event.probabilities])).all()
                       for event in steps):
                    raise AssertionError("stochastic world risk is not finite")
                trials.append({
                    "source_split": group,
                    "public_source_id": source["scenes"][0]["id"],
                    "method": "PR61_random_init_map_on_conditioning_rule",
                    "model_seed": seed, "H": horizon,
                    "legal_world_steps": len(states),
                    "actual_PR61_architecture_forward_calls_cumulative": dynamic.forward_calls,
                    "actual_PR61_task_event_queries_cumulative": dynamic.service_query_calls,
                    "device_changes": sum(len(x.changed_stations) for x in steps),
                    "completed_stages_end": steps[-1].completed_stages_after,
                    "source_truth_status": completed,
                })
    report = {
        "schema": "pr67-original-source-legal-neural-conditioned-stochastic-recursion-cpu-v1",
        "PR64_train_sha256": file_sha(train_path),
        "PR64_dev_sha256": file_sha(dev_path),
        "PR64_quality_sha256": file_sha(quality_path),
        "trials": trials,
        "all_actual_PR61_architecture_forward": True,
        "all_rule_business_states_legal": True,
        "original_trained_PR61_checkpoint_used": False,
        "untrained_random_weights_only": True,
        "no_model_training": True,
        "gpu_used": False,
        "original_BLIND_opened": False,
        "NO_H20_learned_performance_claim": True,
        "verdict": "PASS_CPU_NEURAL_TO_LEGAL_WORLD_WIRING_ONLY_BLOCKED_PRETRAINED_H20",
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n")
    print("PR67_NEURAL_LEGAL_CPU_ADMISSION",json.dumps(report,ensure_ascii=False),flush=True)
    return report


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--train",type=Path,required=True)
    p.add_argument("--validation",type=Path,required=True)
    p.add_argument("--quality",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    x=p.parse_args()
    run(x.train,x.validation,x.quality,x.out)


if __name__=="__main__":
    main()
