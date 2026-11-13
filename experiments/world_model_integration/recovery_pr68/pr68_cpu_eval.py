"""PR68 full CPU evaluator: ORIGINAL PR61 weights only, otherwise HARD STOP.

Stage A: reproduce original PR59 12-scene validation one-step performance
(teacher-forced, checkpoint selection validation, not a new test).
Stage B: original PR64 TRAIN/DEV *development* public-world replay and
H5/H20 rule-legal stochastic world recursion with three paired sources:
Rule, trained PR61 map_off, trained PR61 map_on (3 original model seeds).
The latter is a legal/distributional pilot, NOT ground-truth H20 accuracy:
the original factual CP-SAT action policy differs from rollout nearest.

No GPU, no training, no test/BLIND, no random network fallback.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import hashlib
import json
import math

import torch

from experiments.world_model_integration.recovery_pr68.pr68_checkpoint import (
    ORIGINAL_VAL_SHA256, load_original_models, sha256_file,
)
from experiments.world_model_integration.sealed_pr59.pr59_gpu_train import (
    load_actual_source, evaluate,
)
from experiments.world_model_integration.audit_pr63.pr63_risk_set import extract_records
from experiments.world_model_integration.outage_pr64.pr64_gate import fit
from experiments.world_model_integration.outage_pr64.pr64_source import read_protocol
from experiments.world_model_integration.stochastic_pr66.pr66_cpu_admission import (
    verify as verify_pr64_source, one_actual_public_world,
)
from experiments.world_model_integration.stochastic_pr66.pr66_rule_rollout import (
    PublicHazardRuleRollout,
)
from experiments.world_model_integration.neural_pr67.pr67_neural_world import (
    NeuralTaskServiceRule,
)
from experiments.world_model_integration.stress_pr48.pr48_scenarios import PROTOCOL as PR48_PROTOCOL
from lab_transport.decision_value import candidates_for, nearest_batch


def check_pr64_quality(path: Path) -> dict:
    quality = json.loads(path.read_text())
    if (quality.get("source_quality_verdict") !=
        "PASS_TRUE_SHIFTED_OUTAGE_DATA_TO_REVIEW_MODEL_DESIGN"
        or quality.get("failures")
        or quality.get("original_pr59_unopened_TEST_read") is not False
        or quality.get("original_pr59_unopened_TEST_generated") is not False):
        raise ValueError("PR64 developer source failed its original admission")
    return quality


def run(args):
    """Fail BEFORE writing any result if the actual checkpoint is missing."""
    out = Path(args.output)
    if out.exists():
        raise FileExistsError("never overwrite earlier checkpoint-dependent results")
    torch.set_num_threads(1)
    torch.manual_seed(6810)
    models, provenance = load_original_models(
        args.archive, args.manifest, args.original_validation)
    if sha256_file(args.original_validation) != ORIGINAL_VAL_SHA256:
        raise AssertionError("not original PR59 validation, BLIND is forbidden")
    episodes, sources = load_actual_source(
        [args.original_validation], "validation")
    if len(episodes) != 12:
        raise AssertionError("expected 12 original validation scenarios")
    teacher_forced = []
    with torch.inference_mode():
        for (arm, seed), model in sorted(models.items()):
            evidence = evaluate(model, episodes, "cpu", batch_size=32)
            original = next(x for x in provenance["models"]
                            if (x["arm"], x["seed"]) == (arm, seed))
            prior = original["original_val_metrics"]
            columns = ("macro_node_mse", "macro_device_brier",
                       "macro_new_task_mae")
            delta = {k: evidence[k] - prior[k] for k in columns}
            if any(not math.isfinite(v) or abs(v) > 1e-4 for v in delta.values()):
                raise AssertionError("restored PR61 model does not reproduce original validation")
            teacher_forced.append({
                "arm": arm, "seed": seed,
                "checkpoint_sha256": original["best_pt_sha256"],
                "one_step_validation_scenes": evidence["n_independent_scenarios"],
                "one_step_validation_public_transitions": evidence["all_raw_public_transitions"],
                "metrics": {k: evidence[k] for k in columns},
                "delta_vs_original_PR61_gpu_metrics": delta,
                "per_scenario": evidence["per_real_independent_scenario"],
            })

    quality = check_pr64_quality(args.pr64_quality)
    train = verify_pr64_source(args.pr64_train, quality, "train")
    dev = verify_pr64_source(args.pr64_dev, quality, "validation")
    rows, _ = extract_records(train, "train")
    hazard = fit(rows)  # never fit on DEV
    protocol, old = read_protocol()
    stress = json.loads(PR48_PROTOCOL.read_text())
    comparisons = []
    for split, source in (("train", train), ("developer_validation", dev)):
        source_split = "train" if split == "train" else "validation"
        state, public, geometry, rules, fact_status = one_actual_public_world(
            source, source_split, protocol, old, stress)
        planner = lambda s, _: nearest_batch(s, candidates_for(s), geometry)
        for horizon in (5, 20):
            for seed in (1810, 1811, 1812):
                common_seed = 680000 + seed + horizon * 100
                for arm in ("rule", "map_off", "map_on"):
                    dynamic = None
                    if arm != "rule":
                        dynamic = NeuralTaskServiceRule(
                            geometry, rules, models[(arm, seed)], seed=common_seed)
                        dynamic.weight_origin = "original_PR61_verified_best_pt"
                    rollout = PublicHazardRuleRollout(
                        geometry, rules, hazard, "constant", world_dynamics=dynamic)
                    trace, event_rows = rollout.rollout(
                        state, planner, horizon=horizon, seed=common_seed)
                    if len(trace) != horizon:
                        raise AssertionError("missing legal H steps")
                    last = trace[-1]
                    if any(s.resource_occupied > s.resource_capacity for s in last.stations):
                        raise AssertionError("business resource capacity violated")
                    comparisons.append({
                        "source": split,
                        "source_scene": source["scenes"][0]["id"],
                        "source_original_factual_policy": "PR64_true_CPSAT",
                        "evaluation_policy": "nearest_rule_on_each_predicted_world",
                        "arm": arm, "original_model_seed": seed,
                        "common_exogenous_rng_seed": common_seed,
                        "horizon": horizon,
                        "actual_pretrained_checkpoint_used": arm != "rule",
                        "neural_forward_calls": 0 if dynamic is None else dynamic.forward_calls,
                        "neural_active_service_queries": 0 if dynamic is None else dynamic.service_query_calls,
                        "sampled_device_changes": sum(len(v.changed_stations) for v in event_rows),
                        "terminal_completed_visible_stages": sum(
                            s.status == "completed" for s in last.samples),
                        "terminal_waiting_visible_stages": sum(
                            s.status == "waiting" for s in last.samples),
                        "terminal_busy_ports": sum(
                            int(s.resource_occupied) for s in last.stations),
                        "terminal_active_services": sum(
                            s.service_in_progress for s in last.stations),
                        "terminal_visible_task_count": len(last.samples),
                        "all_states_legal": True,
                    })
                    if dynamic and dynamic.forward_calls != horizon:
                        raise AssertionError("pretrained model skipped a recurrence step")

    # This report is development-stage only: one-step validation was used
    # for checkpoint selection and different action policies prevent any
    # honest factual H20 accuracy estimate from these scene comparisons.
    report = {
        "schema": "pr68-original-PR61-verified-CPU-closed-loop-evaluation-v1",
        "original_archive_provenance": provenance,
        "original_pr59_validation_sha256": sha256_file(args.original_validation),
        "original_pr64_train_sha256": sha256_file(args.pr64_train),
        "original_pr64_developer_validation_sha256": sha256_file(args.pr64_dev),
        "original_pr64_quality_sha256": sha256_file(args.pr64_quality),
        "PR61_teacher_forced_validation_reproductions": teacher_forced,
        "PR61_rule_mapoff_mapon_H5_H20_legal_recursion_pilot": comparisons,
        "teacher_forced_validation_is_not_new_independent_test": True,
        "pr64_developer_validation_is_not_new_independent_test": True,
        "H20_summaries_are_not_true_future_prediction_error": True,
        "PR61_GPU_training_performed": False,
        "actual_trained_PR61_weights_used": True,
        "all_inference_on_CPU": True,
        "BLIND_TEST_read": False,
        "BLIND_TEST_generated": False,
        "next_decision": "REVIEW_REAL_PRETRAINED_H20_TRACES_BEFORE_ANY_GPU_TRAINING",
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print("PR68_ORIGINAL_PR61_WEIGHTS_CPU_REPORT",
          json.dumps({
              "source_validation": 12,
              "original_ckpts": len(models),
              "one_step_models": len(teacher_forced),
              "legal_H5_H20_runs": len(comparisons),
              "actual_GPU_training_performed": False,
              "BLIND_test": False,
          }), flush=True)
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--archive", type=Path, required=True)
    p.add_argument("--manifest", type=Path, default=Path(
        "examples/evidence/pr60-gpu-training-20261112/checkpoint_manifest.csv"))
    p.add_argument("--original-validation", type=Path, required=True)
    p.add_argument("--pr64-train", type=Path, required=True)
    p.add_argument("--pr64-dev", type=Path, required=True)
    p.add_argument("--pr64-quality", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    run(a)


if __name__ == "__main__":
    main()
