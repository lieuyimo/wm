"""PR64 CI-free repeatable stdlib-only reviewer of the original CPU gate.

Reads the already SHA-verified, archived PR64 true-source gate JSON and
prints exact independent-scenario bootstrap uncertainty, preserving the
original raw source and model/test seal. No simulator, no fitting or GPU.
"""
from __future__ import annotations
from pathlib import Path
import argparse
import hashlib
import json


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(gate_path,protocol_path,out):
    if out.exists():
        raise FileExistsError("do not silently replace a reviewer report")
    q=json.loads(gate_path.read_text())
    if (q.get("schema")!="pr64-private-randomized-onset-risk-set-prospective-data-quality-v1"
        or q.get("protocol_sha256")!=sha256(protocol_path)
        or q["source_quality_verdict"]!="PASS_TRUE_SHIFTED_OUTAGE_DATA_TO_REVIEW_MODEL_DESIGN"
        or q["failures"]
        or q["new_GPU_hours"]!=0
        or q["new_NEURAL_training_performed"]
        or q["original_pr59_unopened_TEST_generated"]
        or q["original_pr59_unopened_TEST_read"]):
        raise AssertionError("wrong or unaccepted original source read-only gate")
    result={
        "schema":"pr64-review-readonly-ci-interpretation-v1",
        "original_evidence_sha256":sha256(gate_path),
        "frozen_protocol_sha256":sha256(protocol_path),
        "source_verdict":q["source_quality_verdict"],
        "original_real_dev_events":q["source_counts"],
        "training_risk_set":q["train_risk_set"],
        "validation_risk_set":q["validation_risk_set"],
        "all_tick_validation_brier":{
            m:x["overall"]["brier"] for m,x in q["dev_val_full_risk_set_metrics"].items()
        },
        "independent_scenario_macro_and_bootstrap":{
            method:{
                "scene_macro_brier":value["scene_macro_brier"],
                "delta_vs_train_fitted_constant":value.get("paired_scene_macro_delta_vs_train_constant"),
                "scene_bootstrap_ci95":value.get("bootstrap_scenes_2000_ci95"),
                "scenes_better_than_constant":value.get("Brier_better_than_constant_in_scenes"),
                "independent_development_scenarios":value["n_independent_scenes"]
            } for method,value in q["paired_independent_scene_macro_bootstrap"].items()
        },
        "all_original_12_BLIND_scenarios_remained_unopened":True,
        "neural_GPU_training_or_inference":False,
        "important_limit":"New TRAIN vs new dev VAL have distinct event time supports AND disjoint map families and seeds, so cannot isolate event-time causality alone. Fixed simulator failure start rule was intentionally changed for this source, duration support/robots/CP-SAT fixed. Original PR61 neural checkpoint NOT evaluated on new shifted-time source. These exact shifted-time validation outcomes have now been inspected; they are development data, never a fresh blind test."
    }
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n")
    print("PR64_REVIEW_LOCKED_SOURCE_CONDITIONAL_RISK",json.dumps(result,ensure_ascii=False),flush=True)
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--quality",required=True,type=Path)
    p.add_argument("--protocol",required=True,type=Path)
    p.add_argument("--out",required=True,type=Path)
    a=p.parse_args()
    summarize(a.quality,a.protocol,a.out)


if __name__=="__main__":
    main()
