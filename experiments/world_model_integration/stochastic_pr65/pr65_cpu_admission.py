"""PR65 read-only CPU integration of PR61 one-step graph output + PR64 hazard.

Consumes the ORIGINAL SHA-verified PR64 TRAIN/DEV source. Fits a simple
TRAIN-only hazard with the ORIGINAL PR64 method, and confirms the adapter
reproduces its one-step device probabilities on every visible station risk
record. A RANDOM-INITIALIZED graph model is used only to smoke-test the
output wiring, not to make any model accuracy or pretrained-weight claim.

No new business simulation, no GPU, no neural fitting, no BLIND, no H20.
"""
from __future__ import annotations

from pathlib import Path
import argparse
import hashlib
import json
import math

import torch
from torch_geometric.data import Batch

from experiments.world_model_integration.audit_pr63.pr63_risk_set import extract_records
from experiments.world_model_integration.outage_pr64.pr64_gate import fit, _p
from experiments.world_model_integration.sealed_pr59.pr59_dual_model import (
    PublicMapTemporalRelationalWorld, actual_public_record_to_graph,
)
from experiments.world_model_integration.stochastic_pr65.pr65_public_device import (
    PublicStationHazardAdapter,
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_source(path, split, quality):
    src = json.loads(Path(path).read_text())
    if (src.get("schema") != "pr64-shifted-private-failure-event-original-business-public-source-v1"
        or src.get("split") != split or src.get("BLIND_TEST_RUN") is not False
        or sha(path) != quality["source_counts"][split]["source_sha256"]
        or len(src.get("scenes", [])) != {"train": 24, "validation": 12}[split]):
        raise AssertionError("original PR64 source SHA/split/BLIND contract rejected")
    return src


def run(train_path, dev_path, quality_path, out_path):
    if Path(out_path).exists():
        raise FileExistsError("never overwrite original reviewer results")
    q = json.loads(Path(quality_path).read_text())
    if (q.get("source_quality_verdict") != "PASS_TRUE_SHIFTED_OUTAGE_DATA_TO_REVIEW_MODEL_DESIGN"
        or q.get("failures") or q.get("new_GPU_hours") != 0
        or q.get("original_pr59_unopened_TEST_generated")
        or q.get("original_pr59_unopened_TEST_read")):
        raise AssertionError("original SHA-locked PR64 admission did not pass")
    train = verify_source(train_path, "train", q)
    val = verify_source(dev_path, "validation", q)
    train_records, _ = extract_records(train, "train")
    dev_records, _ = extract_records(val, "validation")
    params = fit(train_records)
    by_record = {(r["scene"], r["tick"], r["station"]): r for r in dev_records}
    if len(by_record) != len(dev_records):
        raise AssertionError("duplicate public scene, tick, station risk record")
    methods = ("constant", "age_h4")
    outputs = {}
    for mode in methods:
        adapter = PublicStationHazardAdapter(params, mode)
        errors = []
        visits = 0
        for scene in val["scenes"]:
            for row in scene["raw_actual_public_world_transitions"]:
                public = row["input"]
                t = public["public_tick"]
                for station in adapter.station_probabilities(public):
                    key = (scene["id"], t, station.station)
                    if key not in by_record:
                        continue  # masked station label is not an evaluation risk row
                    risk = by_record[key]
                    p = station.probability_next_available
                    if abs(p - _p(risk, mode, params)) > 1e-12:
                        raise AssertionError("adapter differs from frozen PR64 TRAIN-fitted risk")
                    if station.public_history_age_h4 != risk["age_h4"]:
                        raise AssertionError("public age differs from PR63 extractor")
                    errors.append((p - risk["target"]) ** 2)
                    visits += 1
        if visits != len(dev_records):
            raise AssertionError("adapter missed actual complete DEV risk set")
        brier = sum(errors) / len(errors)
        baseline = q["dev_val_full_risk_set_metrics"][mode]["overall"]["brier"]
        if not math.isclose(brier, baseline, rel_tol=0, abs_tol=1e-12):
            raise AssertionError("independent output integration changes frozen DEV Brier")
        outputs[mode] = {
            "true_station_risk_rows": visits,
            "adapter_brier": brier,
            "original_pr64_brier": baseline,
            "matched_original_train_only_probability_in_every_risk_row": True,
        }

    # 0 checkpoint loading. Real public graph goes through real PyG relation/map
    # model, but random-init forward tests only that the adapter preserves all
    # other learned channels, NOT whether network predictions are good.
    torch.manual_seed(6500)
    torch.set_num_threads(1)
    model = PublicMapTemporalRelationalWorld(
        hidden=48, history_enabled=True, map_enabled=True).cpu().eval()
    tested = 0
    with torch.no_grad():
        for scene in val["scenes"]:
            rows = scene["raw_actual_public_world_transitions"]
            # Both ordinary initial and outage-era public inputs when available.
            onset = scene["public_observed_fault_onset_next_ticks"][0]
            for t in sorted(set((0, min(len(rows) - 1, onset + 1)))):
                row = rows[t]
                graph = actual_public_record_to_graph(row)
                b = Batch.from_data_list([graph])
                node_next, event_next = model(b)
                mixed = PublicStationHazardAdapter(params, "constant").overlay_one_step(
                    row["input"], node_next)
                mask = torch.ones_like(node_next, dtype=torch.bool)
                for i, node in enumerate(row["input"]["graph"]["nodes"]):
                    if node["kind"] == "station":
                        mask[i, 0] = False
                if (not torch.equal(node_next[mask], mixed[mask])
                    or not bool(torch.isfinite(mixed).all())
                    or event_next.shape != (1, 2)):
                    raise AssertionError("PR61 action/map/state output unexpectedly altered")
                tested += 1
    report = {
        "schema": "pr65-world-device-probability-cpu-only-admission-v1",
        "pr64_source_quality_sha256": sha(quality_path),
        "pr64_train_source_sha256": sha(train_path),
        "pr64_dev_source_sha256": sha(dev_path),
        "train_worlds": len(train["scenes"]),
        "dev_worlds": len(val["scenes"]),
        "dev_full_station_risk_rows": len(dev_records),
        "train_only_fitted_hazard_methods": outputs,
        "random_init_real_PyG_one_step_wiring_smoke_rows": tested,
        "checkpoint_loaded": False,
        "neural_trained_or_finetuned": False,
        "gpu_used": False,
        "BLIND_TEST_opened": False,
        "H5_H20_recursive_legality_claimed": False,
        "final_decision": "PASS_CPU_INTERFACE_ONLY_BLOCKED_GPU_H20_UNTIL_REVIEW",
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print("PR65_STOCHASTIC_DEVICE_WORLD_INTERFACE_CPU", json.dumps(report), flush=True)
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True, type=Path)
    ap.add_argument("--dev", required=True, type=Path)
    ap.add_argument("--quality", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    run(a.train, a.dev, a.quality, a.out)


if __name__ == "__main__":
    main()
