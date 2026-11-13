"""PR72: reanalyze frozen REAL PR70 pretrained arm risk scores, no weights.

Directly compare paired (map_on - map_off) Brier on the SAME 12 developer
scenes and SAME original three trained seed IDs; bootstrap SCENES not ticks.
This is reanalysis of PR70's archived ORIGINAL PR61 predictions, not new
checkpoint inference. PR70 did not save phase-level prediction values.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

ARCHIVE_SHA = "919a0b625659e492811a74bc5cdf76a60ccbf639acb32c33a420b7db30a74e14"
SOURCE_SHA = "31a36e0940db7afe015fe35689cb629102dea45e8d61123ed89250abc01d0b86"
SEEDS = (1810, 1811, 1812)


def score(frozen):
    if (frozen.get("archive_sha256") != ARCHIVE_SHA
        or frozen.get("source_sha256") != SOURCE_SHA
        or frozen.get("BLIND_TEST_read") is not False
        or frozen.get("no_training") is not True
        or len(frozen.get("per_scene", [])) != 12):
        raise ValueError("not original frozen PR70 trained model developer-risk report")
    rows = []
    ids = set()
    for sc in frozen["per_scene"]:
        sid = sc["scene"]
        if sid in ids:
            raise ValueError("duplicate original PR70 scene")
        ids.add(sid)
        paired = []
        for seed in SEEDS:
            off = sc["models"][f"map_off_{seed}"]["model"]
            on = sc["models"][f"map_on_{seed}"]["model"]
            if (off["n"] != on["n"] or off["events"] != on["events"]
                or off["n"] <= 0):
                raise ValueError("PR70 pretrained arms do not share matched risk rows")
            paired.append({
                "seed": seed, "n": off["n"], "actual_events": off["events"],
                "map_off_brier": off["brier"],
                "map_on_brier": on["brier"],
                "map_on_minus_off_brier": on["brier"] - off["brier"],
                "map_off_logloss": off["logloss"],
                "map_on_logloss": on["logloss"],
            })
        rows.append({
            "scene": sid, "n": paired[0]["n"],
            "actual_events": paired[0]["actual_events"],
            "per_original_train_seed": paired,
            "scene_mean_delta_on_minus_off": sum(x["map_on_minus_off_brier"]
                                                   for x in paired) / 3,
        })
    ordered = sorted(rows, key=lambda r: r["scene"])
    diffs = [r["scene_mean_delta_on_minus_off"] for r in ordered]
    rng = random.Random(72001)
    boots = sorted(
        sum(diffs[rng.randrange(len(diffs))] for _ in diffs) / len(diffs)
        for _ in range(5000)
    )
    by_seed = {
        str(seed): sum(
            next(x["map_on_minus_off_brier"]
                 for x in s["per_original_train_seed"] if x["seed"] == seed)
            for s in ordered) / len(ordered)
        for seed in SEEDS
    }
    rule = [x["rule"]["brier"] for x in frozen["per_scene"]]
    return {
        "schema": "pr72-frozen-PR70-pretrained-model-paired-service-risk-reanalysis-v1",
        "original_trained_PR61_archive_sha256": ARCHIVE_SHA,
        "original_PR64_developer_validation_sha256": SOURCE_SHA,
        "actual_trained_PR61_output_reused_from_PR70": True,
        "new_PR61_checkpoint_inference_performed": False,
        "model_retraining": False,
        "original_BLIND_read": False,
        "risk_set_is_mixed_pickup_and_delivery": True,
        "n_independent_scenes": len(ordered),
        "n_active_service_risks": sum(r["n"] for r in ordered),
        "n_true_events": sum(r["actual_events"] for r in ordered),
        "original_seeds": list(SEEDS),
        "rule_scene_brier_all_zero": all(x == 0 for x in rule),
        "map_on_minus_map_off_paired_scene_macro_Brier":
            sum(diffs) / len(diffs),
        "bootstrap_5000_scene_resamples_95CI": [
            boots[124], boots[4874]
        ],
        "n_scenes_map_on_lower_brier": sum(v < 0 for v in diffs),
        "per_seed_delta_on_minus_off": by_seed,
        "per_scene": ordered,
        "limitations": (
            "Original PR70 learned predictions were archived as mixed-phase "
            "aggregate scores only; no phase-specific calibration or per-step "
            "conditional forecasting benefit is established here."
        ),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--frozen-pr70-service", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError("do not overwrite original-model paired analysis")
    result = score(json.loads(a.frozen_pr70_service.read_text()))
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print("PR72_FROZEN_ORIGINAL_TRAINED_ARM_SCENE_PAIRED_RESULT",
          json.dumps({
              "delta": result["map_on_minus_map_off_paired_scene_macro_Brier"],
              "CI": result["bootstrap_5000_scene_resamples_95CI"],
              "scenes": result["n_independent_scenes"],
              "new_neural_inference": False,
          }), flush=True)


if __name__ == "__main__":
    main()
