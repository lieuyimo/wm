"""PR62 reproducible readonly audit of PR61 committed GPU evidence.

No checkpoint, GPU, model execution, new simulation, or blind/test access.
Each of 12 held-out validation *scenarios* has the same three model seeds.
Report scenario-level paired differences, not 36 independent worlds.
This is a diagnosis, never a prospective test of an untuned method.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

ORIGIN="ff6129cbf0c14726185d548459ae7fbe0d2dab3f"
FIELDS={
    "node_mse":"node_mse",
    "device_brier":"device_brier",
    "new_task_mae":"next_new_public_stage_mae",
}
SEEDS={"1810","1811","1812"}
REGIMES={"congestion","multi_fault","staggered_arrivals"}
ARMS={"map_off","map_on"}


def source_sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_rows(path):
    with path.open(newline="") as fp:
        return list(csv.DictReader(fp))


def audit(csv_path, summary_path, out):
    if out.exists():
        raise FileExistsError("do not overwrite a readonly evidence audit")
    rows=read_rows(csv_path)
    summary=json.loads(summary_path.read_text())
    if (len(rows)!=72 or summary.get("schema")!="pr60-a100-gpu-aggregate-v1"
        or summary["runs"]!=6
        or summary["blind_test"]["files_read"]!=0
        or summary["blind_test"]["episodes_generated"]!=0):
        raise AssertionError("wrong or contaminated PR61 evidence")
    by_pair={}
    per_scene=defaultdict(list)
    for row in rows:
        if (row["arm"] not in ARMS or row["seed"] not in SEEDS
            or row["regime"] not in REGIMES or row["family"]!="diagonal_staggered_validation"
            or row["censored"]!="False"):
            raise AssertionError("nonregistered training arm/validation source")
        pair=(row["seed"],row["id"])
        by_pair.setdefault(pair,{})
        if row["arm"] in by_pair[pair]:
            raise AssertionError("duplicate selected model in the same paired scene")
        by_pair[pair][row["arm"]]=row
    scenes={scene for _,scene in by_pair}
    if len(by_pair)!=36 or len(scenes)!=12 or any(set(pair)!=ARMS for pair in by_pair.values()):
        raise AssertionError("unpaired training seed or missing validation scene")
    paired={}
    for (seed,scene),values in by_pair.items():
        a=values["map_on"];b=values["map_off"]
        if (a["regime"]!=b["regime"]
            or a["n_real_steps"]!=b["n_real_steps"]
            or a["family"]!=b["family"]):
            raise AssertionError("not the same original public-state validation trajectory")
        diffs={k:float(a[field])-float(b[field]) for k,field in FIELDS.items()}
        paired[f"{seed}:{scene}"]={
            "seed":seed,"scenario_id":scene,"regime":a["regime"],
            "map_id":scene.split("_")[2],
            "same_n_real_steps":int(a["n_real_steps"]),
            "on_minus_off":diffs,
        }
        per_scene[scene].append(paired[f"{seed}:{scene}"])
    # One average over three seeds per true validation scenario, rather
    # than treating adjacent ticks or seeds as independent source worlds.
    scene_means=[]
    for scene,instances in sorted(per_scene.items()):
        if {x["seed"] for x in instances}!=SEEDS:
            raise AssertionError("incomplete scenario-level paired random seed set")
        scene_means.append({
            "id":scene,
            "map_id":instances[0]["map_id"],
            "regime":instances[0]["regime"],
            "seed_count":3,
            "on_minus_off":{k:mean(x["on_minus_off"][k] for x in instances)
                            for k in FIELDS},
        })

    def metrics(group):
        d={k:[x["on_minus_off"][k] for x in group] for k in FIELDS}
        return {k:{
            "mean_on_minus_off":mean(v),
            "n":len(v),
            "num_map_on_better":sum(a<0 for a in v),
            "num_map_on_worse":sum(a>0 for a in v),
            "sd_between_scenes":stdev(v) if len(v)>1 else None,
        } for k,v in d.items()}
    stats=metrics(scene_means)
    by_map={k:metrics([x for x in scene_means if x["map_id"]==k])
            for k in sorted({x["map_id"] for x in scene_means})}
    by_regime={k:metrics([x for x in scene_means if x["regime"]==k])
               for k in sorted(REGIMES)}
    by_seed={k:metrics([{
        "on_minus_off":x["on_minus_off"]
    } for x in paired.values() if x["seed"]==k]) for k in sorted(SEEDS)}
    if (stats["node_mse"]["num_map_on_better"]!=11
        or stats["node_mse"]["n"]!=12
        or len(by_map)!=2
        or not (by_map["0"]["device_brier"]["mean_on_minus_off"]<0
                <by_map["1"]["device_brier"]["mean_on_minus_off"])
        or by_seed["1810"]["device_brier"]["mean_on_minus_off"]<=0
        or by_seed["1811"]["device_brier"]["mean_on_minus_off"]<=0
        or by_seed["1812"]["device_brier"]["mean_on_minus_off"]>=0):
        raise AssertionError("source data changed from independently read PR61")
    result={
        "schema":"pr62-readonly-pr61-gpu-evidence-validation-paired-scene-audit-v1",
        "original_GPU_PR_head":ORIGIN,
        "per_scenario_evidence_sha256":source_sha(csv_path),
        "summary_evidence_sha256":source_sha(summary_path),
        "n_actual_independent_validation_scenes":12,
        "n_distinct_validation_maps":2,
        "n_trained_model_seeds_per_arm":3,
        "n_seed_by_scenario_paired_observations_not_independent":36,
        "all_records_teacher_forced_one_step":True,
        "all_12_blind_test_scenarios_unopened_by_report":True,
        "public_state_next_tick_MSE":stats["node_mse"],
        "station_device_availability_Brier":stats["device_brier"],
        "next_public_stage_count_MAE":stats["new_task_mae"],
        "per_map":by_map,
        "per_stress":by_regime,
        "per_training_seed":by_seed,
        "per_real_validation_scenario":scene_means,
        "boundary":"No raw checkpoint weights in PR61, and 12 scenes live on only 2 distinct validation maps. Station onset/recovery conditional calibration CANNOT be inferred from aggregated per-scenario device Brier; actual model predictions are required.",
        "next_gate":"Collect local-only trusted checkpoint archive and verify SHA256, replay original PR59 validation 12 using real device-event prediction per tick, compare healthy-hold/fault-onset/down/recovery against persistence.",
        "CUDA_or_model_calls":0,
        "new_episodes_generated":0,
        "blind_test_files_read":0,
    }
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n")
    print("PR62_READONLY_PAIR_SCENARIO_AUDIT",json.dumps({
        k:v for k,v in result.items() if k not in ("per_real_validation_scenario","per_map","per_stress","per_training_seed")
    },ensure_ascii=False))
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--per-scenario",required=True,type=Path)
    p.add_argument("--summary",required=True,type=Path)
    p.add_argument("--out",required=True,type=Path)
    args=p.parse_args()
    audit(args.per_scenario,args.summary,args.out)


if __name__=="__main__":
    main()
