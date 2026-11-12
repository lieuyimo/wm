"""PR50 Poisson vs PUBLIC-history-rate baseline, then EasyTPP dataset export.

Strict order: read TRAIN and VALIDATION, select exactly one model by H20
validation count Poisson NLL, only THEN load untouched TEST and SHIFT_TEST.
No true source generator mechanism, private job counts, future simulator
schedule or future observation is ever an input to the predictor.
"""
from __future__ import annotations
from collections import Counter
from pathlib import Path
import argparse
import hashlib
import json
import math

from experiments.world_model_integration.arrival_pr50.pr50_scenario import (
    PROTOCOL,MECHANISMS,SPLITS,validate_protocol,
)

CANDIDATES=("poisson_global","history_4","history_8","history_16")
ANCHORS=(8,16,24,32,40)
HORIZONS=(5,20)


def load_phase(base,phase):
    episodes=[]
    for m in MECHANISMS:
        p=base/m/f"{phase}.json"
        doc=json.loads(p.read_text())
        if (doc["schema"]!="pr50-business-observer-public-arrivals-v1" or
            doc["mechanism"]!=m or doc["split"]!=phase):
            raise ValueError("not a corresponding simulator-derived public split")
        rows=doc["episodes"]
        if len(rows)!=(24 if phase=="train" else 8):
            raise ValueError("public episode cohort changed")
        for x in rows:
            if x["mechanism"]!=m or x["split"]!=phase or x["public_jobs"]!=sum(x["public_count_by_tick"]):
                raise ValueError("corrupt public history")
            if not 1<=x["public_observed_ticks"]<=75:
                raise ValueError("invalid exposure")
        episodes.extend(rows)
    if len({x["seed"] for x in episodes})!=len(episodes):
        raise ValueError("duplicates in published group")
    return episodes


def train_poisson(rows):
    """Only the number of observed public events/exposure, no private labels."""
    ticks=sum(e["public_observed_ticks"] for e in rows)
    events=sum(e["public_jobs"] for e in rows)
    if ticks<=0 or events<=0:
        raise ValueError("cannot fit nonzero train-only arrival intensity")
    marks=Counter(ev["mark"] for e in rows for ev in e["public_first_job_events"])
    rate=events/ticks
    mark_prob={s:(marks[s]+1)/(sum(marks.values())+2) for s in ("w0","w1")}
    return {"global_arrivals_per_tick":rate,"mark_probs":mark_prob,
            "train_public_event_count":events,"train_exposure_ticks":ticks}


def rate_at(public_counts,current_tick,params,name):
    """All counts used have time <= current_tick; no lookahead."""
    rate=params["global_arrivals_per_tick"]
    if name=="poisson_global":
        return rate
    window=int(name.rsplit("_",1)[1])
    if window not in (4,8,16):
        raise ValueError("unregistered history window")
    historical=public_counts[max(0,current_tick+1-window):current_tick+1]
    prior_exposure=4
    return (sum(historical)+prior_exposure*rate)/(len(historical)+prior_exposure)


def poisson_count_nll(target,lam):
    if target<0 or lam<=0 or not math.isfinite(lam):
        raise ValueError("invalid count or rate")
    return lam-target*math.log(lam)+math.lgamma(target+1)


def per_episode(episode,params,name,h):
    counts=episode["public_count_by_tick"]
    records=[]
    for anchor in ANCHORS:
        if anchor+h>=len(counts):
            continue
        # Labels are from public observations only AFTER anchor. Never sent
        # into rate_at; they are used strictly as heldout evaluation outcomes.
        actual=sum(counts[anchor+1:anchor+h+1])
        predicted=rate_at(counts,anchor,params,name)*h
        records.append({
            "anchor_tick":anchor,"expected":predicted,"actual":actual,
            "nll":poisson_count_nll(actual,predicted),
            "absolute_error":abs(actual-predicted)
        })
    return records


def metric(rows,params,name,h,*,with_per_episode=False):
    pieces=[]
    for e in rows:
        samples=per_episode(e,params,name,h)
        if samples:
            pieces.append({
                "id":e["id"],"mechanism":e["mechanism"],
                "mae":sum(s["absolute_error"] for s in samples)/len(samples),
                "nll":sum(s["nll"] for s in samples)/len(samples),
                "anchors":len(samples),
                "rows":samples if with_per_episode else None
            })
    if not pieces:
        raise ValueError("not enough publicly observed future horizon exposure")
    result={
        "scenario_groups":len(pieces),
        "mean_count_mae":sum(x["mae"] for x in pieces)/len(pieces),
        "mean_count_poisson_nll":sum(x["nll"] for x in pieces)/len(pieces),
        "total_eligible_anchors":sum(x["anchors"] for x in pieces)
    }
    if with_per_episode:
        result["episodes"]=pieces
    return result


def evaluate_all(rows,params,model,with_per_episode=False):
    return {
        str(h):{
            "all":metric(rows,params,model,h,with_per_episode=with_per_episode),
            "by_process":{m:metric([e for e in rows if e["mechanism"]==m],
                                   params,model,h,with_per_episode=False)
                          for m in MECHANISMS}
        } for h in HORIZONS
    }


def export_easytpp_jsonl(source,output):
    output.mkdir(parents=True,exist_ok=True)
    for split in SPLITS:
        dst=output/f"{split}.jsonl"
        if dst.exists():
            raise FileExistsError("EasyTPP source row export must not overwrite")
        with dst.open("w") as f:
            for m in MECHANISMS:
                p=source/m/f"easytpp_{split}.jsonl"
                lines=p.read_text().splitlines()
                if len(lines)!=(24 if split=="train" else 8):
                    raise AssertionError("unregistered EasyTPP event sequence count")
                for raw in lines:
                    item=json.loads(raw)
                    if set(item)!={"dim_process","time_since_start","time_since_last_event","type_event"}:
                        raise AssertionError("private source columns leaked into neural dataset")
                    if item["dim_process"]!=2 or len(item["type_event"])==0:
                        raise AssertionError("unexpected event marks")
                    f.write(json.dumps(item,sort_keys=True)+"\n")


def write_easytpp_config(out,checkpoint):
    """Official EasyTPP; train config deliberately maps test to VALIDATION to protect heldout."""
    content=f"""data:
  pr50_public_arrivals:
    data_format: json
    train_dir: '{(out/"train.jsonl").as_posix()}'
    valid_dir: '{(out/"validation.jsonl").as_posix()}'
    test_dir: '{(out/"validation.jsonl").as_posix()}'
    data_specs:
      num_event_types: 2
      pad_token_id: 2
      padding_side: right
      truncation_side: right
      rescale_time: true
NHP_train:
  base_config:
    stage: train
    backend: torch
    dataset_id: pr50_public_arrivals
    runner_id: std_tpp
    model_id: NHP
    base_dir: '{checkpoint.as_posix()}'
  trainer_config:
    batch_size: 16
    max_epoch: 15
    shuffle: true
    optimizer: adam
    learning_rate: 0.001
    valid_freq: 1
    use_tfb: false
    metrics: [acc, rmse]
    seed: 1810
    gpu: -1
  model_config:
    hidden_size: 32
    loss_integral_num_sample_per_step: 10
    use_ln: false
  thinning:
    num_seq: 5
    num_sample: 1
    num_exp: 200
    look_ahead_time: 10
    patience_counter: 3
    over_sample_rate: 5
    num_samples_boundary: 5
    dtime_max: 20
    num_step_gen: 1
"""
    (out/"easytpp_config.yaml").write_text(content)


def run(source,out):
    protocol=json.loads(PROTOCOL.read_text())
    validate_protocol(protocol)
    if out.exists():
        raise FileExistsError("PR50 aggregate output must be write-once")
    out.mkdir(parents=True)
    # Until model selection, evaluation test files are intentionally not
    # opened at all. These are new simulation seed namespaces.
    train=load_phase(source,"train")
    validation=load_phase(source,"validation")
    if set(x["seed"] for x in train)&set(x["seed"] for x in validation):
        raise AssertionError("train vs validation leakage")
    params=train_poisson(train)
    validation_models={
        name:evaluate_all(validation,params,name,False)
        for name in CANDIDATES
    }
    selected=min(CANDIDATES,key=lambda name:(
        validation_models[name]["20"]["all"]["mean_count_poisson_nll"],
        CANDIDATES.index(name)))
    selection={
        "schema":"pr50-train-and-validation-selection-v1",
        "source_protocol_sha256":hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(),
        "chosen_by_validation_only":selected,
        "selected_horizon":20,
        "train_only_estimate":params,
        "validation_all_models":validation_models,
        "locked_before_independent_test":True,
        "independent_test_scores_used_to_select":False,
    }
    (out/"train-validation-selection.json").write_text(
        json.dumps(selection,indent=2,ensure_ascii=False)+"\n")
    # Independent tests loaded only AFTER the selection was saved.
    tests={}
    used={x["seed"] for x in train+validation}
    for phase in ("test","shift_test"):
        rows=load_phase(source,phase)
        keys={x["seed"] for x in rows}
        if keys&used:
            raise AssertionError("independent test is not independent")
        used.update(keys)
        tests[phase]={
            name:evaluate_all(rows,params,name,True if name==selected else False)
            for name in ("poisson_global",selected)
        }
    output={
        "schema":"pr50-arrival-public-history-identifiability-v1",
        "protocol_sha256":hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(),
        "selection_model":selected,
        "source_splits":{"train":len(train),"validation":len(validation),
                         "test":24,"shift_test":24},
        "selection":selection,"independent_tests":tests,
        "test_set_was_not_used_for_selection":True,
        "gpu":0,"new_neural_training":0,
        "domain_limits":"Same original synthetic map/workflows. Process family label is evaluation metadata only; no claim about unknown real hospital arrivals."
    }
    (out/"baseline_report.json").write_text(json.dumps(output,indent=2,ensure_ascii=False)+"\n")
    export_easytpp_jsonl(source,out/"easytpp")
    write_easytpp_config(out/"easytpp",out/"easytpp-checkpoints")
    print("PR50_PUBLIC_BASELINE_SELECTED",selected)
    print("PR50_IDENTIFIABILITY_RESULT",json.dumps({
        "selected":selected,
        "validation_h20_nll":validation_models[selected]["20"]["all"]["mean_count_poisson_nll"],
        "tests":{
            k:{a:{"h20_nll":v["20"]["all"]["mean_count_poisson_nll"],
                  "h20_mae":v["20"]["all"]["mean_count_mae"],
                  "h5_mae":v["5"]["all"]["mean_count_mae"]}
               for a,v in values.items()} for k,values in tests.items()},
        "easy_tpp_sequences":{"train":72,"validation":24,"test":24,"shift_test":24},
        "future_leakage":False
    },ensure_ascii=False))
    return output


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--source",required=True,type=Path)
    p.add_argument("--out",required=True,type=Path)
    x=p.parse_args()
    run(x.source,x.out)


if __name__=="__main__":
    main()
