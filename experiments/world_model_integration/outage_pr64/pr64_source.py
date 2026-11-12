"""PR64 prospective CPU-only private outage-onset time randomization.

Reuses the original PR48/55/56 true BusinessSimulator/CP-SAT, PR54 public
collector and PR56 map adapter. ONLY the time of PRIVATE station fault
onset changes. Fault duration, map, traffic model, service rules, robots,
jobs, action selection and simulator transitions stay unchanged.

The hidden schedule is not serialized in model input. Actual t+1 is only
a label; audit derives realized events from genuine public state changes.
This work makes no assertion about neural model training or H20 rollout.
"""
from __future__ import annotations
from collections import Counter
from dataclasses import replace
from pathlib import Path
import argparse
import hashlib
import json

from business_transport.simulation import BusinessSimulator,resource_audit
from experiments.business_constraints.run import lifecycle_audit
from experiments.world_model_integration.decision_learning_pr51.pr51_labels import TrueCPsatBranch
from experiments.world_model_integration.quality_pr55.pr55_generate import flag_transition
from experiments.world_model_integration.sealed_pr59.pr59_source import read_protocol as pr59_protocol,validate_layout_business
from experiments.world_model_integration.stress_pr48.pr48_scenarios import PROTOCOL as PR48_PROTOCOL
from experiments.world_model_integration.temporal_pr54.pr54_public_sequence import PublicTransitionCollector
from experiments.world_model_integration.topology_pr56.pr56_public_map import augment_public_transition,build_topology_business
from lab_transport.multisite_experiment import discrete_safety_violations

PROTOCOL=Path(__file__).with_name("protocol.json")
IDENTITY="PR64_RANDOM_OUTAGE_ONSET_EVENT_OBSERVABILITY_PILOT_V1_20261113"
PR63_SHA="95ef4c3bee3824f382166efd0c3d62ce568b06f5"


def read_protocol():
    p=json.loads(PROTOCOL.read_text())
    old=pr59_protocol()
    if p.get("protocol")!=IDENTITY or p.get("parent_pr63_sha")!=PR63_SHA:
        raise AssertionError("PR64 protocol version/provenance changed")
    if p["resource"]["GPU"] or p["resource"]["new_model_training"] or p["resource"]["new_neural_inference"] or p["resource"]["BLIND_TEST"]:
        raise AssertionError("source must stay CPU-only factual simulation")
    if p["public_state_contract"]["history_window"]!=4 or p["public_state_contract"]["max_ticks"]!=260:
        raise AssertionError("original public collector contract changed")
    if len(p["cohort"])!=36 or len({c["seed"] for c in p["cohort"]})!=36:
        raise AssertionError("new independent pilot cohort changed")
    prior_seeds={c["seed"] for c in old["cohort"]}
    if prior_seeds&{c["seed"] for c in p["cohort"]}:
        raise AssertionError("reused old TRAINVAL or frozen BLIND seed")
    layouts={x["id"]:x for x in p["scenario_parent"]}
    original={x["id"]:x for x in old["cohort"]}
    if any(x["family"]=="horizontal_shelf_blind_topology" for x in layouts.values()):
        raise AssertionError("original unseen BLIND geography must remain sealed")
    for split in ("train","validation"):
        n=p["splits"][split]["count"]
        cases=[c for c in p["cohort"] if c["split"]==split]
        if len(cases)!=n or {c["source_reference_case_id"] for c in cases}!={
            c["id"] for c in old["cohort"] if c["split"]==split
        }:
            raise AssertionError("original train/validation case design mapping changed")
        for c in cases:
            src=original[c["source_reference_case_id"]]
            for key in ("regime","profile","layout","map_family","job_count",
                        "mechanism","station_id","secondary_station_id"):
                if c[key]!=src[key]:
                    raise AssertionError("non-outage business setting changed: "+key)
            if c["layout"] not in layouts or c["map_family"]!=layouts[c["layout"]]["family"]:
                raise AssertionError("unregistered known public map")
            if c["seed"]==src["seed"] or c["id"]==src["id"]:
                raise AssertionError("reused original TRAIN/VAL factual simulation")
            if c["id"].split("_")[-2]!=src["id"].split("_")[-2]:
                raise AssertionError("map index source id changed")
        if p["splits"][split]["map_family"]!=old["layouts"][
            next(i for i,m in enumerate(old["layouts"]) if m["id"]==cases[0]["layout"])
        ]["family"]:
            raise AssertionError("map family changed")
    train=set(p["splits"]["train"]["allowed_private_start_ticks"])
    val=set(p["splits"]["validation"]["allowed_private_start_ticks"])
    if train&val or len(train)<6 or len(val)<6:
        raise AssertionError("private fault onset shift supports not disjoint")
    if any(x<6 or x>40 for x in train|val):
        raise AssertionError("unreasonable private outage onset tick")
    if p["splits"]["BLIND_TEST"]!={"count":0,"generate":False,"read":False,
                                     "map_family":"horizontal_shelf_blind_topology"}:
        raise AssertionError("BLIND_TEST must not be touched")
    return p,old


def onset_start(seed,regime,station,ordinal,allowed):
    raw=f"pr64|private_outage_onset|{seed}|{regime}|{station}|{ordinal}".encode()
    index=int.from_bytes(hashlib.sha256(raw).digest()[:8],"big")%len(allowed)
    return allowed[index]


def private_outage_shift(world,case,allowed):
    original=dict(world.scenario.stations)
    stations=dict(original)
    expected=[case["station_id"]]+(
        [case["secondary_station_id"]] if case["regime"]=="multi_fault" else [])
    if len(set(expected))!=len(expected):
        raise AssertionError("duplicated fault station")
    taken=set()
    starts=[]
    duration_source=[]
    for ordinal,sid in enumerate(expected):
        old=original[sid]
        if len(old.outages)!=1:
            raise AssertionError("original PR48 sampled one real private outage per selected station")
        old_start,old_end=old.outages[0]
        duration=old_end-old_start
        if duration<=0:
            raise AssertionError("duration not original positive PR48 support")
        start=onset_start(case["seed"],case["regime"],sid,ordinal,allowed)
        if start in taken:
            # Ensure two physical station events are temporally separated, so
            # source flag counts reflect both; never choose from outcomes.
            start=allowed[(allowed.index(start)+1)%len(allowed)]
        taken.add(start)
        starts.append(start)
        duration_source.append(duration)
        stations[sid]=replace(old,outages=((start,start+duration),))
    if any(original[sid].outages!=stations[sid].outages for sid in original
           if sid not in expected):
        raise AssertionError("unselected original station outage changed")
    if any((stations[sid].service_ticks,stations[sid].prior_ticks)!=
           (original[sid].service_ticks,original[sid].prior_ticks)
           for sid in original):
        raise AssertionError("physical service process illegally modified")
    shifted=replace(world,scenario=replace(world.scenario,stations=stations))
    if shifted.rules!=world.rules or shifted.jobs!=world.jobs or shifted.scenario.robots!=world.scenario.robots or shifted.scenario.grid!=world.scenario.grid:
        raise AssertionError("changed original action/route/world source beyond event time")
    return shifted,starts,duration_source


def real_episode(case,protocol,old,stress):
    topo,detour=validate_layout_business(case,old,stress)
    world,geo=build_topology_business(case,stress,old)
    choices=protocol["splits"][case["split"]]["allowed_private_start_ticks"]
    shifted,starts,durations=private_outage_shift(world,case,choices)
    policy=TrueCPsatBranch(geo,shifted.rules,case["seed"],{"counterfactual_max_candidates":4})
    collector=PublicTransitionCollector(shifted.rules,window=4,max_steps=260)
    fact=BusinessSimulator(shifted,"world",case["seed"],collect_metrics=True).run(
        dispatch_policy=policy,on_transition=collector)
    if (resource_audit(fact,shifted.rules)["violations"]
        or discrete_safety_violations(fact)
        or lifecycle_audit(fact["history"])):
        raise AssertionError("actual original simulator resource/motion/workflow unsafe")
    rows=collector.rows
    if (not rows or len(rows)!=fact["summary"]["ticks"]
        or len(rows)>260 or detour<1):
        raise AssertionError("missing original public t0-to-terminal BusinessSimulator transition")
    for row in rows:
        augment_public_transition(row,geo)
        inp=row["input"];label=row["target"]
        if inp["feature_last_observed_tick"]!=inp["public_tick"] or label["next_public_tick"]!=inp["public_tick"]+1:
            raise AssertionError("future private label leakage")
        if inp["public_map"]["static_map_digest"]!=topo["static_map_digest"]:
            raise AssertionError("public route topology changed unexpectedly")
        if set(inp)!={"public_tick","graph","history","history_visible_mask",
                      "node_persistence","feature_last_observed_tick","input_source",
                      "public_map"}:
            raise AssertionError("private hazard/schedule accidentally in model INPUT")
        row["quality"]={"flags":flag_transition(row)}
    # Determine the TRUE observed failure/recovery ticks only from public
    # t+1 state labels, NEVER from the hidden simulator schedule for the model.
    public_events=Counter()
    true_onsets=[]
    true_recoveries=[]
    for row in rows:
        nodes=row["input"]["graph"]["nodes"]
        targets=row["target"]["node_target"]
        masks=row["target"]["node_mask"]
        for n,y,mask in zip(nodes,targets,masks):
            if n["kind"]!="station" or not mask[0]:
                continue
            cur=int(bool(n["device_available"]))
            nxt=int(y[0])
            if cur==1 and nxt==0:
                public_events["onset"]+=1
                true_onsets.append(row["target"]["next_public_tick"])
            elif cur==0 and nxt==1:
                public_events["recovery"]+=1
                true_recoveries.append(row["target"]["next_public_tick"])
    if sorted(true_onsets)!=sorted(starts) or len(true_recoveries)!=len(starts):
        raise AssertionError("actual public fault/recovery source disagrees with original hidden onset mechanism")
    return {
        "id":case["id"],"case":case,
        "public_map":topo,"real_status":fact["summary"]["status"],
        "observed_ticks":len(rows),
        "public_observed_fault_onset_next_ticks":sorted(true_onsets),
        "public_observed_fault_recovery_next_ticks":sorted(true_recoveries),
        "public_device_event_count":dict(public_events),
        "private_environment_schedule_sha256":hashlib.sha256(json.dumps({
            "seed":case["seed"],"stations":[case["station_id"],case["secondary_station_id"]],
            "starts":starts,"duration":durations
        },sort_keys=True).encode()).hexdigest(),
        "private_schedule_NOT_in_model_graph":True,
        "right_censored_final_makespan":fact["summary"]["status"]!="completed",
        "raw_actual_public_world_transitions":rows
    }


def run(split,output):
    p,old=read_protocol()
    if split not in ("train","validation"):
        raise PermissionError("original BLIND TEST has no simulation entrypoint")
    if output.exists():
        raise FileExistsError("do not overwrite original frozen factual episode files")
    stress=json.loads(PR48_PROTOCOL.read_text())
    cases=[x for x in p["cohort"] if x["split"]==split]
    scenes=[]
    for case in cases:
        scene=real_episode(case,p,old,stress)
        scenes.append(scene)
        print("PR64_TRUE_RANDOM_OUTAGE_SOURCE",json.dumps({
            "id":case["id"],"regime":case["regime"],
            "observed_fault_times":scene["public_observed_fault_onset_next_ticks"],
            "observed_recovery_times":scene["public_observed_fault_recovery_next_ticks"],
            "ticks":scene["observed_ticks"],
            "complete":not scene["right_censored_final_makespan"]
        },ensure_ascii=False),flush=True)
    info={
        "split":split,"scenes":len(scenes),"real_continuous_public_transitions":sum(x["observed_ticks"] for x in scenes),
        "fault_onsets":sum(x["public_device_event_count"].get("onset",0) for x in scenes),
        "fault_recoveries":sum(x["public_device_event_count"].get("recovery",0) for x in scenes),
        "real_episode_completed":sum(not x["right_censored_final_makespan"] for x in scenes),
        "right_censored":sum(x["right_censored_final_makespan"] for x in scenes),
        "distinct_observed_onset_ticks":sorted(set(t for x in scenes for t in x["public_observed_fault_onset_next_ticks"])),
        "GPU":0,"neural_training":0
    }
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps({
        "schema":"pr64-shifted-private-failure-event-original-business-public-source-v1",
        "split":split,"BLIND_TEST_RUN":False,
        "protocol_sha256":hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(),
        "scenes":scenes,"summary":info,
        "limitations":"Private outage start randomized deterministically per source case; unchanged PR48 duration and unchanged original 7x5 simulator. New development validation, NOT original heldout blind test."
    },ensure_ascii=False)+"\n")
    print("PR64_FROZEN_TRUE_SOURCE_SUMMARY",json.dumps(info,ensure_ascii=False),flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--split",choices=("train","validation","test"),required=True)
    p.add_argument("--out",required=True,type=Path)
    x=p.parse_args()
    run(x.split,x.out)


if __name__=="__main__":
    main()
