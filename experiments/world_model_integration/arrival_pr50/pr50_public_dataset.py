"""PR50 observed PUBLIC first-job arrival dataset, no hidden schedule features.

Every event is discovered from a chronological BusinessObserver snapshot,
NEVER from preconfigured jobs, private simulator events, unreleased stage
definitions or future release ticks. Successor workflow stages are excluded.
The private schedule is used only by the actual Simulator constructor.
"""
from __future__ import annotations

from pathlib import Path
import argparse
import hashlib
import json
import time

from business_transport.observation import BusinessObserver
from business_transport.policy import BusinessPolicy
from business_transport.simulation import BusinessSimulator
from experiments.world_model_integration.arrival_pr50.pr50_scenario import (
    PROTOCOL,MECHANISMS,MARKS,SPLITS,true_business,validate_protocol
)


class PublicArrivals:
    """One snapshot per public tick, first-stage job appearances only."""

    def __init__(self,window):
        self.window=int(window)
        self.observer=BusinessObserver()
        self.last_tick=None
        self.seen_jobs=set()
        self.counts=[]
        self.events=[]

    def observe(self,raw):
        if self.last_tick is not None and raw.tick!=self.last_tick+1:
            raise ValueError("public observations must have consecutive time")
        self.last_tick=raw.tick
        if raw.tick>=self.window:
            return
        public=self.observer.observe(raw)
        arrivals=[]
        for sample in public.samples:
            if getattr(sample,"stage_index",-1)!=0:
                continue
            name=getattr(sample,"job_id","")
            workflow=getattr(sample,"workflow","")
            if not name or workflow not in MARKS:
                raise AssertionError("unexpected publicly released first-stage sample")
            if name in self.seen_jobs:
                continue
            if sample.status=="unreleased" or sample.release_tick>raw.tick:
                raise AssertionError("private future stage leaked into public observer")
            self.seen_jobs.add(name)
            arrivals.append({
                "tick":int(raw.tick),"mark":workflow,
                "job_id":name,
                "observed_release_tick":int(sample.release_tick),
            })
        arrivals.sort(key=lambda x:(x["mark"],x["job_id"]))
        self.events.extend(arrivals)
        self.counts.append(len(arrivals))

    def final(self):
        if not self.counts:
            raise ValueError("no public observation window")
        if any(e["tick"]>=len(self.counts) for e in self.events):
            raise ValueError("event is not within observed public exposure")
        return {
            "public_observed_ticks":len(self.counts),
            "public_count_by_tick":list(self.counts),
            "public_first_job_events":[{"tick":x["tick"],"mark":x["mark"]}
                                       for x in self.events],
            "public_jobs":len(self.events),
            "public_mark_counts":{mark:sum(e["mark"]==mark for e in self.events)
                                  for mark in MARKS},
            "source":"BusinessObserver first visible stage0 only",
        }


def one(spec,protocol):
    business,geo=true_business(spec,protocol)
    collector=PublicArrivals(protocol["tick_window"])
    independent_policy=BusinessPolicy(
        "nearest",geo,business.rules,BusinessObserver())
    # Only callbacks receive actual public state; the future schedule is never
    # offered to collector, event predictor, or BusinessPolicy.
    result=BusinessSimulator(
        business,"world",spec["seed"],collect_metrics=False).run(
        dispatch_policy=independent_policy,on_observation=collector.observe)
    record=collector.final()
    if record["public_jobs"]==0:
        raise AssertionError("unobservable future-only arrivals")
    # Offline integrity label is computed after all public features are final,
    # and is NOT added to any training row or EasyTPP representation.
    private_label=len(business.jobs)
    if record["public_jobs"]>private_label:
        raise AssertionError("observed more jobs than factual schedule")
    record.update(id=spec["id"],seed=spec["seed"],
                  mechanism=spec["mechanism"],split=spec["split"],
                  shifted=spec["shift"],
                  factual_episode_status=result["summary"]["status"],
                  # Hash only to verify stable run provenance. Never use as input.
                  public_events_sha256=hashlib.sha256(json.dumps(
                      record["public_first_job_events"],sort_keys=True).encode()).hexdigest())
    return record


def to_easytpp(record):
    """Gatech-derived JSONL sequence; episode/split/mechanism excluded."""
    events=record["public_first_job_events"]
    marks={s:i for i,s in enumerate(MARKS)}
    timestamps=[float(e["tick"]) for e in events]
    delta=[]
    last=0.0
    for i,t in enumerate(timestamps):
        if i and t<timestamps[i-1]:
            raise ValueError("public events not chronological")
        delta.append(t-last)
        last=t
    if not events:
        raise ValueError("EasyTPP requires nonempty event sequence")
    return {
        "time_since_start":timestamps,
        "time_since_last_event":delta,
        "type_event":[marks[e["mark"]] for e in events],
        "dim_process":len(MARKS)
    }


def build(mechanism,output):
    protocol=json.loads(PROTOCOL.read_text())
    validate_protocol(protocol)
    if mechanism not in MECHANISMS:
        raise ValueError("unregistered process")
    if output.exists():
        raise FileExistsError("PR50 raw public events are write-once")
    output.mkdir(parents=True)
    start=time.process_time()
    groups=[]
    for spec in protocol["cohort"]:
        if spec["mechanism"]!=mechanism:
            continue
        groups.append(one(spec,protocol))
        print("PR50_PUBLIC_EPISODE",spec["id"],groups[-1]["public_jobs"],
              "visible_ticks",groups[-1]["public_observed_ticks"],flush=True)
    if len(groups)!=48:
        raise AssertionError("expected 48 public episodes per process")
    for split in SPLITS:
        selected=[x for x in groups if x["split"]==split]
        if len(selected)!=(24 if split=="train" else 8):
            raise AssertionError("wrong public split population")
        (output/f"{split}.json").write_text(
            json.dumps({"schema":"pr50-business-observer-public-arrivals-v1",
                        "mechanism":mechanism,"split":split,"episodes":selected},
                       indent=2,ensure_ascii=False)+"\n")
        with (output/f"easytpp_{split}.jsonl").open("w") as f:
            for row in selected:
                f.write(json.dumps(to_easytpp(row),sort_keys=True)+"\n")
    metadata={
        "schema":"pr50-public-events-one-regime-v1",
        "mechanism":mechanism,"source_protocol_sha256":hashlib.sha256(
            PROTOCOL.read_bytes()).hexdigest(),
        "episodes":len(groups),
        "train_episodes":24,"validation_episodes":8,
        "test_episodes":8,"shift_test_episodes":8,
        "new_training":0,"gpu":0,
        "cpu_process_seconds":time.process_time()-start,
        "public_only":True,"hidden_future_schedule_never_exported":True,
        "exposure_distribution":[e["public_observed_ticks"] for e in groups],
        "total_public_jobs":sum(e["public_jobs"] for e in groups),
    }
    (output/"summary.json").write_text(json.dumps(metadata,indent=2)+"\n")
    print("PR50_PUBLIC_DATASET",json.dumps(metadata,ensure_ascii=False))
    return metadata


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--mechanism",choices=MECHANISMS,required=True)
    p.add_argument("--output",required=True,type=Path)
    x=p.parse_args()
    build(x.mechanism,x.output)


if __name__=="__main__":
    main()
