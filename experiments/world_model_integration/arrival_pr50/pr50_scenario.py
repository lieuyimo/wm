"""PR50 synthetic arrival schedules used only to initialise the factual simulator.

Never pass returned job release tick list, generator family, hidden latent
regime or future release schedule to the world model / dataset feature row.
"""
from __future__ import annotations

from pathlib import Path
import hashlib
import json

from business_transport.scenario import compile_scenario
from experiments.business_constraints.tests.test_business import fixture
from lab_transport.dynamics_model import Geometry

PROTOCOL=Path(__file__).with_name("pr50_protocol.json")
MECHANISMS=("independent","periodic","bursty")
SPLITS=("train","validation","test","shift_test")
MARKS=("w0","w1")


def _hash_uniform(*parts):
    value=hashlib.sha256(("pr50-PRIVATE-actual-simulator|"+
                          "|".join(str(x) for x in parts)).encode()).digest()
    return int.from_bytes(value[:8],"big")/2**64


def validate_protocol(cfg):
    if cfg.get("protocol")!="PR50_PUBLIC_HISTORY_MARKED_ARRIVAL_MODEL_20261112_V1":
        raise ValueError("unexpected PR50 protocol")
    if cfg.get("parent_pr49_source")!="c85c0b02c7ef34d44a9347ffd113ae9fee6e3952":
        raise ValueError("PR49 provenance changed")
    if cfg["tick_window"]!=75 or cfg["workflow_types"]!=["w0","w1"]:
        raise ValueError("public event timeline or marks changed")
    if cfg["easytpp"]["version"]!="0.2.4" or cfg["baselines"]["history_windows"]!=[4,8,16]:
        raise ValueError("model transfer reference changed")
    cohort=cfg["cohort"]
    if len(cohort)!=144 or len({x["seed"] for x in cohort})!=144:
        raise ValueError("not 144 new independent episodes")
    for m in MECHANISMS:
        for split,n in (("train",24),("validation",8),("test",8),("shift_test",8)):
            got=[x for x in cohort if x["mechanism"]==m and x["split"]==split]
            if len(got)!=n or any(x["shift"]!=(split=="shift_test") for x in got):
                raise ValueError("wrong precommitted cohort split")
    if any(x["seed"]<20000000 for x in cohort):
        raise ValueError("unexpected seed overlaps previous study namespace")
    return True


def _private_arrivals(spec,cfg):
    """Only the business scenario constructor calls this function."""
    h=cfg["tick_window"]
    name=spec["mechanism"]
    source=cfg["mechanisms"][name]
    shift=spec["shift"]
    release=[]
    if name=="independent":
        p=source["shift_probability"] if shift else source["base_probability"]
        for t in range(h):
            if _hash_uniform(spec["seed"],"arrival",t)<p:
                release.append((t,None))
    elif name=="periodic":
        t=1+int(_hash_uniform(spec["seed"],"start")*source["first_tick_max"])
        options=source["shift_intervals"] if shift else source["intervals"]
        while t<h:
            release.append((t,None))
            gap=options[int(_hash_uniform(spec["seed"],"gap",len(release))*len(options))]
            t+=gap
    elif name=="bursty":
        hidden=1 if _hash_uniform(spec["seed"],"initial_regime")<0.5 else 0
        for t in range(h):
            if t>0 and t%source["block_length"]==0:
                p=(source["shift_switch_probability"] if shift
                   else source["base_switch_probability"])
                if _hash_uniform(spec["seed"],"state_switch",t)<p:
                    hidden=1-hidden
            probs=source["shift_probs"] if shift else source["base_probs"]
            if _hash_uniform(spec["seed"],"arrival",t)<probs[hidden]:
                release.append((t,hidden))
    else:
        raise ValueError("unregistered private process")
    if not release:
        raise AssertionError("new schedule contains zero jobs")
    jobs=[]
    for i,(t,hidden) in enumerate(release):
        if name=="bursty":
            mark_p=source["mark_probs_w0_by_latent_state"][hidden]
        else:
            mark_p=source["mark_probability_w0"]
        workflow="w0" if _hash_uniform(spec["seed"],"mark",t,i)<mark_p else "w1"
        jobs.append({
            "id":f"PR50J{i+1}", "workflow":workflow,
            "release_tick":t,"deadline_tick":cfg["deadline_tick"]})
    return jobs


def true_business(spec,cfg):
    """Real BusinessSimulator case. Hidden schedule never returned as model features."""
    base,old,_=fixture(capacity=1)
    jobs=_private_arrivals(spec,cfg)
    raw={**old,"name":"pr50-"+spec["id"],"jobs":jobs,
         "max_ticks":cfg["max_business_ticks"]}
    business=compile_scenario(base,raw)
    return business,Geometry(base.grid)
