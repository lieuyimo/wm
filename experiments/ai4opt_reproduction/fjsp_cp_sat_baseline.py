#!/usr/bin/env python3
"""Independent OR-Tools CP-SAT FJSP baseline on Song et al. original test files.

This reuses the standard optional-interval/cumulative-disjunctive formulation,
NOT a newly invented learning method. It uses only publicly available FJSP
data: operation order, eligible machines and known processing times.
No author/wm model parameters are read or saved.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

from ortools.sat.python import cp_model


def read_fjsp(path):
    lines=path.read_text().splitlines()
    first=lines[0].split()
    n,m=int(first[0]),int(first[1])
    if len(lines)<n+1:
        raise ValueError("not enough job lines")
    jobs=[]
    for j,line in enumerate(lines[1:n+1]):
        tokens=[int(v) for v in line.split()]
        num=tokens[0]
        at=1
        operations=[]
        for k in range(num):
            count=tokens[at]
            at+=1
            opts=[]
            for _ in range(count):
                machine=tokens[at]-1
                duration=tokens[at+1]
                at+=2
                if not (0<=machine<m and duration>0):
                    raise ValueError("invalid machine or duration in public FJSP")
                opts.append((machine,duration))
            if len(opts)!=len(set(x[0] for x in opts)):
                raise ValueError("duplicate machine alternative")
            if not opts:
                raise ValueError("unprocessed empty operation")
            operations.append(opts)
        if at!=len(tokens):
            raise ValueError(f"public job {j} has trailing tokens")
        jobs.append(operations)
    return n,m,jobs


def solve(path,seconds,seed):
    n,m,jobs=read_fjsp(path)
    horizon=sum(max(duration for _,duration in alts) for job in jobs for alts in job)
    model=cp_model.CpModel()
    machine_intervals=[[] for _ in range(m)]
    finished=[]
    selected={}
    starts={}
    ends={}
    for j,job in enumerate(jobs):
        prev=None
        for k,alts in enumerate(job):
            start=model.new_int_var(0,horizon,f"s_{j}_{k}")
            end=model.new_int_var(0,horizon,f"e_{j}_{k}")
            starts[j,k]=start
            ends[j,k]=end
            flags=[]
            for machine,duration in alts:
                f=model.new_bool_var(f"a_{j}_{k}_{machine}")
                interval=model.new_optional_interval_var(start,duration,end,f,f"i_{j}_{k}_{machine}")
                machine_intervals[machine].append(interval)
                selected[j,k,machine]=(f,duration)
                flags.append(f)
            model.add_exactly_one(flags)
            if prev is not None:
                model.add(start>=prev)
            prev=end
        finished.append(prev)
    for intervals in machine_intervals:
        model.add_no_overlap(intervals)
    makespan=model.new_int_var(0,horizon,"makespan")
    model.add_max_equality(makespan,finished)
    model.minimize(makespan)
    solver=cp_model.CpSolver()
    solver.parameters.max_time_in_seconds=seconds
    solver.parameters.num_search_workers=1
    solver.parameters.random_seed=seed
    t=time.perf_counter()
    status=solver.solve(model)
    elapsed=time.perf_counter()-t
    name=solver.status_name(status)
    if status not in (cp_model.OPTIMAL,cp_model.FEASIBLE):
        raise RuntimeError(f"no feasible CP-SAT solution for {path.name}: {name}")
    records=[]
    by_machine=[[] for _ in range(m)]
    for (j,k),st in starts.items():
        picked=[(machine,duration) for (jj,kk,machine),(f,duration) in selected.items()
                if (jj,kk)==(j,k) and solver.value(f)]
        if len(picked)!=1:
            raise AssertionError("exactly one selected alternative was violated")
        machine,duration=picked[0]
        a,b=solver.value(st),solver.value(ends[j,k])
        if a+duration!=b:
            raise AssertionError("machine interval invalid")
        if k and a<solver.value(ends[j,k-1]):
            raise AssertionError("job precedence invalid")
        by_machine[machine].append((a,b,j,k))
        records.append((j,k,machine,a,b))
    for mach,rs in enumerate(by_machine):
        ordered=sorted(rs)
        if any(ordered[i][1]>ordered[i+1][0] for i in range(len(ordered)-1)):
            raise AssertionError(f"machine {mach} overlap")
    actual_max=max(x[4] for x in records)
    if actual_max != round(solver.objective_value):
        raise AssertionError("replayed schedule and solver objective differ")
    return {
        "instance":path.name,"jobs":n,"machines":m,
        "operations":len(records),"makespan":actual_max,
        "status":name,"best_bound":float(solver.best_objective_bound),
        "gap_to_lower_bound":(actual_max-float(solver.best_objective_bound))/max(actual_max,1),
        "valid_precedence_and_machine_intervals":True,
        "elapsed_cpu_s":elapsed,"limit_s":seconds,
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--upstream",type=Path,required=True)
    p.add_argument("--count",type=int,default=10)
    p.add_argument("--seconds",type=float,default=2)
    p.add_argument("--seed",type=int,default=1810)
    args=p.parse_args()
    if args.count<1 or args.seconds<=0:
        raise ValueError("invalid budget")
    files=sorted((args.upstream/"data_test/1005").glob("*.fjs"),key=lambda p:p.name)[:args.count]
    if len(files)!=args.count:
        raise AssertionError("source FJSP instances missing")
    rows=[]
    for p in files:
        row=solve(p,args.seconds,args.seed)
        rows.append(row)
        print("FJSP_CP_SAT_INSTANCE",json.dumps(row),flush=True)
    summary={"instances":len(rows),"feasible":sum(x["valid_precedence_and_machine_intervals"] for x in rows),
             "optimal":sum(x["status"]=="OPTIMAL" for x in rows),
             "mean_makespan":sum(x["makespan"] for x in rows)/len(rows),
             "mean_wall_s":sum(x["elapsed_cpu_s"] for x in rows)/len(rows),
             "mean_gap_to_lower_bound":sum(x["gap_to_lower_bound"] for x in rows)/len(rows),
             "time_limit_per_instance_s":args.seconds,"seed":args.seed}
    print("FJSP_CP_SAT_FINISHED",json.dumps(summary),flush=True)


if __name__=="__main__":
    main()
