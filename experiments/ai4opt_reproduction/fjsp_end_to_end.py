"""Measure real wall time from a FJSP request to a validated schedule at a frozen quality target."""

from __future__ import annotations

import argparse
from contextlib import contextmanager, ExitStack, redirect_stdout
from hashlib import sha256
import json
from pathlib import Path
from statistics import mean, median
import subprocess
import sys
from time import perf_counter

from fjsp_hint_diagnostics import add_complete_hint
from fjsp_warm_start import AUTHOR_COMMIT, load_dispatcher, load_solver_model, validate, verify_checkout, verify_instances


ARMS = ("cp_sat", "spt_hint", "hgnn_hint")
PILOT = ("brandimarte/Mk01.fjs", "edata/la21.fjs", "vdata/abz7.fjs", "vdata/la31.fjs")


def emit(document):
    print(json.dumps(document), flush=True)


def initial_worker(args):
    with redirect_stdout(sys.stderr):
        import torch

        dispatcher = load_dispatcher(args.upstream, args.checkpoint, args.seed, args.device)
    device = torch.device(args.device)
    emit({"ready": True, "torch": torch.__version__, "device": str(device),
          "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None})
    for line in sys.stdin:
        with redirect_stdout(sys.stderr):
            schedule, duration = dispatcher(Path(json.loads(line)["path"]), args.arm)
        emit({"schedule": [[*key, *value] for key, value in schedule.items()], "seconds": duration})


@contextmanager
def worker(args, arm):
    hgnn = arm == "hgnn_hint"
    device = args.inference_device if hgnn else "cpu"
    command = [str(args.inference_python), "-u", str(Path(__file__).resolve()), "worker", "--arm", arm,
               "--upstream", str(args.upstream), "--seed", str(args.seed), "--device", device]
    if hgnn:
        command.extend(["--checkpoint", str(args.checkpoint)])
    started = perf_counter()
    proc = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
    try:
        line = proc.stdout.readline()
        if not line:
            raise RuntimeError(f"{arm} worker failed before becoming ready")
        ready = json.loads(line)
        if not ready.get("ready") or ready.get("device") != device or bool(ready.get("gpu")) != device.startswith("cuda"):
            raise RuntimeError(f"Wrong worker or device for {arm}")
        startup = perf_counter() - started

        def initial(path):
            proc.stdin.write(json.dumps({"path": str(path)}) + "\n")
            proc.stdin.flush()
            answer = proc.stdout.readline()
            if not answer:
                raise RuntimeError(f"{arm} failed for {path}")
            record = json.loads(answer)
            schedule = {(job, op): (machine, begin, end) for job, op, machine, begin, end in record["schedule"]}
            return schedule, record["seconds"]

        yield initial, {**ready, "startup_seconds": startup}
    finally:
        proc.stdin.close()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.terminate()
            proc.wait(timeout=15)
        proc.stdout.close()


def measure(path, benchmark_root, generate_initial, target, budget, solver_seed, solver_workers):
    from ortools.sat.python import cp_model

    request_started = perf_counter()
    initial_schedule, generation_seconds = generate_initial(path) if generate_initial else (None, None)
    after_initial = perf_counter()
    model, jobs, starts, presences, options = load_solver_model(path, benchmark_root)
    initial = validate(jobs, initial_schedule) if initial_schedule is not None else None
    if initial_schedule is not None:
        add_complete_hint(model, options, initial_schedule, initial)

    class ReachTarget(cp_model.CpSolverSolutionCallback):
        def __init__(self):
            super().__init__()
            self.reached_at = None

        def on_solution_callback(self):
            if self.reached_at is None and self.objective_value <= target:
                self.reached_at = perf_counter() - request_started
                self.stop_search()

    callback = ReachTarget()
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = budget
    solver.parameters.num_search_workers = solver_workers
    solver.parameters.random_seed = solver_seed
    solve_started = perf_counter()
    status = solver.solve(model, callback)
    solve_ended = perf_counter()
    if status in (cp_model.MODEL_INVALID, cp_model.INFEASIBLE):
        raise RuntimeError(f"Unexpected {solver.status_name(status)} for {path}")
    final = None
    if status in (cp_model.FEASIBLE, cp_model.OPTIMAL):
        schedule = {}
        for key, start in starts.items():
            (machine, duration), = (option for index, option in enumerate(options[key])
                                    if solver.value(presences[(*key, index)]))
            begin = solver.value(start)
            schedule[key] = (machine, begin, begin + duration)
        final = validate(jobs, schedule)
        if final != solver.objective_value:
            raise ValueError(f"Solver objective mismatch for {path}")
    if initial is not None and (final is None or initial < final):
        final = initial
    reached = final is not None and final <= target
    elapsed = perf_counter() - request_started
    return {"target_makespan": target, "target_reached": reached, "initial_makespan": initial,
            "final_makespan": final, "status": solver.status_name(status),
            "initial_worker_seconds": generation_seconds, "initial_with_ipc_seconds": after_initial - request_started,
            "model_and_hint_seconds": solve_started - after_initial, "solve_call_seconds": solve_ended - solve_started,
            "validation_seconds": elapsed - (solve_ended - request_started),
            "target_callback_seconds": callback.reached_at, "end_to_end_seconds": elapsed}


def summarize(rows):
    arms = {}
    pairs = {}
    for arm in ARMS:
        cohort = [row for row in rows if row["arm"] == arm]
        successes = [row["end_to_end_seconds"] for row in cohort if row["target_reached"]]
        arms[arm] = {"cases": len(cohort), "target_reached": len(successes),
                     "median_success_seconds": median(successes) if successes else None,
                     "mean_all_seconds": mean(row["end_to_end_seconds"] for row in cohort) if cohort else None,
                     "median_initial_seconds": median(row["initial_with_ipc_seconds"] for row in cohort) if cohort else None}
    by_key = {(row["case"], row["arm"]): row for row in rows if row["target_reached"]}
    for control in ARMS[:2]:
        common = [(row["end_to_end_seconds"], by_key[row["case"], control]["end_to_end_seconds"])
                  for (case, arm), row in by_key.items() if arm == "hgnn_hint" and (case, control) in by_key]
        pairs[control] = {"both_reached": len(common), "hgnn_faster": sum(a < b for a, b in common),
                          "median_hgnn_seconds": median(a for a, _ in common) if common else None,
                          "median_control_seconds": median(b for _, b in common) if common else None,
                          "median_hgnn_minus_control_seconds": median(a - b for a, b in common) if common else None}
    return {"arms": arms, "hgnn_pairs": pairs}


def frozen_targets(path, inputs, hashes, expected_budget):
    record = json.loads(path.read_text())
    if record["time_limit_seconds"] != expected_budget or len(record["rows"]) != len(inputs):
        raise ValueError("Unexpected reference case count or historical solver budget")
    by_file = {(row["name"], row["sha256"]): row for row in record["rows"]}
    targets = {}
    for family, case in inputs:
        key = f"{family}/{case.name}"
        row = by_file[case.stem, hashes[key]]
        cost = row["objective"]
        if row["status"] not in {"FEASIBLE", "OPTIMAL"} or cost is None or int(cost) != cost:
            raise ValueError(f"No integer feasible reference for {key}")
        targets[key] = int(cost)
    return targets


def run(args):
    import ortools

    verify_checkout(args.upstream, AUTHOR_COMMIT)
    checkpoint_sha = sha256(args.checkpoint.read_bytes()).hexdigest()
    if checkpoint_sha != args.checkpoint_sha256.lower():
        raise ValueError("Checkpoint SHA-256 differs from private training manifest")
    inputs, hashes = verify_instances(args.benchmarks, args.manifest)
    targets = frozen_targets(args.reference, inputs, hashes, args.seconds)
    path_by_case = {f"{family}/{path.name}": path for family, path in inputs}
    cases = (*PILOT, *(case for case in path_by_case if case not in PILOT))
    args.output.mkdir(parents=True, exist_ok=False)
    protocol = {"reference_sha256": sha256(args.reference.read_bytes()).hexdigest(), "targets": targets,
                "instance_sha256": hashes, "cases": cases, "checkpoint_sha256": checkpoint_sha,
                "solver_workers": args.solver_workers, "solver_seed": args.solver_seed, "ortools": ortools.__version__,
                "hgnn_inference_device": args.inference_device,
                "search_budget_seconds": args.seconds, "timer": "case read to validated result; persistent workers"}
    rows = []
    with ExitStack() as stack, (args.output / "results.jsonl").open("x") as log:
        workers = {arm: stack.enter_context(worker(args, arm)) for arm in ARMS[1:]}
        protocol["worker_setup"] = {arm: metadata for arm, (_, metadata) in workers.items()}
        (args.output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
        emit({"stage": "workers_ready", "worker_setup": protocol["worker_setup"]})
        for index, case in enumerate(cases):
            path = path_by_case[case]
            order = ARMS[index % len(ARMS):] + ARMS[:index % len(ARMS)]
            for arm in order:
                initial = workers[arm][0] if arm in workers else None
                record = {"case": case, "arm": arm, **measure(path, args.benchmarks, initial,
                          targets[case], args.seconds, args.solver_seed, args.solver_workers)}
                rows.append(record)
                print(json.dumps(record), file=log, flush=True)
                emit({"progress": len(rows), "case": case, "arm": arm, "reached": record["target_reached"],
                      "seconds": round(record["end_to_end_seconds"], 3)})
    summary = {"overall": summarize(rows), "by_family": {
        family: summarize([row for row in rows if row["case"].startswith(family + "/")])
        for family in sorted({case.split("/", 1)[0] for case in cases})}}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    benchmark = commands.add_parser("run")
    for name in ("upstream", "benchmarks", "manifest", "reference", "checkpoint", "output", "inference-python"):
        benchmark.add_argument("--" + name, type=Path, required=True)
    benchmark.add_argument("--checkpoint-sha256", required=True)
    benchmark.add_argument("--inference-device", choices=("cpu", "cuda:0"), default="cuda:0")
    benchmark.add_argument("--seed", type=int, default=1810)
    benchmark.add_argument("--solver-seed", type=int, default=1)
    benchmark.add_argument("--solver-workers", type=int, default=0)
    benchmark.add_argument("--seconds", type=float, default=60)
    dispatch = commands.add_parser("worker")
    dispatch.add_argument("--arm", choices=ARMS[1:], required=True)
    dispatch.add_argument("--upstream", type=Path, required=True)
    dispatch.add_argument("--checkpoint", type=Path)
    dispatch.add_argument("--seed", type=int, required=True)
    dispatch.add_argument("--device", required=True)
    args = parser.parse_args()
    if args.command == "worker":
        initial_worker(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
