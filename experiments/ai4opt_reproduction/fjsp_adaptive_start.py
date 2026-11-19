"""Measure fixed and adaptive FJSP starts at a common end-to-end deadline."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
from hashlib import sha256
import json
from pathlib import Path
from statistics import median
from time import perf_counter

from fjsp_end_to_end import ARMS, emit, frozen_targets, worker
from fjsp_hint_diagnostics import add_complete_hint
from fjsp_warm_start import AUTHOR_COMMIT, load_solver_model, validate, verify_checkout, verify_instances


BUDGET = 10.0
PROBE_SECONDS = 0.75
SEEDS = (1, 2, 3)
CALIBRATION_SEED = 17
THRESHOLDS = (0.10, 0.20, 0.35)
DEADLINES = (1, 5, 10)


def split_cases(cases):
    def rank(stem):
        return sha256(f"wm-fjsp-adaptive-v1:{stem}".encode()).digest()

    brandimarte = {Path(case).stem for case in cases if case.startswith("brandimarte/")}
    hurink = {Path(case).stem for case in cases if not case.startswith("brandimarte/")}
    assert len(brandimarte) == len(hurink) == 10 and not brandimarte & hurink
    dev = set(sorted(brandimarte, key=rank)[:5] + sorted(hurink, key=rank)[:5])
    return {case: "dev" if Path(case).stem in dev else "test" for case in cases}


def solve(model_data, target, deadline, seed, workers):
    from ortools.sat.python import cp_model

    model, jobs, starts, presences, options = model_data

    class ReachTarget(cp_model.CpSolverSolutionCallback):
        def on_solution_callback(self):
            if self.objective_value <= target:
                self.stop_search()

    solver = cp_model.CpSolver()
    callback = ReachTarget()
    solver.parameters.random_seed = seed
    solver.parameters.num_search_workers = workers
    remaining = deadline - perf_counter()
    if remaining <= 0:
        return None, "SKIPPED", None, None
    solver.parameters.max_time_in_seconds = remaining
    status = solver.solve(model, callback)
    status_name = solver.status_name(status)
    if status in (cp_model.INFEASIBLE, cp_model.MODEL_INVALID):
        raise RuntimeError(f"Unexpected solver status {status_name}")
    answer = objective_disagreement = None
    bound = solver.best_objective_bound
    if status in (cp_model.FEASIBLE, cp_model.OPTIMAL):
        schedule = {}
        for key, var in starts.items():
            (machine, duration), = (option for index, option in enumerate(options[key])
                                    if solver.value(presences[(*key, index)]))
            start = solver.value(var)
            schedule[key] = (machine, start, start + duration)
        makespan = validate(jobs, schedule)
        makespan_var = model.get_int_var_from_proto_index(len(model.proto.variables) - 1)
        if makespan_var.name != "makespan" or solver.value(makespan_var) != makespan:
            raise ValueError("Makespan variable disagrees with the checked schedule")
        if bound > makespan or (status == cp_model.OPTIMAL and bound != makespan):
            raise ValueError("Solver bound disagrees with the checked schedule")
        # Short parallel solves can report an objective from a different incumbent.
        if solver.objective_value != makespan:
            objective_disagreement = solver.objective_value
        answer = makespan, schedule
    return answer, status_name, bound, objective_disagreement


def measure(path, root, generators, arm, target, seed, workers, threshold=None, budget=BUDGET, route=None):
    start = perf_counter()
    deadline = start + budget
    data = load_solver_model(path, root)
    model, jobs, _, _, options = data
    result = None
    probe_status = probe_makespan = probe_bound = probe_gap = None
    probe_objective_disagreement = final_objective_disagreement = None
    probe_seconds = 0.0
    final_status = "SKIPPED"
    initial_arm = arm if arm in ARMS[1:] else None
    route_seconds = None
    if route is not None:
        route_start = perf_counter()
        initial_arm = route(jobs)
        route_seconds = perf_counter() - route_start
    if arm in ("adaptive", "probe_cp"):
        probe_start = perf_counter()
        probe_end = min(deadline, probe_start + PROBE_SECONDS)
        result, probe_status, probe_bound, probe_objective_disagreement = solve(data, target, probe_end, seed, workers)
        probe_seconds = perf_counter() - probe_start
        if result is not None:
            probe_makespan = result[0]
            probe_gap = max(0.0, (probe_makespan - probe_bound) / probe_makespan)
        needs_search = result is None or (result[0] > target and probe_status != "OPTIMAL")
        if arm == "adaptive" and needs_search and (probe_gap is None or probe_gap > threshold):
            initial_arm = "hgnn_hint"

    initial_seconds = 0.0
    initial_makespan = None
    if initial_arm is not None and perf_counter() < deadline:
        initial_start = perf_counter()
        schedule, _ = generators[initial_arm](path)
        initial_makespan = validate(jobs, schedule)
        initial_seconds = perf_counter() - initial_start
        if result is None or initial_makespan < result[0]:
            result = initial_makespan, schedule

    if (result is None or result[0] > target) and probe_status != "OPTIMAL" and perf_counter() < deadline:
        if result is not None:
            add_complete_hint(model, options, result[1], result[0])
        answer, final_status, _, final_objective_disagreement = solve(data, target, deadline, seed, workers)
        if answer is not None and (result is None or answer[0] < result[0]):
            result = answer

    makespan = result[0] if result is not None else None
    elapsed = perf_counter() - start
    quality_reached = makespan is not None and makespan <= target
    return {"end_to_end_seconds": elapsed, "target_makespan": target, "final_makespan": makespan,
            "quality_reached": quality_reached, "hit": quality_reached and elapsed <= budget,
            "initial_arm": initial_arm if initial_makespan is not None else None, "initial_makespan": initial_makespan,
            "initial_seconds": initial_seconds, "probe_status": probe_status, "probe_seconds": probe_seconds,
            "probe_makespan": probe_makespan, "probe_bound": probe_bound, "probe_gap": probe_gap,
            "probe_objective_disagreement": probe_objective_disagreement,
            "final_objective_disagreement": final_objective_disagreement,
            "route_seconds": route_seconds,
            "final_status": final_status, "threshold": threshold}


def hits(rows, seconds):
    return sum(row["quality_reached"] and row["end_to_end_seconds"] <= seconds for row in rows)


def metrics(rows):
    attained = [row["end_to_end_seconds"] for row in rows if row["hit"]]
    return {"requests": len(rows), "hits_by_deadline_seconds": {s: hits(rows, s) for s in DEADLINES},
            "median_hit_seconds": median(attained) if attained else None,
            "hgnn_calls": sum(row["initial_arm"] == "hgnn_hint" for row in rows),
            "past_budget": sum(row["end_to_end_seconds"] > BUDGET for row in rows),
            "reached_quality_past_budget": sum(row["quality_reached"] and not row["hit"] for row in rows)}


def calibration_score(rows):
    return (hits(rows, BUDGET), hits(rows, 1),
            -sum(row["end_to_end_seconds"] if row["hit"] else BUDGET for row in rows))


def run(args):
    import ortools

    verify_checkout(args.upstream, AUTHOR_COMMIT)
    checkpoint_digest = sha256(args.checkpoint.read_bytes()).hexdigest()
    if checkpoint_digest != args.checkpoint_sha256.lower():
        raise ValueError("Unexpected checkpoint digest")
    inputs, hashes = verify_instances(args.benchmarks, args.manifest)
    paths = {f"{family}/{path.name}": path for family, path in inputs}
    targets = frozen_targets(args.reference, inputs, hashes, expected_budget=60)
    split = split_cases(paths)
    args.output.mkdir(parents=True, exist_ok=args.resume)
    protocol = {"benchmark_sha256": hashes, "reference_sha256": sha256(args.reference.read_bytes()).hexdigest(),
                "targets": targets, "split": split, "checkpoint_sha256": checkpoint_digest,
                "budget_seconds": BUDGET, "probe_seconds": PROBE_SECONDS, "thresholds": THRESHOLDS,
                "solver_seed": SEEDS, "calibration_seed": CALIBRATION_SEED,
                "solver_workers": args.solver_workers, "ortools": ortools.__version__}
    protocol_path = args.output / "protocol.json"
    if args.resume:
        previous = json.loads(protocol_path.read_text())
        previous.pop("initial_workers")
        if previous != json.loads(json.dumps(protocol)):
            raise ValueError("Cannot resume with a different benchmark protocol")
    with ExitStack() as stack:
        workers = {arm: stack.enter_context(worker(args, arm)) for arm in ARMS[1:]}
        generators = {arm: entry[0] for arm, entry in workers.items()}
        protocol["initial_workers"] = {arm: entry[1] for arm, entry in workers.items()}
        if not args.resume:
            protocol_path.write_text(json.dumps(protocol, indent=2) + "\n")

        def execute_phase(phase, cases, seeds, variants):
            planned = []
            for seed_index, seed in enumerate(seeds):
                for index, case in enumerate(cases):
                    offset = (index + seed_index) % len(variants)
                    planned.extend((seed, case, arm, t) for arm, t in variants[offset:] + variants[:offset])
            log = args.output / f"{phase}.jsonl"
            rows = [json.loads(line) for line in log.read_text().splitlines()] if args.resume and log.exists() else []
            completed = [(r["seed"], r["case"], r["arm"], r["threshold"]) for r in rows]
            if completed != planned[:len(rows)]:
                raise ValueError(f"Stored {phase} results do not match the planned run order")
            with log.open("a" if args.resume else "x") as stream:
                for seed, case, arm, threshold in planned[len(rows):]:
                    outcome = measure(paths[case], args.benchmarks, generators, arm,
                                      targets[case], seed, args.solver_workers, threshold)
                    row = {"case": case, "split": split[case], "seed": seed, "arm": arm, **outcome}
                    rows.append(row)
                    print(json.dumps(row), file=stream, flush=True)
                    emit({"phase": phase, "count": len(rows), "case": case, "seed": seed,
                          "arm": arm, "hit": outcome["hit"], "seconds": round(outcome["end_to_end_seconds"], 3)})
            return rows

        baseline = execute_phase("baseline", tuple(paths), SEEDS, tuple((arm, None) for arm in ARMS))
        development = tuple(case for case in paths if split[case] == "dev")
        heldout = tuple(case for case in paths if split[case] == "test")
        calibration = execute_phase("calibration", development, (CALIBRATION_SEED,),
                                    tuple(("adaptive", threshold) for threshold in THRESHOLDS))
        candidates = {threshold: [r for r in calibration if r["threshold"] == threshold] for threshold in THRESHOLDS}
        selected = max(THRESHOLDS, key=lambda threshold: (*calibration_score(candidates[threshold]), threshold))
        policy = {"gap_threshold": selected, "calibration": {t: metrics(rows) for t, rows in candidates.items()}}
        policy_path = args.output / "policy.json"
        if args.resume and policy_path.exists():
            if json.loads(policy_path.read_text()) != json.loads(json.dumps(policy)):
                raise ValueError("Stored policy disagrees with the calibrated policy")
        else:
            policy_path.write_text(json.dumps(policy, indent=2) + "\n")
        emit({"phase": "policy_frozen", "gap_threshold": selected})
        adaptive = execute_phase("heldout", heldout, SEEDS, (("adaptive", selected), ("probe_cp", None)))

    test_rows = [row for row in baseline if row["split"] == "test"] + adaptive
    arms = (*ARMS, "probe_cp", "adaptive")
    summary = {"baseline_by_seed": {seed: {arm: metrics([r for r in baseline if r["seed"] == seed and r["arm"] == arm])
                                           for arm in ARMS} for seed in SEEDS},
               "heldout": {arm: metrics([r for r in test_rows if r["arm"] == arm]) for arm in arms},
               "heldout_by_seed": {seed: {arm: metrics([r for r in test_rows if r["seed"] == seed and r["arm"] == arm])
                                          for arm in arms} for seed in SEEDS}}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("upstream", "benchmarks", "manifest", "reference", "checkpoint", "output", "inference-python"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--solver-workers", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.set_defaults(seed=1810, inference_device="cpu")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
