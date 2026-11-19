"""Trace partial and complete CP-SAT hints on four cases from the frozen FJSP comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

from fjsp_warm_start import add_partial_hint, load_solver_model, validate, verify_instances


CASES = ("brandimarte/Mk01.fjs", "edata/la21.fjs", "vdata/abz7.fjs", "vdata/la31.fjs")
ARMS = (
    ("cp_sat", None, None),
    ("spt_partial", "spt_hint", "partial"),
    ("hgnn_partial", "hgnn_hint", "partial"),
    ("spt_complete", "spt_hint", "complete"),
    ("hgnn_complete", "hgnn_hint", "complete"),
)
MILESTONES = (1, 5, 10, 60)


def add_complete_hint(model, opts, schedule, makespan):
    values = {"makespan": makespan}
    for (job, op), (machine, begin, end) in schedule.items():
        suffix = f"_j{job}_t{op}"
        values.update({"start" + suffix: begin, "end" + suffix: end, "duration" + suffix: end - begin})
        alternatives = opts[job, op]
        if len(alternatives) > 1:
            for index, (alternative, _) in enumerate(alternatives):
                selected = alternative == machine
                alt_suffix = f"{suffix}_a{index}"
                values["presence" + alt_suffix] = int(selected)
                values["start" + alt_suffix] = begin if selected else 0
                values["end" + alt_suffix] = end if selected else 0
    for index, variable in enumerate(model.proto.variables):
        if variable.name in values:
            value = values.pop(variable.name)
        elif len(variable.domain) == 2 and variable.domain[0] == variable.domain[1]:
            value = variable.domain[0]
        else:
            raise ValueError(f"No complete hint value for {variable.name} at variable {index}")
        if not any(lower <= value <= upper for lower, upper in zip(variable.domain[::2], variable.domain[1::2])):
            raise ValueError(f"Hint value {value} is outside {variable.name}'s domain")
        model.add_hint(model.get_int_var_from_proto_index(index), value)
    if values:
        raise ValueError(f"Unexpected TU/e variable names in complete hint: {sorted(values)[:8]}")


def validate_complete_hint(model, expected):
    from ortools.sat.python import cp_model

    check = cp_model.CpSolver()
    check.parameters.num_search_workers = 1
    check.parameters.max_time_in_seconds = 5
    check.parameters.fix_variables_to_their_hinted_value = True
    status = check.solve(model)
    if status != cp_model.OPTIMAL or check.objective_value != expected:
        raise ValueError(f"Complete hint failed fixed-variable feasibility check: {check.status_name(status)}")


def trace_case(path, benchmark_root, hint_kind, schedule, seed, workers, budget):
    from ortools.sat.python import cp_model

    model, jobs, starts, presences, options = load_solver_model(path, benchmark_root)
    initial = validate(jobs, schedule) if schedule is not None else None
    if hint_kind == "partial":
        add_partial_hint(model, starts, presences, options, schedule)
    elif hint_kind == "complete":
        add_complete_hint(model, options, schedule, initial)
        validate_complete_hint(model, initial)

    class Incumbents(cp_model.CpSolverSolutionCallback):
        def __init__(self):
            super().__init__()
            self.events = []

        def on_solution_callback(self):
            cost = round(self.objective_value)
            self.events.append({"seconds": self.wall_time, "makespan": cost, "bound": self.best_objective_bound})

    effective_workers = workers or None
    messages = []

    def on_log(message):
        nonlocal effective_workers
        for line in message.splitlines():
            match = re.search(r"Setting number of workers to (\d+)|Starting search[^\n]* with (\d+) workers", line)
            if match:
                effective_workers = int(match.group(1) or match.group(2))
            if "solution hint" in line.lower() or re.match(r"^#\d+\s", line):
                messages.append(line)

    callback = Incumbents()
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = budget
    solver.parameters.num_search_workers = workers
    solver.parameters.random_seed = seed
    solver.parameters.log_search_progress = True
    solver.parameters.log_to_stdout = False
    solver.log_callback = on_log
    status = solver.solve(model, callback)
    if status not in (cp_model.FEASIBLE, cp_model.OPTIMAL):
        raise RuntimeError(f"CP-SAT {solver.status_name(status)} for {path.name} with {hint_kind} hint")

    solution = {}
    for key, start in starts.items():
        (machine, duration), = (option for index, option in enumerate(options[key])
                                if solver.value(presences[(*key, index)]))
        begin = solver.value(start)
        solution[key] = (machine, begin, begin + duration)
    final = validate(jobs, solution)
    if final != solver.objective_value:
        raise ValueError(f"Solver objective mismatch for {path.name}")

    events = callback.events
    return {"initial_makespan": initial, "first_solution": events[0] if events else None,
            "by_seconds": {str(t): min((event["makespan"] for event in events if event["seconds"] <= t), default=None)
                           for t in MILESTONES if t <= budget},
            "final_makespan": final, "status": solver.status_name(status), "best_bound": solver.best_objective_bound,
            "search_wall_seconds": solver.wall_time, "effective_solver_workers": effective_workers,
            "hinted_variables": len(model.proto.solution_hint.vars), "model_variables": len(model.proto.variables),
            "incumbents": events, "solver_log": messages}


def read_starts(previous):
    starts = {}
    for family, path, arm, entries, _ in json.loads((previous / "starts.json").read_text())["rows"]:
        if entries is not None:
            key = f"{family}/{Path(path).name}", arm
            starts[key] = {(job, op): (machine, begin, end) for job, op, machine, begin, end in entries}
    return starts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous", type=Path, required=True)
    parser.add_argument("--benchmarks", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import ortools

    prior = json.loads((args.previous / "protocol.json").read_text())
    inputs, digest = verify_instances(args.benchmarks, args.manifest)
    if digest != prior["instances"] or ortools.__version__ != prior["ortools"]:
        raise ValueError("Cases or OR-Tools version differ from the prior comparison")
    paths = {f"{family}/{path.name}": path for family, path in inputs}
    starts = read_starts(args.previous)
    cached_rows = [json.loads(line) for line in (args.previous / "results.jsonl").read_text().splitlines()]
    for row in cached_rows:
        key = f"{row['family']}/{row['instance']}", row["arm"]
        if key in starts:
            final_time = max(end for _, _, end in starts[key].values())
            if final_time != row["initial_makespan"]:
                raise ValueError(f"Cached initial makespan differs from prior results for {key}")

    args.output.mkdir(parents=True, exist_ok=False)
    protocol = {"cases": {case: digest[case] for case in CASES}, "arms": [arm for arm, _, _ in ARMS],
                "solver_seed": prior["solver_seed"], "solver_workers": prior["solver_workers"],
                "budget_seconds": prior["seconds_per_search"], "concurrent_searches": 1,
                "checkpoint_sha256": prior["checkpoint_sha256"], "ortools": ortools.__version__,
                "initial_schedule_source": "cached default-thread comparison",
                "milestones_seconds": MILESTONES}
    (args.output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    summary = []
    with (args.output / "trace.jsonl").open("x") as log:
        for case in CASES:
            for arm, initial_arm, hint_kind in ARMS:
                schedule = starts[case, initial_arm] if initial_arm else None
                result = trace_case(paths[case], args.benchmarks.resolve(), hint_kind, schedule,
                                    protocol["solver_seed"], protocol["solver_workers"], protocol["budget_seconds"])
                record = {"case": case, "arm": arm, **result}
                print(json.dumps(record), file=log, flush=True)
                summary.append({key: value for key, value in record.items() if key not in {"incumbents", "solver_log"}})
                print(f"{len(summary)}/{len(CASES)*len(ARMS)} {case} {arm} {record['final_makespan']}", flush=True)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
