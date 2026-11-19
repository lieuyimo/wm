"""Compare three CP-SAT starts on the 40 frozen standard FJSP instances."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from hashlib import sha256
import json
import multiprocessing
from pathlib import Path
import platform
import re
from statistics import mean
import subprocess
import sys
import time


AUTHOR_COMMIT = "d7637b70029be6cb8fe78e450dcb07767876daea"
BENCHMARK_COMMIT = "d088a26582f5cb79c6b4f2ad94f68ac7c50878cb"
AUTHOR_MODEL_SHA = "594f93a9f9838f89bf508297b3019ebaae8a0482a6f3abf91a3be7412198f0ce"
HURINK = ("la01", "la06", "la11", "la16", "la21", "la26", "la36", "abz7", "la31", "mt06")
ARMS = ("cp_sat", "spt_hint", "hgnn_hint")


def verify_checkout(root, expected):
    commit = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    diff = subprocess.run(["git", "-C", str(root), "diff", "--quiet", "HEAD", "--", "*.py", "config.json"], check=False)
    if commit != expected or diff.returncode:
        raise ValueError(f"Expected unmodified checkout at {expected}: {root}")


def cases(root):
    data = root / "data/fjsp"
    yield from (("brandimarte", data / f"brandimarte/Mk{number:02}.fjs") for number in range(1, 11))
    for family in ("edata", "rdata", "vdata"):
        yield from ((family, data / f"hurink/{family}/{name}.fjs") for name in HURINK)


def verify_instances(root, manifest_path):
    manifest = json.loads(manifest_path.read_text())["instances"]
    expected = {(Path(row["path"]).parent.name, Path(row["path"]).name): row["sha256"] for row in manifest}
    inputs = list(cases(root))
    actual = {(family, path.name): sha256(path.read_bytes()).hexdigest() for family, path in inputs}
    if len(manifest) != 40 or expected != actual:
        raise ValueError("The selected instances differ from benchmark v1")
    return inputs, {f"{family}/{name}": digest for (family, name), digest in actual.items()}


def read_case(path, *, allow_zero_duration=False):
    lines = [line for line in path.read_text().splitlines() if line.strip()]
    num_jobs, num_machines = map(int, lines[0].split()[:2])
    if len(lines) != num_jobs + 1:
        raise ValueError(f"Wrong job count in {path}")
    minimum_duration = 0 if allow_zero_duration else 1
    jobs = []
    for line in lines[1:]:
        tokens = line.split()
        cursor = 1
        operations = []
        for _ in range(int(tokens[0])):
            count = int(tokens[cursor])
            raw = list(map(int, tokens[cursor + 1:cursor + 1 + 2 * count]))
            options = {machine - 1: duration for machine, duration in zip(raw[::2], raw[1::2], strict=True)}
            if not options or len(options) != count or any(
                machine < 0 or machine >= num_machines or duration < minimum_duration for machine, duration in options.items()
            ):
                raise ValueError(f"Invalid machine alternatives in {path}")
            operations.append(options)
            cursor += 1 + 2 * count
        if not operations or cursor != len(tokens):
            raise ValueError(f"Invalid operation count in {path}")
        jobs.append(operations)
    return num_machines, jobs


def validate(jobs, schedule):
    if set(schedule) != {(job, op) for job, operations in enumerate(jobs) for op in range(len(operations))}:
        raise ValueError("Schedule does not cover exactly the instance operations")
    machine_intervals = {}
    for (job, op), (machine, start, end) in schedule.items():
        duration = jobs[job][op].get(machine)
        if duration is None or start < 0 or end - start != duration:
            raise ValueError("Schedule violates machine eligibility or processing time")
        if op and start < schedule[job, op - 1][2]:
            raise ValueError("Schedule violates job precedence")
        machine_intervals.setdefault(machine, []).append((start, end))
    for intervals in machine_intervals.values():
        intervals.sort()
        if any(left[1] > right[0] for left, right in zip(intervals, intervals[1:])):
            raise ValueError("Schedule has overlapping machine intervals")
    return max(end for _, _, end in schedule.values())


def spt_action(state, torch):
    choices = []
    for job in range(state.mask_job_finish_batch.size(1)):
        if state.mask_job_procing_batch[0, job] or state.mask_job_finish_batch[0, job]:
            continue
        op = int(state.ope_step_batch[0, job])
        for machine in range(state.mask_ma_procing_batch.size(1)):
            if not state.mask_ma_procing_batch[0, machine] and state.ope_ma_adj_batch[0, op, machine]:
                choices.append((float(state.proc_times_batch[0, op, machine]), job, machine, op))
    _, job, machine, op = min(choices)
    return torch.tensor([[op], [machine], [job]], dtype=torch.long)


def load_dispatcher(upstream, checkpoint, seed, inference_device):
    import torch

    sys.path.insert(0, str(upstream))
    from PPO_model import Memory, PPO
    from env.fjsp_env import FJSPEnv

    torch.set_num_threads(1)
    torch.manual_seed(seed)
    config = json.loads((upstream / "config.json").read_text())
    device = torch.device(inference_device)
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("Inference device must be cpu, cuda or cuda:N")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA inference requested but CUDA is unavailable")
    network = None
    if checkpoint is not None:
        model_settings = dict(config["model_paras"], device=device)
        model_settings["critic_in_dim"] = model_settings["out_size_ma"] + model_settings["out_size_ope"]
        model_settings["actor_in_dim"] = 2 * model_settings["critic_in_dim"]
        with device:
            network = PPO(model_settings, config["train_paras"])
        network.policy_old.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)
        network.policy_old.eval()
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    def dispatch(path, arm):
        started = time.perf_counter()
        machines, jobs = read_case(path)
        operations = [(job, op) for job, entries in enumerate(jobs) for op in range(len(entries))]
        arm_device = device if arm == "hgnn_hint" else torch.device("cpu")
        settings = dict(config["env_paras"], device=arm_device, num_jobs=len(jobs), num_mas=machines,
                        batch_size=1, show_mode="none")
        with arm_device:
            environment = FJSPEnv(case=[str(path)], env_paras=settings, data_source="file")
            memory = Memory()
            with torch.inference_mode():
                for _ in operations:
                    action = (spt_action(environment.state, torch) if arm == "spt_hint" else
                              network.policy_old.act(environment.state, memory, environment.done_batch,
                                                     flag_sample=False, flag_train=False))
                    environment.step(action)
                valid, gantt = environment.validate_gantt()
        if not bool(environment.done_batch.all()) or not valid:
            raise ValueError(f"Invalid {arm} start for {path}")
        flat = gantt[0].tolist()
        if any(any(value != int(value) for value in row) for row in flat):
            raise ValueError(f"Non-integer schedule field for {path}")
        schedule = {op: tuple(map(int, row[1:])) for op, row in zip(operations, flat, strict=True)}
        if validate(jobs, schedule) != int(environment.makespan_batch[0]):
            raise ValueError(f"Makespan mismatch for {path}")
        if arm_device.type == "cuda":
            torch.cuda.synchronize(arm_device)
        return schedule, time.perf_counter() - started

    return dispatch


def generate_starts(inputs, upstream, checkpoint, seed, inference_device):
    dispatch = load_dispatcher(upstream, checkpoint, seed, inference_device)
    prepared = []
    for family, path in inputs:
        for arm in ARMS[1:]:
            schedule, elapsed = dispatch(path, arm)
            prepared.append((family, path, arm, schedule, elapsed))
        prepared.append((family, path, "cp_sat", None, None))
    return prepared


def load_solver_model(path, benchmark_root):
    if str(benchmark_root) not in sys.path:
        sys.path.insert(0, str(benchmark_root))
    from data.data_parsers import parser_fjsp
    from scheduling_environment.jobShop import JobShop
    from solution_methods.cp_sat.models.FJSPmodel import fjsp_cp_sat_model

    _, jobs = read_case(path)
    environment = parser_fjsp.parse_fjsp(JobShop(), str(path), from_absolute_path=True)
    model, variables = fjsp_cp_sat_model(environment)
    starts, presences = variables["starts"], variables["presences"]
    options = {(job, op): list(task.processing_times.items())
               for job, entry in enumerate(environment.jobs) for op, task in enumerate(entry.operations)}
    if any(dict(alternatives) != jobs[job][op] for (job, op), alternatives in options.items()):
        raise ValueError(f"Benchmark parsers disagree for {path}")
    return model, jobs, starts, presences, options


def add_partial_hint(model, starts, presences, options, hint):
    for key, start in starts.items():
        machine, begin, _ = hint[key]
        model.add_hint(start, begin)
        if len(options[key]) > 1:
            for index, (alternative, _) in enumerate(options[key]):
                model.add_hint(presences[(*key, index)], int(machine == alternative))


def solve_case(item):
    from ortools.sat.python import cp_model

    family, path, arm, hint, hint_seconds, budget, seed, benchmark_root, solver_workers = item
    model, jobs, starts, presences, options = load_solver_model(path, benchmark_root)

    initial = None
    if hint is not None:
        initial = validate(jobs, hint)
        add_partial_hint(model, starts, presences, options, hint)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = budget
    solver.parameters.num_search_workers = solver_workers
    solver.parameters.random_seed = seed
    effective_workers = solver_workers or None
    if solver_workers == 0:
        def capture_workers(message):
            nonlocal effective_workers
            match = re.search(r"Setting number of workers to (\d+)|Starting search[^\n]* with (\d+) workers", message)
            if match:
                effective_workers = int(match.group(1) or match.group(2))

        solver.parameters.log_search_progress = True
        solver.parameters.log_to_stdout = False
        solver.log_callback = capture_workers
    status = solver.solve(model)
    if status in (cp_model.MODEL_INVALID, cp_model.INFEASIBLE):
        raise RuntimeError(f"Unexpected CP-SAT {solver.status_name(status)} for {family}/{path.name}")
    solver_cost = None
    if status in (cp_model.FEASIBLE, cp_model.OPTIMAL):
        schedule = {}
        for key, start in starts.items():
            (machine, duration), = (option for index, option in enumerate(options[key])
                                    if solver.value(presences[(*key, index)]))
            begin = solver.value(start)
            schedule[key] = (machine, begin, begin + duration)
        solver_cost = validate(jobs, schedule)
        if solver_cost != solver.objective_value:
            raise ValueError(f"CP-SAT objective mismatch for {family}/{path.name}")

    available_costs = [cost for cost in (initial, solver_cost) if cost is not None]
    return {"family": family, "instance": path.name, "arm": arm,
            "initial_makespan": initial, "solver_makespan": solver_cost,
            "final_makespan": min(available_costs, default=None),
            "status": solver.status_name(status), "best_bound": solver.best_objective_bound,
            "initial_wall_seconds": hint_seconds, "search_wall_seconds": solver.wall_time,
            "effective_solver_workers": effective_workers}


def summarize(rows):
    costs = {(row["family"], row["instance"], row["arm"]): row["final_makespan"] for row in rows}
    keys = sorted({(family, instance) for family, instance, _ in costs})
    by_arm = {}
    for arm in ARMS:
        results = [costs[(*case, arm)] for case in keys if costs[(*case, arm)] is not None]
        by_arm[arm] = {"feasible": len(results), "total": len(keys), "mean_makespan": mean(results) if results else None}
    paired = {}
    for treatment, control in (("spt_hint", "cp_sat"), ("hgnn_hint", "cp_sat"), ("hgnn_hint", "spt_hint")):
        deltas = [costs[(*case, treatment)] - costs[(*case, control)] for case in keys
                  if costs[(*case, treatment)] is not None and costs[(*case, control)] is not None]
        paired[f"{treatment}_vs_{control}"] = {"pairs": len(deltas), "mean_delta": mean(deltas) if deltas else None,
                                               "win": sum(delta < 0 for delta in deltas),
                                               "tie": sum(delta == 0 for delta in deltas),
                                               "loss": sum(delta > 0 for delta in deltas)}
    return {"arms": by_arm, "paired": paired}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--benchmarks", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=60)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1810)
    parser.add_argument("--solver-workers", type=int, default=1, help="CP-SAT threads per search; 0 uses the solver default")
    parser.add_argument("--solver-seed", type=int, help="Solver seed; defaults to the inference seed")
    parser.add_argument("--inference-python", type=Path, default=Path(sys.executable))
    parser.add_argument("--inference-device", default="cpu", help="HGNN inference device: cpu, cuda or cuda:N")
    parser.add_argument("--prepare-only", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.seconds <= 0 or args.workers < 1 or args.solver_workers < 0:
        parser.error("seconds and workers must be positive; solver-workers cannot be negative")
    if args.solver_workers == 0 and args.workers != 1:
        parser.error("run one search at a time when using CP-SAT's default thread count")
    solver_seed = args.seed if args.solver_seed is None else args.solver_seed
    checkpoint_sha = sha256(args.checkpoint.read_bytes()).hexdigest()
    if checkpoint_sha != args.checkpoint_sha256.lower():
        raise ValueError("Checkpoint SHA-256 does not match the selected checkpoint")
    inputs, digests = verify_instances(args.benchmarks, args.manifest)
    if args.prepare_only:
        import torch

        verify_checkout(args.upstream, AUTHOR_COMMIT)
        prepared = generate_starts(inputs, args.upstream.resolve(), args.checkpoint, args.seed, args.inference_device)
        device = torch.device(args.inference_device)
        payload = {"torch": torch.__version__, "inference_device": str(device),
                   "inference_gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None, "rows": [
            [family, str(path), arm, [[*key, *entry] for key, entry in schedule.items()] if schedule else None, spent]
            for family, path, arm, schedule, spent in prepared
        ]}
        (args.output / "starts.json").write_text(json.dumps(payload) + "\n")
        print(f"Validated {sum(bool(item[3]) for item in prepared)} initial schedules", flush=True)
        return

    import ortools

    args.output.mkdir(parents=True, exist_ok=False)
    subprocess.run([str(args.inference_python), str(Path(__file__).resolve()), *sys.argv[1:], "--prepare-only"], check=True)
    payload = json.loads((args.output / "starts.json").read_text())
    prepared = [(family, Path(path), arm, {(job, op): (machine, start, end) for job, op, machine, start, end in rows}
                 if rows is not None else None, spent) for family, path, arm, rows, spent in payload["rows"]]
    provenance = {"benchmark_commit": BENCHMARK_COMMIT, "author_commit": AUTHOR_COMMIT,
                  "checkpoint_sha256": checkpoint_sha, "author_published_checkpoint": checkpoint_sha == AUTHOR_MODEL_SHA,
                  "seconds_per_search": args.seconds, "seed": args.seed, "solver_seed": solver_seed,
                  "solver_workers": args.solver_workers,
                  "concurrent_searches": args.workers, "python": platform.python_version(),
                  "ortools": ortools.__version__, "torch": payload["torch"],
                  "inference_device": payload["inference_device"], "inference_gpu": payload["inference_gpu"],
                  "instances": digests}
    (args.output / "protocol.json").write_text(json.dumps(provenance, indent=2) + "\n")
    tasks = (item + (args.seconds, solver_seed, args.benchmarks.resolve(), args.solver_workers) for item in prepared)
    rows = []
    with (args.output / "results.jsonl").open("x") as log, ProcessPoolExecutor(
        max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        for row in pool.map(solve_case, tasks):
            rows.append(row)
            print(json.dumps(row), file=log, flush=True)
            print(f"{len(rows)}/{len(prepared)} {row['family']}/{row['instance']} {row['arm']} {row['final_makespan']}",
                  flush=True)
    summary = {"overall": summarize(rows), "by_family": {
        family: summarize([row for row in rows if row["family"] == family]) for family in sorted({row["family"] for row in rows})
    }}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
