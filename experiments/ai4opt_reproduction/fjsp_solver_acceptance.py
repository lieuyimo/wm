"""Verify the public FJSP command and compare its three strategies on fresh inputs."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from time import perf_counter

from fjsp_adaptive_start import BUDGET, DEADLINES, SEEDS
from fjsp_end_to_end import emit
from fjsp_solver import FjspSolver, InferenceWorker, RULE
from fjsp_warm_start import BENCHMARK_COMMIT, read_case, validate


ARM_ORDER = ("cp-sat", "hgnn", "auto")


def check_result(path, result):
    _, jobs = read_case(path)
    schedule = {(r["job"], r["operation"]): (r["machine"], r["start"], r["end"]) for r in result["operations"]}
    if result["makespan"] is None:
        assert not schedule
    else:
        assert len(schedule) == len(result["operations"]) and validate(jobs, schedule) == result["makespan"]
    expected = (result["target_makespan"] is not None and result["makespan"] is not None
                and result["makespan"] <= result["target_makespan"])
    assert result["quality_reached"] == expected
    assert result["hit"] == (expected and result["end_to_end_seconds"] <= result["budget_seconds"])


def smoke(args, new_cases, model, solver):
    mk01 = args.benchmarks / "data/fjsp/brandimarte/Mk01.fjs"
    with TemporaryDirectory() as directory:
        temp = Path(directory)
        exported = temp / "schedule.json"
        subprocess.run([sys.executable, str(Path(__file__).with_name("fjsp_solver.py")), str(mk01),
                        "--benchmarks", str(args.benchmarks), "--seconds", "3", "--output", str(exported)], check=True)
        result = json.loads(exported.read_text())
        check_result(mk01, result)
        assert result["makespan"] == 40 and result["target_makespan"] is None
        assert result["selected_strategy"] == "cp-sat" and result["within_budget"]
        (args.output / "example-schedule.json").write_bytes(exported.read_bytes())

        expired = solver.solve(new_cases[-1], 0.035, "cp-sat", 1)
        check_result(new_cases[-1], expired)
        assert expired["solver_status"] in {"DEADLINE", "UNKNOWN"} and not expired["hit"]
        assert expired["end_to_end_seconds"] < 0.15

        absent = InferenceWorker(args.inference_python, args.upstream, temp / "missing.pt", args.checkpoint_sha256)
        recovered = FjspSolver(args.benchmarks, absent, args.solver_workers).solve(mk01, 3, "hgnn", 40)
        check_result(mk01, recovered)
        assert recovered["fallback_reason"] == "model_load_failed" and recovered["strategy_used"] == "cp-sat" and recovered["hit"]

        hanging = temp / "hanging-worker"
        hanging.write_text(f"#!{sys.executable}\nimport sys,time\n"
                           'print(\'{"ready":true,"device":"cpu","gpu":null}\',flush=True)\n'
                           "sys.stdin.readline()\ntime.sleep(60)\n")
        hanging.chmod(0o700)
        stalled = InferenceWorker(hanging, args.upstream, args.checkpoint, args.checkpoint_sha256)
        try:
            timed = FjspSolver(args.benchmarks, stalled, args.solver_workers).solve(mk01, 4, "hgnn", 40)
            check_result(mk01, timed)
            assert timed["fallback_reason"] == "inference_timeout" and timed["strategy_used"] == "cp-sat" and timed["hit"]
        finally:
            stalled.close()

    model.start(perf_counter() + 25)
    neural = solver.solve(mk01, 4, "hgnn", 40)
    check_result(mk01, neural)
    assert neural["hit"] and neural["inference_completed"] and neural["fallback_reason"] is None
    return {"cli_export_without_target": result["makespan"], "forced_timeout_seconds": expired["end_to_end_seconds"],
            "load_failure_recovered_seconds": recovered["end_to_end_seconds"],
            "inference_timeout_recovered_seconds": timed["end_to_end_seconds"],
            "real_hgnn_seconds": neural["end_to_end_seconds"]}


def summarize(rows):
    result = {}
    for strategy in ARM_ORDER:
        group = [row for row in rows if row["requested_strategy"] == strategy]
        result[strategy] = {
            "requests": len(group),
            "hits": {str(s): sum(row["quality_reached"] and row["end_to_end_seconds"] <= s for row in group) for s in DEADLINES},
            "invocations": sum(row["selected_strategy"] == "hgnn" for row in group),
            "completed_inferences": sum(row["inference_completed"] for row in group),
            "fallbacks": sum(row["fallback_reason"] is not None for row in group),
            "missed_deadlines": sum(not row["within_budget"] for row in group),
        }
    return result


def run(args):
    import ortools

    args.output.mkdir(parents=True, exist_ok=args.resume)
    manifest = json.loads(args.manifest.read_text())
    targets = {row["case"]: row["makespan"] for row in map(json.loads, args.reference.read_text().splitlines())}
    if len(manifest) != 18 or set(manifest) != set(targets):
        raise ValueError("Expected the complete 18-case Dauzere–Paulli benchmark")
    paths = {case: args.cases_dir / case for case in sorted(manifest)}
    for case, path in paths.items():
        if sha256(path.read_bytes()).hexdigest() != manifest[case]["sha256"]:
            raise ValueError(f"Changed benchmark case {case}")
    protocol = {"benchmark_commit": BENCHMARK_COMMIT, "instances": manifest, "targets": targets, "rule": list(RULE),
                "seeds": list(SEEDS), "workers": args.solver_workers, "budget_seconds": BUDGET,
                "reference_search_seconds": 60, "reference_seed": 11, "ortools": ortools.__version__}
    protocol_file = args.output / "protocol.json"
    if args.resume and protocol_file.exists():
        if json.loads(protocol_file.read_text()) != protocol:
            raise ValueError("Resume protocol differs")
    else:
        with protocol_file.open("x") as stream:
            stream.write(json.dumps(protocol, indent=2) + "\n")
    model = InferenceWorker(args.inference_python, args.upstream, args.checkpoint, args.checkpoint_sha256)
    solver = FjspSolver(args.benchmarks, model, args.solver_workers)
    try:
        smoke_file = args.output / "smoke.json"
        if not (args.resume and smoke_file.exists()):
            smoke_results = smoke(args, tuple(paths.values()), model, solver)
            smoke_file.write_text(json.dumps(smoke_results, indent=2) + "\n")
            emit({"phase": "smoke", **smoke_results})
        model.start(perf_counter() + 25)
        planned = []
        for i, seed in enumerate(SEEDS):
            for j, case in enumerate(paths):
                offset = (i + j) % len(ARM_ORDER)
                planned.extend((case, seed, arm) for arm in ARM_ORDER[offset:] + ARM_ORDER[:offset])
        log = args.output / "requests.jsonl"
        rows = list(map(json.loads, log.read_text().splitlines())) if args.resume and log.exists() else []
        if [(row["instance"], row["seed"], row["requested_strategy"]) for row in rows] != planned[:len(rows)]:
            raise ValueError("Resume rows differ from planned run order")
        with log.open("a" if args.resume else "x") as stream:
            for case, seed, arm in planned[len(rows):]:
                result = solver.solve(paths[case], BUDGET, arm, targets[case], seed)
                check_result(paths[case], result)
                del result["operations"]
                result["seed"] = seed
                rows.append(result)
                print(json.dumps(result), file=stream, flush=True)
                emit({"phase": "evaluation", "count": len(rows), "case": case, "seed": seed, "strategy": arm,
                      "hit": result["hit"], "seconds": round(result["end_to_end_seconds"], 3)})
    finally:
        model.close()
    summary = {"overall": summarize(rows), "by_seed": {seed: summarize([r for r in rows if r["seed"] == seed]) for seed in SEEDS}}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("benchmarks", "cases-dir", "manifest", "reference", "upstream", "checkpoint", "inference-python", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--solver-workers", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
