"""Solve a standard FJSP request and export a validated schedule as JSON."""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
from hashlib import sha256
import json
import math
import multiprocessing
import os
from pathlib import Path
import select
import subprocess
import sys
from time import perf_counter

from fjsp_hint_diagnostics import add_complete_hint
from fjsp_static_start import features, use_hgnn
from fjsp_warm_start import AUTHOR_COMMIT, load_solver_model, read_case, validate, verify_checkout


RULE = (150, 267)
STRATEGIES = ("auto", "cp-sat", "hgnn")


class InferenceError(RuntimeError):
    pass


class InferenceWorker:
    def __init__(self, python, upstream, checkpoint, digest):
        self.config = (python, upstream, checkpoint, digest)
        self.process = None
        self.pending = bytearray()

    def receive(self, deadline):
        descriptor = self.process.stdout.fileno()
        while (end := self.pending.find(b"\n")) < 0:
            remaining = deadline - perf_counter()
            if remaining <= 0 or not select.select([descriptor], [], [], remaining)[0]:
                raise InferenceError("inference_timeout")
            chunk = os.read(descriptor, 65536)
            if not chunk:
                raise InferenceError("inference_worker_exited")
            self.pending.extend(chunk)
        line = bytes(self.pending[:end])
        del self.pending[:end + 1]
        message = json.loads(line)
        if not isinstance(message, dict):
            raise InferenceError("invalid_inference_response")
        return message

    def start(self, deadline):
        if self.process is not None:
            if self.process.poll() is None:
                return
            self.close()
        python, upstream, checkpoint, digest = self.config
        if None in self.config:
            raise InferenceError("model_not_configured")
        try:
            verify_checkout(upstream, AUTHOR_COMMIT)
            with checkpoint.open("rb") as stream:
                actual = sha256()
                for chunk in iter(lambda: stream.read(65536), b""):
                    if perf_counter() >= deadline:
                        raise InferenceError("inference_timeout")
                    actual.update(chunk)
            if actual.hexdigest() != digest.lower():
                raise InferenceError("checkpoint_digest_mismatch")
            if perf_counter() >= deadline:
                raise InferenceError("inference_timeout")
            script = Path(__file__).with_name("fjsp_end_to_end.py")
            self.process = subprocess.Popen([str(python), "-u", str(script), "worker", "--arm", "hgnn_hint",
                                             "--upstream", str(upstream), "--checkpoint", str(checkpoint),
                                             "--seed", "1810", "--device", "cpu"], stdin=subprocess.PIPE,
                                            stdout=subprocess.PIPE, bufsize=0)
            ready = self.receive(deadline)
            if ready.get("ready") is not True or ready.get("device") != "cpu" or ready.get("gpu") is not None:
                raise InferenceError("invalid_inference_worker")
        except (OSError, ValueError, subprocess.SubprocessError) as error:
            self.close()
            raise InferenceError("model_load_failed") from error
        except InferenceError:
            self.close()
            raise

    def generate(self, path, deadline):
        self.start(deadline)
        try:
            payload = (json.dumps({"path": str(path)}) + "\n").encode()
            self.process.stdin.write(payload)
            answer = self.receive(deadline)
            return {(job, op): (machine, start, end) for job, op, machine, start, end in answer["schedule"]}
        except (OSError, ValueError, TypeError, KeyError) as error:
            self.close()
            raise InferenceError("inference_failed") from error
        except InferenceError:
            self.close()
            raise

    def close(self):
        if self.process is not None:
            if self.process.poll() is None:
                self.process.kill()
            self.process.wait()
            self.process.stdin.close()
            self.process.stdout.close()
            self.process = None
        self.pending.clear()


def cp_search(send, path, benchmarks, available_until, seed, workers, initial, target):
    from ortools.sat.python import cp_model

    try:
        with redirect_stdout(sys.stderr):
            model, jobs, starts, presences, options = load_solver_model(path, benchmarks)
        if initial is not None:
            add_complete_hint(model, options, initial[1], initial[0])

        class Incumbent(cp_model.CpSolverSolutionCallback):
            def on_solution_callback(self):
                schedule = {}
                for key, start in starts.items():
                    (machine, duration), = (option for index, option in enumerate(options[key])
                                            if self.value(presences[(*key, index)]))
                    begin = self.value(start)
                    schedule[key] = (machine, begin, begin + duration)
                cost = validate(jobs, schedule)
                send.send(("candidate", (cost, schedule)))
                if target is not None and cost <= target:
                    self.stop_search()

        solver = cp_model.CpSolver()
        solver.parameters.random_seed = seed
        solver.parameters.num_search_workers = workers
        remaining = available_until - perf_counter()
        if remaining > 0:
            solver.parameters.max_time_in_seconds = remaining
            status = solver.solve(model, Incumbent())
            send.send(("status", solver.status_name(status)))
    except Exception as error:
        print(f"CP-SAT worker: {error}", file=sys.stderr)
        send.send(("status", "ERROR"))
    finally:
        send.close()


class FjspSolver:
    def __init__(self, benchmarks, inference, workers=0):
        # Import once before requests. Process isolation bounds late native search shutdown.
        from ortools.sat.python import cp_model  # noqa: F401

        self.benchmarks = benchmarks.resolve()
        self.inference = inference
        self.workers = workers
        self.context = multiprocessing.get_context("fork")
        self.pending_search = None

    def reap_search(self, timeout=None):
        if self.pending_search is None:
            return
        self.pending_search.join(timeout)
        if self.pending_search.exitcode is not None:
            self.pending_search.close()
            self.pending_search = None

    def search(self, path, cutoff, deadline, jobs, seed, best, target):
        self.reap_search(max(0.0, cutoff - perf_counter()))
        if self.pending_search is not None or perf_counter() >= cutoff:
            return best, "DEADLINE"
        receiver, sender = self.context.Pipe(duplex=False)
        process = self.context.Process(target=cp_search, args=(sender, path, self.benchmarks, cutoff,
                                                               seed, self.workers, best, target))
        status = "DEADLINE"
        process.start()
        sender.close()
        try:
            while (remaining := cutoff - perf_counter()) > 0 and receiver.poll(remaining):
                try:
                    kind, value = receiver.recv()
                except EOFError:
                    status = "ERROR"
                    break
                if kind == "status":
                    status = value
                    break
                cost, schedule = value
                if validate(jobs, schedule) != cost:
                    raise ValueError("CP-SAT worker returned a mismatched makespan")
                if perf_counter() >= cutoff:
                    break
                if best is None or cost < best[0]:
                    best = cost, schedule
                if target is not None and best[0] <= target:
                    status = "TARGET"
                    break
        finally:
            if process.is_alive():
                process.kill()
            receiver.close()
            self.pending_search = process
            self.reap_search(max(0.0, deadline - perf_counter() - 0.005))
        return best, status

    def solve(self, path, seconds, strategy="auto", target=None, seed=1):
        started = perf_counter()
        if not math.isfinite(seconds) or seconds <= 0 or strategy not in STRATEGIES or (target is not None and target <= 0):
            raise ValueError("Expected a positive finite time limit, a supported strategy and a positive target")
        deadline = started + seconds
        cutoff = deadline - min(0.25, seconds * 0.4)
        path = path.resolve()
        machines, jobs = read_case(path)
        if not jobs or machines <= 0:
            raise ValueError("The instance must contain jobs and machines")
        counts = features(jobs)
        selected = ("hgnn" if use_hgnn(counts, RULE) else "cp-sat") if strategy == "auto" else strategy
        reason = ("both_thresholds_met" if selected == "hgnn" else "below_static_thresholds") if strategy == "auto" else "explicit"
        fallback = None
        best = None
        inference_seconds = 0.0
        if selected == "hgnn":
            before = perf_counter()
            try:
                # Reserve one fifth of the request for a CP-SAT fallback.
                inference_deadline = min(started + seconds * 0.8, cutoff - min(0.02, seconds * 0.1))
                schedule = self.inference.generate(path, inference_deadline)
                cost = validate(jobs, schedule)
                if perf_counter() >= deadline:
                    raise InferenceError("inference_timeout")
                best = cost, schedule
            except InferenceError as error:
                fallback = str(error)
            except (TypeError, ValueError):
                fallback = "invalid_neural_schedule"
            inference_seconds = perf_counter() - before

        inference_completed = best is not None
        status = "TARGET" if best is not None and target is not None and best[0] <= target else "DEADLINE"
        if status != "TARGET" and perf_counter() < cutoff:
            best, status = self.search(path, cutoff, deadline, jobs, seed, best, target)
        cost, schedule = best if best is not None else (None, {})
        operations = [{"job": job, "operation": op, "machine": machine, "start": start, "end": end}
                      for (job, op), (machine, start, end) in sorted(schedule.items())]
        elapsed = perf_counter() - started
        reached = target is not None and cost is not None and cost <= target
        return {"instance": path.name, "requested_strategy": strategy, "selected_strategy": selected,
                "strategy_used": "cp-sat" if fallback else selected, "decision_reason": reason, "fallback_reason": fallback,
                "total_operations": counts[0], "optional_intervals": counts[1], "budget_seconds": seconds,
                "target_makespan": target, "quality_reached": reached, "hit": reached and elapsed <= seconds,
                "end_to_end_seconds": elapsed, "inference_seconds": inference_seconds, "solver_status": status,
                "inference_completed": inference_completed, "within_budget": elapsed <= seconds,
                "cleanup_deferred": self.pending_search is not None,
                "makespan": cost, "operations": operations}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("instance", type=Path)
    parser.add_argument("--benchmarks", type=Path, required=True)
    parser.add_argument("--seconds", type=float, required=True)
    parser.add_argument("--strategy", choices=STRATEGIES, default="auto")
    parser.add_argument("--target", type=int)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--solver-workers", type=int, default=0)
    parser.add_argument("--upstream", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--checkpoint-sha256")
    parser.add_argument("--inference-python", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    worker = InferenceWorker(args.inference_python, args.upstream, args.checkpoint, args.checkpoint_sha256)
    solver = None
    try:
        solver = FjspSolver(args.benchmarks, worker, args.solver_workers)
        result = solver.solve(args.instance, args.seconds, args.strategy, args.target, args.seed)
        text = json.dumps(result, indent=2) + "\n"
        if args.output is None:
            print(text, end="", flush=True)
        else:
            with args.output.open("x") as out:
                out.write(text)
    except (OSError, ValueError, IndexError) as error:
        parser.error(str(error))
    finally:
        worker.close()
        if solver is not None:
            solver.reap_search()


if __name__ == "__main__":
    main()
