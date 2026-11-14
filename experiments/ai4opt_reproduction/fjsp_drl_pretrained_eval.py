#!/usr/bin/env python3
"""Faithful public Song et al. FJSP-DRL pretrained evaluation, no training.

The paper's published HGNN+PPO and published checkpoint are used unchanged.
We invoke their original FJSPEnv directly, avoiding test.py's unconditional
NVIDIA NVML and pandas.ExcelWriter.save() assumptions. SPT/LPT are explicitly
separate classic rules evaluated in the SAME ORIGINAL environment.
No trained parameters, gradients, optimizer, checkpoint, or model state is
saved, printed, uploaded or added to the owning repository.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch

UPSTREAM_COMMIT = "d7637b70029be6cb8fe78e450dcb07767876daea"


def sha256(path: Path):
    d = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            d.update(b)
    return d.hexdigest()


def setup_upstream(root: Path):
    sys.path.insert(0, str(root))
    from PPO_model import Memory, PPO
    from env.fjsp_env import FJSPEnv
    from env.load_data import nums_detec
    return Memory, PPO, FJSPEnv, nums_detec


def eligible_actions(state, method: str):
    """SPT/LPT on paper's current legal machine-operation pairs, no oracle."""
    if len(state.batch_idxes) != 1 or int(state.batch_idxes[0]) != 0:
        raise AssertionError("one independent instance per rollout required")
    jobs = int(state.mask_job_finish_batch.size(1))
    machines = int(state.mask_ma_procing_batch.size(1))
    choices = []
    for j in range(jobs):
        if bool(state.mask_job_procing_batch[0, j]) or bool(state.mask_job_finish_batch[0, j]):
            continue
        op = int(state.ope_step_batch[0, j])
        end = int(state.end_ope_biases_batch[0, j])
        if op > end:
            continue
        for m in range(machines):
            if bool(state.mask_ma_procing_batch[0, m]):
                continue
            if int(state.ope_ma_adj_batch[0, op, m]) != 1:
                continue
            pt = float(state.proc_times_batch[0, op, m])
            if pt <= 0:
                raise AssertionError("zero duration is not a legal alternative")
            choices.append((pt, j, m, op))
    if not choices:
        raise ValueError("original environment has no legal action before all jobs finish")
    if method == "SPT":
        pt, j, m, op = min(choices)
    elif method == "LPT":
        pt, j, m, op = min(choices, key=lambda x: (-x[0], x[1], x[2], x[3]))
    else:
        raise ValueError(method)
    return torch.tensor([[op], [m], [j]], dtype=torch.long)


def evaluate_instance(root, filename, method, model, Memory, FJSPEnv, nums_detec, config):
    lines = filename.read_text().splitlines(keepends=True)
    n_jobs, n_machines, n_ops = nums_detec(lines)
    env_paras = dict(config["env_paras"])
    env_paras.update({"device": torch.device("cpu"), "num_jobs": n_jobs,
                      "num_mas": n_machines, "batch_size": 1, "show_mode": "none"})
    env = FJSPEnv(case=[str(filename)], env_paras=env_paras, data_source="file")
    memory = Memory()
    steps = 0
    tic = time.perf_counter()
    while not bool(env.done_batch.all()):
        if steps >= n_ops + 2:
            raise AssertionError("paper environment did not complete in allotted operations")
        with torch.inference_mode():
            action = (model.policy_old.act(env.state, memory, env.done_batch,
                                           flag_sample=False, flag_train=False)
                      if method == "HGNN-PPO-pretrained"
                      else eligible_actions(env.state, method))
        if action.shape != (3, 1):
            raise AssertionError("illegal action tensor shape")
        _, _, done = env.step(action)
        steps += 1
    elapsed = time.perf_counter() - tic
    valid = bool(env.validate_gantt()[0])
    cost = float(env.makespan_batch[0].item())
    if not valid or steps != n_ops or not np.isfinite(cost):
        raise AssertionError(f"illegal/nonterminal schedule: valid={valid} steps={steps}/{n_ops}")
    return {"instance": filename.name, "jobs": n_jobs, "machines": n_machines,
            "operations": n_ops, "makespan": cost, "valid_gantt": valid,
            "decisions": steps, "elapsed_cpu_s": elapsed}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--upstream", type=Path, required=True)
    p.add_argument("--count", type=int, default=10)
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--seed", type=int, default=1810)
    args = p.parse_args()
    if not 1 <= args.count <= 100:
        raise ValueError("limit sample to first 1..100 published instances")
    root = args.upstream.resolve()
    version = os.popen(f'git -C "{root}" rev-parse HEAD').read().strip()
    if version != UPSTREAM_COMMIT:
        raise AssertionError(f"upstream version drift: {version}, expected {UPSTREAM_COMMIT}")
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    Memory, PPO, FJSPEnv, nums_detec = setup_upstream(root)
    config = json.loads((root / "config.json").read_text())
    model_paras = dict(config["model_paras"])
    model_paras["device"] = torch.device("cpu")
    model_paras["actor_in_dim"] = model_paras["out_size_ma"] * 2 + model_paras["out_size_ope"] * 2
    model_paras["critic_in_dim"] = model_paras["out_size_ma"] + model_paras["out_size_ope"]
    model = PPO(model_paras, config["train_paras"])
    published_checkpoint = root / "model/save_10_5.pt"
    # The author's checkpoint is ALREADY publicly released in its own repo;
    # we never read, copy, upload, log or serialize any privately trained wm weight.
    data = torch.load(published_checkpoint, map_location="cpu", weights_only=True)
    model.policy.load_state_dict(data, strict=True)
    model.policy_old.load_state_dict(data, strict=True)
    model.policy.eval()
    model.policy_old.eval()
    del data
    files = sorted((root / "data_test/1005").glob("*.fjs"), key=lambda q: q.name)[:args.count]
    if len(files) != args.count:
        raise AssertionError("missing source test instances")
    print("PAPER_REPRO_SOURCE", json.dumps({
        "repo": "songwenas12/fjsp-drl",
        "commit": version,
        "published_model_file_sha256": sha256(published_checkpoint),
        "model_sha_is_only_identity_not_weights": True,
        "pytorch": torch.__version__,
        "numpy": np.__version__,
        "environment": "author FJSPEnv, single-instance, CPU",
        "policy": "author PPO HGNN pretrained greedy",
        "paper_model_privacy": "no model tensors persist beyond run",
        "seed": args.seed,
        "test_count": len(files),
    }), flush=True)
    output = {"paper": "Song et al. Flexible Job Shop Scheduling via Graph Neural Network and Deep Reinforcement Learning (IEEE TII 2023)",
              "type": "published pretrained checkpoint inference reproduction, not training",
              "commit": version, "published_checkpoint_sha256": sha256(published_checkpoint),
              "seed": args.seed, "torch": torch.__version__, "numpy": np.__version__,
              "source_files": {x.name: sha256(x) for x in files},
              "rows": {}, "summaries": {}}
    for method in ["HGNN-PPO-pretrained", "SPT", "LPT"]:
        rows = []
        for f in files:
            x = evaluate_instance(root, f, method, model, Memory, FJSPEnv, nums_detec, config)
            rows.append(x)
            print("PAPER_REPRO_INSTANCE", json.dumps({"method": method, **x}), flush=True)
        output["rows"][method] = rows
        output["summaries"][method] = {
            "instances": len(rows), "feasible": sum(x["valid_gantt"] for x in rows),
            "mean_makespan": sum(x["makespan"] for x in rows) / len(rows),
            "mean_wall_s": sum(x["elapsed_cpu_s"] for x in rows) / len(rows),
            "sum_wall_s": sum(x["elapsed_cpu_s"] for x in rows),
        }
    baseline = output["rows"]["SPT"]
    drl = output["rows"]["HGNN-PPO-pretrained"]
    output["paired_vs_spt"] = {
        "mean_makespan_delta": sum(drl[i]["makespan"] - baseline[i]["makespan"] for i in range(len(files))) / len(files),
        "wins": sum(drl[i]["makespan"] < baseline[i]["makespan"] for i in range(len(files))),
        "ties": sum(drl[i]["makespan"] == baseline[i]["makespan"] for i in range(len(files))),
        "losses": sum(drl[i]["makespan"] > baseline[i]["makespan"] for i in range(len(files))),
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n")
    print("PAPER_REPRO_FINISHED", json.dumps({
        "summaries": output["summaries"], "paired_vs_spt": output["paired_vs_spt"]
    }), flush=True)


if __name__ == "__main__":
    main()
