"""Reproduce official Song et al. FJSP-DRL HGNN/PPO on public 10x5 instances.

Source: https://github.com/songwenas12/fjsp-drl
Pinned commit: d7637b70029be6cb8fe78e450dcb07767876daea
Paper: DOI 10.1109/TII.2022.3189725 (Song et al., IEEE TII, 2023).

This harness imports the ORIGINAL env / PPO / graph modules, trains only
in memory, and never imports, downloads, writes, publishes, or uploads
ANY model checkpoint. This is a bounded reproducibility test of the
published implementation, not a paper-scale training claim. Baseline:
standard shortest processing time (SPT) from the published FJSP rules.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch


UPSTREAM_SHA = "d7637b70029be6cb8fe78e450dcb07767876daea"
WEIGHT_EXTENSIONS = {".pt", ".pth", ".ckpt", ".safetensors", ".bin"}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)


def assert_no_weights(root: Path) -> None:
    bad = [
        str(p.relative_to(root))
        for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in WEIGHT_EXTENSIONS
    ]
    if bad:
        raise PermissionError(
            "PRIVATE_WEIGHT_POLICY: reproduction MUST NOT stage any "
            "model checkpoint, even official upstream files: " + repr(bad[:8])
        )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as src:
        for chunk in iter(lambda: src.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def official_policy_action(model, state, dones, memory_module):
    with torch.inference_mode():
        return model.policy_old.act(
            state, memory_module.Memory(), dones,
            flag_sample=False, flag_train=False,
        )


def spt_action(state):
    """Standard non-learning shortest-processing-time dispatching rule.

    Enumerates only currently legal operation-machine pairs from the ORIGINAL
    FJSP environment, chooses the shortest available processing duration.
    Ties are deterministically broken by job and machine index.
    """
    jobs_count = int(state.mask_job_finish_batch.shape[1])
    machines_count = int(state.mask_ma_procing_batch.shape[1])
    actions = []
    for b in state.batch_idxes.tolist():
        eligible = []
        for j in range(jobs_count):
            if bool(state.mask_job_finish_batch[b, j]):
                continue
            if bool(state.mask_job_procing_batch[b, j]):
                continue
            o = int(state.ope_step_batch[b, j])
            for m in range(machines_count):
                if bool(state.mask_ma_procing_batch[b, m]):
                    continue
                if bool(state.ope_ma_adj_batch[b, o, m]) and float(state.proc_times_batch[b, o, m]) > 0:
                    eligible.append(
                        (float(state.proc_times_batch[b, o, m]), j, m, o)
                    )
        if not eligible:
            raise AssertionError("ORIGINAL FJSP state has no legal SPT action")
        _, j, m, o = min(eligible)
        actions.append((o, m, j))
    return torch.tensor(actions, dtype=torch.long).T


def solve_file(
    instance: Path, env_class, env_paras, *,
    model=None, memory_module=None, method="SPT",
):
    env_p = dict(env_paras)
    env_p["batch_size"] = 1
    with instance.open(encoding="utf-8") as stream:
        header = stream.readline().split()
    if len(header) < 2:
        raise ValueError("invalid original public FJSP instance")
    env_p["num_jobs"] = int(header[0])
    env_p["num_mas"] = int(header[1])
    env = env_class([str(instance)], env_p, data_source="file")
    state = env.state
    dones = env.done_batch
    steps = 0
    horizon = int(env.nums_opes.max()) + 2
    while not bool(dones.all()):
        if steps > horizon:
            raise AssertionError("published FJSP policy did not complete all operations")
        if method == "SPT":
            action = spt_action(state)
        else:
            action = official_policy_action(model, state, dones, memory_module)
        state, _, dones = env.step(action)
        steps += 1
    lawful, _ = env.validate_gantt()
    if not lawful:
        raise AssertionError("published FJSP environment rejected schedule Gantt")
    makespan = float(env.makespan_batch[0].item())
    if makespan <= 0:
        raise AssertionError("invalid original-public-FJSP makespan")
    env.close()
    return {"makespan": makespan, "operations": steps, "gantt_valid": True}


def evaluate(model, test_files, env_class, env_paras, memory_module, method):
    result = []
    for item in test_files:
        r = solve_file(
            item, env_class, env_paras,
            model=model, memory_module=memory_module, method=method,
        )
        result.append({"case": item.name, **r})
    return result


def rollouts_for_ppo(
    ppo, source, env_class, env_paras, train_paras,
    memory_module, n_iterations, seed,
):
    completed = []
    all_losses = []
    total_steps = 0
    for iteration in range(n_iterations):
        # Exactly upstream's random instance generator; fixed op counts per
        # job allow original vectorized 10x5 parallel-instance execution.
        # Reconstruct the source case every iteration; do not overwrite data.
        case = source.CaseGenerator(
            env_paras["num_jobs"], env_paras["num_mas"], 4, 6,
            nums_ope=[
                random.randint(4, 6)
                for _ in range(env_paras["num_jobs"])
            ],
        )
        env = env_class(case, dict(env_paras), data_source="case")
        memory = memory_module.Memory()
        state = env.state
        dones = env.done_batch
        guard = int(env.nums_opes.max()) + 2
        steps = 0
        while not bool(dones.all()):
            if steps > guard:
                raise AssertionError("original PPO episode did not terminate")
            with torch.no_grad():
                action = ppo.policy_old.act(
                    state, memory, dones, flag_sample=True, flag_train=True
                )
            state, rewards, dones = env.step(action)
            memory.rewards.append(rewards)
            memory.is_terminals.append(dones)
            steps += 1
        lawful, _ = env.validate_gantt()
        if not lawful:
            raise AssertionError("original FJSP PPO rollout violated Gantt feasibility")
        before = {
            k: v.detach().clone()
            for k, v in ppo.policy.state_dict().items()
        }
        loss, reward = ppo.update(memory, env_paras, train_paras)
        if not np.isfinite(float(loss)) or not np.isfinite(float(reward)):
            raise AssertionError("nonfinite original PPO optimizer result")
        delta = sum(
            int(not torch.equal(v, before[k]))
            for k, v in ppo.policy.state_dict().items()
        )
        if delta == 0:
            raise AssertionError("original FJSP PPO produced no parameter update")
        all_losses.append(float(loss))
        total_steps += steps * int(env_paras["batch_size"])
        completed.append({
            "iteration": iteration + 1,
            "training_batch": int(env_paras["batch_size"]),
            "operations_per_instance": steps,
            "valid_gantt": True,
            "changed_state_tensors": delta,
            "ppo_loss": float(loss),
            "mean_training_makespan": float(env.makespan_batch.mean()),
        })
        memory.clear_memory()
        env.close()
        print(
            "AI4OPT_FJSP_TRAIN_STEP",
            json.dumps(completed[-1], sort_keys=True),
            flush=True,
        )
    return completed, total_steps


def summarize(test_files, rows):
    vals = [r["makespan"] for r in rows]
    return {
        "cases": len(vals),
        "valid_schedules": sum(bool(r["gantt_valid"]) for r in rows),
        "mean_makespan": float(np.mean(vals)),
        "std_makespan": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
        "min_makespan": float(min(vals)),
        "max_makespan": float(max(vals)),
        "per_case_makespan": [
            {"name": p.name, "makespan": r["makespan"]}
            for p, r in zip(test_files, rows)
        ],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--eval-count", type=int, default=12)
    parser.add_argument("--seed", type=int, default=1810)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    root = args.upstream_root.resolve()
    if args.iterations < 1 or args.eval_count < 4:
        raise ValueError("reproduction needs positive real training and >=4 held-out cases")
    assert_no_weights(root)
    os.chdir(root)
    sys.path.insert(0, str(root))
    import env.fjsp_env as fjsp_env
    from env import case_generator
    ppo_module = importlib.import_module("PPO_model")
    with (root / "config.json").open(encoding="utf-8") as stream:
        original_cfg = json.load(stream)
    pars = original_cfg["env_paras"].copy()
    pars.update({
        "batch_size": 2, "num_jobs": 10, "num_mas": 5,
        "valid_batch_size": args.eval_count,
        "device": torch.device("cpu"),
    })
    mp = original_cfg["model_paras"].copy()
    mp["device"] = torch.device("cpu")
    mp["actor_in_dim"] = 2 * mp["out_size_ma"] + 2 * mp["out_size_ope"]
    mp["critic_in_dim"] = mp["out_size_ma"] + mp["out_size_ope"]
    train_pars = original_cfg["train_paras"].copy()
    train_pars["viz"] = False
    test_files = sorted((root / "data_test" / "1005").glob("*.fjs"))[:args.eval_count]
    if len(test_files) != args.eval_count:
        raise RuntimeError("official published 10x5 held-out benchmark missing")
    if any("save_" in p.name for p in test_files):
        raise AssertionError("invalid benchmark sample list")
    seed_everything(args.seed)
    model = ppo_module.PPO(mp, train_pars, num_envs=int(pars["batch_size"]))
    parameter_count = sum(p.numel() for p in model.policy.parameters())
    before = evaluate(
        model, test_files, fjsp_env.FJSPEnv, pars, ppo_module, "HGNN_PPO_BEFORE"
    )
    spt = evaluate(
        model, test_files, fjsp_env.FJSPEnv, pars, ppo_module, "SPT"
    )
    training, transitions = rollouts_for_ppo(
        model, case_generator, fjsp_env.FJSPEnv, pars, train_pars,
        ppo_module, args.iterations, args.seed,
    )
    after = evaluate(
        model, test_files, fjsp_env.FJSPEnv, pars, ppo_module, "HGNN_PPO_AFTER"
    )
    assert_no_weights(root)
    report = {
        "kind": "bounded_original_HGNN_PPO_implementation_reproduction",
        "not_paper_scale": True,
        "not_trained_converged": True,
        "upstream": "songwenas12/fjsp-drl",
        "source_commit": UPSTREAM_SHA,
        "model_checkpoint_loaded": False,
        "model_checkpoint_written": False,
        "published_10x5_test_files": args.eval_count,
        "benchmark_source_sha256": {
            p.name: sha256_file(p) for p in test_files
        },
        "random_seed": args.seed,
        "train_iterations": args.iterations,
        "train_batch": int(pars["batch_size"]),
        "training_env_steps": transitions,
        "training_episodes_solved_legally": args.iterations * int(pars["batch_size"]),
        "algorithm": "Original published HGNN+PPO, unchanged loss and optimizer",
        "baseline": "SPT eligible original FJSP op-machine dispatch pairs",
        "paper_hyperparameters_except_iterations_and_batch": {
            "K_epochs": train_pars["K_epochs"],
            "lr": train_pars["lr"],
            "minibatch_size": train_pars["minibatch_size"],
            "eps_clip": train_pars["eps_clip"],
        },
        "torch": torch.__version__,
        "numpy": np.__version__,
        "device": "cpu",
        "network_parameters": parameter_count,
        "spt": summarize(test_files, spt),
        "hgnn_ppo_before": summarize(test_files, before),
        "hgnn_ppo_after": summarize(test_files, after),
        "training_history": training,
    }
    assert all(
        report[name]["valid_schedules"] == args.eval_count
        for name in ("spt", "hgnn_ppo_before", "hgnn_ppo_after")
    )
    print("AI4OPT_FJSP_REAL_REPRO_REPORT", json.dumps(report, sort_keys=True), flush=True)
    if args.output is not None:
        if args.output.suffix != ".json":
            raise ValueError("only a public metrics JSON output is supported")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        # No model weights, private data, or optimizer state in JSON.
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print("AI4OPT_FJSP_REPRO_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
