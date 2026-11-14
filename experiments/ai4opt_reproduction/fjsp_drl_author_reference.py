"""Evaluate the ORIGINAL AUTHOR-PUBLISHED HGNN/PPO checkpoint on public FJSP.

Unlike our training harness, this reads the author's already-public
model/save_10_5.pt from pinned upstream source into ephemeral runner memory.
It NEVER copies or publishes that model to the user's repository and never
loads, writes or touches PR61/PR78 private checkpoints.

Source: songwenas12/fjsp-drl d7637b70029be6cb8fe78e450dcb07767876daea
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import subprocess
from pathlib import Path
import sys
import time

import torch

from fjsp_drl_repro import (
    UPSTREAM_SHA,
    seed_everything,
    sha256_file,
    evaluate,
    summarize,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--eval-count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1810)
    args = parser.parse_args()
    root = args.upstream_root.resolve()
    checkpoint = root / "model" / "save_10_5.pt"
    if not checkpoint.is_file() or checkpoint.stat().st_size != 849961:
        raise FileNotFoundError("the exact author's PUBLIC 10x5 model file is absent")
    found = [p for p in root.rglob("*.pt") if p.is_file()]
    if found != [checkpoint]:
        raise PermissionError("only the exact single already-public author reference may be loaded")
    original_sha = sha256_file(checkpoint)
    original_code_commit = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    if original_code_commit != UPSTREAM_SHA:
        raise ValueError("paper checkpoint is not paired with pinned upstream code")
    os.chdir(root)
    sys.path.insert(0, str(root))
    from env.fjsp_env import FJSPEnv
    ppo_module = importlib.import_module("PPO_model")
    with (root / "config.json").open(encoding="utf-8") as src:
        cfg = json.load(src)
    env_paras = dict(cfg["env_paras"])
    env_paras["device"] = torch.device("cpu")
    model_paras = dict(cfg["model_paras"])
    model_paras["device"] = torch.device("cpu")
    model_paras["actor_in_dim"] = 2 * model_paras["out_size_ma"] + 2 * model_paras["out_size_ope"]
    model_paras["critic_in_dim"] = model_paras["out_size_ma"] + model_paras["out_size_ope"]
    torch.set_num_threads(1)
    seed_everything(args.seed)
    model = ppo_module.PPO(model_paras, cfg["train_paras"], num_envs=1)
    original_state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(original_state, dict) or not all(
        isinstance(t, torch.Tensor) for t in original_state.values()
    ):
        raise TypeError("author checkpoint is not a tensor-only state_dict")
    model.policy_old.load_state_dict(original_state, strict=True)
    model.policy.load_state_dict(original_state, strict=True)
    # Never print tensor values, copy or checkpoint the model.
    test_files = sorted((root / "data_test" / "1005").glob("*.fjs"))[:args.eval_count]
    if len(test_files) != args.eval_count:
        raise RuntimeError("original 100 public 10x5 test problems missing")
    t = time.perf_counter()
    author = evaluate(model, test_files, FJSPEnv, env_paras, ppo_module, "AUTHOR_PRETRAINED")
    author_time = time.perf_counter()-t
    t = time.perf_counter()
    spt = evaluate(model, test_files, FJSPEnv, env_paras, ppo_module, "SPT")
    spt_time = time.perf_counter()-t
    report = {
        "kind": "original_author_published_pretrained_HGNN_PPO_inference",
        "not_our_model_weights": True,
        "original_public_model_loaded_only_in_ephemeral_runner": True,
        "no_user_model_checkpoints_loaded_or_written": True,
        "no_checkpoint_output_or_artifact": True,
        "source_commit": UPSTREAM_SHA,
        "author_published_checkpoint_size_bytes": checkpoint.stat().st_size,
        "author_published_checkpoint_sha256": original_sha,
        "reference_seed": args.seed,
        "eval_cases": args.eval_count,
        "cpu_torch": torch.__version__,
        "author_pretrained": summarize(test_files, author),
        "spt": summarize(test_files, spt),
        "author_cpu_eval_wall_seconds": author_time,
        "spt_cpu_eval_wall_seconds": spt_time,
    }
    assert report["author_pretrained"]["valid_schedules"] == args.eval_count
    assert report["spt"]["valid_schedules"] == args.eval_count
    print("AI4OPT_AUTHOR_PRETRAINED_RESULT", json.dumps(report, sort_keys=True), flush=True)
    print("AI4OPT_AUTHOR_PRETRAINED_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
