"""PR68: strict trust gate for the ORIGINAL PR61 local-only A100 archive.

No checkpoint bytes are bundled with this repository. The archive must be
explicitly supplied by its owner. Reject missing, wrong, truncated, swapped,
path-traversing, unverified and incompatible files *before* creating a model.

PR62 independently verified these source weights on the GPU host. PR68 reads
the original PR61 SHA/manifest and uses safe PyTorch weights-only loading on
CPU. Never use torch.load(weights_only=False) or silent random initialization.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import tarfile

import torch

from experiments.world_model_integration.sealed_pr59.pr59_dual_model import (
    PublicMapTemporalRelationalWorld,
    parameter_count,
)

ARCHIVE_NAME = "pr60-gpu-checkpoints-20261112.tar.gz"
ARCHIVE_BYTES = 6661520
ARCHIVE_SHA256 = "919a0b625659e492811a74bc5cdf76a60ccbf639acb32c33a420b7db30a74e14"
ORIGINAL_VAL_SHA256 = "c7a0a0081e0c901faea23b37816eb54b560efb9c8b66118387c3fbb0a9509712"
ORIGINAL_MODEL_HEAD = "e32c402ecbf67d19788c0a5c99e5282d6b5017e5"
CKPT_SCHEMA = "pr59-public-map-world-actual-gpu-checkpoint-v1"
ARMS = ("map_off", "map_on")
SEEDS = (1810, 1811, 1812)
CHECKPOINT_COUNT = 12
MODEL_PARAMETERS = 52953
MAX_MEMBER_BYTES = 3 * 1024 * 1024
MAX_ARCHIVE_UNPACKED_BYTES = 25 * 1024 * 1024


class AdmissionError(ValueError):
    """Reject evidence that cannot be proven to be the original PR61 model."""


def sha256_file(file_path: Path) -> str:
    h = hashlib.sha256()
    with file_path.open("rb") as reader:
        for block in iter(lambda: reader.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def sha256_bytes(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def read_manifest(manifest: Path):
    with manifest.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 6:
        raise AdmissionError("PR61 manifest must list exactly six controlled arm/seed runs")
    table = {}
    for row in rows:
        arm = row["arm"]
        seed = int(row["seed"])
        key = (arm, seed)
        if arm not in ARMS or seed not in SEEDS or key in table:
            raise AdmissionError("PR61 manifest has duplicate/unregistered model")
        if int(row["params"]) != MODEL_PARAMETERS:
            raise AdmissionError("wrong original PR61 parameter count")
        for stage in ("best", "last"):
            digest = row[stage + "_sha256"]
            expected_size = int(row[stage + "_bytes"])
            if len(digest) != 64 or expected_size <= 0 or expected_size > MAX_MEMBER_BYTES:
                raise AdmissionError("malformed checkpoint manifest digest/size")
        table[key] = row
    if set(table) != {(arm, seed) for arm in ARMS for seed in SEEDS}:
        raise AdmissionError("PR61 controlled map_on/off model roster incomplete")
    return table


def unpack_verified_archive(archive: Path, manifest: Path) -> tuple[dict, dict]:
    """Return 6 best.pt bytes + reproducible verification report, never extract."""
    if not archive.is_file():
        raise FileNotFoundError(
            "BLOCKED_ORIGINAL_PR61_ARCHIVE_MISSING: " + str(archive))
    if archive.stat().st_size != ARCHIVE_BYTES or sha256_file(archive) != ARCHIVE_SHA256:
        raise AdmissionError("original PR61 archive SHA256 or byte-size mismatch")
    rows = read_manifest(manifest)
    matched: dict[tuple[str, int, str], bytes] = {}
    seen_names = set()
    total_size = 0
    with tarfile.open(archive, "r:gz") as tf:
        for member in tf:
            if (member.name.startswith("/") or
                ".." in Path(member.name).parts or
                "\\" in member.name or member.name in seen_names):
                raise AdmissionError("archive contains duplicate or unsafe path")
            seen_names.add(member.name)
            if member.isdir():
                continue
            if not member.isfile():
                raise AdmissionError("original checkpoint archive has a non-regular member")
            total_size += member.size
            if (member.size > MAX_MEMBER_BYTES
                or total_size > MAX_ARCHIVE_UNPACKED_BYTES):
                raise AdmissionError("archive uncompressed bytes exceed safe bound")
            name_parts = Path(member.name).parts
            if len(name_parts) < 3 or name_parts[-1] not in ("best.pt", "last.pt"):
                continue
            stage = name_parts[-1][:-3]
            arm = name_parts[-3]
            if arm not in ARMS:
                continue
            try:
                seed = int(name_parts[-2])
            except ValueError:
                continue
            key = (arm, seed)
            if key not in rows:
                raise AdmissionError("unregistered arm/seed found in checkpoint archive")
            identity = (arm, seed, stage)
            if identity in matched:
                raise AdmissionError("ambiguous duplicate checkpoint path")
            expected = rows[key]
            if member.size != int(expected[stage + "_bytes"]):
                raise AdmissionError(f"PR61 manifest byte mismatch for {identity}")
            source = tf.extractfile(member)
            if source is None:
                raise AdmissionError("tar member has no regular file data")
            data = source.read(MAX_MEMBER_BYTES + 1)
            if len(data) != member.size:
                raise AdmissionError("tar member unexpectedly truncated")
            if sha256_bytes(data) != expected[stage + "_sha256"]:
                raise AdmissionError(f"PR61 manifest SHA mismatch for {identity}")
            matched[identity] = data
    expected_members = {
        (arm, seed, stage)
        for arm in ARMS for seed in SEEDS for stage in ("best", "last")
    }
    if set(matched) != expected_members:
        raise AdmissionError("archive lacks original PR61 complete 6 best / 6 last checkpoints")
    report = {
        "archive_filename": archive.name,
        "archive_bytes": archive.stat().st_size,
        "archive_sha256": ARCHIVE_SHA256,
        "verified_best_and_last_files": CHECKPOINT_COUNT,
        "total_unpacked_member_bytes": total_size,
        "manifest_sha256": sha256_file(manifest),
        "original_train_head": ORIGINAL_MODEL_HEAD,
        "original_validation_sha256": ORIGINAL_VAL_SHA256,
    }
    return {k: v for k, v in matched.items() if k[2] == "best"}, report


def load_original_models(archive: Path, manifest: Path,
                         original_validation: Path | None = None):
    if original_validation is not None:
        if (not original_validation.is_file()
            or sha256_file(original_validation) != ORIGINAL_VAL_SHA256):
            raise AdmissionError("wrong/missing frozen PR59 validation source")
    blobs, report = unpack_verified_archive(archive, manifest)
    models = {}
    checkpoints = []
    for arm in ARMS:
        for seed in SEEDS:
            # weights_only=True avoids the arbitrary Python pickle objects
            # accepted by the older PR62 weights_only=False reader.
            data = torch.load(
                io.BytesIO(blobs[(arm, seed, "best")]), map_location="cpu",
                weights_only=True)
            if (data.get("schema") != CKPT_SCHEMA
                or data.get("arm") != arm or data.get("seed") != seed
                or data.get("git_head_source") != ORIGINAL_MODEL_HEAD
                or data.get("hidden") != 48
                or data.get("train_map_vs_masked_same_parameter_count") is not True
                or data.get("architecture") !=
                   "real PyG SAGEConv public map / PR54 GRU-RGCN relational state transition"
                or not (0 <= data.get("epoch", -1) < 20)
                or ORIGINAL_VAL_SHA256 not in
                    data.get("source_sha256", {}).get("validation", {}).values()):
                raise AdmissionError(f"PR61 checkpoint source/architecture mismatch: {arm}/{seed}")
            for key in (
                "model_state_dict", "optimizer_state_dict", "torch_rng_state",
                "cuda_rng_state_all", "frozen_validation_metrics",
            ):
                if key not in data:
                    raise AdmissionError(f"original training/resume field missing: {key}")
            model = PublicMapTemporalRelationalWorld(
                hidden=48, history_enabled=True, map_enabled=(arm == "map_on"))
            if parameter_count(model) != MODEL_PARAMETERS:
                raise AdmissionError("PR61 source network parameter count changed")
            model.load_state_dict(data["model_state_dict"], strict=True)
            if any(not bool(torch.isfinite(p).all()) for p in model.parameters()):
                raise AdmissionError("original trained checkpoint contains non-finite weights")
            models[(arm, seed)] = model.cpu().eval()
            checkpoints.append({
                "arm": arm, "seed": seed,
                "epoch_1_based": data["epoch"] + 1,
                "best_pt_sha256": sha256_bytes(blobs[(arm, seed, "best")]),
                "hidden": 48, "parameters": parameter_count(model),
                "original_val_metrics": data["frozen_validation_metrics"],
            })
    report.update({
        "weights_loaded_from_original_pr61": True,
        "weights_only_safe_loader": True,
        "original_gpu_model_training_repeated": False,
        "models": checkpoints,
    })
    return models, report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--archive", type=Path, required=True)
    p.add_argument("--manifest", type=Path, default=Path(
        "examples/evidence/pr60-gpu-training-20261112/checkpoint_manifest.csv"))
    p.add_argument("--original-validation", type=Path)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    if args.out.exists():
        raise FileExistsError("do not overwrite an earlier PR61 verification")
    models, report = load_original_models(
        args.archive, args.manifest, args.original_validation)
    assert len(models) == 6
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print("PR68_TRUSTED_ORIGINAL_PR61_CHECKPOINT_LOADED_CPU",
          json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
