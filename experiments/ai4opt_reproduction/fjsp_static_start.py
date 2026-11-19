"""Calibrate a pre-solve HGNN choice and assess it on fresh FJSP instances."""

from __future__ import annotations

import argparse
from hashlib import sha1, sha256
from itertools import product
import json
from pathlib import Path
from time import perf_counter

from fjsp_adaptive_start import BUDGET, SEEDS, hits, measure, metrics, solve
from fjsp_end_to_end import emit, worker
from fjsp_warm_start import (
    AUTHOR_COMMIT, BENCHMARK_COMMIT, load_solver_model, read_case, verify_checkout, verify_instances,
)


FAMILIES = ("edata", "rdata", "vdata")
ARMS = ("cp_sat", "hgnn_hint", "adaptive", "static")
REFERENCE_SECONDS = 60
REFERENCE_SEED = 11


def features(jobs):
    operations = [eligible for job in jobs for eligible in job]
    return len(operations), sum(len(eligible) for eligible in operations if len(eligible) > 1)


def use_hgnn(values, rule):
    return rule is not None and all(value >= cutoff for value, cutoff in zip(values, rule, strict=True))


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def calibrate(paths, prior, dev_rows):
    sizes = {case: features(read_case(paths[case])[1]) for case, split in prior["split"].items() if split == "dev"}
    arms = ARMS[:2]
    by_key = {(row["case"], row["seed"], row["arm"]): row for row in dev_rows}
    if len(sizes) != 20 or len(dev_rows) != len(by_key) or set(by_key) != {
        (case, seed, arm) for case in sizes for seed in SEEDS for arm in arms
    }:
        raise ValueError("Expected a complete development-only paired calibration dataset")
    for row in dev_rows:
        if row["split"] != "dev" or row["target_makespan"] != prior["targets"][row["case"]]:
            raise ValueError("Calibration rows disagree with the frozen development protocol")

    cutoffs = []
    for column in zip(*sizes.values(), strict=True):
        values = sorted(column)
        cutoffs.append(sorted({0, *(values[(len(values) - 1) * i // 4] for i in (1, 2, 3))}))

    direct_hits = {seconds: hits([row for row in dev_rows if row["arm"] == "cp_sat"], seconds) for seconds in (1, 10)}
    candidates = []
    for rule in (None, *product(*cutoffs)):
        decisions = {case: "hgnn_hint" if use_hgnn(size, rule) else "cp_sat" for case, size in sizes.items()}
        chosen = [by_key[case, seed, arm] for case, arm in decisions.items() for seed in SEEDS]
        candidates.append({"rule": rule, "hits_at_1": hits(chosen, 1), "hits_at_10": hits(chosen, 10),
                           "hgnn_calls": sum(arm == "hgnn_hint" for arm in decisions.values()) * len(SEEDS)})

    def score(record):
        rule = record["rule"] or (0, 0)
        return (record["hits_at_10"], -record["hgnn_calls"], record["hits_at_1"],
                -sum(cutoff > 0 for cutoff in rule), *rule)

    best = max((r for r in candidates if r["hits_at_1"] >= direct_hits[1]), key=score)
    return {"chosen": best, "cp_sat_hits": direct_hits, "development_features": sizes, "candidates": candidates}


def public_instances(root, source_tree, prior_paths):
    blobs = {}
    for line in source_tree.read_text().splitlines():
        metadata, path = line.split("\t", 1)
        _, kind, digest = metadata.split()
        if kind == "blob":
            blobs[path] = digest

    def read_public(path):
        content = path.read_bytes()
        digest = sha1(f"blob {len(content)}\0".encode() + content).hexdigest()
        if digest != blobs[path.relative_to(root).as_posix()]:
            raise ValueError(f"Public instance differs from the pinned Git blob: {path}")
        return sha256(content).hexdigest()

    used = {path.stem for path in prior_paths.values()}
    by_family = {}
    for family in FAMILIES:
        dirs = root / "data/fjsp/hurink" / family
        by_family[family] = {path.stem: path for path in dirs.glob("*.fjs") if path.stem not in used}
    common = set.intersection(*(set(paths) for paths in by_family.values()))
    size = {}
    excluded = {}
    verified = {}
    for stem in sorted(common):
        variants = {}
        zero_duration_variants = []
        for family in FAMILIES:
            path = by_family[family][stem]
            digest = read_public(path)
            _, jobs = read_case(path, allow_zero_duration=True)
            if any(0 in eligible.values() for job in jobs for eligible in job):
                zero_duration_variants.append(family)
            variants[family] = {"sha256": digest, "features": features(jobs)}
        if zero_duration_variants:
            excluded[stem] = zero_duration_variants
            continue
        op_counts = {row["features"][0] for row in variants.values()}
        (size[stem],) = op_counts
        verified[stem] = variants
    ordered = sorted(size, key=lambda stem: (size[stem], stem))

    def rank(stem):
        return sha256(f"wm-fjsp-static-v1:{stem}".encode()).digest()

    selected = {}
    metadata = {}
    for index, band in enumerate(("small", "middle", "large")):
        stratum = ordered[index * len(ordered) // 3:(index + 1) * len(ordered) // 3]
        stems = sorted(stratum, key=rank)[:6]
        for stem, family in zip(stems, FAMILIES * 2, strict=True):
            path = by_family[family][stem]
            case = f"{family}/{path.name}"
            selected[case] = path
            metadata[case] = {**verified[stem][family], "size_band": band}
    return selected, metadata, excluded


def freeze(path, record, resume):
    encoded = json.dumps(record, indent=2) + "\n"
    if resume and path.exists():
        if json.loads(path.read_text()) != json.loads(encoded):
            raise ValueError(f"Resume input disagrees with saved {path.name}")
    else:
        with path.open("x") as stream:
            stream.write(encoded)


def run(args):
    import ortools

    verify_checkout(args.upstream, AUTHOR_COMMIT)
    digest = sha256(args.checkpoint.read_bytes()).hexdigest()
    if digest != args.checkpoint_sha256.lower():
        raise ValueError("Unexpected checkpoint digest")
    inputs, hashes = verify_instances(args.benchmarks, args.manifest)
    previous = json.loads(args.prior_protocol.read_text())
    if (previous["benchmark_sha256"] != hashes or previous["checkpoint_sha256"] != digest
            or previous["budget_seconds"] != BUDGET or previous["solver_seed"] != list(SEEDS)
            or previous["solver_workers"] != args.solver_workers or previous["ortools"] != ortools.__version__):
        raise ValueError("Calibration and fresh evaluation use a different setup")
    prior_paths = {f"{family}/{path.name}": path for family, path in inputs}
    calibration = calibrate(prior_paths, previous, read_rows(args.development))
    paths, metadata, excluded = public_instances(args.benchmarks, args.source_tree, prior_paths)
    probe_threshold = json.loads(args.prior_policy.read_text())["gap_threshold"]
    args.output.mkdir(parents=True, exist_ok=args.resume)
    freeze(args.output / "protocol.json", {
        "benchmark_commit": BENCHMARK_COMMIT, "checkpoint_sha256": digest, "test_cases": metadata,
        "exclude_stems": sorted({p.stem for p in prior_paths.values()}), "solver_seeds": SEEDS,
        "solver_workers": args.solver_workers, "ortools": ortools.__version__, "budget_seconds": BUDGET,
        "reference_search_seconds": REFERENCE_SECONDS, "reference_seed": REFERENCE_SEED,
        "development_sha256": sha256(args.development.read_bytes()).hexdigest(), "probe_threshold": probe_threshold,
        "excluded_zero_duration_variants": excluded,
    }, args.resume)
    freeze(args.output / "rule.json", calibration, args.resume)
    rule = calibration["chosen"]["rule"]
    emit({"phase": "rule_frozen", **calibration["chosen"], "fresh_cases": len(paths)})

    references = args.output / "reference.jsonl"
    targets = read_rows(references) if args.resume else []
    cases = tuple(paths)
    if [row["case"] for row in targets] != list(cases[:len(targets)]):
        raise ValueError("Unexpected saved reference order")
    with references.open("a" if args.resume else "x") as stream:
        for case in cases[len(targets):]:
            data = load_solver_model(paths[case], args.benchmarks)
            started = perf_counter()
            answer, status, bound, disagreement = solve(data, 0, started + REFERENCE_SECONDS, REFERENCE_SEED, args.solver_workers)
            if answer is None:
                raise RuntimeError(f"No feasible reference schedule for {case}")
            record = {"case": case, "target": answer[0], "bound": bound, "status": status,
                      "reported_objective_if_different": disagreement, "seconds": perf_counter() - started}
            targets.append(record)
            print(json.dumps(record), file=stream, flush=True)
            emit({"phase": "reference", "count": len(targets), "case": case, "target": answer[0], "status": status})
    by_case = {row["case"]: row["target"] for row in targets}

    def route(jobs):
        return "hgnn_hint" if use_hgnn(features(jobs), rule) else None

    planned = []
    for i, seed in enumerate(SEEDS):
        for j, case in enumerate(cases):
            offset = (i + j) % len(ARMS)
            planned.extend((seed, case, arm) for arm in ARMS[offset:] + ARMS[:offset])
    log = args.output / "results.jsonl"
    rows = read_rows(log) if args.resume else []
    if [(row["seed"], row["case"], row["arm"]) for row in rows] != planned[:len(rows)]:
        raise ValueError("Unexpected saved evaluation order")
    with worker(args, "hgnn_hint") as (generate, info), log.open("a" if args.resume else "x") as stream:
        generators = {"hgnn_hint": generate}
        for seed, case, arm in planned[len(rows):]:
            outcome = measure(paths[case], args.benchmarks, generators, arm, by_case[case], seed, args.solver_workers,
                              threshold=probe_threshold if arm == "adaptive" else None,
                              route=route if arm == "static" else None)
            record = {"case": case, "seed": seed, "arm": arm, **outcome}
            rows.append(record)
            print(json.dumps(record), file=stream, flush=True)
            emit({"phase": "evaluation", "count": len(rows), "case": case, "seed": seed,
                  "arm": arm, "hit": outcome["hit"], "seconds": round(outcome["end_to_end_seconds"], 3)})
    summary = {"overall": {arm: metrics([r for r in rows if r["arm"] == arm]) for arm in ARMS},
               "by_seed": {seed: {arm: metrics([r for r in rows if r["seed"] == seed and r["arm"] == arm])
                                 for arm in ARMS} for seed in SEEDS}, "initial_worker": info}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("upstream", "benchmarks", "manifest", "prior-protocol", "prior-policy", "development", "source-tree",
                 "checkpoint", "output", "inference-python"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--solver-workers", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.set_defaults(seed=1810, inference_device="cpu")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
