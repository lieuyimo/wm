"""PR63 read-only risk-set and event-time observability audit.

Fits immutable device-transition hazard baselines on raw TRAIN only, then
evaluates on the untouched PR59 validation split. Also evaluates the existing
PR61 checkpoints on the complete validation risk sets (CPU by default; no
training, no optimizer, no BLIND/TEST access).

The primary risk sets are:
  * all currently-UP station ticks  -> next available / onset
  * all currently-DOWN station ticks -> next available / recovery

The realized-transition phase groups are reported descriptively. They are not
used to fit any parameter and are not a calibration test.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

import torch
from torch_geometric.loader import DataLoader

from experiments.world_model_integration.audit_pr62.pr62_device_events import read_ckpts
from experiments.world_model_integration.sealed_pr59.pr59_gpu_train import load_actual_source

SPLITS = ("train", "validation")
SCHEMAS = {
    "pr59-sha-sealed-natural-public-world-map-family-split-v1",
    "pr60-additional-real-public-map-train-source-v1",
}
PHASES = ("healthy_hold", "fault_onset", "continuing_down", "recovery")
BASELINES = ("constant", "age_h4", "age_exact", "abs_tick")
CLIP = 1e-6
BOOTSTRAP_B = 2000
BOOTSTRAP_SEED = 20261113

# The 16 public node features are defined by pr51_train.node_features.  None of
# them is the absolute public clock; the history tensor is 4 snapshots long.
NODE_FEATURE_DIM = 16
HISTORY_WINDOW = 4
DEVICE_AVAILABLE_INDEX = 6


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def station_stage(current: int, target: int) -> str:
    if current == 1 and target == 1:
        return "healthy_hold"
    if current == 1 and target == 0:
        return "fault_onset"
    if current == 0 and target == 0:
        return "continuing_down"
    return "recovery"


def age_h4_bucket(age: int) -> str:
    return str(min(HISTORY_WINDOW, max(1, int(age))))


def age_exact_bucket(age: int) -> str:
    age = int(age)
    if age <= 7:
        return str(age)
    if age <= 12:
        return "8-12"
    if age == 13:
        return "13"
    if age <= 20:
        return "14-20"
    return "21+"


def tick_bucket(tick: int) -> str:
    tick = int(tick)
    if tick <= 5:
        return "0-5"
    if tick == 6:
        return "6"
    if tick <= 11:
        return "7-11"
    if tick == 12:
        return "12"
    if tick <= 20:
        return "13-20"
    return "21+"


def _read_source(path: Path, expected: str) -> dict:
    if expected not in SPLITS:
        raise ValueError("only TRAIN/VALIDATION are readable")
    src = json.loads(Path(path).read_text())
    if src.get("split") != expected:
        raise AssertionError(f"source split mismatch for {path}")
    if src.get("BLIND_TEST_RUN") is not False:
        raise AssertionError("BLIND source rejected")
    if src.get("schema") not in SCHEMAS:
        raise ValueError(f"unsupported source schema: {src.get('schema')}")
    if not isinstance(src.get("scenes"), list) or not src["scenes"]:
        raise AssertionError("empty source")
    return src


def _station_ages(history: list, visible: list, index: int, current: int) -> int:
    """Number of trailing visible snapshots in the same public state, capped at 4."""
    row = history[index]
    vis = visible[index]
    if len(row) != HISTORY_WINDOW or len(vis) != HISTORY_WINDOW:
        raise AssertionError("public history dimension differs from model input")
    age = 1
    if int(round(row[-1][DEVICE_AVAILABLE_INDEX])) != current:
        raise AssertionError("current history state differs from current graph")
    for k in range(HISTORY_WINDOW - 2, -1, -1):
        if not vis[k] or int(round(row[k][DEVICE_AVAILABLE_INDEX])) != current:
            break
        age += 1
    return age


def extract_records(src: dict, split: str):
    """Extract complete public device risk sets.  Labels never enter features."""
    records = []
    scene_meta = []
    for scene in src["scenes"]:
        case = scene.get("case", {})
        if case.get("split") != split:
            raise AssertionError("scene split mismatch")
        scene_id = scene["id"]
        parts = scene_id.split("_")
        if len(parts) < 3 or not parts[2].isdigit():
            raise AssertionError("unregistered scene id/map id")
        map_id = parts[2]
        family = case.get("map_family")
        regime = case.get("regime")
        rows = scene.get("raw_actual_public_world_transitions") or []
        if not rows:
            raise AssertionError("empty scene")
        station_indices = None
        exact_ages = {}
        scene_counts = Counter()
        for row in rows:
            inp = row["input"]
            label = row["target"]
            if label.get("next_public_tick") != inp.get("public_tick") + 1:
                raise AssertionError("non-consecutive public transition")
            nodes = inp["graph"]["nodes"]
            target = label["node_target"]
            mask = label["node_mask"]
            history = inp["history"]
            visible = inp["history_visible_mask"]
            if not (len(nodes) == len(target) == len(mask) == len(history) == len(visible)):
                raise AssertionError("node/target/history misalignment")
            current_stations = [i for i, n in enumerate(nodes) if n["kind"] == "station"]
            if station_indices is None:
                station_indices = current_stations
                if not station_indices:
                    raise AssertionError("no station nodes")
            elif current_stations != station_indices:
                raise AssertionError("station node order changed; age alignment unsafe")
            for i in station_indices:
                sid = nodes[i]["id"]
                current = 1 if nodes[i].get("device_available") else 0
                if sid not in exact_ages:
                    exact_age = 1
                else:
                    prev_state, prev_age = exact_ages[sid]
                    exact_age = prev_age + 1 if prev_state == current else 1
                exact_ages[sid] = (current, exact_age)
                if not mask[i][0]:
                    continue
                target_i = target[i][0]
                if target_i not in (0.0, 1.0):
                    raise AssertionError("non-binary next device state")
                y = int(target_i)
                age_h4 = _station_ages(history, visible, i, current)
                risk = "up" if current == 1 else "down"
                phase = station_stage(current, y)
                records.append({
                    "split": split, "scene": scene_id, "map": map_id,
                    "family": family, "regime": regime,
                    "tick": int(inp["public_tick"]),
                    "next_tick": int(label["next_public_tick"]),
                    "station": sid, "current": current, "target": y,
                    "risk_set": risk, "phase": phase,
                    "age_h4": age_h4, "age_exact": exact_age,
                })
                scene_counts[phase] += 1
        scene_meta.append({
            "scene": scene_id, "map": map_id, "family": family, "regime": regime,
            "transitions": len(rows), "phase_counts": dict(scene_counts),
        })
    return records, scene_meta


def load_records(paths, split: str):
    if split not in SPLITS:
        raise ValueError("only train/validation sources are allowed")
    records = []
    sources = {}
    scenes = []
    for path in paths:
        path = Path(path)
        src = _read_source(path, split)
        source_sha = sha256(path)
        recs, meta = extract_records(src, split)
        records.extend(recs)
        sources[str(path.resolve())] = source_sha
        scenes.extend(meta)
    if not records:
        raise AssertionError("no risk-set records")
    return records, {
        "split": split,
        "scenes": len(scenes),
        "transitions": sum(x["transitions"] for x in scenes),
        "sources": sources,
        "scene_meta": scenes,
    }


def _event(risk: str, target: int) -> bool:
    return target == 0 if risk == "up" else target == 1


def fit_hazards(records, bucket_fn=None, alpha: float = 1.0):
    if bucket_fn is None:
        out = {}
        for risk in ("up", "down"):
            risk_records = [r for r in records if r["risk_set"] == risk]
            n = len(risk_records)
            events = sum(_event(risk, r["target"]) for r in risk_records)
            if n == 0:
                raise AssertionError(f"empty {risk} risk set")
            out[risk] = {
                "n": n, "events": events,
                "hazard": events / n if alpha == 0 else (events + alpha) / (n + 2 * alpha),
            }
        return out
    out = {}
    for risk in ("up", "down"):
        buckets = {}
        for r in records:
            if r["risk_set"] != risk:
                continue
            b = bucket_fn(r)
            bucket = buckets.setdefault(b, {"n": 0, "events": 0})
            bucket["n"] += 1
            bucket["events"] += int(_event(risk, r["target"]))
        for b, v in buckets.items():
            v["hazard"] = (v["events"] + alpha) / (v["n"] + 2 * alpha)
        out[risk] = buckets
    return out


def fit_baselines(train_records):
    params = {
        "schema": "pr63-train-fitted-device-hazards-v1",
        "fit_split": "train",
        "alpha_age_buckets": 1.0,
        "constant": fit_hazards(train_records, None, alpha=0.0),
        "age_h4": fit_hazards(train_records, lambda r: age_h4_bucket(r["age_h4"])),
        "age_exact": fit_hazards(train_records, lambda r: age_exact_bucket(r["age_exact"])),
        "abs_tick": fit_hazards(train_records, lambda r: tick_bucket(r["tick"])),
        "no_validation_labels_used": True,
    }
    return params


def baseline_probability(record, method: str, params: dict) -> float:
    risk = record["risk_set"]
    if method == "constant":
        hazard = params["constant"][risk]["hazard"]
    elif method == "age_h4":
        hazard = params["age_h4"][risk][age_h4_bucket(record["age_h4"])]["hazard"]
    elif method == "age_exact":
        hazard = params["age_exact"][risk][age_exact_bucket(record["age_exact"])]["hazard"]
    elif method == "abs_tick":
        hazard = params["abs_tick"][risk][tick_bucket(record["tick"])]["hazard"]
    else:
        raise ValueError(method)
    return 1.0 - hazard if risk == "up" else hazard


def _safe_logloss(p: float, y: int) -> float:
    p = min(1.0 - CLIP, max(CLIP, p))
    return -(y * math.log(p) + (1 - y) * math.log(1.0 - p))


def _mean(vals):
    return sum(vals) / len(vals) if vals else None


def _reliability(rows):
    edges = [0.0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0]
    buckets = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sub = [(p, y) for p, y in rows if lo <= p < hi or (hi == 1.0 and p == 1.0)]
        if sub:
            buckets.append({
                "low": lo, "high": hi, "n": len(sub),
                "pred_mean": _mean([p for p, _ in sub]),
                "empirical": _mean([float(y) for _, y in sub]),
            })
    return buckets


def score_rows(rows):
    """rows: list of (p, y, record)."""
    def pack(sub):
        n = len(sub)
        if not n:
            return {"n": 0, "available": False}
        brier = _mean([(p - y) ** 2 for p, y, _ in sub])
        logloss = _mean([_safe_logloss(p, y) for p, y, _ in sub])
        return {
            "n": n, "available": True, "brier": brier, "logloss": logloss,
            "pred_mean": _mean([p for p, _, _ in sub]),
            "empirical_next_available": _mean([float(y) for _, y, _ in sub]),
        }
    out = {
        "overall": pack(rows),
        "risk_set": {
            risk: pack([x for x in rows if x[2]["risk_set"] == risk])
            for risk in ("up", "down")
        },
        "phase": {
            phase: pack([x for x in rows if x[2]["phase"] == phase])
            for phase in PHASES
        },
        "by_map": {
            m: pack([x for x in rows if x[2]["map"] == m])
            for m in sorted({x[2]["map"] for x in rows})
        },
        "by_regime": {
            r: pack([x for x in rows if x[2]["regime"] == r])
            for r in sorted({x[2]["regime"] for x in rows})
        },
        "per_scene": {},
        "reliability": _reliability([(p, y) for p, y, _ in rows]),
        "event_ranking": {
            "n_events": sum(1 for _, y, r in rows if _event(r["risk_set"], y)),
            "reported": False,
            "reason": "16 validation onset/recovery events; AP/AUC/lift not statistically reliable",
        },
    }
    for sid in sorted({x[2]["scene"] for x in rows}):
        out["per_scene"][sid] = pack([x for x in rows if x[2]["scene"] == sid])
    return out


def evaluate_baselines(params, records):
    out = {}
    for method in BASELINES:
        rows = []
        for r in records:
            p = baseline_probability(r, method, params)
            rows.append((p, r["target"], r))
        out[method] = score_rows(rows)
    return out


def _map_id(scene_id: str) -> str:
    return scene_id.split("_")[2]


def evaluate_models(validation_path: Path, checkpoint_root: Path, manifest: Path,
                    device: str, batch_size: int):
    episodes, sources = load_actual_source([validation_path], "validation")
    if len(episodes) != 12:
        raise AssertionError("not the 12 frozen validation scenes")
    validation_sha = sha256(validation_path)
    models = read_ckpts(checkpoint_root, manifest, validation_sha, device)
    if len(models) != 6:
        raise AssertionError("six original checkpoints required")
    reports = {}
    for (arm, seed), (model, ckpt_path) in sorted(models.items()):
        rows = []
        model.eval()
        with torch.no_grad():
            for ep in episodes:
                for batch in DataLoader(ep["graphs"], batch_size=batch_size, shuffle=False):
                    b = batch.to(device)
                    pred, _ = model(b)
                    station = (b.x[:, 2] > 0.5) & (b.target_mask[:, 0] > 0.5)
                    for index in station.nonzero(as_tuple=False).flatten().tolist():
                        p = float(pred[index, 0].item())
                        y = int(b.node_target[index, 0].item())
                        x = int(b.x[index, DEVICE_AVAILABLE_INDEX].item())
                        if not (0.0 <= p <= 1.0 and y in (0, 1) and x in (0, 1)):
                            raise AssertionError("invalid model prediction or label")
                        rows.append((p, y, {
                            "scene": ep["id"], "map": _map_id(ep["id"]),
                            "family": ep["family"], "regime": ep["regime"],
                            "risk_set": "up" if x == 1 else "down",
                            "phase": station_stage(x, y),
                        }))
        if len(rows) != 7585:
            raise AssertionError(f"unexpected validation station-tick count: {len(rows)}")
        reports[f"model_{arm}_{seed}"] = {
            "arm": arm, "seed": seed,
            "checkpoint_sha256": sha256(ckpt_path),
            **score_rows(rows),
        }
    # Mean across the three training seeds for each arm.  This is a descriptive
    # training-seed aggregate; scenes remain the independent evaluation unit.
    for arm in ("map_off", "map_on"):
        keys = [k for k, v in reports.items() if v["arm"] == arm]
        if len(keys) != 3:
            raise AssertionError("three seeds per arm required")
        per_scene = {}
        for sid in sorted(reports[keys[0]]["per_scene"]):
            per_scene[sid] = {
                "brier": _mean([reports[k]["per_scene"][sid]["brier"] for k in keys]),
                "logloss": _mean([reports[k]["per_scene"][sid]["logloss"] for k in keys]),
                "n": reports[keys[0]]["per_scene"][sid]["n"],
            }
        reports[f"model_{arm}_mean"] = {
            "arm": arm, "seed": None,
            "overall": {
                "brier": _mean([reports[k]["overall"]["brier"] for k in keys]),
                "logloss": _mean([reports[k]["overall"]["logloss"] for k in keys]),
                "n": reports[keys[0]]["overall"]["n"],
            },
            "risk_set": {
                risk: {
                    "brier": _mean([reports[k]["risk_set"][risk]["brier"] for k in keys]),
                    "logloss": _mean([reports[k]["risk_set"][risk]["logloss"] for k in keys]),
                    "n": reports[keys[0]]["risk_set"][risk]["n"],
                } for risk in ("up", "down")
            },
            "phase": {
                phase: {
                    "brier": _mean([reports[k]["phase"][phase]["brier"] for k in keys]),
                    "logloss": _mean([reports[k]["phase"][phase]["logloss"] for k in keys]),
                    "n": reports[keys[0]]["phase"][phase]["n"],
                } for phase in PHASES
            },
            "per_scene": per_scene,
        }
    return reports, {"validation": str(validation_path.resolve()),
                     "validation_sha256": validation_sha,
                     "checkpoint_manifest": str(manifest.resolve()),
                     "checkpoint_manifest_sha256": sha256(manifest),
                     "device": device, "new_training": False}


def _bootstrap(values_by_scene, b=BOOTSTRAP_B, seed=BOOTSTRAP_SEED):
    ids = sorted(values_by_scene)
    if not ids:
        return None
    rng = random.Random(seed)
    means = []
    for _ in range(b):
        sample = [ids[rng.randrange(len(ids))] for _ in range(len(ids))]
        means.append(_mean([values_by_scene[s] for s in sample]))
    means.sort()
    return {
        "n_scenes": len(ids), "bootstrap": b,
        "mean": _mean([values_by_scene[s] for s in ids]),
        "ci95_low": means[int(0.025 * (b - 1))],
        "ci95_high": means[int(0.975 * (b - 1))],
    }


def _method_scene_values(metrics):
    return {sid: v["brier"] for sid, v in metrics["per_scene"].items()}


def uncertainty(baselines, models):
    methods = {}
    for name, m in baselines.items():
        methods[name] = _method_scene_values(m)
    for name, m in models.items():
        if name.endswith("_mean"):
            methods[name] = _method_scene_values(m)
    base = methods.get("constant")
    scene_out = {}
    paired = {}
    for name, vals in methods.items():
        scene_out[name] = _bootstrap(vals)
        if base is not None:
            diff = {sid: vals[sid] - base[sid] for sid in vals}
            paired[name] = _bootstrap(diff)
    # Map-block uncertainty: bootstrap six scenes within each map.
    map_out = {}
    for name, vals in methods.items():
        map_out[name] = {}
        for map_id in ("0", "1"):
            sub = {sid: val for sid, val in vals.items()
                   if sid.split("_")[2] == map_id}
            map_out[name][map_id] = _bootstrap(sub, b=1000)
    return {"scene_bootstrap": scene_out, "paired_minus_constant": paired,
            "map_block": map_out, "independent_unit": "12 source scenes"}


def _event_summary(records):
    out = {
        "risk_set": {},
        "phase": Counter(),
        "onset_next_tick": Counter(),
        "recovery_next_tick": Counter(),
        "onset_current_tick": Counter(),
        "recovery_current_tick": Counter(),
        "onset_age_exact": Counter(),
        "recovery_age_exact": Counter(),
        "onset_age_h4": Counter(),
        "recovery_age_h4": Counter(),
        "by_map": defaultdict(lambda: Counter()),
        "by_regime": defaultdict(lambda: Counter()),
    }
    for risk in ("up", "down"):
        sub = [r for r in records if r["risk_set"] == risk]
        ev = sum(_event(risk, r["target"]) for r in sub)
        out["risk_set"][risk] = {"n": len(sub), "events": ev,
                                 "hazard": ev / len(sub) if sub else None}
    for r in records:
        phase = r["phase"]
        out["phase"][phase] += 1
        if phase == "fault_onset":
            out["onset_next_tick"][r["next_tick"]] += 1
            out["onset_current_tick"][r["tick"]] += 1
            out["onset_age_exact"][r["age_exact"]] += 1
            out["onset_age_h4"][r["age_h4"]] += 1
        if phase == "recovery":
            out["recovery_next_tick"][r["next_tick"]] += 1
            out["recovery_current_tick"][r["tick"]] += 1
            out["recovery_age_exact"][r["age_exact"]] += 1
            out["recovery_age_h4"][r["age_h4"]] += 1
        out["by_map"][r["map"]][phase] += 1
        out["by_regime"][r["regime"]][phase] += 1
    for key in ("phase", "onset_next_tick", "recovery_next_tick",
                "onset_current_tick", "recovery_current_tick",
                "onset_age_exact", "recovery_age_exact", "onset_age_h4",
                "recovery_age_h4"):
        out[key] = dict(sorted(out[key].items(), key=lambda kv: str(kv[0])))
    out["by_map"] = {m: dict(v) for m, v in sorted(out["by_map"].items())}
    out["by_regime"] = {r: dict(v) for r, v in sorted(out["by_regime"].items())}
    return out


def observability(train_records, val_records, train_meta, val_meta, pr48_protocol: Path):
    pr48 = json.loads(pr48_protocol.read_text())
    meta = pr48.get("predefined_scenario_mechanisms", {})
    return {
        "schema": "pr63-event-time-observability-v1",
        "train": _event_summary(train_records),
        "validation": _event_summary(val_records),
        "train_meta": {k: v for k, v in train_meta.items() if k != "scene_meta"},
        "validation_meta": {k: v for k, v in val_meta.items() if k != "scene_meta"},
        "pr48_protocol": {
            "schema": pr48.get("protocol"),
            "outage_start_tick": meta.get("outage_start_tick"),
            "secondary_outage_start_tick": meta.get("secondary_outage_start_tick"),
            "source": str(pr48_protocol.resolve()),
            "sha256": sha256(pr48_protocol),
        },
        "model_input": {
            "node_feature_dim": NODE_FEATURE_DIM,
            "history_window": HISTORY_WINDOW,
            "absolute_public_tick_in_input_json": True,
            "absolute_public_tick_used_by_model": False,
            "full_age_used_by_model": False,
            "history_age_cap": HISTORY_WINDOW,
            "note": "pr51 node_features has no clock; pr54 history is t-3..t; PR48 outage starts are fixed at ticks 7/13",
        },
        "BLIND_TEST_read": False,
        "BLIND_TEST_generated": False,
    }


def phase_csv_rows(baselines, models):
    rows = []
    for method, metrics in baselines.items():
        scopes = [("overall", metrics["overall"]), ("up", metrics["risk_set"]["up"]),
                  ("down", metrics["risk_set"]["down"])]
        scopes += [(p, metrics["phase"][p]) for p in PHASES]
        for scope, v in scopes:
            if v.get("n"):
                rows.append([method, scope, v["n"], v["brier"], v["logloss"],
                             v.get("pred_mean"), v.get("empirical_next_available")])
    for method, metrics in models.items():
        scopes = [("overall", metrics["overall"]), ("up", metrics["risk_set"]["up"]),
                  ("down", metrics["risk_set"]["down"])]
        scopes += [(p, metrics["phase"][p]) for p in PHASES]
        for scope, v in scopes:
            if v.get("n"):
                rows.append([method, scope, v["n"], v["brier"], v["logloss"],
                             v.get("pred_mean"), v.get("empirical_next_available")])
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train-source", type=Path, action="append", required=True)
    p.add_argument("--validation", type=Path, required=True)
    p.add_argument("--checkpoint-root", type=Path, default=None)
    p.add_argument("--checkpoint-manifest", type=Path, default=None)
    p.add_argument("--pr48-protocol", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--cpu-threads", type=int, default=2)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite risk-set evidence")
    if (args.checkpoint_root is None) != (args.checkpoint_manifest is None):
        raise ValueError("checkpoint root and manifest must be supplied together")
    torch.set_num_threads(args.cpu_threads)

    train_records, train_meta = load_records(args.train_source, "train")
    val_records, val_meta = load_records([args.validation], "validation")
    if train_meta["scenes"] != 60 or train_meta["transitions"] != 9046:
        raise AssertionError("unexpected frozen TRAIN source")
    if val_meta["scenes"] != 12 or val_meta["transitions"] != 1517:
        raise AssertionError("unexpected frozen validation source")
    if sum(r["phase"] == "fault_onset" for r in val_records) != 16:
        raise AssertionError("unexpected validation onset cardinality")
    if sum(r["phase"] == "recovery" for r in val_records) != 16:
        raise AssertionError("unexpected validation recovery cardinality")

    params = fit_baselines(train_records)
    params["train_meta"] = {k: v for k, v in train_meta.items() if k != "scene_meta"}
    params["train_meta"]["source_sha256"] = train_meta["sources"]
    baselines = evaluate_baselines(params, val_records)

    model_reports = {}
    model_meta = {}
    if args.checkpoint_root is not None:
        model_reports, model_meta = evaluate_models(
            args.validation, args.checkpoint_root, args.checkpoint_manifest,
            args.device, args.batch_size)

    result = {
        "schema": "pr63-risk-set-observability-v1",
        "baseline_params": params,
        "baselines": baselines,
        "model_meta": model_meta,
        "models": model_reports,
        "uncertainty": uncertainty(baselines, model_reports),
        "validation_meta": {k: v for k, v in val_meta.items() if k != "scene_meta"},
        "observability": observability(train_records, val_records, train_meta, val_meta,
                                       args.pr48_protocol),
        "limitations": [
            "12 independent validation scenes; 16 onset and 16 recovery events",
            "station ticks are correlated; no tick-level independent inference",
            "phase-conditioned metrics are descriptive, not calibration",
            "absolute-tick and exact-age diagnostics are not model input features",
            "no H20/autoregressive/online/decision claim",
        ],
        "BLIND_TEST_read": False,
        "BLIND_TEST_generated": False,
        "new_training_performed": False,
        "source_model_parameter_updates": 0,
    }

    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "baseline_params.json").write_text(
        json.dumps(params, ensure_ascii=False, indent=2) + "\n")
    (args.output / "observability.json").write_text(
        json.dumps(result["observability"], ensure_ascii=False, indent=2) + "\n")
    (args.output / "risk_set_metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    with (args.output / "phase_metrics.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "scope", "n", "brier", "logloss", "pred_mean",
                    "empirical_next_available"])
        w.writerows(phase_csv_rows(baselines, model_reports))
    files = ["baseline_params.json", "observability.json",
             "risk_set_metrics.json", "phase_metrics.csv"]
    with (args.output / "SHA256SUMS").open("w") as f:
        for name in files:
            f.write(f"{sha256(args.output / name)}  {name}\n")
    print("PR63_RISK_SET_WRITTEN", str(args.output), flush=True)
    print(json.dumps({
        "train_scenes": train_meta["scenes"],
        "validation_scenes": val_meta["scenes"],
        "baseline_brier": {k: v["overall"]["brier"] for k, v in baselines.items()},
        "model_brier": {k: v["overall"]["brier"] for k, v in model_reports.items()
                        if k.endswith("_mean")},
        "event_counts": {"onset": 16, "recovery": 16},
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
