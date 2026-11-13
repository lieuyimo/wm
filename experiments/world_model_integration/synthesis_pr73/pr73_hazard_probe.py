"""Diagnostic TRAIN-only standard discrete-time event hazard controls.

Reuse the PR63 existing count/Laplace smoothing idea; this is an explicitly
simple diagnostic baseline, NOT an original algorithm, new neural training or
a learned world model. Thresholds/denominators never use DEV labels.
"""
from collections import defaultdict

PHASES = ("to_pickup", "to_dropoff")
ALPHA = 1.0


def _count(rows, key):
    groups = defaultdict(lambda: [0, 0])
    for r in rows:
        k = key(r)
        groups[k][0] += 1
        groups[k][1] += int(r["event"])
    return {k: tuple(v) for k, v in groups.items()}


def _hazard(pair):
    n, events = pair
    return (events + ALPHA) / (n + 2 * ALPHA)


def fit_train_only(train_rows):
    if not train_rows or any(r["phase"] not in PHASES for r in train_rows):
        raise ValueError("TRAIN active-service risk rows required")
    phase = _count(train_rows, lambda r: r["phase"])
    if set(phase) != set(PHASES):
        raise ValueError("each phase must exist in TRAIN")
    joint = _count(train_rows, lambda r: (r["station"], r["phase"], r["active_elapsed"]))
    return {
        "phase": phase, "joint": joint,
        "alpha": ALPHA, "train_only": True,
    }


def probabilities(control, row):
    phase = row["phase"]
    if phase not in control["phase"]:
        raise ValueError("unexpected out-of-support phase")
    key = row["station"], phase, row["active_elapsed"]
    p_constant = _hazard(control["phase"][phase])
    # Existing discrete-time risk-set / Laplace-smoothed empirical hazard,
    # backed off to the TRAIN phase-only rate if this bucket was not seen.
    p_conditioned = _hazard(control["joint"][key]) if key in control["joint"] else p_constant
    return {"phase_constant": p_constant, "public_elapsed_hazard": p_conditioned,
            "original_rule": float(row["rule_event"])}


def compare(control, dev_rows):
    if not dev_rows:
        raise ValueError("nonempty independent developer risk rows required")
    preds = [probabilities(control, r) for r in dev_rows]
    methods = ("original_rule", "phase_constant", "public_elapsed_hazard")
    return {
        "n": len(dev_rows),
        "events": sum(r["event"] for r in dev_rows),
        "all_scored_against_same_dev_rows": True,
        "dev_used_for_fit": False,
        "train_only_original_discrete_hazard_smoothing": True,
        "method": {
            k: {
                "brier": sum((p[k] - r["event"]) ** 2
                             for p, r in zip(preds, dev_rows)) / len(dev_rows),
                "phase": {
                    phase: {
                        "n": sum(r["phase"] == phase for r in dev_rows),
                        "brier": (
                            sum((p[k] - r["event"]) ** 2
                                for p, r in zip(preds, dev_rows)
                                if r["phase"] == phase)
                            / sum(r["phase"] == phase for r in dev_rows)
                            if any(r["phase"] == phase for r in dev_rows) else None
                        ),
                    }
                    for phase in PHASES
                },
            }
            for k in methods
        },
        "train_risk_rows": sum(n for n, _ in control["phase"].values()),
        "n_train_public_age_station_buckets": len(control["joint"]),
    }


def evaluate_train_dev(train_rows, dev_rows):
    return compare(fit_train_only(train_rows), dev_rows)
