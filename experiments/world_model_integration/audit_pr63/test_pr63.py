"""CPU-only unit tests for PR63 risk-set extraction and train-fitted baselines."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from experiments.world_model_integration.audit_pr63.pr63_risk_set import (
    _bootstrap, _read_source, age_exact_bucket, age_h4_bucket,
    baseline_probability, evaluate_baselines, extract_records, fit_baselines,
    score_rows, station_stage, tick_bucket,
)


def rec(scene, risk, target, phase, age_h4=1, age_exact=1, tick=0, map_id="0",
        regime="congestion"):
    return {
        "scene": scene, "map": map_id, "family": "test", "regime": regime,
        "tick": tick, "next_tick": tick + 1, "station": "station:0",
        "current": 1 if risk == "up" else 0, "target": target,
        "risk_set": risk, "phase": phase, "age_h4": age_h4,
        "age_exact": age_exact,
    }


class Pr63Tests(unittest.TestCase):
    def test_buckets_and_phases(self):
        self.assertEqual(station_stage(1, 1), "healthy_hold")
        self.assertEqual(station_stage(1, 0), "fault_onset")
        self.assertEqual(station_stage(0, 0), "continuing_down")
        self.assertEqual(station_stage(0, 1), "recovery")
        self.assertEqual([age_h4_bucket(a) for a in (1, 2, 3, 4, 9)],
                         ["1", "2", "3", "4", "4"])
        self.assertEqual([age_exact_bucket(a) for a in (1, 7, 8, 13, 21)],
                         ["1", "7", "8-12", "13", "21+"])
        self.assertEqual([tick_bucket(t) for t in (5, 6, 7, 12, 13)],
                         ["0-5", "6", "7-11", "12", "13-20"])

    def test_fit_and_evaluate(self):
        train = [
            rec("s1", "up", 1, "healthy_hold"),
            rec("s1", "up", 0, "fault_onset"),
            rec("s1", "down", 0, "continuing_down"),
            rec("s1", "down", 1, "recovery"),
        ]
        params = fit_baselines(train)
        self.assertAlmostEqual(params["constant"]["up"]["hazard"], 0.5)
        self.assertAlmostEqual(params["constant"]["down"]["hazard"], 0.5)
        self.assertTrue(params["no_validation_labels_used"])
        val = [
            rec("s1", "up", 1, "healthy_hold", age_h4=1),
            rec("s1", "up", 0, "fault_onset", age_h4=1),
            rec("s1", "down", 0, "continuing_down", age_h4=1),
            rec("s1", "down", 1, "recovery", age_h4=1),
        ]
        metrics = evaluate_baselines(params, val)
        # UP p=0.5, DOWN p=0.5 -> every squared error is 0.25.
        self.assertAlmostEqual(metrics["constant"]["overall"]["brier"], 0.25)
        self.assertEqual(metrics["constant"]["overall"]["n"], 4)
        self.assertEqual(metrics["constant"]["risk_set"]["up"]["n"], 2)
        self.assertAlmostEqual(metrics["constant"]["phase"]["fault_onset"]["brier"], 0.25)
        self.assertEqual(metrics["constant"]["per_scene"]["s1"]["n"], 4)
        self.assertEqual(metrics["constant"]["event_ranking"]["n_events"], 2)
        self.assertFalse(metrics["constant"]["event_ranking"]["reported"])

    def test_score_rows_and_bootstrap(self):
        rows = [
            (0.8, 1, rec("s1", "up", 1, "healthy_hold")),
            (0.2, 0, rec("s1", "down", 0, "continuing_down")),
        ]
        m = score_rows(rows)
        self.assertAlmostEqual(m["overall"]["brier"], 0.04)
        self.assertGreater(m["overall"]["logloss"], 0)
        vals = {"s1": 0.1, "s2": 0.3, "s3": 0.5}
        a = _bootstrap(vals, b=200, seed=7)
        b = _bootstrap(vals, b=200, seed=7)
        self.assertEqual(a, b)
        self.assertLessEqual(a["ci95_low"], a["mean"])
        self.assertGreaterEqual(a["ci95_high"], a["mean"])

    def test_source_rejects_blind_and_test(self):
        base = {"schema": "pr59-sha-sealed-natural-public-world-map-family-split-v1",
                "split": "validation", "BLIND_TEST_RUN": False, "scenes": [{}]}
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "data.json"
            p.write_text(json.dumps(base))
            self.assertEqual(_read_source(p, "validation")["split"], "validation")
            bad = dict(base, BLIND_TEST_RUN=True)
            p.write_text(json.dumps(bad))
            with self.assertRaises(AssertionError):
                _read_source(p, "validation")
            bad = dict(base, split="test")
            p.write_text(json.dumps(bad))
            with self.assertRaises(AssertionError):
                _read_source(p, "validation")
            with self.assertRaises(ValueError):
                _read_source(p, "test")

    def test_extract_minimal_source(self):
        nodes = [
            {"id": "robot:0", "kind": "robot"},
            {"id": "station:0", "kind": "station", "device_available": True},
        ]
        # History: four current station snapshots, all UP; robot row may be zeros.
        history = [[[0.0] * 16 for _ in range(4)] for _ in nodes]
        for k in range(4):
            history[1][k][6] = 1.0
        visible = [[0, 0, 0, 1], [1, 1, 1, 1]]
        src = {
            "split": "validation", "BLIND_TEST_RUN": False,
            "scenes": [{
                "id": "PR59_VALIDATION_0_0",
                "case": {"split": "validation", "map_family": "test",
                         "regime": "congestion"},
                "raw_actual_public_world_transitions": [{
                    "input": {"public_tick": 6, "graph": {"nodes": nodes},
                              "history": history,
                              "history_visible_mask": visible},
                    "target": {"next_public_tick": 7,
                               "node_target": [[1.0, 0, 0, 0, 0, 0],
                                               [0.0, 0, 0, 0, 0, 0]],
                               "node_mask": [[0, 0, 0, 0, 0, 0],
                                             [1, 0, 0, 0, 0, 0]]},
                }],
            }],
        }
        records, meta = extract_records(src, "validation")
        self.assertEqual(len(records), 1)
        r = records[0]
        self.assertEqual((r["tick"], r["next_tick"]), (6, 7))
        self.assertEqual((r["current"], r["target"]), (1, 0))
        self.assertEqual((r["risk_set"], r["phase"]), ("up", "fault_onset"))
        self.assertEqual(r["age_h4"], 4)
        self.assertEqual(r["age_exact"], 1)
        self.assertEqual(meta[0]["phase_counts"], {"fault_onset": 1})

    def test_baseline_probability(self):
        params = {
            "constant": {"up": {"hazard": 0.1}, "down": {"hazard": 0.2}},
            "age_h4": {"up": {"1": {"hazard": 0.3}}, "down": {"1": {"hazard": 0.4}}},
            "age_exact": {"up": {"1": {"hazard": 0.5}}, "down": {"1": {"hazard": 0.6}}},
            "abs_tick": {"up": {"0-5": {"hazard": 0.7}}, "down": {"0-5": {"hazard": 0.8}}},
        }
        self.assertAlmostEqual(
            baseline_probability(rec("s", "up", 1, "healthy_hold", 1, 1, 0),
                                 "constant", params), 0.9)
        self.assertAlmostEqual(
            baseline_probability(rec("s", "down", 0, "continuing_down", 1, 1, 0),
                                 "constant", params), 0.2)
        self.assertAlmostEqual(
            baseline_probability(rec("s", "up", 1, "healthy_hold", 1, 1, 0),
                                 "age_h4", params), 0.7)


if __name__ == "__main__":
    unittest.main()
