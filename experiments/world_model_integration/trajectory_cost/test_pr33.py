"""Verify unchanged PR30 progress scoring and source-derived trajectory cost."""
from __future__ import annotations
import json
import os
from pathlib import Path
import unittest

from experiments.world_model_integration.trajectory_cost.pr33_evaluate import (
    PROTOCOL, ARMS, TrajectoryCostPolicy, _no_seen_seed, evaluate_group,
)
from experiments.world_model_integration.online_horizon.pr30_run import OnlineHorizonPolicy
from experiments.world_model_integration.dynamic_identifiability.pr28_experiment import build_business


class TrajectoryReferenceTests(unittest.TestCase):
    def test_frozen_scenes_and_workload_independence(self):
        proto=json.loads(PROTOCOL.read_text())
        self.assertEqual(proto["id"],"RHWM_PR33_TRAJECTORY_COST_REFERENCE_TRANSFER_20261112_V1")
        self.assertEqual(proto["arms"],list(ARMS))
        self.assertEqual(len(proto["scenario_groups"]),24)
        self.assertEqual({g["regime"] for g in proto["scenario_groups"]},
                         {"light","medium","heavy","shift"})
        self.assertEqual({reg:sum(s["regime"]==reg for s in proto["scenario_groups"])
                         for reg in ("light","medium","heavy","shift")},
                         {reg:6 for reg in ("light","medium","heavy","shift")})
        _no_seen_seed(proto)
        self.assertEqual(proto["budget"]["horizon"],20)
        self.assertEqual(proto["budget"]["per_decision_cpu_seconds"],0.5)

    def test_original_progress_score_unchanged_and_cost_tiebreak_fixed(self):
        # Source PR30 progress criterion and first-plan tie are unchanged.
        self.assertEqual(OnlineHorizonPolicy._choose_scored_index(None,[1.,1.]),0)
        self.assertEqual(OnlineHorizonPolicy._choose_scored_index(None,[1.,1.25]),1)
        self.assertEqual(TrajectoryCostPolicy._choose_scored_index(
            None,[(7.,5.),(7.,5.)]),0)
        self.assertEqual(TrajectoryCostPolicy._choose_scored_index(
            None,[(7.,5.),(7.,3.)]),1)
        self.assertEqual(TrajectoryCostPolicy._choose_scored_index(
            None,[(8.,1.),(7.,5.)]),1)

    @unittest.skipUnless(os.environ.get("RHWM_V4_ORIGINAL") and os.environ.get("RHWM_V4_SCRATCH"),
                         "CI restores exact frozen published model checkpoints")
    def test_actual_full_online_six_policy_legal_regression(self):
        cfg=json.loads(PROTOCOL.read_text())
        scene=next(g for g in cfg["scenario_groups"] if g["regime"]=="medium")
        result=evaluate_group(
            scene,cfg,
            Path(os.environ["RHWM_V4_ORIGINAL"]),
            Path(os.environ["RHWM_V4_SCRATCH"]))
        self.assertEqual(set(result["arms"]),set(ARMS))
        self.assertTrue(all(x["safety_violations"]==0 for x in result["arms"].values()))
        self.assertTrue(all(x["prediction_errors"]==0 for x in result["arms"].values()))
        self.assertTrue(all(not x["job_metrics"]["censored"] for x in result["arms"].values()))
        self.assertTrue(all(x["policy_calls"]>0 for x in result["arms"].values()))


if __name__=="__main__":
    unittest.main()
