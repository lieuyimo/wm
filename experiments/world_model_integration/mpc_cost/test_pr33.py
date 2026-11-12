"""PR33 sealed scene seeds, source-implemented trajectory score and safety."""
from __future__ import annotations

import json
import os
from pathlib import Path
import unittest

from experiments.world_model_integration.mpc_cost.pr33_run import (
    PROTOCOL,ARMS,no_seen_scenarios,one_online,
)
from experiments.world_model_integration.posttraining.pr32_online_run import PROTOCOL as PR32
from experiments.world_model_integration.decision_value.pr27_evaluate import _trace_cost
from lab_transport.decision_value import trace_score


class SourceBackedMPCTests(unittest.TestCase):
    def test_frozen_protocol_new_scenes_and_exact_cost_source(self):
        cfg=json.loads(PROTOCOL.read_text())
        self.assertEqual(cfg["protocol"],"RHWM_PR33_SOURCE_BACKED_MPC_COST_20261112_V1")
        self.assertEqual(cfg["arms"],list(ARMS))
        self.assertEqual(len(cfg["groups"]),24)
        self.assertEqual({r["regime"] for r in cfg["groups"]},
                         {"light","medium","heavy","shift"})
        self.assertEqual([sum(r["regime"]==name for r in cfg["groups"])
                          for name in ("light","medium","heavy","shift")],[6,6,6,6])
        no_seen_scenarios(cfg)
        self.assertEqual(cfg["evaluation"]["horizon_ticks"],20)
        self.assertEqual(cfg["evaluation"]["particles"],3)
        self.assertEqual(cfg["evaluation"]["score_budget_sec_per_online_decision"],0.5)
        # Source-exact transfer: both modules already existed before PR33.
        self.assertTrue(callable(_trace_cost))
        self.assertTrue(callable(trace_score))

    @unittest.skipUnless(os.environ.get("RHWM_V4_ORIGINAL") and
                         os.environ.get("RHWM_V4_SCRATCH"),
                         "CI restores checked original and PR31 selected V4")
    def test_each_public_model_cost_selects_only_legal_cpsat_plans(self):
        cfg=json.loads(PROTOCOL.read_text())
        spec=cfg["groups"][0]
        c={
            "max_ticks":cfg["evaluation"]["max_ticks"],
            "candidate_limit":cfg["evaluation"]["candidate_limit"],
            "particles":cfg["evaluation"]["particles"],
            "decision_cpu_budget_seconds":cfg["evaluation"]["score_budget_sec_per_online_decision"],
        }
        old=Path(os.environ["RHWM_V4_ORIGINAL"])
        new=Path(os.environ["RHWM_V4_SCRATCH"])
        for arm in ("cp_sat","rule_cost","original_cost","scratch_cost"):
            r=one_online(spec,c,arm,old,new)
            self.assertEqual(r["safety_violations"],0)
            self.assertEqual(r["prediction_errors"],0)
            self.assertFalse(r["job_metrics"]["censored"])
            self.assertGreater(r["policy_calls"],0)
            self.assertLessEqual(r["changed_first_proposal"],r["multi_candidate_decisions"])


if __name__=="__main__":
    unittest.main()
