"""Independent corrected-budget offline-replay guardrails and real online smoke."""
from __future__ import annotations
import json
import os
from pathlib import Path
import unittest

from experiments.world_model_integration.trajectory_cost.pr36_evaluate import (
    PROTOCOL, _verify_new_scene_group, _admission_audit, _check_report
)
from experiments.world_model_integration.trajectory_cost.pr33_evaluate import (
    ARMS, evaluate_group,
)


class StrictIndependentGeneralizationTests(unittest.TestCase):
    def test_new_24_seed_groups_and_cost_comparators_are_sealed(self):
        cfg=json.loads(PROTOCOL.read_text())
        self.assertEqual(cfg["protocol"],"RHWM_PR36_STRICT_POSTCANDIDATE_BUDGET_20261112_V1")
        _verify_new_scene_group(cfg)
        self.assertEqual(cfg["arms"],list(ARMS))
        self.assertEqual(cfg["budget"]["horizon"],20)
        self.assertEqual(cfg["budget"]["per_decision_cpu_seconds"],0.5)

    def test_audit_rejects_late_model_action(self):
        fictitious={
            "changed_plan_calls":1,
            "decision_records":[{
                "tick":0,"selected_index":1,
                "model_selection_cpu_seconds":0.5001,
            }],
        }
        with self.assertRaisesRegex(AssertionError,"after CPU deadline"):
            _admission_audit(fictitious,0.5)
        fictitious["decision_records"][0]["model_selection_cpu_seconds"]=None
        with self.assertRaisesRegex(AssertionError,"overrode CP-SAT baseline"):
            _admission_audit(fictitious,0.5)

    @unittest.skipUnless(os.environ.get("RHWM_V4_ORIGINAL") and
                         os.environ.get("RHWM_V4_SCRATCH"),
                         "GitHub CI restores verified original and scratch checkpoints")
    def test_real_six_arm_online_simulator_admits_no_late_choice(self):
        cfg=json.loads(PROTOCOL.read_text())
        sample=next(s for s in cfg["new_sealed_groups"] if s["regime"]=="medium")
        g=evaluate_group(sample,cfg,
                         Path(os.environ["RHWM_V4_ORIGINAL"]),
                         Path(os.environ["RHWM_V4_SCRATCH"]))
        admitted=_check_report(g,cfg["budget"]["per_decision_cpu_seconds"])
        self.assertEqual(set(admitted),set(ARMS))
        self.assertEqual(admitted["cp_sat"],0)
        self.assertTrue(all(v["safety_violations"]==0 for v in g["arms"].values()))
        self.assertTrue(all(v["prediction_errors"]==0 for v in g["arms"].values()))


if __name__=="__main__":
    unittest.main()
