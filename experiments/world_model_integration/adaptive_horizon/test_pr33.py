"""Actual public-trigger, source model, fixed protocol and online legal actions."""
from __future__ import annotations

import json
import os
from pathlib import Path
import unittest

from experiments.world_model_integration.adaptive_horizon.pr33_run import (
    PROTOCOL,ARMS,validate_new_groups,_unfinished_jobs,_best,adaptive_one,
)
from experiments.world_model_integration.dynamic_identifiability.pr28_experiment import (
    build_business,run_factual,
)


class AdaptiveHorizonTests(unittest.TestCase):
    def setUp(self):
        self.cfg=json.loads(PROTOCOL.read_text())

    def test_independent_scenes_and_protected_lookahead(self):
        validate_new_groups(self.cfg)
        self.assertEqual(len(self.cfg["groups"]),24)
        self.assertEqual(self.cfg["arms"],list(ARMS))
        self.assertEqual(self.cfg["resource"]["GPU"],0)
        self.assertEqual(self.cfg["resource"]["train"],0)
        self.assertEqual(self.cfg["adaptation"]["screen_horizon"],5)
        self.assertEqual(self.cfg["adaptation"]["escalation_horizon"],20)
        self.assertEqual(self.cfg["adaptation"]["max_cpu_per_decision_seconds"],0.5)

    def test_tie_break_and_public_workload_only(self):
        self.assertEqual(_best([0.,0.,0.]),0)
        self.assertEqual(_best([0.,2.,1.]),1)
        self.assertEqual(_best([4.,4.]),0)
        spec=next(s for s in self.cfg["groups"] if s["regime"]=="heavy")
        from experiments.world_model_integration.horizon_generalization.pr29_evaluate import first_action_config
        config={"max_ticks":240,"bounded_counterfactual":{
            "candidates":{"limit":4,"wall_seconds":0.08,
                          "deterministic_seconds":0.04}}}
        state=run_factual(spec,config,0)["first_state"]
        self.assertEqual(_unfinished_jobs(state),5)

    @unittest.skipUnless(os.environ.get("RHWM_V4_CHECKPOINT"),
                         "CI restores immutable original V4")
    def test_live_adaptive_policy_executes_safe_business_steps(self):
        original=Path(os.environ["RHWM_V4_CHECKPOINT"])
        spec=next(s for s in self.cfg["groups"] if s["regime"]=="medium")
        value=adaptive_one(spec,self.cfg,original)
        self.assertEqual(value["safety_violations"],0)
        self.assertFalse(value["job_metrics"]["censored"])
        self.assertGreater(value["policy_calls"],0)
        self.assertLessEqual(value["changed_plan_calls"],value["multiple_candidates"])
        self.assertGreaterEqual(value["screened_decisions"],value["escalated_decisions"])
        self.assertEqual(value["escalated_decisions"],
                         value["escalated_completed"]+value["escalated_incomplete"])
        self.assertGreaterEqual(value["full_episode_cpu_seconds"],0.)


if __name__=="__main__":
    unittest.main()
