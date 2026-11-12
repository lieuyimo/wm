"""Artificial CPU clock regression: last predicted candidate may overrun.

These tests reproduce the exact former error in the PR30/32 policy:
the final candidate may return after >0.5 seconds. The repaired policy
must NEVER accept an alternate plan after this, even if its score wins.
"""
from __future__ import annotations

from types import SimpleNamespace
import json
from unittest import TestCase
from unittest.mock import patch

from experiments.world_model_integration.online_horizon.pr30_run import OnlineHorizonPolicy
from experiments.world_model_integration.strict_budget.pr35_run import (
    PROTOCOL,_new_cohort,
)


def call_with_virtual_clock(durations):
    clock=[0.0]
    score=[1.,5.]
    class FakeModel:
        def __init__(self):self.call_count=0
        def rollout(self,*args,**kwargs):
            i=self.call_count
            self.call_count+=1
            clock[0]+=durations[i]
            return SimpleNamespace(distributions=[
                SimpleNamespace(expected_completed_samples=score[i])])
    model=FakeModel()
    policy=object.__new__(OnlineHorizonPolicy)
    policy.model=model
    policy.observer=SimpleNamespace(observe=lambda raw:SimpleNamespace(tick=0))
    policy.solver=object()
    policy.forecast_solver=object()
    policy.geometry=object()
    policy.seed=111
    policy.horizon=20
    policy.arm="v4_h20"
    policy.protocol={"decision_cpu_budget_seconds":0.5,"particles":3}
    policy.n_calls=0
    policy.n_multi=0
    policy.n_flip=0
    policy.n_model_queries=0
    policy.n_model_steps=0
    policy.n_budget_exceeded=0
    policy.n_fallback=0
    policy.n_errors=0
    policy.decisions=[]
    policy.cpu_decision_seconds=0.
    with patch(
        "experiments.world_model_integration.online_horizon.pr30_run._unique_plans",
        return_value=[("A",),("B",)]),patch(
        "experiments.world_model_integration.online_horizon.pr30_run.time.process_time",
        side_effect=lambda:clock[0]):
        selected=policy(None,())
    return selected,policy,model


class PR35StrictPostCandidateGateTests(TestCase):
    def test_frozen_holdout_cohort_does_not_reuse_prior_scenes(self):
        proto=json.loads(PROTOCOL.read_text())
        _new_cohort(proto)
        self.assertEqual(len(proto["groups"]),24)
        self.assertEqual(proto["decision_cpu_budget_seconds"],0.5)

    def test_last_candidate_overrun_never_accepted(self):
        selected,p,m=call_with_virtual_clock([0.2,0.4])
        self.assertEqual(m.call_count,2)
        self.assertEqual(selected,("A",))
        self.assertEqual(p.n_budget_exceeded,1)
        self.assertEqual(p.n_fallback,1)
        self.assertEqual(p.n_flip,0)
        self.assertEqual(p.decisions[-1]["selected_index"],0)
        self.assertFalse(p.decisions[-1]["process_cpu_seconds"]<0.5)
        self.assertTrue(p.decisions[-1]["budget_exceeded"])

    def test_early_candidate_overrun_stops_before_next_candidate(self):
        selected,p,m=call_with_virtual_clock([0.51,0.05])
        self.assertEqual(m.call_count,1)
        self.assertEqual(selected,("A",))
        self.assertEqual(p.n_budget_exceeded,1)
        self.assertEqual(p.n_fallback,1)

    def test_valid_completed_candidate_ranking_remains_unchanged(self):
        selected,p,m=call_with_virtual_clock([0.2,0.1])
        self.assertEqual(m.call_count,2)
        self.assertEqual(selected,("B",))
        self.assertEqual(p.n_flip,1)
        self.assertEqual(p.n_budget_exceeded,0)
        self.assertEqual(p.n_fallback,0)
        self.assertTrue(p.decisions[-1]["decision_acceptance_within_budget"])

    def test_exact_boundary_is_rejected(self):
        selected,p,m=call_with_virtual_clock([0.2,0.3])
        self.assertEqual(selected,("A",))
        self.assertEqual(p.n_budget_exceeded,1)
        self.assertEqual(p.n_fallback,1)
        self.assertEqual(p.n_flip,0)
