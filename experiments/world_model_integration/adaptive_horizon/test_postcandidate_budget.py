"""Deterministic CPU gate regression for the documented PR33 post-candidate bug.

A fake clock jumps after the LAST candidate rollout; the correct scheduler
must reject its ranking and retain the CP-SAT first legal proposal.
No GPU or model inference is needed for this time/contract test.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch
import unittest

from experiments.world_model_integration.online_horizon.pr30_run import OnlineHorizonPolicy


class _ClockModel:
    def __init__(self,clock,second_cpu):
        self.clock=clock
        self.calls=0
        self.second_cpu=second_cpu

    def rollout(self,*args,**kwargs):
        self.calls+=1
        self.clock[0]=0.18 if self.calls==1 else self.second_cpu
        return SimpleNamespace(distributions=[SimpleNamespace(
            expected_completed_samples=0.0 if self.calls==1 else 3.0)])


def _policy(clock,second_cpu):
    """Minimal real OnlineHorizonPolicy with stubbed lawful CP-SAT proposals."""
    policy=OnlineHorizonPolicy.__new__(OnlineHorizonPolicy)
    policy.observer=SimpleNamespace(observe=lambda raw:SimpleNamespace(tick=0))
    policy.solver=SimpleNamespace(proposals=lambda state,legal:[("a",),("b",)])
    policy.forecast_solver=object()
    policy.model=_ClockModel(clock,second_cpu)
    policy.seed=123
    policy.horizon=20
    policy.geometry=object()
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
    policy.cpu_decision_seconds=0.0
    return policy


class BudgetPostCandidateTests(unittest.TestCase):
    def test_last_candidate_exceeds_budget_and_cannot_rerank(self):
        clock=[0.0]
        policy=_policy(clock,0.66)
        with patch("experiments.world_model_integration.online_horizon.pr30_run.time.process_time",
                   side_effect=lambda:clock[0]):
            result=policy.__call__(None,("a","b"))
        self.assertEqual(result,("a",))
        self.assertEqual(policy.n_budget_exceeded,1)
        self.assertEqual(policy.n_fallback,1)
        self.assertEqual(policy.n_flip,0)
        self.assertEqual(policy.n_model_queries,2)
        self.assertTrue(any(x.get("post_candidate_budget_exceeded") for x in policy.decisions))

    def test_complete_legal_rerank_within_budget_is_allowed(self):
        clock=[0.0]
        policy=_policy(clock,0.40)
        with patch("experiments.world_model_integration.online_horizon.pr30_run.time.process_time",
                   side_effect=lambda:clock[0]):
            result=policy.__call__(None,("a","b"))
        self.assertEqual(result,("b",))
        self.assertEqual(policy.n_budget_exceeded,0)
        self.assertEqual(policy.n_fallback,0)
        self.assertEqual(policy.n_flip,1)


if __name__=="__main__":
    unittest.main()
