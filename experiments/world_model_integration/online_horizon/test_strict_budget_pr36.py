"""Deterministic PR36 CPU-admission gate regression.

PR33 audit found that an over-budget *final* H20 candidate was still allowed
to change the online dispatch. All tests inject a controlled CPU clock,
without waiting or modifying the model/simulator. They validate the actual
OnlineHorizonPolicy used by both progress and trajectory-cost ranking.
"""
from __future__ import annotations
from types import SimpleNamespace
from unittest.mock import patch
import unittest

from lab_transport.transition_schema import Assignment
from experiments.world_model_integration.online_horizon.pr30_run import OnlineHorizonPolicy


class _FakeModel:
    def rollout(self,initial,first_action,**kwargs):
        # An otherwise good second action must not be accepted after deadline.
        expected=100.0 if first_action else 0.0
        return SimpleNamespace(distributions=[
            SimpleNamespace(expected_completed_samples=expected)])


def _policy():
    # Bypass external model loading: exercise real PR30 online __call__,
    # its candidate loop, score acceptance and exact fallback behavior.
    p=OnlineHorizonPolicy.__new__(OnlineHorizonPolicy)
    assignment=Assignment("R1","S1")
    p.arm="v4_h20"
    p.geometry=None
    p.rules=None
    p.protocol={"decision_cpu_budget_seconds":0.5,"particles":3}
    p.seed=1
    p.horizon=20
    p.observer=SimpleNamespace(observe=lambda raw:SimpleNamespace(tick=0))
    p.solver=SimpleNamespace(proposals=lambda *args:[(),(assignment,)])
    p.forecast_solver=SimpleNamespace()
    p.model=_FakeModel()
    p.n_calls=p.n_multi=p.n_flip=p.n_model_queries=p.n_model_steps=0
    p.n_budget_exceeded=p.n_fallback=p.n_errors=0
    p.cpu_decision_seconds=0.
    p.decisions=[]
    return p,assignment


def _evaluate(clock):
    p,a=_policy()
    with patch("experiments.world_model_integration.online_horizon.pr30_run.time.process_time",
               side_effect=clock):
        chosen=p(None,(a,))
    return p,a,chosen


class StrictCPUAdmissionTests(unittest.TestCase):
    def test_final_candidate_exceeds_budget_and_must_fallback(self):
        # First rollout <0.5s; final rollout consumes budget before the
        # next decision. Previous implementation selected model anyway.
        p,a,chosen=_evaluate([0.,0.01,0.12,0.19,0.52,0.53])
        self.assertEqual(chosen,())
        self.assertEqual(p.n_budget_exceeded,1)
        self.assertEqual(p.n_fallback,1)
        self.assertEqual(p.n_flip,0)
        self.assertIsNone(p.decisions[-1]["model_selection_cpu_seconds"])
        self.assertEqual(p.decisions[-1]["predicted_completed"],[])

    def test_selection_after_complete_forecasts_cannot_cross_deadline(self):
        # Even if all candidates finished just before deadline, the
        # complete-candidate ranking step must be admitted before it.
        p,a,chosen=_evaluate([0.,0.01,0.12,0.19,0.49,0.49,0.501,0.502])
        self.assertEqual(chosen,())
        self.assertEqual(p.n_budget_exceeded,1)
        self.assertEqual(p.n_fallback,1)
        self.assertEqual(p.n_flip,0)
        self.assertIsNone(p.decisions[-1]["model_selection_cpu_seconds"])

    def test_valid_completed_ranking_can_select_better_candidate(self):
        p,a,chosen=_evaluate([0.,0.01,0.12,0.19,0.30,0.31,0.32,0.33])
        self.assertEqual(chosen,(a,))
        self.assertEqual(p.n_budget_exceeded,0)
        self.assertEqual(p.n_fallback,0)
        self.assertEqual(p.n_flip,1)
        accepted=p.decisions[-1]["model_selection_cpu_seconds"]
        self.assertIsNotNone(accepted)
        self.assertLess(accepted,0.5)

    def test_non_model_default_returns_first_legal_plan(self):
        p,a=_policy()
        p.model=None
        with patch("experiments.world_model_integration.online_horizon.pr30_run.time.process_time",
                   side_effect=[0.,0.01]):
            chosen=p(None,(a,))
        self.assertEqual(chosen,())
        self.assertEqual(p.n_model_queries,0)
        self.assertEqual(p.n_budget_exceeded,0)
        self.assertEqual(p.n_fallback,0)


if __name__=="__main__":
    unittest.main()
