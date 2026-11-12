"""Unit tests for correctly limiting per-action claims to shared factual prefixes."""
from __future__ import annotations
import unittest
from experiments.world_model_integration.decision_bottleneck.pr33_readonly import (
    _first_common_prefix_divergence,_records,
)


def policy(records,events="same"):
    return {
        "decision_records":records,
        "policy_calls":len(records),
        "changed_plan_calls":sum(x["selected_index"]!=0 for x in records),
        "multiple_candidates":sum(x["legal_plans"]>1 for x in records),
        "factual_event_sha256":events,
    }


def rec(tick,selected_index=0,values=(1.0,2.0),first="A",selected=None):
    return {
        "tick":tick,
        "selected_index":selected_index,
        "legal_plans":2,
        "first_proposal":first,
        "selected":selected if selected is not None else ("A" if selected_index==0 else "B"),
        "predicted_completed":list(values),
        "process_cpu_seconds":0.02,
    }


class PR33ReadOnlyContractTests(unittest.TestCase):
    def test_initial_divergence_is_compared_only_at_matched_public_prefix(self):
        a=policy([rec(0),rec(1),rec(2)])
        b=policy([rec(0),rec(1,1),rec(2,0,first="unrelated")])
        x=_first_common_prefix_divergence(a,b)
        self.assertEqual(x["status"],"first_changed_action_on_common_prefix")
        self.assertEqual(x["tick"],1)
        self.assertEqual(x["compared_common_steps"],1)

    def test_never_compare_unmatched_candidate_context(self):
        a=policy([rec(0),rec(1,first="A")])
        b=policy([rec(0),rec(1,first="C")])
        x=_first_common_prefix_divergence(a,b)
        self.assertEqual(x["status"],"unmatched_decision_context")
        self.assertEqual(x["compared_common_steps"],1)

    def test_identical_action_and_events_are_recorded(self):
        a=policy([rec(0),rec(1)],events="same")
        b=policy([rec(0),rec(1)],events="same")
        x=_first_common_prefix_divergence(a,b)
        self.assertEqual(x["status"],"same_actions_same_factual_events")

    def test_inconsistent_policy_count_fails_closed(self):
        a=policy([rec(0)])
        a["changed_plan_calls"]=3
        with self.assertRaises(AssertionError):_records(a)


if __name__=="__main__": unittest.main()
