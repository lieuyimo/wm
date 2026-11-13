"""PR67 deterministic causal tests for PR61 graph -> legal service transition."""
from dataclasses import replace
import unittest

import torch

from business_transport.observation import capacity_view
from business_transport.workflow_transition import check_business_transition
from experiments.workflow_prediction.tests.test_workflow import case, factual
from experiments.world_model_integration.neural_pr67.pr67_neural_world import (
    NeuralTaskServiceRule, PublicGraphHistory,
)
from experiments.world_model_integration.sealed_pr59.pr59_dual_model import (
    PublicMapTemporalRelationalWorld,
)
from experiments.world_model_integration.stochastic_pr66.pr66_rule_rollout import (
    PublicHazardRuleRollout,
)
from lab_transport.decision_value import nearest_batch, candidates_for


class ControlledPR61Output(torch.nn.Module):
    """Untrained unit-test fixture: force task completion signal only."""
    def __init__(self, p):
        super().__init__()
        self.p = p

    def forward(self, batch):
        nodes = torch.zeros((batch.x.shape[0], 6), dtype=torch.float32)
        task = batch.x[:, 1] > .5
        nodes[task, 2] = self.p
        nodes[task, 3] = self.p
        events = torch.tensor([[0.01, 0.02]], dtype=torch.float32)
        return nodes, events


def fitted():
    return {"fit_only_on_train": True, "constant": {
        "up": {"hazard": 0.04}, "down": {"hazard": 0.15},
    }, "age_h4": {"up": {}, "down": {}}}


class NeuralConstrainedWorldTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        case0 = case(2, jobs=1)
        # Keep fixture business physics except for an explicitly long
        # station service: creates genuine observable active service windows.
        original = case0.scenario
        stations = {sid: replace(s, service_ticks=8, prior_ticks=8, jitter_ticks=0)
                    for sid, s in original.stations.items()}
        cls.business = replace(case0, scenario=replace(original, stations=stations))
        _, cls.rows, cls.geometry = factual(cls.business)
        cls.rules = cls.business.rules

    def _active(self):
        from business_transport.session_events import eligible_sessions
        return next(row for row in self.rows
                    if any(reason is None for _, reason in
                           eligible_sessions(row.observation, row.assignments,
                                             require_active=True)))

    def test_public_graph_history_uses_exact_pr61_width_and_time(self):
        row = self.rows[0]
        h = PublicGraphHistory(self.geometry, self.rules)
        graph, keys = h.encode(row.observation, row.assignments)
        self.assertEqual(graph.x.shape[0], len(keys))
        self.assertEqual(graph.x.shape[1], 16)
        self.assertEqual(graph.history.shape[1:], (4,16))
        self.assertTrue(torch.all(graph.history_mask[:, :3] == 0))
        self.assertTrue(torch.all(graph.history_mask[:, -1] == 1))
        with self.assertRaisesRegex(ValueError, "chronological"):
            h.encode(row.observation, row.assignments)
        h.reset()
        self.assertEqual(len(h.prior), 0)

    def test_very_high_vs_low_neural_task_head_changes_legal_service_state(self):
        row = self._active()
        source = row.observation
        from business_transport.session_events import eligible_sessions
        query = next(q for q, reason in eligible_sessions(
            source, row.assignments, require_active=True) if reason is None)
        models = {}
        for p in (0.0, 1.0):
            wrapped = NeuralTaskServiceRule(
                self.geometry, self.rules, ControlledPR61Output(p), seed=6701)
            predicted = wrapped.predict_next(source, row.assignments)
            check_business_transition(source, predicted, self.geometry,
                                      self.rules, row.assignments)
            self.assertEqual(wrapped.forward_calls, 1)
            self.assertTrue(wrapped.last_neural_events)
            self.assertEqual(wrapped.last_neural_events[0].sample_id, query["sample_id"])
            self.assertEqual(wrapped.last_neural_events[0].sampled_complete, bool(p))
            models[p] = predicted
        sid = query["sample_id"]
        prior = next(s for s in source.samples if s.id == sid)
        low = next(s for s in models[0.0].samples if s.id == sid)
        high = next(s for s in models[1.0].samples if s.id == sid)
        self.assertEqual(low.status, prior.status)
        self.assertNotEqual(high.status, low.status)
        self.assertEqual([s.id for s in models[0.0].stations],
                         [s.id for s in models[1.0].stations])
        self.assertTrue(all(s.resource_occupied <= s.resource_capacity
                            for output in models.values() for s in output.stations))

    def test_bad_prediction_shape_or_probability_is_not_silently_projected(self):
        class Broken(torch.nn.Module):
            def forward(self, batch):
                return torch.zeros((batch.x.shape[0],5)), torch.zeros((1,2))
        row = self._active()
        with self.assertRaisesRegex(ValueError, "output contract"):
            NeuralTaskServiceRule(self.geometry, self.rules, Broken()).predict_next(
                row.observation, row.assignments)
        class Nonprob(torch.nn.Module):
            def forward(self, batch):
                n = torch.ones((batch.x.shape[0],6))*1.5
                return n, torch.zeros((1,2))
        with self.assertRaisesRegex(ValueError, "probability"):
            NeuralTaskServiceRule(self.geometry, self.rules, Nonprob()).predict_next(
                row.observation, row.assignments)

    def test_real_PyG_pr61_architecture_forward_and_event_history(self):
        row = self.rows[0]
        torch.manual_seed(6722)
        network = PublicMapTemporalRelationalWorld(hidden=48, history_enabled=True,
                                                    map_enabled=True)
        dynamics = NeuralTaskServiceRule(self.geometry, self.rules, network, seed=67)
        next_state = dynamics.predict_next(row.observation, row.assignments)
        check_business_transition(row.observation, next_state, self.geometry,
                                  self.rules, row.assignments)
        self.assertEqual(dynamics.forward_calls, 1)
        self.assertEqual(len(dynamics.last_event_head), 2)
        dynamics.reset_seed(67)
        nxt2 = dynamics.predict_next(row.observation, row.assignments)
        self.assertEqual(next_state, nxt2)

    def test_recurrent_neural_action_world_rule_and_exogenous_h20(self):
        from business_transport.workflow_transition import check_business_transition
        row = self.rows[0]
        torch.manual_seed(6732)
        net = PublicMapTemporalRelationalWorld(
            hidden=48, history_enabled=True, map_enabled=True)
        neural = NeuralTaskServiceRule(self.geometry, self.rules, net, seed=67)
        stochastic = PublicHazardRuleRollout(
            self.geometry, self.rules, fitted(), "constant", world_dynamics=neural)
        policy = lambda state, _: nearest_batch(
            state, candidates_for(state), self.geometry)
        trace, logs = stochastic.rollout(row.observation, policy, horizon=20, seed=6702)
        trace_again, logs_again = stochastic.rollout(
            row.observation, policy, horizon=20, seed=6702)
        self.assertEqual(trace, trace_again)
        self.assertEqual(logs, logs_again)
        self.assertEqual(len(trace), 20)
        self.assertGreaterEqual(neural.service_query_calls, 0)
        self.assertEqual(neural.forward_calls, 40)
        self.assertTrue(all(st.tick == row.observation.tick+i+1
                            for i, st in enumerate(trace)))
        self.assertTrue(all(s.resource_occupied <= s.resource_capacity
                            for st in trace for s in st.stations))


if __name__ == "__main__":
    unittest.main()
