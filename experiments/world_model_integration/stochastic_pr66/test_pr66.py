"""CPU rule legality checks for sampled device events at next tick boundary."""
from dataclasses import replace
import unittest

from business_transport.dynamics import WorkflowDynamics
from business_transport.observation import capacity_view
from business_transport.workflow_transition import check_business_transition
from experiments.workflow_prediction.tests.test_workflow import case, factual
from experiments.world_model_integration.stochastic_pr66.pr66_rule_rollout import (
    apply_next_device_boundary, lawful_rule_then_event, PublicHazardRuleRollout,
    initial_public_station_history,
)
from lab_transport.decision_value import candidates_for, nearest_batch
from lab_transport.torch_structured_adapter import ProgressAwareRule


def params():
    # Deterministic controlled test-only frequencies, NOT fitted event evidence.
    return {
        "fit_only_on_train": True,
        "constant": {
            "up": {"hazard": 0.27}, "down": {"hazard": 0.64},
        },
        "age_h4": {
            "up": {"4": {"hazard": 0.4}},
            "down": {"4": {"hazard": 0.7}},
        },
    }


def fake_current_public(state):
    # Synthetic test fixture: tests time/history alignment without pretending
    # it was the full authentic PR64 model graph.
    nodes = []
    histories = []
    masks = []
    for i, s in enumerate(sorted(state.stations, key=lambda st: st.id)):
        nodes.append({"kind": "station", "id": f"station:{i}",
                      "pos": list(s.position),
                      "device_available": bool(s.device_available)})
        row = [0.0] * 16
        row[6] = float(s.device_available)
        histories.append([row[:] for _ in range(4)])
        masks.append([1, 1, 1, 1])
    return {
        "public_tick": state.tick,
        "feature_last_observed_tick": state.tick,
        "graph": {"nodes": nodes},
        "history": histories,
        "history_visible_mask": masks,
        "node_persistence": [[0] * 6 for _ in nodes],
        "input_source": "synthetic_unit_contract_only",
        "public_map": {"static_map_digest": "synthetic-fixture-not-real-model-input"},
    }


class RuleConstrainedStochasticDeviceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        torch.set_num_threads(1)
        cls.business = case(2)
        _, cls.rows, cls.geometry = factual(cls.business)
        cls.rules = cls.business.rules
        cls.core = WorkflowDynamics(ProgressAwareRule(cls.geometry), cls.rules)

    def _next_device(self, state, overrides):
        return {s.id: overrides.get(s.id, bool(s.device_available)) for s in state.stations}

    def test_event_boundary_changes_no_robots_or_task_lifecycle(self):
        row = self.rows[0]
        original = self.core.predict_next(row.observation, row.assignments)
        check_business_transition(row.observation, original, self.geometry,
                                  self.rules, row.assignments)
        future = self._next_device(original, {"B0": False})
        altered = lawful_rule_then_event(row.observation, row.assignments,
                                         next_device=future, geometry=self.geometry,
                                         rules=self.rules)
        self.assertEqual(original.tick, altered.tick)
        self.assertEqual(original.robots, altered.robots)
        self.assertEqual(original.samples, altered.samples)
        self.assertFalse(next(s.device_available for s in altered.stations if s.id == "B0"))
        with self.assertRaisesRegex(ValueError, "unknown future device"):
            check_business_transition(row.observation, altered,
                                      self.geometry, self.rules, row.assignments)

    def test_external_boundary_guards_station_identity_and_values(self):
        core = self.core.predict_next(self.rows[0].observation, self.rows[0].assignments)
        statuses = self._next_device(core, {})
        with self.assertRaises(ValueError):
            apply_next_device_boundary(core, {"B0": True}, self.rules)
        broken = dict(statuses)
        broken["B0"] = 1
        with self.assertRaises(ValueError):
            apply_next_device_boundary(core, broken, self.rules)
        self.assertEqual(core, apply_next_device_boundary(core, statuses, self.rules))

    def test_down_at_next_boundary_pauses_next_step_service_and_holds_slot(self):
        # Choose a real observed busy transition at B0 and make its public
        # expected duration long in both controlled forecast branches.
        busy = next(row for row in self.rows
                    if any(r.task_id == "J1::stage1" and
                           r.phase == "to_dropoff" and
                           r.position == next(s.position for s in row.observation.stations
                                             if s.id == "B0")
                           for r in row.observation.robots))
        state = replace(busy.observation, stations=tuple(
            replace(s, expected_service_ticks=12.0,
                    observed_effective_service_mean=12.0)
            if s.id == "B0" else s for s in busy.observation.stations))
        state = capacity_view(state, (), self.rules)[0]
        up = lawful_rule_then_event(state, (), next_device=self._next_device(state, {}),
                                    geometry=self.geometry, rules=self.rules)
        down = lawful_rule_then_event(state, (), next_device=self._next_device(state, {"B0": False}),
                                      geometry=self.geometry, rules=self.rules)
        up_b = next(s for s in up.stations if s.id == "B0")
        down_b = next(s for s in down.stations if s.id == "B0")
        self.assertEqual(up_b.resource_occupied, down_b.resource_occupied)
        self.assertGreater(down_b.resource_occupied, 0)
        self.assertTrue(down_b.service_session_active)
        self.assertTrue(up_b.service_in_progress)
        self.assertFalse(down_b.device_available)
        self.assertFalse(down_b.available)
        self.assertFalse(down_b.service_in_progress)
        self.assertEqual({r.id: r.service_elapsed for r in up.robots},
                         {r.id: r.service_elapsed for r in down.robots})
        # At t+1 old/current availability matters for t+1->t+2 service.
        up_next = self.core.predict_next(up, ())
        down_next = self.core.predict_next(down, ())
        check_business_transition(up, up_next, self.geometry, self.rules, ())
        check_business_transition(down, down_next, self.geometry, self.rules, ())
        self.assertEqual({s.id: s.status for s in up_next.samples},
                         {s.id: s.status for s in down_next.samples})
        self.assertFalse(next(s.device_available for s in down_next.stations if s.id == "B0"))
        up_r = next(r for r in up_next.robots if r.task_id == "J1::stage1")
        down_r = next(r for r in down_next.robots if r.task_id == "J1::stage1")
        self.assertIsNotNone(up_r.service_active_elapsed)
        self.assertIsNotNone(down_r.service_active_elapsed)
        self.assertGreater(up_r.service_active_elapsed, down_r.service_active_elapsed)
        self.assertGreater(down_r.service_paused_elapsed, up_r.service_paused_elapsed)

    def test_h20_rollout_is_seeded_and_all_steps_rule_legal(self):
        row = self.rows[0]
        run = PublicHazardRuleRollout(self.geometry, self.rules, params())
        planner = lambda state, _: nearest_batch(state, candidates_for(state), self.geometry)
        first, logs = run.rollout(row.observation, planner, horizon=20, seed=66)
        second, logs2 = run.rollout(row.observation, planner, horizon=20, seed=66)
        self.assertEqual(first, second)
        self.assertEqual(logs, logs2)
        self.assertEqual(len(first), 20)
        self.assertGreater(sum(len(e.changed_stations) for e in logs), 0)
        self.assertTrue(all(s.resource_occupied <= s.resource_capacity
                            for state in first for s in state.stations))
        self.assertEqual([s.tick for s in first],
                         list(range(row.observation.tick+1,row.observation.tick+21)))

    def test_age_h4_requires_actual_four_frames_and_recurses_causally(self):
        row = self.rows[0]
        public = fake_current_public(row.observation)
        reference = initial_public_station_history(public, row.observation)
        self.assertEqual(set(reference), {s.id for s in row.observation.stations})
        r = PublicHazardRuleRollout(self.geometry, self.rules, params(), "age_h4")
        with self.assertRaises(ValueError):
            r.rollout(row.observation, lambda *_: (), horizon=1, seed=1)
        _, logs = r.rollout(row.observation, lambda *_: (), horizon=5,
                            seed=1, initial_public_input=public)
        self.assertAlmostEqual(dict(logs[0].probabilities)["B0"], 0.6)
        self.assertEqual(len(logs), 5)
        bad = dict(public)
        bad["public_tick"] = row.observation.tick+1
        with self.assertRaises(ValueError):
            initial_public_station_history(bad, row.observation)


if __name__ == "__main__":
    unittest.main()
