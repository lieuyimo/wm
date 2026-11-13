"""PR77 original model-wiring and cross-domain public-contract tests.

Untrained PR61 network or controlled fake output is used ONLY for shape/legality
tests; actual trained PR61 inference requires the original SHA-verified archive.
Railway schema examples assert public-graph compatibility, NOT a rail simulator
or a trained train-delay model.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import unittest

import torch

from world_model_core.contracts import (
    ActionConditionedWorldQuery, LAB_READINESS, RAIL_READINESS,
    PublicField, PublicAction, PublicWorldFrame,
    RelationalEdge, RelationalEntity, require_trained_forecasting,
)
from business_transport.world_model.relational_gnn import RelationalGNNSystemWorldModel
from experiments.world_model_integration.neural_pr67.test_pr67 import ControlledPR61Output
from experiments.world_model_integration.sealed_pr59.pr59_dual_model import (
    PublicMapTemporalRelationalWorld, parameter_count,
)
from experiments.workflow_prediction.tests.test_workflow import case, factual
from lab_transport.decision_value import candidates_for, nearest_batch


def lab_query():
    nodes = (
        RelationalEntity("robot:R1", "robot", {
            "busy": PublicField(False, 0),
            "service_elapsed": PublicField(0, 0),
        }),
        RelationalEntity("task:J1::stage1", "task", {
            "status": PublicField("waiting", 0),
            "release_tick": PublicField(0, 0),
        }),
        RelationalEntity("station:A", "station", {
            "resource_occupied": PublicField(0, 0),
        }),
        RelationalEntity("resource:R", "resource", {
            "capacity": PublicField(1, 0),
        }),
    )
    frame = PublicWorldFrame(
        "lab_sample_transport", 0, 1., nodes,
        relations=(RelationalEdge("robot:R1", "task:J1::stage1", "currently_assigned"),),
        topology=(RelationalEdge("station:A", "resource:R", "resource_conflict"),))
    action = PublicAction("lab_sample_transport", 0, "assign_robot_task",
                          "robot:R1", ("task:J1::stage1",))
    return ActionConditionedWorldQuery((frame,), action, 5, particles=3)


def rail_query():
    nodes = (
        RelationalEntity("train:T1", "train", {
            "speed_kmh": PublicField(40.0, 4),
            "delay_seconds": PublicField(35, 4),
            # A future PLANNED arrival is allowed if the timetable was
            # published at t=4. It is not an observed actual future arrival.
            "published_arrival_second": PublicField(620, 1),
        }),
        RelationalEntity("track:S1", "track_section", {
            "occupied": PublicField(True, 4),
            "block_capacity": PublicField(1, 1),
        }),
        RelationalEntity("track:S2", "track_section", {
            "occupied": PublicField(False, 4),
        }),
        RelationalEntity("signal:SIG1", "signal", {
            "aspect": PublicField("stop", 4),
        }),
        RelationalEntity("station:ST1", "station", {
            "delay_seconds": PublicField(10, 4),
        }),
    )
    frame = PublicWorldFrame(
        "rail_transit", 4, 30.0, nodes,
        relations=(RelationalEdge("train:T1", "track:S1", "occupies"),),
        topology=(RelationalEdge("track:S1", "track:S2", "directed_to"),))
    action = PublicAction("rail_transit", 4, "hold_train", "train:T1")
    return ActionConditionedWorldQuery((frame,), action, 20, particles=3)


class DomainContractTests(unittest.TestCase):
    def test_lab_and_rail_both_accept_public_graphs_and_action_horizons(self):
        a, b = lab_query(), rail_query()
        self.assertEqual(a.history[-1].domain, "lab_sample_transport")
        self.assertEqual(b.history[-1].domain, "rail_transit")
        self.assertEqual(b.history[-1].tick_seconds, 30.)
        self.assertEqual(b.action.action_kind, "hold_train")
        self.assertEqual(a.horizon_ticks, 5)

    def test_rail_data_schema_not_falsely_claimed_as_trained_rail_model(self):
        self.assertTrue(RAIL_READINESS.public_graph_contract)
        self.assertFalse(RAIL_READINESS.domain_encoder)
        self.assertFalse(RAIL_READINESS.legal_transition)
        self.assertFalse(RAIL_READINESS.supports_trained_one_step_forecasting)
        with self.assertRaisesRegex(NotImplementedError, "not available"):
            require_trained_forecasting(RAIL_READINESS)
        self.assertTrue(LAB_READINESS.supports_trained_one_step_forecasting)
        self.assertFalse(LAB_READINESS.supports_verified_decision_rollout)

    def test_rail_future_private_event_cannot_be_encoded_as_current_public(self):
        q=rail_query()
        frame=q.history[-1]
        nodes=list(frame.entities)
        original=nodes[0]
        corrupted=replace(original,public_features={
            **original.public_features,
            "future_realized_arrival":PublicField(650,5),
        })
        nodes[0]=corrupted
        with self.assertRaisesRegex(ValueError, "future disclosure"):
            replace(frame,entities=tuple(nodes))

    def test_rail_wrong_topology_actor_and_timetable_are_rejected(self):
        q=rail_query()
        world=q.history[-1]
        with self.assertRaisesRegex(ValueError, "topology edge"):
            replace(world,topology=(
                RelationalEdge("track:S1","track:S2","robot_grid_step"),))
        with self.assertRaisesRegex(ValueError, "unseen entities"):
            replace(q,action=PublicAction(
                "rail_transit",4,"route_train","train:T1",("track:MISSING",)))
        with self.assertRaisesRegex(ValueError, "not registered"):
            PublicAction("rail_transit",4,"assign_robot_task","train:T1")

    def test_clock_domain_history_and_public_action_prohibit_mixing(self):
        a,b=lab_query(),rail_query()
        with self.assertRaisesRegex(ValueError, "unrelated transport domains"):
            ActionConditionedWorldQuery((a.history[-1],b.history[-1]),
                                        b.action,5)
        with self.assertRaisesRegex(ValueError, "current public domain"):
            ActionConditionedWorldQuery(b.history,
                                        PublicAction("rail_transit",5,"hold_train","train:T1"),
                                        5)
        with self.assertRaisesRegex(ValueError, "strictly chronological"):
            ActionConditionedWorldQuery((b.history[-1],b.history[-1]),b.action,5)


class ExistingPR61WorldModelBridgeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        base = case(2, jobs=1)
        stations = {sid: replace(s,service_ticks=8,prior_ticks=8,jitter_ticks=0)
                    for sid,s in base.scenario.stations.items()}
        world=replace(base,scenario=replace(base.scenario,stations=stations))
        _,cls.rows,cls.geometry = factual(world)
        cls.rules=world.rules

    def test_existing_gnn_is_actual_52953_parameter_graph_not_deepsets(self):
        neural=PublicMapTemporalRelationalWorld(
            hidden=48,history_enabled=True,map_enabled=True)
        self.assertEqual(parameter_count(neural),52953)
        self.assertIsNotNone(neural.entity_world.relation_step1)
        self.assertIsNotNone(neural.map_conv1)

    def test_real_checkpoint_not_available_must_hard_fail(self):
        with self.assertRaisesRegex(FileNotFoundError,"BLOCKED_ORIGINAL_PR61_ARCHIVE_MISSING"):
            RelationalGNNSystemWorldModel.from_original_archive(
                self.geometry,self.rules,
                Path("/nonexistent/pr77-original-pr61.tar.gz"),
                Path("examples/evidence/pr60-gpu-training-20261112/checkpoint_manifest.csv"))
        with self.assertRaisesRegex(ValueError,"requires original verified"):
            RelationalGNNSystemWorldModel(
                self.geometry,self.rules,ControlledPR61Output(0),
                model_version="fake-pretrained")

    def test_existing_pr67_to_original_rhwm_forecast_api_has_real_legal_states(self):
        fake=ControlledPR61Output(0.25)
        world=RelationalGNNSystemWorldModel._for_untrained_contract_test_only(
            self.geometry,self.rules,fake)
        state=self.rows[0].observation
        first_action=tuple(self.rows[0].assignments)
        step=world.predict_next(state,first_action,sampling_seed=1810)
        self.assertEqual(step.next_state.tick,state.tick+1)
        self.assertIn("UNTRAINED",step.model_version)
        policy=lambda s:tuple(nearest_batch(s,candidates_for(s),self.geometry))
        result=world.rollout(
            state,first_action,
            continuation_policy=policy,
            horizon_ticks=3,particles=2,sampling_seed=1810,
            continuation_policy_name="nearest_original_public")
        self.assertEqual(len(result.particles),2)
        self.assertEqual(len(result.distributions),3)
        self.assertEqual(len(result.particles[0].steps),3)
        self.assertEqual(result.distributions[-1].tick,state.tick+3)
        self.assertTrue(all(p.steps[-1].next_state.tick==state.tick+3
                            for p in result.particles))
        self.assertTrue(result.unknown_exogenous)
        self.assertTrue(all("UNTRAINED" in p.steps[0].model_version
                            for p in result.particles))


if __name__=="__main__":
    unittest.main()
