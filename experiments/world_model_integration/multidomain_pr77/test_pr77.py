"""PR77 original model wiring and joint AGV + conveyor graph contract tests.

PR61 untrained fixtures are ONLY code path/legality tests; actual weights
still require SHA admission. Conveyor schema DOES NOT imply a fitted predictor.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import unittest

import torch

from world_model_core.contracts import (
    ActionConditionedWorldQuery, LAB_READINESS, CONVEYOR_READINESS,
    ORIGINAL_PR61_NODE_KINDS, CONVEYOR_NODE_KINDS,
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


def conveyor_query():
    """One lab world with both robot handoff and conveyor carrier movement."""
    nodes = (
        RelationalEntity("robot:R1","robot",{
            "busy":PublicField(False,4),
        }),
        RelationalEntity("task:J1","task",{
            "status":PublicField("in_transit",4),
            "transport_mode":PublicField("conveyor",4),
        }),
        RelationalEntity("carrier:P1","sample_carrier",{
            "sample_id":PublicField("J1",4),
            "carrier_status":PublicField("in_motion",4),
        }),
        RelationalEntity("segment:S1","conveyor_section",{
            "occupied":PublicField(True,4),
            "capacity":PublicField(1,1),
            "belt_speed":PublicField(0.25,4),
        }),
        RelationalEntity("segment:S2","conveyor_section",{
            "occupied":PublicField(False,4),
            "capacity":PublicField(1,1),
        }),
        RelationalEntity("diverter:D1","diverter",{
            "branch":PublicField("to_analyzer",4),
            "available":PublicField(True,4),
        }),
        RelationalEntity("buffer:B1","buffer_slot",{
            "occupied":PublicField(False,4),
        }),
        RelationalEntity("port:X1","transfer_port",{
            "robot_accessible":PublicField(True,4),
            "carrier_present":PublicField(False,4),
        }),
        RelationalEntity("station:A1","station",{
            "device_available":PublicField(True,4),
        }),
    )
    frame=PublicWorldFrame(
        "lab_sample_transport",4,0.2,nodes,
        relations=(
            RelationalEdge("carrier:P1","task:J1","carries_sample"),
            RelationalEdge("carrier:P1","segment:S1","occupies_section"),
            RelationalEdge("robot:R1","port:X1","can_transfer_at"),
        ),
        topology=(
            RelationalEdge("segment:S1","diverter:D1","feeds_into"),
            RelationalEdge("diverter:D1","segment:S2","branches_to"),
            RelationalEdge("segment:S2","buffer:B1","buffered_by"),
            RelationalEdge("buffer:B1","station:A1","serves_station"),
            RelationalEdge("buffer:B1","port:X1","transfers_at"),
        ))
    action=PublicAction("lab_sample_transport",4,"route_carrier",
                        "carrier:P1",("segment:S2",))
    return ActionConditionedWorldQuery((frame,),action,20,particles=3)


class DomainContractTests(unittest.TestCase):
    def test_same_lab_domain_accepts_robot_and_conveyor_actions_and_graph(self):
        robot=lab_query()
        conveyor=conveyor_query()
        self.assertEqual(robot.history[-1].domain,conveyor.history[-1].domain)
        self.assertEqual(conveyor.action.action_kind,"route_carrier")
        self.assertEqual(conveyor.history[-1].tick_seconds,0.2)
        kinds={e.entity_kind for e in conveyor.history[-1].entities}
        self.assertTrue({"robot","task","station","sample_carrier",
                         "conveyor_section","buffer_slot","transfer_port"}<=kinds)
        # A joint public state can ALSO be queried by a mobile robot action.
        state=conveyor.history[-1]
        robot_action=PublicAction(
            "lab_sample_transport",4,"assign_robot_task","robot:R1",("task:J1",))
        mixed=ActionConditionedWorldQuery((state,),robot_action,5,particles=2)
        self.assertEqual(mixed.action.actor_id,"robot:R1")

    def test_conveyor_graph_is_not_a_trained_pr61_model(self):
        self.assertTrue(CONVEYOR_READINESS.public_graph_contract)
        self.assertFalse(CONVEYOR_READINESS.domain_encoder)
        self.assertFalse(CONVEYOR_READINESS.legal_transition)
        self.assertFalse(CONVEYOR_READINESS.supports_trained_one_step_forecasting)
        with self.assertRaisesRegex(NotImplementedError,"not available"):
            require_trained_forecasting(CONVEYOR_READINESS)
        self.assertTrue(LAB_READINESS.supports_trained_one_step_forecasting)
        self.assertFalse(LAB_READINESS.supports_verified_decision_rollout)
        self.assertTrue(CONVEYOR_NODE_KINDS.isdisjoint(ORIGINAL_PR61_NODE_KINDS))

    def test_future_public_schedule_allowed_but_future_actual_sample_position_forbidden(self):
        q=conveyor_query()
        frame=q.history[-1]
        nodes=list(frame.entities)
        carrier=nodes[2]
        nodes[2]=replace(carrier,public_features={
            **carrier.public_features,
            "published_route_target":PublicField("station:A1",1),
        })
        updated=replace(frame,entities=tuple(nodes))
        self.assertEqual(updated.entities[2].public_features["published_route_target"].known_at_tick,1)
        nodes[2]=replace(nodes[2],public_features={
            **nodes[2].public_features,
            "future_actual_jam_at_tick_6":PublicField(True,6),
        })
        with self.assertRaisesRegex(ValueError,"future disclosure"):
            replace(frame,entities=tuple(nodes))

    def test_conveyor_bad_segment_or_unregistered_rail_action_are_rejected(self):
        q=conveyor_query()
        frame=q.history[-1]
        with self.assertRaisesRegex(ValueError,"topology edge"):
            replace(frame,topology=(
                RelationalEdge("segment:S1","segment:S2","train_signal_route"),))
        with self.assertRaisesRegex(ValueError,"unobserved entities"):
            replace(q,action=PublicAction(
                "lab_sample_transport",4,"route_carrier","carrier:P1",("segment:UNKNOWN",)))
        with self.assertRaisesRegex(ValueError,"not registered"):
            PublicAction("lab_sample_transport",4,"route_train","carrier:P1")

    def test_clock_history_and_action_time_must_match(self):
        q=conveyor_query()
        lab=lab_query()
        with self.assertRaisesRegex(ValueError,"clock units"):
            ActionConditionedWorldQuery((lab.history[-1],q.history[-1]),q.action,5)
        with self.assertRaisesRegex(ValueError,"current public domain"):
            ActionConditionedWorldQuery(
                q.history,PublicAction("lab_sample_transport",5,"hold_carrier","carrier:P1"),5)
        with self.assertRaisesRegex(ValueError,"strictly chronological"):
            ActionConditionedWorldQuery((q.history[-1],q.history[-1]),q.action,5)


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
