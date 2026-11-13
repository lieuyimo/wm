"""PR78: keep the original PR61 trained graph path and test PUBLIC event repair.

The controlled duration variants are NOT factual training samples. They
demonstrate PR72's identified alias is removed by the new explicitly-known
service sidecar. Only real t+1 data may enter service_event_label.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import unittest

import torch
from torch_geometric.loader import DataLoader

from business_transport.session_events import eligible_sessions
from experiments.workflow_prediction.tests.test_workflow import case,factual
from experiments.world_model_integration.gnn_pr78.pr78_model import (
    PublicServiceGraphHistory, PublicMapRelationalServiceWorld,
    service_event_loss, admit_original_pr61_backbone,
    PublicRelationalServiceRule, QUERY_WIDTH,
)
from experiments.world_model_integration.decision_learning_pr51.pr51_train import (
    node_features,
)
from lab_transport.torch_structured_adapter import ProgressAwareRule
from business_transport.dynamics import WorkflowDynamics


class OriginalPR61EventAlignment(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        old=case(2,jobs=1)
        stations={name:replace(s,service_ticks=8,prior_ticks=8,jitter_ticks=0)
                  for name,s in old.scenario.stations.items()}
        cls.world=replace(old,scenario=replace(old.scenario,stations=stations))
        _,cls.rows,cls.geometry=factual(cls.world)
        cls.rules=cls.world.rules

    @classmethod
    def _risk(cls):
        for i,row in enumerate(cls.rows):
            if i+1<len(cls.rows):
                risk=[q for q,reason in eligible_sessions(
                    row.observation,row.assignments,require_active=True)
                    if reason is None]
                if risk:
                    return i,row,risk[0]
        raise AssertionError("no real active service in original physical source")

    def test_real_original_graph_history_and_public_service_event_truth(self):
        index,row,query=self._risk()
        hist=PublicServiceGraphHistory(self.geometry,self.rules)
        # The public history must be consumed in strictly chronological order
        # exactly as the original PR67 four-frame encoder expects.
        for earlier in self.rows[:index]:
            hist.encode(earlier.observation,earlier.assignments)
        input_graph,keys=hist.encode(row.observation,row.assignments)
        self.assertEqual(tuple(input_graph.x.shape)[1],16)
        self.assertEqual(tuple(input_graph.history.shape)[1:],(4,16))
        self.assertEqual(input_graph.service_public_query.shape[1],QUERY_WIDTH)
        self.assertEqual(int(input_graph.service_risk_mask.sum()),1)
        self.assertEqual(input_graph.service_public_tick.item(),row.observation.tick)
        self.assertTrue(torch.all(input_graph.service_public_query[
            ~input_graph.service_risk_mask]==0))
        future=self.rows[index+1].observation
        data=PublicServiceGraphHistory.attach_factual_next_service_labels(
            input_graph,keys,future)
        self.assertTrue(torch.equal(data.service_event_mask,data.service_risk_mask))
        self.assertEqual(data.service_label_public_tick.item(),future.tick)
        self.assertEqual(int(data.service_event_mask.sum()),1)
        task_idx=keys.index(("task",query["sample_id"]))
        correct="in_transit" if query["phase"]=="to_pickup" else "completed"
        self.assertEqual(float(data.service_event_label[task_idx]),
                         float(next(s.status for s in future.samples
                                    if s.id==query["sample_id"])==correct))
        with self.assertRaisesRegex(ValueError,"requires public t\\+1"):
            PublicServiceGraphHistory.attach_factual_next_service_labels(
                data,keys,row.observation)

    def test_pr72_same_full_old_graph_different_public_mean_is_now_visible(self):
        index,row,query=self._risk()
        obs=row.observation
        robot=next(r for r in obs.robots if r.id==query["robot_id"])
        self.assertIsNotNone(robot.service_active_elapsed)
        low=float(robot.service_active_elapsed+1)
        high=float(robot.service_active_elapsed+7)
        def variant(expected):
            return replace(obs,stations=tuple(
                replace(s,expected_service_ticks=expected,
                        observed_effective_service_mean=expected)
                if s.id==query["station_id"] else s for s in obs.stations))
        old_rule=WorkflowDynamics(ProgressAwareRule(self.geometry),self.rules)
        a,b=variant(low),variant(high)
        pa=old_rule.predict_next(a,row.assignments)
        pb=old_rule.predict_next(b,row.assignments)
        sa=next(s.status for s in pa.samples if s.id==query["sample_id"])
        sb=next(s.status for s in pb.samples if s.id==query["sample_id"])
        self.assertNotEqual(sa,sb)
        x=PublicServiceGraphHistory(self.geometry,self.rules)
        y=PublicServiceGraphHistory(self.geometry,self.rules)
        for old in self.rows[:index]:
            x.encode(old.observation,old.assignments)
            y.encode(old.observation,old.assignments)
        ga,ka=x.encode(a,row.assignments)
        gb,kb=y.encode(b,row.assignments)
        self.assertEqual(ka,kb)
        for key in ("x","history","history_mask","edge_index","edge_type",
                    "map_x","map_edge_index","entity_to_map_index"):
            self.assertTrue(torch.equal(getattr(ga,key),getattr(gb,key)),key)
        self.assertTrue(torch.equal(ga.service_risk_mask,gb.service_risk_mask))
        self.assertFalse(torch.equal(ga.service_public_query,gb.service_public_query))
        task_idx=ka.index(("task",query["sample_id"]))
        self.assertGreater(float((ga.service_public_query[task_idx]-
                                  gb.service_public_query[task_idx]).abs().sum()),0.0)

    def test_supervised_bce_reaches_original_graph_and_map_features(self):
        i,row,query=self._risk()
        h=PublicServiceGraphHistory(self.geometry,self.rules)
        for earlier in self.rows[:i]:
            h.encode(earlier.observation,earlier.assignments)
        cur,keys=h.encode(row.observation,row.assignments)
        data=PublicServiceGraphHistory.attach_factual_next_service_labels(
            cur,keys,self.rows[i+1].observation)
        torch.manual_seed(7810)
        model=PublicMapRelationalServiceWorld(hidden=48,map_enabled=True)
        self.assertEqual(model.original_pr61_parameters(),52953)
        self.assertGreater(model.service_head_parameters(),0)
        node,event,logits=model(DataLoader([data],batch_size=1).__iter__().__next__())
        self.assertEqual(node.shape,(len(keys),6))
        self.assertEqual(event.shape,(1,2))
        self.assertEqual(logits.shape,(len(keys),))
        loss=service_event_loss(logits,data)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertGreater(float(model.service_event_head[0].weight.grad.abs().sum()),0)
        self.assertGreater(float(model.graph_world.entity_world.relation_step1.weight.grad.abs().sum()),0)
        self.assertGreater(float(model.graph_world.map_conv1.lin_l.weight.grad.abs().sum()),0)

    def test_nonactive_nodes_cannot_become_negative_service_labels(self):
        i,row,query=self._risk()
        g,keys=PublicServiceGraphHistory(self.geometry,self.rules).encode(
            row.observation,row.assignments)
        g=PublicServiceGraphHistory.attach_factual_next_service_labels(
            g,keys,self.rows[i+1].observation)
        model=PublicMapRelationalServiceWorld()
        _,_,logits=model(DataLoader([g],batch_size=1).__iter__().__next__())
        loss=service_event_loss(logits,g)
        self.assertGreater(float(loss),0)
        corrupt=replace(g) if False else g.clone()
        corrupt.service_event_mask=~corrupt.service_event_mask
        with self.assertRaisesRegex(AssertionError,"SAME public active risk"):
            service_event_loss(logits,corrupt)

    def test_new_public_service_head_reuses_original_PR67_lawful_world_transition(self):
        from business_transport.workflow_transition import check_business_transition
        i,row,query=self._risk()
        torch.manual_seed(7812)
        net=PublicMapRelationalServiceWorld(hidden=48,map_enabled=True)
        with self.assertRaisesRegex(PermissionError,"verified original"):
            PublicRelationalServiceRule(self.geometry,self.rules,net)
        world=PublicRelationalServiceRule._for_untrained_contract_test_only(
            self.geometry,self.rules,net,seed=7812)
        next_public=world.predict_next(row.observation,row.assignments)
        check_business_transition(
            row.observation,next_public,self.geometry,self.rules,row.assignments)
        self.assertEqual(next_public.tick,row.observation.tick+1)
        self.assertEqual(world.forward_calls,1)
        self.assertEqual(world.service_query_calls,1)
        self.assertTrue(world.last_neural_events)
        self.assertEqual(world.last_neural_events[0].sample_id,query["sample_id"])
        self.assertTrue(0<=world.last_neural_events[0].probability_complete<=1)
        self.assertIn("UNTRAINED",world.weight_origin)
        world.reset_seed(7812)
        repeated=world.predict_next(row.observation,row.assignments)
        self.assertEqual(next_public,repeated)

    def test_verified_new_head_cannot_load_without_real_artifact_and_matching_SHA(self):
        i,row,query=self._risk()
        with self.assertRaises(FileNotFoundError):
            PublicRelationalServiceRule.from_pretrained_output(
                self.geometry,self.rules,Path("/nonexistent/pr61.tar.gz"),
                Path("/nonexistent/manifest.csv"),
                Path("/nonexistent/pr78-head.pt"),"0"*64)

    def test_pr61_checkpoint_cannot_be_fabricated_from_nonexistent_archive(self):
        model=PublicMapRelationalServiceWorld()
        with self.assertRaises(FileNotFoundError):
            admit_original_pr61_backbone(
                model,Path("/nonexistent/original-pr61.tar.gz"),
                Path("/nonexistent/checkpoint_manifest.csv"))


if __name__=="__main__":
    unittest.main()
