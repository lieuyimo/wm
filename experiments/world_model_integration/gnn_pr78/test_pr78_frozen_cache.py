"""Frozen original PR61 graph forward versus cached service risk equivalence.

The test uses RANDOM model parameters explicitly as a structural fixture,
never as original pretrained PR61 checkpoint evidence or scored results.
"""
from __future__ import annotations

from dataclasses import replace
import unittest

import torch
from torch_geometric.loader import DataLoader

from experiments.workflow_prediction.tests.test_workflow import case,factual
from business_transport.session_events import eligible_sessions
from experiments.world_model_integration.gnn_pr78.pr78_model import (
    PublicMapRelationalServiceWorld,PublicServiceGraphHistory,
    original_gnn_state_sha256,
)
from experiments.world_model_integration.gnn_pr78.pr78_frozen_cache import (
    frozen_public_service_rows,cached_dev_metrics,
)
from experiments.world_model_integration.gnn_pr78.pr78_pretrained_train import eval_dev


class PR78FrozenGraphCacheContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        initial=case(2,jobs=1)
        stations={
            station:replace(cfg,service_ticks=8,prior_ticks=8,jitter_ticks=0)
            for station,cfg in initial.scenario.stations.items()
        }
        cls.world=replace(initial,scenario=replace(initial.scenario,stations=stations))
        _,cls.rows,cls.geometry=factual(cls.world)
        cls.rules=cls.world.rules

    def true_public_service_examples(self):
        history=PublicServiceGraphHistory(self.geometry,self.rules)
        by_phase={}
        for t,row in enumerate(self.rows[:-1]):
            graph,keys=history.encode(row.observation,row.assignments)
            if not graph.service_risk_mask.any():
                continue
            j=int(torch.where(graph.service_risk_mask)[0][0])
            phase="to_pickup" if graph.service_phase[j,0]>0.5 else "to_dropoff"
            if phase not in by_phase:
                data=PublicServiceGraphHistory.attach_factual_next_service_labels(
                    graph,keys,self.rows[t+1].observation)
                by_phase[phase]=data
        if set(by_phase)!={"to_pickup","to_dropoff"}:
            raise AssertionError("original factual sample needs both active service phases")
        graphs=[by_phase["to_pickup"],by_phase["to_dropoff"]]
        count=sum(int(g.service_risk_mask.sum()) for g in graphs)
        return [{"id":"PR78_TEST_ONLY_ORIGINAL_2_PHASE",
                 "regime":"test_fixture",
                 "graphs":graphs,"n_risk_rows":count}]

    def test_frozen_cached_risk_features_match_exact_original_graph_forward(self):
        scenes=self.true_public_service_examples()
        torch.manual_seed(7890)
        for arm in (False,True):
            net=PublicMapRelationalServiceWorld(hidden=48,map_enabled=arm).cpu()
            for p in net.graph_world.parameters():
                p.requires_grad_(False)
            net.eval()
            cache=frozen_public_service_rows(net.graph_world,scenes,torch.device("cpu"))
            self.assertEqual(cache["features"].shape,(2,54))
            self.assertEqual(cache["phase_ids"].tolist(),[0,1])
            self.assertEqual(len(cache["scenes"]),1)
            with torch.inference_mode():
                for idx,graph in enumerate(scenes[0]["graphs"]):
                    b=next(iter(DataLoader([graph],batch_size=1)))
                    original_next,original_new,normal_logits=net(b)
                    direct_feature=torch.cat((
                        original_next[b.service_risk_mask],
                        b.service_public_query[b.service_risk_mask]),dim=1)
                    self.assertTrue(torch.allclose(cache["features"][idx:idx+1],
                                                   direct_feature,atol=1e-6,rtol=1e-6))
                    self.assertTrue(torch.allclose(
                        normal_logits[b.service_risk_mask],
                        net.service_event_head(cache["features"][idx:idx+1]).flatten(),
                        atol=1e-6,rtol=1e-6))
            summary=cached_dev_metrics(net.service_event_head,cache,torch.device("cpu"))
            original=eval_dev(net,scenes,torch.device("cpu"))
            self.assertAlmostEqual(summary["risk_weighted_brier"],
                                   original["risk_weighted_brier"],places=6)
            self.assertEqual(summary["real_original_eligible_service_risks"],2)

    def test_head_only_provenance_digest_is_stable_and_detects_trunk_change(self):
        net=PublicMapRelationalServiceWorld(hidden=48,map_enabled=True)
        original=net.graph_world.state_dict()
        before=original_gnn_state_sha256(original)
        copied={k:v.detach().cpu().clone() for k,v in original.items()}
        self.assertEqual(before,original_gnn_state_sha256(copied))
        reverse=dict(reversed(list(copied.items())))
        self.assertEqual(before,original_gnn_state_sha256(reverse))
        self.assertEqual(len(before),64)
        first=next(iter(copied))
        mutated={k:v.clone() for k,v in copied.items()}
        mutated[first].view(-1)[0]+=0.125
        self.assertNotEqual(before,original_gnn_state_sha256(mutated))
        with self.assertRaisesRegex(ValueError,"empty"):
            original_gnn_state_sha256({})

    def test_frozen_cache_refuses_a_live_trainable_original_backbone(self):
        net=PublicMapRelationalServiceWorld(hidden=48,map_enabled=True)
        with self.assertRaisesRegex(AssertionError,"fully frozen"):
            frozen_public_service_rows(net.graph_world,self.true_public_service_examples(),
                                       torch.device("cpu"))

    def test_cpu_head_only_updates_without_touching_original_graph(self):
        scenes=self.true_public_service_examples()
        torch.manual_seed(78001)
        net=PublicMapRelationalServiceWorld(hidden=48,map_enabled=True)
        for p in net.graph_world.parameters():
            p.requires_grad_(False)
        original_state={k:v.detach().clone() for k,v in net.graph_world.state_dict().items()}
        cache=frozen_public_service_rows(net.graph_world,scenes,torch.device("cpu"))
        optimizer=torch.optim.Adam(net.service_event_head.parameters(),lr=1e-3)
        old_head=net.service_event_head[0].weight.detach().clone()
        for _ in range(3):
            optimizer.zero_grad(set_to_none=True)
            logits=net.service_event_head(cache["features"]).flatten()
            loss=torch.nn.functional.binary_cross_entropy_with_logits(
                logits,cache["labels"])
            loss.backward()
            optimizer.step()
        self.assertFalse(torch.equal(net.service_event_head[0].weight,old_head))
        self.assertTrue(all(torch.equal(v,original_state[k])
                            for k,v in net.graph_world.state_dict().items()))
        self.assertTrue(all(p.grad is None for p in net.graph_world.parameters()))


if __name__=="__main__":
    unittest.main()
