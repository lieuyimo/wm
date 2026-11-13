"""PR67: PR61 graph world predictions -> existing legal service-event rule transition.

Existing implementation reused, not an invented model/optimizer:
  - PR54 graph_public_features, node_features, 4-frame history semantics;
  - PR56 original public static map-to-entity dual graph;
  - PR59/61 PublicMapTemporalRelationalWorld SAGEConv/GRU/RGCN predictor;
  - existing SessionRuleDynamics service-duration intervention pattern;
  - original WorkflowDynamics + business check_transition;
  - PR66 exogenous station event boundary, separately.

This maps the task head's next-state probabilities to the **currently active,
publicly observed** task service completion event (pickup -> in_transit,
dropoff -> completed). It does not decode arbitrary robot trajectories or
fabricate unseen tasks. Unknown/publicly unobservable service progress falls
back to the original rules. Model event counts are diagnostics ONLY.

The predictor is injectable to make controlled deterministic unit tests
possible; random weights and synthetic task probabilities are NOT trained
models and may not be used for a performance/decision claim.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
import math
import random

import torch
from torch_geometric.data import Batch
from experiments.world_model_integration.temporal_pr54.pr54_public_sequence import (
    graph_public_features,
)
from experiments.world_model_integration.decision_learning_pr51.pr51_train import candidate_graph
from experiments.world_model_integration.sealed_pr59.pr59_dual_model import (
    PublicTopologyTemporalData,
)
from experiments.world_model_integration.topology_pr56.pr56_public_map import (
    public_topology_for_graph,
)
from business_transport.dynamics import WorkflowDynamics
from business_transport.session_events import eligible_sessions
from lab_transport.torch_structured_adapter import ProgressAwareRule


class PublicGraphHistory:
    """PR54 causal 4-frame per-entity current action-conditioned graph encoding.

    Memory contains past *publicly emitted* or predicted observations only.
    Names are used solely to align entities across frames, never fed to GNN.
    """

    def __init__(self, geometry, rules):
        self.geometry = geometry
        self.rules = rules
        self.prior = deque(maxlen=3)
        self.last_tick = None

    def reset(self):
        self.prior.clear()
        self.last_tick = None

    def encode(self, current, assignments):
        graph, keys, features = graph_public_features(
            current, self.rules, tuple(assignments))
        if self.last_tick is not None and current.tick != self.last_tick + 1:
            raise ValueError("neural history must be chronological with no skipped ticks")
        if self.prior and self.prior[-1][0] >= current.tick:
            raise ValueError("stale or future neural history")
        entity = candidate_graph(graph, 0, True)
        n = len(keys)
        if len(features) != n or entity.x.shape != (n, 16):
            raise AssertionError("PR61 public entity feature width/order changed")
        history = torch.zeros((n, 4, 16), dtype=torch.float32)
        mask = torch.zeros((n, 4), dtype=torch.float32)
        for offset, (_, values) in enumerate(self.prior):
            target = 3 - len(self.prior) + offset
            for i, key in enumerate(keys):
                if key in values:
                    history[i, target] = torch.tensor(values[key], dtype=torch.float32)
                    mask[i, target] = 1
        history[:, -1] = entity.x
        mask[:, -1] = 1
        topology = public_topology_for_graph(self.geometry.grid, self.geometry, graph)
        map_x = torch.tensor(topology["map_node_features"], dtype=torch.float32)
        map_edges = torch.tensor(topology["directed_map_edges"], dtype=torch.long).T.contiguous()
        links = torch.tensor(topology["public_entity_to_map_cell_index"],
                             dtype=torch.long).T.contiguous()
        data = PublicTopologyTemporalData(
            **entity.to_dict(),
            history=history,
            history_mask=mask,
            map_x=map_x,
            map_edge_index=map_edges,
            entity_to_map_index=links,
            map_node_count=torch.tensor([len(map_x)], dtype=torch.long),
        )
        # The previous executed-action graph features are recorded AFTER the
        # current forecast is made, as in the original PR54 transition source.
        self.prior.append((current.tick, dict(zip(keys, features))))
        self.last_tick = current.tick
        return Batch.from_data_list([data]), keys


@dataclass(frozen=True)
class NeuralServiceEvent:
    robot_id: str
    sample_id: str
    station_id: str
    phase: str
    probability_complete: float
    sampled_complete: bool


class NeuralTaskServiceRule:
    """GNN next task states drive service completion, original rule drives legality.

    An actual pretrained PR61 model is accepted via the unchanged forward()
    signature, but NO checkpoint is silently loaded or promoted here.
    """

    def __init__(self, geometry, rules, predictor, *, seed=67):
        if not callable(getattr(predictor, "forward", None)):
            raise TypeError("a PR61-compatible PyG forward predictor is required")
        self.geometry = geometry
        self.rules = rules
        self.predictor = predictor.cpu().eval()
        self.history = PublicGraphHistory(geometry, rules)
        self.rule = WorkflowDynamics(ProgressAwareRule(geometry), rules)
        self.rng = random.Random(seed)
        self.forward_calls = 0
        self.service_query_calls = 0
        self.last_neural_events = ()
        self.last_event_head = None
        self.weight_origin = "caller_supplied_not_verified"

    def reset_seed(self, seed):
        self.rng.seed(seed)
        self.history.reset()
        self.last_neural_events = ()
        self.last_event_head = None

    def predict_next(self, observation, assignments):
        actions = tuple(assignments)
        pyg, keys = self.history.encode(observation, actions)
        with torch.inference_mode():
            nodes, events = self.predictor(pyg)
        self.forward_calls += 1
        if nodes.shape != (len(keys), 6) or events.shape != (1, 2):
            raise ValueError("neural world predictor changed PR61 output contract")
        if not torch.isfinite(nodes).all() or not torch.isfinite(events).all():
            raise ValueError("non-finite world prediction")
        self.last_event_head = tuple(float(x) for x in events[0])
        index = {key: i for i, key in enumerate(keys)}
        overrides = {}
        selected = []
        for query, reason in eligible_sessions(observation, actions, require_active=True):
            if reason is not None:
                continue
            task = index.get(("task", query["sample_id"]))
            if task is None:
                raise AssertionError("observed active service task absent from public graph")
            head = 2 if query["phase"] == "to_pickup" else 3
            probability = float(nodes[task, head])
            if not 0 <= probability <= 1:
                raise ValueError("invalid learned task next-status probability")
            station = query["station_id"]
            robot = next(r for r in observation.robots if r.id == query["robot_id"])
            progress = robot.service_active_elapsed
            if not isinstance(progress, int) or progress < 0:
                raise AssertionError("active eligible service has invalid public progress")
            if station in overrides:
                raise ValueError("multiple physical service sessions on one port")
            sampled = self.rng.random() < probability
            # EXACTLY the existing SessionRuleDynamics intervention:
            # set observed remaining service threshold; the original rule
            # itself executes or withholds the legal service completion.
            overrides[station] = progress + (1 if sampled else 2)
            selected.append(NeuralServiceEvent(
                robot_id=query["robot_id"],
                sample_id=query["sample_id"],
                station_id=station,
                phase=query["phase"],
                probability_complete=probability,
                sampled_complete=sampled,
            ))
        self.last_neural_events = tuple(selected)
        if overrides:
            changed = replace(observation, stations=tuple(
                replace(s, expected_service_ticks=float(overrides[s.id]),
                        observed_effective_service_mean=float(overrides[s.id]))
                if s.id in overrides else s for s in observation.stations
            ))
        else:
            changed = observation
        # WorkflowDynamics enforces capacity and legal workflow release.
        predicted = self.rule.predict_next(changed, actions)
        originals = {s.id: s for s in observation.stations}
        predicted = replace(predicted, stations=tuple(
            replace(s,
                    expected_service_ticks=originals[s.id].expected_service_ticks,
                    observed_effective_service_mean=originals[s.id].observed_effective_service_mean)
            for s in predicted.stations
        ))
        self.service_query_calls += len(selected)
        return predicted
