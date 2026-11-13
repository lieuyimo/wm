"""PR78: original PR61 relational GNN with PUBLIC conditional service-event target.

This is *engineering alignment* of existing verified interfaces, not a novel
predictor or an assertion that the PR61 16d checkpoint was trained with these
new fields. Keep the entire existing GRU/RGCN/SAGEConv entity/map world model
unchanged and add a conventional BCE service-event head conditional on
business_transport.session_events.encode_query (48d current public input).

Unchanged original PR61 weights can be initialized ONLY via the existing
PR68 original SHA/manifest admission API. The added service head REQUIRES new
training. A randomly initialized model is permitted only in unit/CPU gradient
smoke and never reported as actual pretrained-GNN performance.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import torch
from torch_geometric.data import Batch
from torch import nn
import torch.nn.functional as F

from business_transport.session_events import (
    QUERY_WIDTH, QUERY_LABEL_VERSION, eligible_sessions, encode_query,
)
from business_transport.state_features import build_business_features
from experiments.world_model_integration.neural_pr67.pr67_neural_world import (
    PublicGraphHistory, NeuralTaskServiceRule,
)
from experiments.world_model_integration.recovery_pr68.pr68_checkpoint import (
    ARCHIVE_SHA256, load_original_models,
)
from experiments.world_model_integration.sealed_pr59.pr59_dual_model import (
    PublicMapTemporalRelationalWorld,
)

SOURCE_SCHEMA="pr78-original-pr61-gnn-plus-public-service-query-v1"
SERVICE_TARGETS={"to_pickup":"in_transit","to_dropoff":"completed"}


class PublicServiceGraphHistory:
    """Original chronological PR61 public graph plus a separate public sidecar.

    The 16d node input and 4-frame original history are EXACTLY preserved for
    original PR61 weights, while the currently active business service's
    public 48d query is bound to its TASK node (never arbitrary task indices).
    Unobserved/no-active services get a strict zero query and zero risk mask.
    """

    def __init__(self,geometry,rules):
        self.original=PublicGraphHistory(geometry,rules)
        self.geometry=geometry
        self.rules=rules

    def reset(self):
        self.original.reset()

    def encode(self,current,assignments):
        actions=tuple(assignments)
        old_pyg, keys=self.original.encode(current,actions)
        if old_pyg.num_graphs!=1:
            raise AssertionError("original PR61 history must encode one scene at a time")
        data=old_pyg.to_data_list()[0]
        n=len(keys)
        q=torch.zeros((n,QUERY_WIDTH),dtype=torch.float32)
        eligible=torch.zeros(n,dtype=torch.bool)
        phase=torch.zeros((n,2),dtype=torch.float32)
        task_ids={key:i for i,key in enumerate(keys) if key[0]=="task"}
        if any(k[0]!="task" and (data.x[i,1]>0.5) for i,k in enumerate(keys)):
            raise AssertionError("original PR61 entity ID/order does not match one-hot task kind")
        sessions=[query for query,reason in eligible_sessions(
            current,actions,require_active=True) if reason is None]
        payload=None
        if sessions:
            payload=build_business_features(current,actions,self.geometry,self.rules)
            for forbidden in ("true_remaining_service_countdown",
                              "future_fault_schedule","unreleased_external_jobs",
                              "counterfactual_outcome"):
                if payload["unknown"][forbidden]["known"]:
                    raise AssertionError("hidden future leaked into public service query")
            if payload["tick"]!=current.tick:
                raise AssertionError("service query public clock mismatch")
        for session in sessions:
            idx=task_ids.get(("task",session["sample_id"]))
            if idx is None:
                raise AssertionError("current public service session task absent in original graph")
            if eligible[idx]:
                raise AssertionError("duplicate sample task active service query")
            vector=encode_query(payload,session)
            if len(vector)!=QUERY_WIDTH:
                raise AssertionError("original public service input changed width")
            q[idx]=torch.tensor(vector,dtype=torch.float32)
            eligible[idx]=True
            phase[idx,0]=float(session["phase"]=="to_pickup")
            phase[idx,1]=float(session["phase"]=="to_dropoff")
        if int(eligible.sum())!=len(sessions):
            raise AssertionError("service query count lost in original GNN task graph")
        data.service_public_query=q
        data.service_risk_mask=eligible
        data.service_phase=phase
        data.service_public_tick=torch.tensor([current.tick],dtype=torch.long)
        return data,keys

    @staticmethod
    def attach_factual_next_service_labels(data, keys, next_public):
        """Only used by supervised DATASET code, never in predictor input."""
        if int(data.service_public_tick[0])+1!=next_public.tick:
            raise ValueError("true next service event label requires public t+1")
        next_tasks={s.id:s for s in next_public.samples}
        target=torch.zeros(data.x.shape[0],dtype=torch.float32)
        mask=data.service_risk_mask.clone()
        if len(keys)!=data.x.shape[0]:
            raise AssertionError("labels and current public graph node identity misaligned")
        for i,key in enumerate(keys):
            if not bool(mask[i]):
                continue
            if key[0]!="task" or key[1] not in next_tasks:
                raise AssertionError("active service label cannot match a future unknown task")
            if bool(data.service_phase[i,0])==bool(data.service_phase[i,1]):
                raise AssertionError("every active service needs exactly one observed phase")
            to_status="in_transit" if data.service_phase[i,0]>0.5 else "completed"
            target[i]=float(next_tasks[key[1]].status==to_status)
        data.service_event_label=target
        data.service_event_mask=mask
        data.service_label_public_tick=torch.tensor([next_public.tick],dtype=torch.long)
        return data


class PublicMapRelationalServiceWorld(nn.Module):
    """Same PR61 graph backbone plus standard conditioned binary event head.

    Old graph shape, task-status logits and map_on/map_off options are unmodified.
    A conventional 48d-query + original GNN task-status probability MLP learns
    P(service finishes on next public tick | current graph/action/service data).
    The existing PR67/WorkflowDynamics lawful event projection can be updated
    to consume this probability after actual comparative training.
    """

    def __init__(self,hidden=48,map_enabled=True):
        super().__init__()
        self.graph_world=PublicMapTemporalRelationalWorld(
            hidden=hidden,history_enabled=True,map_enabled=map_enabled)
        self.service_event_head=nn.Sequential(
            nn.Linear(QUERY_WIDTH+6,hidden),
            nn.ReLU(),
            nn.Linear(hidden,1),
        )

    def forward(self,batch):
        next_nodes, next_visible=self.graph_world(batch)
        if (batch.service_public_query.shape!=(len(next_nodes),QUERY_WIDTH)
            or batch.service_risk_mask.shape!=(len(next_nodes),)):
            raise AssertionError("missing PR78 current public service features/eligibility")
        logits=self.service_event_head(torch.cat(
            (next_nodes,batch.service_public_query),dim=-1)).squeeze(-1)
        return next_nodes,next_visible,logits

    def original_pr61_parameters(self):
        return sum(p.numel() for p in self.graph_world.parameters())

    def service_head_parameters(self):
        return sum(p.numel() for p in self.service_event_head.parameters())


def service_event_loss(logits,batch):
    """Only currently ELIGIBLE and factually active sessions count in BCE."""
    if not hasattr(batch,"service_event_mask") or not hasattr(batch,"service_event_label"):
        raise ValueError("missing next-public supervised service event labels")
    eligible=batch.service_risk_mask.bool()
    mask=batch.service_event_mask.bool()
    if logits.shape!=mask.shape or not torch.equal(eligible,mask):
        raise AssertionError("training target is not the SAME public active risk set")
    if not torch.any(mask):
        return logits.sum()*0.
    labels=batch.service_event_label[mask]
    if not torch.all((labels==0)|(labels==1)):
        raise ValueError("service event target must be t+1 binary factual status")
    return F.binary_cross_entropy_with_logits(logits[mask],labels)


def admit_original_pr61_backbone(
    model: PublicMapRelationalServiceWorld,
    archive: Path, manifest: Path, *,
    arm="map_on", seed=1810, original_validation=None,
):
    """Reuse original PR68 six-ckpt SHA; never silently initialize as pretrained."""
    models,report=load_original_models(
        Path(archive),Path(manifest),original_validation)
    if (not report.get("weights_loaded_from_original_pr61")
        or report.get("archive_sha256")!=ARCHIVE_SHA256):
        raise ValueError("original PR61 archive source attestation failed")
    if (arm,seed) not in models:
        raise ValueError("original PR61 map/seed checkpoint not in verified manifest")
    if (arm=="map_on")!=bool(model.graph_world.map_enabled):
        raise ValueError("cannot load map_off checkpoint as map_on control")
    model.graph_world.load_state_dict(models[(arm,seed)].state_dict(),strict=True)
    return {
        "original_pr61_checkpoint_verified":True,
        "original_pr61_archive_sha256":ARCHIVE_SHA256,
        "original_pr61_arm":arm,
        "original_pr61_seed":seed,
        "new_service_head_trained":False,
        "not_original_pr61_checkpoint":True,
    }

class _CurrentPublicServiceBatchHistory:
    """Preserve PR67's original temporal graph state and add PR78 query sidecar."""
    def __init__(self,geometry,rules):
        self.encoder=PublicServiceGraphHistory(geometry,rules)

    def reset(self):
        self.encoder.reset()

    def encode(self,current,assignments):
        data,keys=self.encoder.encode(current,assignments)
        return Batch.from_data_list([data]),keys


class _ConditionalEventAsOriginalTaskHead(nn.Module):
    """Adapt new conditional event head to the ORIGINAL legal PR67 protocol.

    The old PR61 absolute next-status output is still available for analysis.
    For each *currently active service* task, only its corresponding phase
    completion probability is replaced by PR78's conditional event probability.
    PR67's existing public-eligible sampling, physical service intervention,
    WorkflowDynamics and all original legality checks are then reused AS-IS.
    """

    def __init__(self,model):
        super().__init__()
        self.model=model

    def forward(self,graph):
        predicted_nodes,predicted_new_tasks,service_logits=self.model(graph)
        if (not hasattr(graph,"service_risk_mask")
            or not hasattr(graph,"service_phase")):
            raise ValueError("PR78 original-world adapter missing public service queries")
        probs=torch.sigmoid(service_logits)
        output=predicted_nodes.clone()
        if not torch.all((graph.service_phase[:,0]+graph.service_phase[:,1]
                          == graph.service_risk_mask.float())):
            raise AssertionError("PR78 task event phase or public eligibility mismatch")
        pickup=graph.service_risk_mask.bool() & (graph.service_phase[:,0]>0.5)
        delivery=graph.service_risk_mask.bool() & (graph.service_phase[:,1]>0.5)
        output[pickup,2]=probs[pickup]
        output[delivery,3]=probs[delivery]
        return output,predicted_new_tasks


class PublicRelationalServiceRule(NeuralTaskServiceRule):
    """Original PR61 GNN + PR78 event head -> original PR67 lawful world step.

    Untrained code-path testing is explicit; actual inference must load the
    ORIGINAL PR61 six-weight archive plus the SHA-verified *newly trained*
    conditional head. No old-checkpoint-only or random fallback.
    """

    def __init__(self,geometry,rules,network,*,seed=7810,_fixture=False):
        if not _fixture:
            raise PermissionError(
                "PR78 conditional service head needs verified original+new "
                "trained checkpoint; use from_pretrained_output")
        super().__init__(geometry,rules,_ConditionalEventAsOriginalTaskHead(network),seed=seed)
        self.history=_CurrentPublicServiceBatchHistory(geometry,rules)
        self.weight_origin="UNTRAINED_PR78_INTERFACE_TEST_ONLY"

    @classmethod
    def _for_untrained_contract_test_only(cls,geometry,rules,network,*,seed=7810):
        return cls(geometry,rules,network,seed=seed,_fixture=True)

    @classmethod
    def from_pretrained_output(cls,geometry,rules,archive,manifest,
                               trained_head_file,expected_head_sha256,*,
                               seed=7810,original_validation=None):
        import hashlib
        digest=hashlib.sha256(Path(trained_head_file).read_bytes()).hexdigest()
        if not expected_head_sha256 or expected_head_sha256!=digest:
            raise ValueError("PR78 newly trained public service head SHA mismatch")
        checkpoint=torch.load(trained_head_file,map_location="cpu",weights_only=True)
        if (checkpoint.get("schema")!=SOURCE_SCHEMA or
            checkpoint.get("original_PR61_archive_sha256")!=ARCHIVE_SHA256 or
            checkpoint.get("original_pretrained_backbone_unchanged") is not True or
            checkpoint.get("online_verified") is not False or
            checkpoint.get("BLIND_TEST_read") is not False):
            raise ValueError("PR78 trained conditional service head provenance invalid")
        arm=checkpoint["original_PR61_arm"]
        original_seed=checkpoint["original_PR61_seed"]
        model=PublicMapRelationalServiceWorld(
            hidden=48,map_enabled=(arm=="map_on"))
        admission=admit_original_pr61_backbone(
            model,archive,manifest,arm=arm,seed=original_seed,
            original_validation=original_validation)
        stored=checkpoint["original_pretrained_graph_state"]
        if any(not torch.equal(v,stored[k])
               for k,v in model.graph_world.state_dict().items()):
            raise ValueError("PR78 claimed original PR61 frozen weights do not match archive")
        model.service_event_head.load_state_dict(
            checkpoint["trained_new_public_service_head_state"],strict=True)
        obj=cls(geometry,rules,model,seed=seed,_fixture=True)
        obj.weight_origin=f"VERIFIED_PR61_{arm}_{original_seed}_PR78_{digest}"
        obj.predictor.eval()
        return obj
