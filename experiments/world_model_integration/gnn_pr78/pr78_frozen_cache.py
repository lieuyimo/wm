"""Exact public risk-row cache for a FROZEN original PR61 GNN backbone.

For fixed PR61 weights, the 4-frame public graph / PyG map encoding and
candidate action do NOT change across conditional service-head epochs.
Extracting the original GNN's 6d task predictions once (in eval mode)
and concatenating the original PR73/78 48d currently public service query
is algebraically equivalent to repeating the same frozen GNN forward at
every optimizer step. Train the original MLP service-event head on 54d
risk rows with standard BCE/Adam. Never cache or encode factual t+1 in x.

This cache does not remove the requirement for the original PR61
SHA-verified pretrained checkpoint; random inputs are TEST ONLY.
"""
from __future__ import annotations

from collections import defaultdict

import torch
from torch_geometric.loader import DataLoader


def frozen_public_service_rows(graph_world, scenes, device, *, batch_size=16):
    """Return exact risk-only feature rows, targets and original scene/phase IDs.

    All supervision targets are selected AFTER graph_world computes node
    forecasts from current public inputs. False future observations cannot
    enter feature construction.
    """
    if any(p.requires_grad for p in graph_world.parameters()):
        raise AssertionError("PR61 backbone must be fully frozen for exact caching")
    graph_world.eval()
    features=[]
    labels=[]
    scene_indices=[]
    phase_indices=[]
    per_scene=[]
    with torch.inference_mode():
        for scene_i,scene in enumerate(scenes):
            count=0
            positives=0
            by_phase={"to_pickup":[0,0],"to_dropoff":[0,0]}
            for original in DataLoader(scene["graphs"],batch_size=batch_size,shuffle=False):
                batch=original.to(device)
                next_nodes,_=graph_world(batch)
                mask=batch.service_risk_mask.bool()
                if not torch.equal(mask,batch.service_event_mask.bool()):
                    raise AssertionError("unseen task must not become a service-event negative")
                if next_nodes.shape != (len(mask),6):
                    raise AssertionError("original PR61 node-next head output changed")
                if batch.service_public_query.shape != (len(mask),48):
                    raise AssertionError("existing public 48d query shape changed")
                if batch.service_phase.shape != (len(mask),2):
                    raise AssertionError("phase labels are not bound to task identities")
                x=torch.cat((next_nodes[mask],
                             batch.service_public_query[mask]),dim=1)
                y=batch.service_event_label[mask]
                phase=batch.service_phase[mask]
                if not bool(torch.isfinite(x).all()):
                    raise ValueError("nonfinite original public graph/query features")
                if not bool(((y==0)|(y==1)).all()):
                    raise ValueError("factual t+1 event target not binary")
                if not torch.equal(phase.sum(dim=1),torch.ones(len(y),device=device)):
                    raise AssertionError("each active risk row requires one service phase")
                n=len(y)
                features.append(x.cpu())
                labels.append(y.cpu())
                scene_indices.extend([scene_i]*n)
                phase_indices.extend(phase.argmax(dim=1).cpu().tolist())
                count+=n
                positives+=int(y.sum())
                for index,name in enumerate(("to_pickup","to_dropoff")):
                    subset=phase[:,index].bool()
                    by_phase[name][0]+=int(subset.sum())
                    by_phase[name][1]+=int(y[subset].sum())
            if count != scene["n_risk_rows"]:
                raise AssertionError("cache changed original per-scene public risk counts")
            if not count:
                raise AssertionError("original scene contains no active service risk")
            per_scene.append({
                "scene":scene["id"],"regime":scene["regime"],
                "n":count,"events":positives,
                "by_phase":by_phase,
            })
    if not features:
        raise AssertionError("no true original service risk examples")
    x=torch.cat(features,dim=0)
    y=torch.cat(labels,dim=0)
    if x.shape!=(len(y),54):
        raise AssertionError("public task next-state 6d + service query 48d mismatch")
    if len(scene_indices)!=len(y) or len(phase_indices)!=len(y):
        raise AssertionError("task phase/scene labels detached from risk rows")
    return {
        "features":x,
        "labels":y,
        "scene_ids":torch.tensor(scene_indices,dtype=torch.long),
        "phase_ids":torch.tensor(phase_indices,dtype=torch.long),
        "scenes":per_scene,
        "exact_original_backbone_reused":True,
        "next_tick_only_supervision_not_in_features":True,
    }


def cached_dev_metrics(head,cache,device,batch_size=256):
    """All 12 original scenario metrics and phase Brier on cached true risks."""
    head.eval()
    x=cache["features"]
    y=cache["labels"]
    pred=[]
    with torch.inference_mode():
        for part in x.split(batch_size):
            pred.append(torch.sigmoid(head(part.to(device)).flatten()).cpu())
    probs=torch.cat(pred)
    if probs.shape!=y.shape:
        raise AssertionError("cached head returned wrong risk output shape")
    err=(probs-y).square()
    scenarios=[]
    for scene_i,scene in enumerate(cache["scenes"]):
        mask=cache["scene_ids"]==scene_i
        if not bool(mask.any()) or int(mask.sum())!=scene["n"]:
            raise AssertionError("cached original scene marker lost")
        scenarios.append({
            "scene":scene["scene"],"regime":scene["regime"],
            "n":int(mask.sum()),"events":int(y[mask].sum()),
            "brier":float(err[mask].mean()),
        })
    by_phase={}
    for phase_i,phase in enumerate(("to_pickup","to_dropoff")):
        mask=cache["phase_ids"]==phase_i
        if not bool(mask.any()):
            raise AssertionError("missing original pickup/delivery phase")
        by_phase[phase]={
            "n":int(mask.sum()),"events":int(y[mask].sum()),
            "brier":float(err[mask].mean()),
        }
    return {
        "scene_count":len(scenarios),
        "real_original_eligible_service_risks":len(y),
        "real_next_step_events":int(y.sum()),
        "risk_weighted_brier":float(err.mean()),
        "scene_macro_brier":sum(s["brier"] for s in scenarios)/len(scenarios),
        "phase":by_phase,
        "per_original_scene":scenarios,
    }
