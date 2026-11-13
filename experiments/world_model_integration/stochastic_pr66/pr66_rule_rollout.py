"""PR66 rule-constrained stochastic exogenous device boundary for business worlds.

Discrete hazard: Suresh et al. (2022). Stochastic trajectory propagation:
Chua et al. (2018, PETS), conceptual reference only; this is a CPU RULE
rollout, not an implementation of PETS or a trained PR61 neural H20 rollout.

Timeline:
  1. Observe public state x_t and propose a legal assignment a_t.
  2. Evolve [t,t+1) under current *observed* device_t using the existing
     WorkflowDynamics(ProgressAwareRule) and full business legality checker.
  3. Sample exogenous device_{t+1} from TRAIN-only public conditional risk.
  4. Apply that event at the NEXT observation boundary and recompute
     workstation capacity/active-session observables. Service at step t is
     not retroactively changed by an outage first seen at t+1.
  5. Subsequent steps see the new device status and can pause service;
     sequential workflow successors remain governed by original rules.

No future simulator outage schedule, unknown jobs or target labels are inputs.
"""
from __future__ import annotations

from dataclasses import dataclass, fields, replace
import math
import random

from business_transport.dynamics import WorkflowDynamics
from business_transport.observation import capacity_view
from business_transport.workflow_transition import check_business_transition
from lab_transport.state_v2 import rebuild
from lab_transport.torch_structured_adapter import ProgressAwareRule

from experiments.world_model_integration.stochastic_pr65.pr65_public_device import (
    DEVICE_INDEX, PublicStationHazardAdapter, visible_age_h4,
)

DERIVED_STATION_FIELDS = (
    "occupant_robot_id", "service_task_id", "service_session_active",
    "service_in_progress", "waiting_pickups", "approaching_pickups",
    "approaching_dropoffs",
)
EVENT_CHANGE_FIELDS = frozenset((
    "available", "device_available", "capacity_blocked", "resource_occupied",
) + DERIVED_STATION_FIELDS)


def apply_next_device_boundary(next_rule_state, next_device: dict[str, bool], rules):
    """Apply exogenous device changes after lawful business t->t+1 transition.

    Recompute resource admission and visible service relations using the
    original capacity_view + rebuild functions. No robot or sample is edited.
    """
    stations = {s.id: s for s in next_rule_state.stations}
    if set(stations) != set(next_device) or any(type(x) is not bool for x in next_device.values()):
        raise ValueError("event must supply a boolean next state for each existing public station")
    pre, _ = capacity_view(next_rule_state, (), rules, device_status=next_device)
    derived = rebuild(pre, pre.robots)
    by_id = {s.id: s for s in derived.stations}
    result = replace(pre, stations=tuple(
        replace(s, **{key: getattr(by_id[s.id], key) for key in DERIVED_STATION_FIELDS})
        for s in pre.stations
    ))
    # The event boundary is not permitted to alter entity identity, robot
    # phase/progress, task lifecycle, or any station's fixed characteristics.
    if result.robots != next_rule_state.robots or result.samples != next_rule_state.samples:
        raise AssertionError("external device event illegally changed robots or tasks")
    for old, new in zip(next_rule_state.stations, result.stations):
        if old.id != new.id or any(
            getattr(old, key.name) != getattr(new, key.name)
            for key in fields(old) if key.name not in EVENT_CHANGE_FIELDS
        ):
            raise AssertionError("external event modified non-device station structure")
    recalculated, _ = capacity_view(result, (), rules)
    for actual, expected in zip(result.stations, recalculated.stations):
        if any(getattr(actual, name) != getattr(expected, name) for name in (
            "device_available", "available", "resource_occupied", "capacity_blocked"
        )):
            raise AssertionError("exogenous event produced invalid capacity observation")
        if actual.service_in_progress != bool(actual.service_session_active and actual.available):
            raise AssertionError("service continued despite unavailable device")
        if actual.resource_occupied > actual.resource_capacity:
            raise AssertionError("station pool was over capacity")
    return result


def lawful_rule_then_event(before, assignments, *, next_device, geometry, rules):
    """One legal business tick followed by *observed next-boundary* device events.

    check_business_transition intentionally forbids unknown future device
    changes. Therefore check the lawful rule step BEFORE adding the external
    boundary event; do not weaken or bypass the original checker itself.
    """
    actions = tuple(assignments)
    base = WorkflowDynamics(ProgressAwareRule(geometry), rules)
    core = base.predict_next(before, actions)
    check_business_transition(before, core, geometry, rules, actions)
    proposed = apply_next_device_boundary(core, next_device, rules)
    if proposed.tick != before.tick + 1:
        raise AssertionError("event boundary changed the world clock")
    return proposed


def initial_public_station_history(public_input, state):
    """Extract exactly the PR54 t-3..t visible device bits, never future ticks."""
    # Also validates the complete PR65 public-input contract and t boundary.
    adapter_input_checker = PublicStationHazardAdapter({
        "fit_only_on_train": True,
        "constant": {
            "up": {"hazard": 0.5}, "down": {"hazard": 0.5},
        },
    })
    adapter_input_checker.station_probabilities(public_input)
    if state.tick != public_input["public_tick"]:
        raise ValueError("model public input and business observation do not share t")
    nodes = public_input["graph"]["nodes"]
    histories = public_input["history"]
    masks = public_input["history_visible_mask"]
    current = {s.id: int(bool(s.device_available)) for s in state.stations}
    out = {}
    for i, node in enumerate(nodes):
        if node["kind"] != "station":
            continue
        sid = node["id"]
        if sid not in current or int(bool(node["device_available"])) != current[sid]:
            raise ValueError("public graph device status differs from observed business station")
        visible_age_h4(histories[i], masks[i], current[sid])
        out[sid] = [
            int(round(float(v[DEVICE_INDEX]))) if m else None
            for v, m in zip(histories[i], masks[i])
        ]
    if set(out) != set(current):
        raise ValueError("public graph and business observation station sets differ")
    return out


@dataclass(frozen=True)
class StepRiskRecord:
    tick: int
    probabilities: tuple[tuple[str, float], ...]
    next_devices: tuple[tuple[str, bool], ...]
    changed_stations: tuple[str, ...]
    active_services_before: int
    active_services_after: int
    completed_stages_after: int


class PublicHazardRuleRollout:
    """Genuine legal CPU rule-state recursion, not PR61 pretrained inference.

    Uses only PR64 TRAIN-fitted hazard tables from PR65 adapter. For age_h4,
    initial full public-history observations are required; predicted future
    history is advanced using the model's own sampled next device events.
    """

    def __init__(self, geometry, rules, train_fitted_params, method="constant"):
        self.geometry = geometry
        self.rules = rules
        self.adapter = PublicStationHazardAdapter(train_fitted_params, method)
        self.method = method

    def _probabilities(self, state, history):
        table = self.adapter.params
        result = {}
        for s in sorted(state.stations, key=lambda s: s.id):
            current = int(bool(s.device_available))
            if history is not None:
                seq = history[s.id]
                if len(seq) != 4 or seq[-1] != current:
                    raise ValueError("recursive public device history is inconsistent")
                age = 1
                for v in reversed(seq[:-1]):
                    if v is None or v != current:
                        break
                    age += 1
            else:
                age = 1
            risk = "up" if current else "down"
            constant = float(table["constant"][risk]["hazard"])
            hazard = (table["age_h4"][risk].get(str(age), {}).get("hazard", constant)
                      if self.method == "age_h4" else constant)
            if not math.isfinite(float(hazard)) or not 0 <= hazard <= 1:
                raise ValueError("unregistered public-history risk")
            result[s.id] = 1.0 - float(hazard) if current else float(hazard)
        return result

    def rollout(self, state, actions_fn, *, horizon: int, seed: int,
                initial_public_input=None):
        if not isinstance(horizon, int) or not 1 <= horizon <= 20:
            raise ValueError("CPU legal pilot supports H1 to H20 only")
        if self.method == "age_h4" and initial_public_input is None:
            raise ValueError("no age-h4 approximation from a single snapshot")
        history = (initial_public_station_history(initial_public_input, state)
                   if initial_public_input is not None else None)
        rng = random.Random(seed)
        trace = []
        records = []
        for step in range(horizon):
            actions = tuple(actions_fn(state, step))
            probabilities = self._probabilities(state, history)
            if step == 0 and initial_public_input is not None:
                reference = self.adapter.station_probabilities(initial_public_input)
                if any(abs(probabilities[r.station] - r.probability_next_available) > 1e-12
                       for r in reference):
                    raise AssertionError("first-step probability changed from PR65 public contract")
            # Use a reproducible station-ID order and common random numbers.
            next_devices = {sid: rng.random() < p for sid, p in probabilities.items()}
            nxt = lawful_rule_then_event(state, actions, next_device=next_devices,
                                         geometry=self.geometry, rules=self.rules)
            before_device = {s.id: bool(s.device_available) for s in state.stations}
            records.append(StepRiskRecord(
                tick=state.tick,
                probabilities=tuple(probabilities.items()),
                next_devices=tuple(next_devices.items()),
                changed_stations=tuple(sid for sid in next_devices
                                       if next_devices[sid] != before_device[sid]),
                active_services_before=sum(s.service_in_progress for s in state.stations),
                active_services_after=sum(s.service_in_progress for s in nxt.stations),
                completed_stages_after=sum(s.status == "completed" for s in nxt.samples),
            ))
            if history is not None:
                for sid, status in next_devices.items():
                    history[sid] = history[sid][1:] + [int(status)]
            trace.append(nxt)
            state = nxt
        return tuple(trace), tuple(records)
