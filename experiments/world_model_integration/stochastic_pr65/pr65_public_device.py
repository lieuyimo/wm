"""PR65 minimal public-history exogenous device kernel for the existing world model.

Reference: Suresh et al. (2022), discrete-time conditional hazard.
All fitted probabilities are supplied by PR64 TRAIN-only hazard estimation.
This adapter modifies the station availability *probability* output of the
existing action-conditioned graph predictor; it is NOT a simulator, legal
world-state recursion, pretrained checkpoint, or an event-time oracle.

No labels, private outage schedule, future task releases, or absolute clock
are read by this inference adapter. A sampled device bit is an EVENT PROPOSAL
only: never substitute it for an executed BusinessSimulator transition.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Mapping

DEVICE_INDEX = 6  # PR51 16-dimensional public node feature contract
PUBLIC_INPUT_KEYS = frozenset({
    "public_tick", "graph", "history", "history_visible_mask",
    "node_persistence", "feature_last_observed_tick", "input_source",
    "public_map",
})
ALLOWED_METHODS = ("constant", "age_h4")


@dataclass(frozen=True)
class DeviceProbability:
    index: int
    station: str
    current_available: int
    public_history_age_h4: int
    probability_next_available: float

    @property
    def probability_transition(self) -> float:
        p = self.probability_next_available
        return 1.0 - p if self.current_available else p


def visible_age_h4(history, mask, current: int) -> int:
    """Identical public t-3..t 4-frame age definition to PR63 risk-set reader."""
    if len(history) != 4 or len(mask) != 4:
        raise ValueError("PR54 requires four visible-history slots")
    if not mask[-1] or abs(float(history[-1][DEVICE_INDEX])-current) > 1e-6:
        raise ValueError("last observed public station state must match current")
    age = 1
    for k in (2, 1, 0):
        if not mask[k] or abs(float(history[k][DEVICE_INDEX])-current) > 1e-6:
            break
        age += 1
    return age


class PublicStationHazardAdapter:
    """Non-trainable probabilistic device extension of a relational world output.

    Model input is the current PR64 public observation ONLY. The fitted tables
    must be derived exclusively from TRAIN via pr64_gate.fit.
    """

    def __init__(self, train_fitted_params: Mapping, method: str = "constant"):
        if method not in ALLOWED_METHODS:
            raise ValueError("clock/exact-age/hidden features not admitted")
        if train_fitted_params.get("fit_only_on_train") is not True:
            raise ValueError("unverified TRAIN-only hazard table")
        for risk in ("up", "down"):
            row = train_fitted_params.get("constant", {}).get(risk, {})
            p = row.get("hazard")
            if not isinstance(p, (int, float)) or not math.isfinite(p) or not 0 <= p <= 1:
                raise ValueError("missing or invalid conditional constant hazard")
        self.params = train_fitted_params
        self.method = method

    def station_probabilities(self, public_input: Mapping) -> list[DeviceProbability]:
        if set(public_input) != PUBLIC_INPUT_KEYS:
            raise ValueError("not the registered PR64 public input contract")
        if public_input["feature_last_observed_tick"] != public_input["public_tick"]:
            raise ValueError("input observed after its declared t")
        nodes = public_input["graph"]["nodes"]
        history = public_input["history"]
        visible = public_input["history_visible_mask"]
        if len(nodes) != len(history) or len(nodes) != len(visible):
            raise ValueError("misaligned public station history")
        result = []
        for i, node in enumerate(nodes):
            if node["kind"] != "station":
                continue
            current = int(bool(node["device_available"]))
            age = visible_age_h4(history[i], visible[i], current)
            risk = "up" if current else "down"
            constant = float(self.params["constant"][risk]["hazard"])
            if self.method == "age_h4":
                bucket = str(age)
                hazard = self.params["age_h4"][risk].get(bucket, {}).get("hazard", constant)
            else:
                hazard = constant
            if not math.isfinite(float(hazard)) or not 0 <= hazard <= 1:
                raise ValueError("invalid TRAIN-fitted hazard probability")
            p_next = 1.0 - float(hazard) if current else float(hazard)
            result.append(DeviceProbability(
                i, str(node["id"]), current, age, p_next))
        if not result:
            raise ValueError("no public station entities")
        return result

    def overlay_one_step(self, public_input: Mapping, node_next):
        """Keep every PR61 learned entity/event output except station device bit.

        node_next is [num_entities,6] tensor from PublicMapTemporalRelationalWorld.
        Its station field 0 is device_available. No hidden label is accessed.
        """
        station_rows = self.station_probabilities(public_input)
        if len(node_next.shape) != 2 or node_next.shape[0] != len(public_input["graph"]["nodes"]) or node_next.shape[1] != 6:
            raise ValueError("PR61 masked node-next output has incompatible shape")
        merged = node_next.clone()
        for r in station_rows:
            merged[r.index, 0] = r.probability_next_available
        return merged

    def sample_device_event_proposals(self, public_input: Mapping, *, seed: int,
                                      particles: int) -> list[dict[str, int]]:
        """Seeded Bernoulli proposals; NOT a legal future state or H20 rollout."""
        if not isinstance(particles, int) or particles < 1:
            raise ValueError("positive particle count required")
        rng = random.Random(seed)
        rows = self.station_probabilities(public_input)
        return [{r.station: int(rng.random() < r.probability_next_available)
                 for r in rows} for _ in range(particles)]
