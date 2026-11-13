"""Public relational state/action contract for laboratory sample transport.

The SAME laboratory workflow can use mobile robots and sample-carrier
conveyor/track lines. This is a schema/interface expansion, NOT a learned
conveyor dynamics model, a new planning algorithm or a change to PR61 weights.

Existing PR61 GRU/RGCN/SAGEConv uses robot/task/station/resource with fixed
16-dimensional nodes and grid-map 3d features. Conveyor entities can be
expressed at this boundary but MUST NOT enter existing PR61 encoding until
a separately audited feature/edge encoder, prediction head, legal transition
and factual conveyor dataset are available. Track movement and admission
are not simulated here.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Mapping, Protocol, Any


SCHEMA_VERSION = "public-laboratory-multimodal-sample-world-interface-v2"

# These are PUBLIC entity and graph relation vocabularies, not PyG tensor IDs.
# A domain encoder is responsible for mapping these symbols to the concrete
# input dimensions / relation IDs of its own trained model.
# A single laboratory transport domain supports both AMR and conveyor
# carriers. These are schema vocabularies, not trained PR61 tensor indices.
DOMAIN_NODES = {
    "lab_sample_transport": frozenset({
        "robot", "task", "station", "resource",
        "sample_carrier", "conveyor_section", "track_junction",
        "diverter", "buffer_slot", "transfer_port",
    }),
}
DOMAIN_ACTIONS = {
    "lab_sample_transport": frozenset({
        "assign_robot_task", "hold_assignment",
        "route_carrier", "hold_carrier", "release_buffer",
        "set_diverter", "transfer_to_robot", "transfer_to_conveyor",
    }),
}
DOMAIN_TOPOLOGY = {
    "lab_sample_transport": frozenset({
        "connected_to", "traversable_to", "resource_conflict",
        "feeds_into", "branches_to", "serves_station",
        "buffered_by", "transfers_at",
    }),
}

# Already learned by the original lab-only PR61 backbone. Explicitly
# exclude all new conveyor entities from that old checkpoint.
ORIGINAL_PR61_NODE_KINDS = frozenset({
    "robot", "task", "station", "resource",
})
CONVEYOR_NODE_KINDS = frozenset({
    "sample_carrier", "conveyor_section", "track_junction",
    "diverter", "buffer_slot", "transfer_port",
})



@dataclass(frozen=True)
class PublicField:
    """Current public value, including a public schedule of a *future* time.

    The known_at_tick is when that schedule/measurement was disclosed, not
    the numeric time (which is allowed to refer to a planned future event).
    The private realized future result must NEVER be represented as known
    at current tick.
    """
    value: str | float | int | bool
    known_at_tick: int

    def __post_init__(self):
        if type(self.known_at_tick) is not int or self.known_at_tick < 0:
            raise ValueError("public information timestamp must be a nonnegative integer")
        if type(self.value) not in (str, float, int, bool):
            raise TypeError("only scalar public observable values supported")
        if isinstance(self.value, float) and not math.isfinite(self.value):
            raise ValueError("NaN/inf cannot be a public model input")


@dataclass(frozen=True)
class RelationalEntity:
    entity_id: str
    entity_kind: str
    public_features: Mapping[str, PublicField]

    def __post_init__(self):
        if not self.entity_id or not self.entity_kind:
            raise ValueError("entity identity and kind are required")
        if not self.public_features or any(
            not name or not isinstance(field, PublicField)
            for name, field in self.public_features.items()
        ):
            raise ValueError("an entity needs explicitly timestamped public feature fields")


@dataclass(frozen=True)
class RelationalEdge:
    src: str
    dst: str
    relation: str

    def __post_init__(self):
        if not self.src or not self.dst or not self.relation:
            raise ValueError("typed graph edge identity is required")


@dataclass(frozen=True)
class PublicWorldFrame:
    domain: str
    tick: int
    tick_seconds: float
    entities: tuple[RelationalEntity, ...]
    relations: tuple[RelationalEdge, ...] = ()
    topology: tuple[RelationalEdge, ...] = ()

    def __post_init__(self):
        if self.domain not in DOMAIN_NODES:
            raise ValueError("domain requires an explicitly registered node/action adapter")
        if type(self.tick) is not int or self.tick < 0:
            raise ValueError("public observation tick must be nonnegative")
        if not math.isfinite(self.tick_seconds) or self.tick_seconds <= 0:
            raise ValueError("time unit must be finite and positive")
        ids = [entity.entity_id for entity in self.entities]
        if not ids or len(set(ids)) != len(ids):
            raise ValueError("nonempty public graph requires unique entity IDs")
        if any(e.entity_kind not in DOMAIN_NODES[self.domain]
               for e in self.entities):
            raise ValueError("entity kind does not belong to declared domain")
        if any(v.known_at_tick > self.tick
               for e in self.entities for v in e.public_features.values()):
            raise ValueError("future disclosure or simulator oracle leaked into current observation")
        existing = set(ids)
        if any(edge.src not in existing or edge.dst not in existing
               for edge in (*self.relations, *self.topology)):
            raise ValueError("graph relation references unseen entities")
        if any(edge.relation not in DOMAIN_TOPOLOGY[self.domain]
               for edge in self.topology):
            raise ValueError("topology edge uses unknown domain physical relation")


@dataclass(frozen=True)
class PublicAction:
    domain: str
    decided_at_tick: int
    action_kind: str
    actor_id: str
    target_ids: tuple[str, ...] = ()

    def __post_init__(self):
        if self.domain not in DOMAIN_ACTIONS:
            raise ValueError("unknown action domain")
        if self.action_kind not in DOMAIN_ACTIONS[self.domain]:
            raise ValueError("action kind is not registered for this domain")
        if type(self.decided_at_tick) is not int or self.decided_at_tick < 0:
            raise ValueError("action selection tick invalid")
        if not self.actor_id:
            raise ValueError("action actor identity missing")


@dataclass(frozen=True)
class ActionConditionedWorldQuery:
    """Public graph history, one candidate action, and prediction horizon.

    The model must use only history <= current tick, and may output a
    multi-step DISTRIBUTION under an explicit future policy/exogenous model.
    """
    history: tuple[PublicWorldFrame, ...]
    action: PublicAction
    horizon_ticks: int
    particles: int = 1

    def __post_init__(self):
        if not self.history:
            raise ValueError("world query must have at least one public state")
        current = self.history[-1]
        if any(s.domain != current.domain or s.tick_seconds != current.tick_seconds
               for s in self.history):
            raise ValueError("history mixes unrelated transport domains or clock units")
        ticks = [s.tick for s in self.history]
        if ticks != sorted(set(ticks)):
            raise ValueError("public history must be strictly chronological")
        if current.domain != self.action.domain or current.tick != self.action.decided_at_tick:
            raise ValueError("action must be selected from the current public domain/tick")
        identities = {s.entity_id for s in current.entities}
        if (self.action.actor_id not in identities
                or any(s not in identities for s in self.action.target_ids)):
            raise ValueError("action cannot reference unobserved entities")
        if type(self.horizon_ticks) is not int or self.horizon_ticks < 1:
            raise ValueError("prediction horizon must be positive integer ticks")
        if type(self.particles) is not int or self.particles < 1:
            raise ValueError("particle count must be positive integer")


@dataclass(frozen=True)
class WorldModelReadiness:
    domain: str
    public_graph_contract: bool
    pretrained_graph_weights: bool
    domain_encoder: bool
    domain_prediction_heads: bool
    legal_transition: bool
    factual_prediction_validated: bool
    factual_multistep_validated: bool = False
    decision_value_validated: bool = False
    notes: str = ""

    @property
    def supports_trained_one_step_forecasting(self) -> bool:
        return all((
            self.public_graph_contract,
            self.pretrained_graph_weights,
            self.domain_encoder,
            self.domain_prediction_heads,
            self.legal_transition,
            self.factual_prediction_validated,
        ))

    @property
    def supports_verified_decision_rollout(self) -> bool:
        return (
            self.supports_trained_one_step_forecasting
            and self.factual_multistep_validated
            and self.decision_value_validated
        )


# This is an honest evidence ledger for WHAT IS ALREADY PRESENT on PR76.
# PR61 trained weights exist in a separately SHA-controlled local archive,
# but original lab GNN omitted public service variables shown in PR72.
LAB_READINESS = WorldModelReadiness(
    domain="lab_sample_transport", public_graph_contract=True,
    pretrained_graph_weights=True, domain_encoder=True,
    domain_prediction_heads=True, legal_transition=True,
    factual_prediction_validated=True,
    notes=(
        "PR61 actual pretrained map-on/off exists for the robot-only lab slice. "
        "and status heads were evaluated on developer scenes. PR72 found omitted "
        "effective service progress; PR76 did not establish reliable factual H20 "
        "benefit. Do not equate presence of trained weights with operational readiness."
    ),
)
CONVEYOR_READINESS = WorldModelReadiness(
    domain="lab_sample_transport.conveyor_extension",
    public_graph_contract=True, pretrained_graph_weights=False,
    domain_encoder=False, domain_prediction_heads=False,
    legal_transition=False, factual_prediction_validated=False,
    notes=(
        "Same sample/task workflow and public state-action graph can represent "
        "pucks, conveyor sections, switches, buffers and robot-conveyor handoffs. "
        "The original PR61 trained 16d entity encoder and 3d grid topology do "
        "not encode these new nodes. No conveyor simulator/actuation, trained "
        "checkpoint, sample handoff safety or factual conveyor evaluation yet."
    ),
)



class DomainWorldModelAdapter(Protocol):
    """Extension point for existing GNN encoders and domain-specific dynamics.

    An adapter MUST specify an audited encoding from PublicWorldFrame to its
    concrete PyG tensors, a state/task-specific predictive head, and verified
    legal projection. Merely satisfying this typing protocol is not proof of
    operational or safety validity.
    """
    domain: str

    def encode(self, query: ActionConditionedWorldQuery) -> Any: ...

    def predict_step(self, encoded: Any, *, seed: int) -> Any: ...

    def project_legal(
        self, query: ActionConditionedWorldQuery, predicted: Any
    ) -> PublicWorldFrame: ...


def require_trained_forecasting(readiness: WorldModelReadiness) -> None:
    if not readiness.supports_trained_one_step_forecasting:
        raise NotImplementedError(
            f"{readiness.domain}: trained/validated domain adapter not available; "
            "a graph schema alone is NOT a working world model"
        )
