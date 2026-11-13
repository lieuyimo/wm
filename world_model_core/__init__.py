"""Common sample-world state/action contract for AGVs and conveyor carriers.

Existing PR61 robot GNN inference is unchanged. Conveyor nodes require their
own verified input encoding and legal dynamics before trained inference.
"""
from .contracts import (
    ActionConditionedWorldQuery, DomainWorldModelAdapter,
    LAB_READINESS, CONVEYOR_READINESS, PublicAction, PublicField,
    PublicWorldFrame, RelationalEdge, RelationalEntity,
    WorldModelReadiness, require_trained_forecasting,
)

__all__ = (
    "ActionConditionedWorldQuery", "DomainWorldModelAdapter",
    "LAB_READINESS", "CONVEYOR_READINESS", "PublicAction", "PublicField",
    "PublicWorldFrame", "RelationalEdge", "RelationalEntity",
    "WorldModelReadiness", "require_trained_forecasting",
)
