"""Shared public relational state/action contract for transport world models.

This is the common *interface*, not a newly trained model. Use existing
PR61 GRU/RGCN/SAGEConv + legal LAB transition for sample transport; railway
prediction requires separately validated domain encoding and safety rules.
"""
from .contracts import (
    ActionConditionedWorldQuery, DomainWorldModelAdapter,
    LAB_READINESS, RAIL_READINESS, PublicAction, PublicField,
    PublicWorldFrame, RelationalEdge, RelationalEntity,
    WorldModelReadiness, require_trained_forecasting,
)

__all__ = (
    "ActionConditionedWorldQuery", "DomainWorldModelAdapter",
    "LAB_READINESS", "RAIL_READINESS", "PublicAction", "PublicField",
    "PublicWorldFrame", "RelationalEdge", "RelationalEntity",
    "WorldModelReadiness", "require_trained_forecasting",
)
