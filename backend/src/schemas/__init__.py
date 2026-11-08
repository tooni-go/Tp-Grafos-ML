"""Módulo de esquemas de datos Pydantic para validación y serialización."""

from src.schemas.catalog import DepartmentOut, ExamListResponse, ExamOut
from src.schemas.graph import ConflictPairResponse, GraphEdgeOut, GraphResponse
from src.schemas.schedule import (
    ScheduleRequest,
    ScheduleResponse,
    ScheduleStrategyOut,
)

__all__ = [
    "ConflictPairResponse",
    "DepartmentOut",
    "ExamListResponse",
    "ExamOut",
    "GraphEdgeOut",
    "GraphResponse",
    "ScheduleRequest",
    "ScheduleResponse",
    "ScheduleStrategyOut",
]
