"""Esquemas Pydantic para la generación y validación de cronogramas de exámenes."""

from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field

from src.schemas.catalog import ExamOut


class ScheduleRequest(BaseModel):
    """Solicitud de generación de cronograma mediante coloreo de grafos."""

    threshold: Optional[float] = Field(
        default=None,
        gt=0.0,
        lt=1.0,
        description="Umbral de probabilidad de conflicto en (0, 1). Por defecto: calibrado oficial (0.20)",
        examples=[0.20],
    )
    strategy: str = Field(
        default="largest_first",
        description="Heurística de coloreo greedy (ej. 'largest_first', 'saturation_largest_first')",
        examples=["largest_first"],
    )


class ScheduleResponse(BaseModel):
    """Respuesta con el cronograma de turnos asignados y validación formal."""

    threshold: float = Field(..., description="Umbral de probabilidad aplicado", examples=[0.20])
    strategy: str = Field(..., description="Estrategia de coloreo utilizada", examples=["largest_first"])
    turns_count: int = Field(..., description="Cantidad total de turnos de examen asignados", examples=[11])
    max_allowed_turns: int = Field(default=32, description="Límite máximo de turnos en el calendario (32)")
    fits_in_calendar: bool = Field(..., description="Indica si turns_count <= max_allowed_turns (32 turnos)")
    is_valid_coloring: bool = Field(..., description="Garantiza formalmente que ningún par en conflicto comparte turno")
    total_exams_scheduled: int = Field(..., description="Total de materias asignadas en el cronograma", examples=[800])
    turn_distribution: Dict[str, int] = Field(..., description="Cantidad de exámenes asignados a cada turno")
    schedule: Dict[str, List[ExamOut]] = Field(
        ..., description="Asignación completa de materias agrupadas por turno ('0', '1', ...)"
    )


class ScheduleStrategyOut(BaseModel):
    """Descripción de una estrategia o heurística de coloreo disponible."""

    strategy: str = Field(..., description="Identificador técnico de la estrategia", examples=["largest_first"])
    name: str = Field(..., description="Nombre formal del algoritmo", examples=["Welsh-Powell (Largest First)"])
    description: str = Field(..., description="Criterio de ordenamiento de los vértices")
