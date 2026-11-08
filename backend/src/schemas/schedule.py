"""Esquemas Pydantic para la generación y validación de cronogramas de exámenes."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

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


class ScheduleEvaluationResponse(BaseModel):
    """Métricas de evaluación del cronograma contra los conflictos reales observados en 1994."""

    threshold: float = Field(..., description="Umbral de probabilidad aplicado", examples=[0.20])
    strategy: str = Field(..., description="Estrategia de coloreo utilizada", examples=["largest_first"])
    turns_count: int = Field(..., description="Cantidad total de turnos de examen asignados (turnos)", examples=[11])
    max_allowed_turns: int = Field(default=32, description="Límite máximo de turnos en el calendario (32)")
    fits_in_calendar: bool = Field(..., description="Indica si turns_count <= max_allowed_turns (cabe_en_calendario)")
    is_valid_coloring: bool = Field(..., description="Garantiza formalmente que ningún par en conflicto predicho comparte turno")

    # Métricas de choques reales de 1994 (incidencias de conflicto estudiante-par)
    real_conflict_pairs: int = Field(
        ...,
        description="Total de pares de materias no dirigidos con choques reales en 1994 (pares_reales)",
        examples=[4468],
    )
    clashing_pairs: int = Field(
        ...,
        description="Pares reales de materias que quedaron asignados al mismo turno (pares_reales_en_choque)",
        examples=[830],
    )
    total_affected_students: int = Field(
        ...,
        description="Total acumulado de incidencias potenciales de conflicto estudiante-par en 1994 (suma de estudiantes compartidos en todos los pares reales)",
        examples=[28677],
    )
    clashing_students: int = Field(
        ...,
        description="Total acumulado de incidencias de conflicto estudiante-par que permanecen en el mismo turno (penalización agregada de solapamiento)",
        examples=[3723],
    )
    clash_student_percentage: float = Field(
        ...,
        description="Proporción de incidencias de conflicto estudiante-par que permanecen en el cronograma respecto al peor caso teórico (12.98%)",
        examples=[12.98],
    )

    # Métricas de personas únicas (headcount)
    unique_students_affected: int = Field(
        ...,
        description="Cantidad de personas únicas de la cohorte con al menos un conflicto de examen en el cronograma",
        examples=[1945],
    )
    unique_students_percentage: float = Field(
        ...,
        description="Porcentaje de personas únicas afectadas sobre el total de estudiantes de la cohorte evaluada (1945 / 3086 = 63.03%)",
        examples=[63.03],
    )

    # Calidad y matriz de confusión de aristas del grafo predicho
    predicted_edges: int = Field(..., description="Cantidad de aristas predichas en el grafo (aristas_predichas)", examples=[2032])
    true_positives: int = Field(..., description="Aristas predichas que efectivamente tuvieron conflicto real (TP)", examples=[759])
    false_positives: int = Field(..., description="Aristas predichas que no tuvieron conflicto real (FP)", examples=[1273])
    false_negatives: int = Field(..., description="Conflictos reales que el modelo no predijo (FN)", examples=[3709])
    edge_precision: float = Field(..., description="Precisión de las aristas predichas (precision_aristas = TP / (TP + FP))", examples=[0.3735])
    edge_recall: float = Field(..., description="Exhaustividad de las aristas predichas (recall_aristas = TP / (TP + FN))", examples=[0.1699])

    raw_metrics: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Diccionario crudo de métricas retornado directamente por evaluate_against_real",
    )
