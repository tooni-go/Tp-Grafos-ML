"""Esquemas Pydantic para el catálogo de exámenes y departamentos."""

from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field


class ExamOut(BaseModel):
    """Representación de un examen individual del catálogo."""

    exam_id: str = Field(..., description="Identificador único del examen", examples=["ACC1"])
    exam_name: str = Field(..., description="Nombre o título de la materia", examples=["ACCOUNTING 1"])
    department_code: str = Field(..., description="Código de departamento académico", examples=["AC"])
    duration: str = Field(..., description="Duración en formato texto (HH:MM)", examples=["2:00"])
    duration_minutes: int = Field(..., description="Duración total en minutos", examples=[120])


class ExamListResponse(BaseModel):
    """Respuesta paginada del listado de exámenes."""

    total: int = Field(..., description="Cantidad total de exámenes que cumplen con el filtro")
    limit: int = Field(..., description="Límite de registros solicitados por página")
    offset: int = Field(..., description="Desplazamiento inicial de la paginación")
    items: List[ExamOut] = Field(..., description="Lista de exámenes correspondientes a la página")


class DepartmentOut(BaseModel):
    """Resumen de un departamento académico."""

    department_code: str = Field(..., description="Código del departamento", examples=["AC"])
    exams_count: int = Field(..., description="Cantidad de exámenes asignados a este departamento", examples=[12])
