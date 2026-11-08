"""Endpoints para consulta y exploración del catálogo de materias y exámenes."""

from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter, HTTPException, Path, Query, Request

from src.schemas.catalog import DepartmentOut, ExamListResponse, ExamOut
from src.services.catalog_service import get_exam_by_id, list_departments, list_exams

router = APIRouter()


@router.get("/exams", response_model=ExamListResponse, summary="Listar y buscar exámenes")
def get_exams(
    request: Request,
    department_code: Optional[str] = Query(
        None, description="Filtrar por código de departamento (ej. 'AC', 'GM')"
    ),
    search: Optional[str] = Query(
        None, description="Buscar por texto en código o nombre de la materia"
    ),
    limit: int = Query(
        100, ge=1, le=1000, description="Cantidad máxima de exámenes por página"
    ),
    offset: int = Query(
        0, ge=0, description="Desplazamiento de registros para paginación"
    ),
) -> ExamListResponse:
    """Retorna la lista de exámenes que coinciden con los criterios de búsqueda y paginación."""
    catalog = getattr(request.app.state, "catalog", {})
    total, items = list_exams(
        catalog=catalog,
        department_code=department_code,
        search=search,
        limit=limit,
        offset=offset,
    )
    return ExamListResponse(
        total=total,
        limit=limit,
        offset=offset,
        items=[ExamOut(**item) for item in items],
    )


@router.get("/exams/{exam_id}", response_model=ExamOut, summary="Obtener detalle de un examen")
def get_exam(
    request: Request,
    exam_id: str = Path(..., description="Identificador único del examen", examples=["ACC1"]),
) -> ExamOut:
    """Retorna los datos detallados de un examen específico o 404 si no existe."""
    catalog = getattr(request.app.state, "catalog", {})
    exam = get_exam_by_id(catalog=catalog, exam_id=exam_id)
    if exam is None:
        raise HTTPException(
            status_code=404,
            detail=f"Examen con id '{exam_id}' no encontrado en el catálogo.",
        )
    return ExamOut(**exam)


@router.get("/departments", response_model=List[DepartmentOut], summary="Listar departamentos académicos")
def get_departments(
    request: Request,
) -> List[DepartmentOut]:
    """Retorna la lista de todos los departamentos y la cantidad de exámenes asignados a cada uno."""
    catalog = getattr(request.app.state, "catalog", {})
    departments = list_departments(catalog=catalog)
    return [DepartmentOut(**dept) for dept in departments]
