"""Servicio de consultas sobre el catálogo de exámenes en memoria."""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Optional, Tuple


def list_exams(
    catalog: Dict[str, Dict[str, Any]],
    department_code: Optional[str] = None,
    search: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
) -> Tuple[int, List[Dict[str, Any]]]:
    """Filtra, busca y pagina los exámenes del catálogo precargado en memoria."""
    items = list(catalog.values())

    # Filtro opcional por código de departamento
    if department_code is not None and department_code.strip():
        code_clean = department_code.strip().upper()
        items = [exam for exam in items if exam.get("department_code", "").upper() == code_clean]

    # Filtro opcional de búsqueda textual por ID o nombre de examen
    if search is not None and search.strip():
        term = search.strip().lower()
        items = [
            exam
            for exam in items
            if term in exam.get("exam_id", "").lower() or term in exam.get("exam_name", "").lower()
        ]

    # Ordenamiento determinista por ID de examen
    items.sort(key=lambda x: x["exam_id"])
    total = len(items)

    # Paginación
    paginated_items = items[offset : offset + limit]
    return total, paginated_items


def get_exam_by_id(
    catalog: Dict[str, Dict[str, Any]],
    exam_id: str,
) -> Optional[Dict[str, Any]]:
    """Obtiene los detalles de un examen por su ID único."""
    return catalog.get(exam_id.strip())


def list_departments(
    catalog: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Agrupa y cuenta los exámenes por cada departamento académico disponible."""
    counts = Counter(exam.get("department_code", "") for exam in catalog.values())
    departments = [
        {"department_code": dept or "UNKNOWN", "exams_count": count}
        for dept, count in counts.items()
    ]
    departments.sort(key=lambda x: x["department_code"])
    return departments
