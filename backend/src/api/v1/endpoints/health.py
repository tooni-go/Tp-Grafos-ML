"""Endpoint de verificación de estado y salud del backend."""

from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter, Request

from src.core.config import settings

router = APIRouter()


@router.get("/health", summary="Verificar estado del servicio")
def get_health(request: Request) -> Dict[str, Any]:
    """Retorna el estado operativo básico, versión y conteo del catálogo cargado."""
    catalog = getattr(request.app.state, "catalog", {})
    return {
        "status": "healthy",
        "version": settings.VERSION,
        "project_name": settings.PROJECT_NAME,
        "catalog_exams_count": len(catalog),
    }
