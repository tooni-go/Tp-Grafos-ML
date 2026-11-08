"""Endpoint de verificación de estado y salud del backend."""

from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter

from src.core.config import settings

router = APIRouter()


@router.get("/health", summary="Verificar estado del servicio")
def get_health() -> Dict[str, Any]:
    """Retorna el estado operativo básico y la versión de la API."""
    return {
        "status": "healthy",
        "version": settings.VERSION,
        "project_name": settings.PROJECT_NAME,
    }
