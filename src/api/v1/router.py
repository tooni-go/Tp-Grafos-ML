"""Agregador principal de rutas de la API v1."""

from __future__ import annotations

from fastapi import APIRouter

from src.api.v1.endpoints import health

api_router = APIRouter()
api_router.include_router(health.router, tags=["Health"])
