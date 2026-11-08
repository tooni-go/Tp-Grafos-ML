"""Agregador principal de rutas de la API v1."""

from __future__ import annotations

from fastapi import APIRouter

from src.api.v1.endpoints import catalog, graph, health, schedule

api_router = APIRouter()
api_router.include_router(health.router, tags=["Health"])
api_router.include_router(catalog.router, tags=["Catálogo de Exámenes"])
api_router.include_router(graph.router, tags=["Grafo de Conflictos"])
api_router.include_router(schedule.router, tags=["Cronograma y Coloreo"])
