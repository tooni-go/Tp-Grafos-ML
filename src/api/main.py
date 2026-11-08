"""Punto de entrada de la aplicación FastAPI."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Dict

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.v1.router import api_router
from src.core.config import settings
from src.graphs.conflict_graph import load_exams_catalog


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Ciclo de vida de la aplicación: inicializa recursos en memoria durante el startup."""
    if settings.EXAMS_CLEAN_PATH.exists():
        app.state.catalog = load_exams_catalog(settings.EXAMS_CLEAN_PATH)
    else:
        app.state.catalog = {}
    yield
    # Limpieza durante el shutdown
    if hasattr(app.state, "catalog"):
        app.state.catalog.clear()


def create_app() -> FastAPI:
    """Fábrica de la aplicación FastAPI con middlewares, rutas y lifespan configurados."""
    app = FastAPI(
        title=settings.PROJECT_NAME,
        version=settings.VERSION,
        openapi_url=f"{settings.API_V1_STR}/openapi.json",
        docs_url=f"{settings.API_V1_STR}/docs",
        redoc_url=f"{settings.API_V1_STR}/redoc",
        lifespan=lifespan,
    )

    # Middleware para intercambio de recursos de origen cruzado (CORS)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Registro de rutas bajo el prefijo /api/v1
    app.include_router(api_router, prefix=settings.API_V1_STR)

    @app.get("/", include_in_schema=False)
    def root() -> Dict[str, Any]:
        return {
            "message": settings.PROJECT_NAME,
            "version": settings.VERSION,
            "docs": f"{settings.API_V1_STR}/docs",
        }

    return app


app = create_app()
