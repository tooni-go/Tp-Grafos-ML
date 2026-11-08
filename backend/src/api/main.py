"""Punto de entrada de la aplicación FastAPI."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Dict

import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.v1.router import api_router
from src.core.config import settings
from src.graphs.conflict_graph import (
    load_exams_catalog,
    load_student_exams_1994,
)
from src.graphs.predicted_pairs import default_threshold


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Ciclo de vida de la aplicación: inicializa recursos en memoria durante el startup."""
    # 1. Carga del catálogo de exámenes
    if settings.EXAMS_CLEAN_PATH.exists():
        app.state.catalog = load_exams_catalog(settings.EXAMS_CLEAN_PATH)
    else:
        app.state.catalog = {}

    # 2. Umbral oficial calibrado desde metrics_summary.json
    try:
        app.state.default_threshold = default_threshold()
    except Exception:
        app.state.default_threshold = 0.20

    # 3. Carga e indexación de la matriz de probabilidades de pares OOF
    if settings.PAIR_PROBABILITIES_PATH.exists():
        df = pd.read_csv(settings.PAIR_PROBABILITIES_PATH)
        app.state.probabilities_df = df
        app.state.pairs_prob_map = dict(
            zip(zip(df["materia_id_1"], df["materia_id_2"]), df["prob_conflicto"])
        )
    else:
        app.state.probabilities_df = None
        app.state.pairs_prob_map = {}

    # 4. Carga del dataset de pares reales para evaluación vs cohorte 1994
    if settings.SUBJECT_PAIRS_PATH.exists():
        app.state.real_pairs_df = pd.read_csv(
            settings.SUBJECT_PAIRS_PATH,
            usecols=["materia_id_1", "materia_id_2", "cant_estudiantes_afectados"],
        )
    else:
        app.state.real_pairs_df = None

    # 5. Carga de mapa de inscripciones de estudiantes 1994 para cálculo de alumnos únicos afectados
    if settings.STUDENTS_CLEAN_PATH.exists() and settings.ENROLMENTS_CLEAN_PATH.exists():
        app.state.student_exams_1994 = load_student_exams_1994(
            settings.STUDENTS_CLEAN_PATH,
            settings.ENROLMENTS_CLEAN_PATH,
        )
    else:
        app.state.student_exams_1994 = {}

    yield

    # Limpieza durante el shutdown
    if hasattr(app.state, "catalog"):
        app.state.catalog.clear()
    if hasattr(app.state, "pairs_prob_map"):
        app.state.pairs_prob_map.clear()
    if hasattr(app.state, "student_exams_1994"):
        app.state.student_exams_1994.clear()
    app.state.probabilities_df = None
    app.state.real_pairs_df = None


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
