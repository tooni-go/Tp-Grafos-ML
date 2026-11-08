"""Configuración global y variables de entorno del backend."""

from __future__ import annotations

import os
from pathlib import Path
from typing import List

from src.common.schema import PROJECT_ROOT


class Settings:
    """Ajustes de configuración del servidor y rutas del proyecto."""

    PROJECT_NAME: str = "API de Asignación de Turnos de Exámenes"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"

    # Orígenes CORS permitidos para desarrollo e integración con frontend SPA
    @property
    def cors_origins(self) -> List[str]:
        raw = os.getenv("BACKEND_CORS_ORIGINS", "")
        if raw.strip():
            return [origin.strip() for origin in raw.split(",") if origin.strip()]
        return ["*"]

    # Rutas canónicas a datasets y artefactos
    EXAMS_CLEAN_PATH: Path = PROJECT_ROOT / "data" / "interim" / "exams_clean.csv"
    STUDENTS_CLEAN_PATH: Path = PROJECT_ROOT / "data" / "interim" / "students_clean.csv"
    ENROLMENTS_CLEAN_PATH: Path = PROJECT_ROOT / "data" / "interim" / "enrolments_clean.csv"
    SUBJECT_PAIRS_PATH: Path = PROJECT_ROOT / "data" / "processed" / "subject_pairs_poc.csv"
    PAIR_PROBABILITIES_PATH: Path = PROJECT_ROOT / "results" / "predictions" / "pair_probabilities_oof.csv"
    METRICS_SUMMARY_PATH: Path = PROJECT_ROOT / "results" / "metrics" / "metrics_summary.json"


settings = Settings()
