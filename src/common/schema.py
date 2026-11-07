"""Esquema centralizado de columnas, rutas base y validaciones de tipos para Machine Learning.

Este módulo actúa como fuente única de verdad (SSOT) entre:
- `src/transformation/prepare_ml_dataset.py` (Fase 2.5)
- `src/models/compare_models.py` (Fase 3: entrenamiento y evaluación)
- `src/inference/predict_conflicts.py` (Fase 3 -> 4: inferencia para el grafo)

No importa algoritmos de entrenamiento ni métricas de `scikit-learn` para mantener
el módulo de inferencia liviano y desacoplado del código experimental.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

# Raíz del repositorio (independiente del directorio de trabajo desde donde se invoque Python)
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

# Directorios estándar de artefactos de resultados
RESULTS_METRICS_DIR: Path = PROJECT_ROOT / "results" / "metrics"
RESULTS_MODELS_DIR: Path = PROJECT_ROOT / "results" / "models"
RESULTS_PREDICTIONS_DIR: Path = PROJECT_ROOT / "results" / "predictions"

# Columnas identificadoras de cada par de materias (metadato de enlace para aristas; NO usar en X)
ID_COLUMNS: List[str] = [
    "materia_id_1",
    "materia_id_2",
]

# 10 variables predictoras aprobadas para los modelos de clasificación (en su orden exacto)
FEATURE_COLUMNS: List[str] = [
    "mismo_departamento",
    "mismo_prefijo_curricular",
    "anio_confirmado",
    "mismo_anio",
    "misma_carrera",
    "cant_carreras_compartidas_hist",
    "cant_inscriptos_hist_1",
    "cant_inscriptos_hist_2",
    "cant_estudiantes_compartidos_historico",
    "jaccard_historico",
]

# 5 variables predictoras booleanas/binarias {0, 1}
BOOLEAN_FEATURE_COLUMNS: List[str] = [
    "mismo_departamento",
    "mismo_prefijo_curricular",
    "anio_confirmado",
    "mismo_anio",
    "misma_carrera",
]

# 4 variables predictoras numéricas de conteo histórico (enteros/flotantes >= 0)
COUNT_FEATURE_COLUMNS: List[str] = [
    "cant_carreras_compartidas_hist",
    "cant_inscriptos_hist_1",
    "cant_inscriptos_hist_2",
    "cant_estudiantes_compartidos_historico",
]

# Variable predictora continua de similitud de Jaccard histórica en [0, 1]
JACCARD_FEATURE_COLUMN: str = "jaccard_historico"

# Columna excluida de X por colinealidad exacta (conteo / 4810)
COLLINEAR_EXCLUDED_COLUMN: str = "tasa_compartidos_historico"

# Variables objetivo (calculadas sobre la cohorte 1994)
TARGET_REGRESSION: str = "cant_estudiantes_afectados"
TARGET_CLASSIFICATION: str = "hubo_conflicto_potencial"

STRICT_BINARY_MAP: Dict[Any, int] = {
    True: 1,
    False: 0,
    1: 1,
    0: 0,
    1.0: 1,
    0.0: 0,
    "True": 1,
    "False": 0,
    "true": 1,
    "false": 0,
    "1": 1,
    "0": 0,
}


def parse_strict_binary_column(
    series: pd.Series, col_name: str, source_desc: Path | str = "<dataframe>"
) -> pd.Series:
    """Convierte una serie booleana/binaria a int {0, 1} sin conversiones silenciosas."""
    if series.isna().any():
        raise ValueError(
            f"La columna '{col_name}' en {source_desc} contiene valores nulos."
        )
    if pd.api.types.is_bool_dtype(series):
        return series.astype(int)
    mapped = series.map(STRICT_BINARY_MAP)
    if mapped.isna().any():
        invalid_vals = series[mapped.isna()].unique().tolist()[:5]
        raise ValueError(
            f"La columna '{col_name}' en {source_desc} contiene valores no binarios inesperados: {invalid_vals}"
        )
    return mapped.astype(int)
