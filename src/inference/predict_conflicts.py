"""Inferencia de conflictos potenciales entre pares de materias (Fase 3 -> Fase 4).

Este módulo provee una interfaz pública, autocontenida y desacoplada del script de
entrenamiento para ejecutar inferencia con el modelo seleccionado en la Fase 3:
- Esquema compartido: importado exclusivamente desde `src.common.schema`.
- Modelo: `KNN_Full_Unweighted` (`results/models/knn_full_unweighted.joblib`), que
  encapsula en un `sklearn.pipeline.Pipeline` tanto el `StandardScaler` ajustado en
  entrenamiento como el `KNeighborsClassifier(n_neighbors=15, weights='uniform')`.
- Umbral de decisión: `0.21`, leído desde la fuente única de verdad
  `results/metrics/metrics_summary.json` (`calibrated_thresholds_from_validation`).
- Regla de decisión: `predict_proba(X)[:, 1] >= 0.21` (nunca se utiliza `model.predict(X)`,
  dado que `predict()` aplica por defecto el umbral `0.50`).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Tuple

import joblib
import numpy as np
import pandas as pd

from src.common.schema import (
    BOOLEAN_FEATURE_COLUMNS,
    COUNT_FEATURE_COLUMNS,
    FEATURE_COLUMNS,
    ID_COLUMNS,
    JACCARD_FEATURE_COLUMN,
    RESULTS_METRICS_DIR,
    RESULTS_MODELS_DIR,
    parse_strict_binary_column,
)

DEFAULT_MODEL_NAME: str = "KNN_Full_Unweighted"
DEFAULT_MODEL_PATH: Path = RESULTS_MODELS_DIR / "knn_full_unweighted.joblib"
DEFAULT_METRICS_SUMMARY_PATH: Path = RESULTS_METRICS_DIR / "metrics_summary.json"

PROBABILITY_COLUMN: str = "prob_conflicto"
PREDICTION_COLUMN: str = "pred_conflicto"


def _validate_threshold_value(threshold: Any) -> float:
    """Valida que un umbral sea un número flotante finito en el intervalo abierto (0, 1)."""
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float, np.number)):
        raise ValueError(
            f"El umbral debe ser un valor numérico entre 0 y 1; se recibió: {threshold!r}"
        )
    th_float = float(threshold)
    if np.isnan(th_float) or np.isinf(th_float) or not (0.0 < th_float < 1.0):
        raise ValueError(
            f"El umbral debe ser finito y pertenecer al intervalo (0, 1); se recibió: {threshold!r}"
        )
    return th_float


def load_calibrated_threshold(
    model_name: str = DEFAULT_MODEL_NAME,
    metrics_summary_path: Path | str = DEFAULT_METRICS_SUMMARY_PATH,
) -> float:
    """Obtiene el umbral seleccionado en validación desde `metrics_summary.json`.

    Maneja explícitamente la ausencia del archivo, errores de formato JSON y la
    ausencia de las claves requeridas en la configuración.
    """
    summary_path = Path(metrics_summary_path)
    if not summary_path.exists() or not summary_path.is_file():
        raise FileNotFoundError(
            f"No se encontró el archivo de resumen de métricas para leer el umbral: {summary_path}"
        )

    try:
        with summary_path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"No se pudo leer o parsear el archivo JSON de métricas en {summary_path}: {exc}"
        ) from exc

    if not isinstance(payload, dict):
        raise ValueError(
            f"El contenido de {summary_path} no tiene la estructura de objeto JSON esperada."
        )

    thresholds_map = payload.get("calibrated_thresholds_from_validation")
    if not isinstance(thresholds_map, dict):
        raise KeyError(
            f"El archivo {summary_path} no contiene el diccionario 'calibrated_thresholds_from_validation'."
        )

    if model_name not in thresholds_map:
        raise KeyError(
            f"El modelo '{model_name}' no figura en 'calibrated_thresholds_from_validation' dentro de {summary_path}."
        )

    return _validate_threshold_value(thresholds_map[model_name])


def _validate_loaded_model(model: Any, source_desc: str) -> Any:
    """Verifica que el estimador exponga `predict_proba` y coincida con `FEATURE_COLUMNS`."""
    if not hasattr(model, "predict_proba") or not callable(getattr(model, "predict_proba")):
        raise ValueError(
            f"El modelo cargado desde {source_desc} no implementa el método 'predict_proba'."
        )

    if hasattr(model, "feature_names_in_"):
        expected_features = list(model.feature_names_in_)
        if expected_features != FEATURE_COLUMNS:
            raise ValueError(
                f"Las features del modelo ({expected_features}) no coinciden con "
                f"FEATURE_COLUMNS ({FEATURE_COLUMNS})."
            )
    return model


def load_conflict_model(model_path: Path | str = DEFAULT_MODEL_PATH) -> Any:
    """Carga y valida el pipeline entrenado (`.joblib`) desde disco."""
    path = Path(model_path)
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(
            f"No se encontró el archivo del modelo entrenado en: {path}. "
            "Verifique que el artefacto .joblib esté disponible en results/models/."
        )

    try:
        model = joblib.load(path)
    except Exception as exc:
        raise ValueError(
            f"Error al cargar el modelo serializado desde {path}: {exc}"
        ) from exc

    return _validate_loaded_model(model, str(path))


def validate_and_prepare_features(
    pairs_df: pd.DataFrame,
    *,
    require_ids: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame | None]:
    """Valida el DataFrame de entrada y devuelve `(X_ordenado, ids_df)`.

    Validaciones realizadas:
    1. Que `pairs_df` sea un `pandas.DataFrame` no vacío.
    2. Que estén presentes las 10 columnas de `FEATURE_COLUMNS` (y `ID_COLUMNS`
       si `require_ids=True`).
    3. Que no existan valores nulos (`NaN`/`None`) en las columnas requeridas antes
       de convertir tipos.
    4. Si hay columnas identificadoras (`materia_id_1`, `materia_id_2`):
       - Que no contengan cadenas vacías.
       - Que no existan auto-pares (`materia_id_1 == materia_id_2`).
       - Que no existan pares duplicados (ni directos `(u, v)` ni simétricos `(v, u)`).
    5. Que las 5 columnas booleanas contengan valores binarios estrictos (`0/1`, `True/False`).
    6. Que las 10 features sean numéricas y finitas (sin `inf` ni `-inf`).
    7. Que los 4 conteos históricos (`COUNT_FEATURE_COLUMNS`) sean no negativos (`>= 0`)
       y que `jaccard_historico` pertenezca al intervalo `[0, 1]`.
    8. Que las columnas de `X` queden exactamente en el orden canónico de `FEATURE_COLUMNS`.
    """
    if not isinstance(pairs_df, pd.DataFrame):
        raise TypeError(
            f"Se esperaba un pandas.DataFrame como entrada; se recibió {type(pairs_df).__name__}."
        )

    if pairs_df.empty:
        raise ValueError("El DataFrame de entrada está vacío (0 filas).")

    has_all_ids = all(col in pairs_df.columns for col in ID_COLUMNS)
    if require_ids and not has_all_ids:
        missing_ids = [col for col in ID_COLUMNS if col not in pairs_df.columns]
        raise ValueError(
            f"Faltan las columnas identificadoras del par requeridas: {missing_ids}"
        )

    missing_features = [col for col in FEATURE_COLUMNS if col not in pairs_df.columns]
    if missing_features:
        raise ValueError(
            f"Faltan las siguientes features requeridas para el modelo: {missing_features}"
        )

    required_cols = (ID_COLUMNS if (require_ids or has_all_ids) else []) + FEATURE_COLUMNS

    # Validar ausencia de nulos antes de cualquier conversión de tipos
    null_counts = pairs_df[required_cols].isna().sum()
    total_nulls = int(null_counts.sum())
    if total_nulls > 0:
        cols_with_nulls = null_counts[null_counts > 0].index.tolist()
        raise ValueError(
            f"Los datos de entrada contienen {total_nulls} valores nulos en las columnas: {cols_with_nulls}"
        )

    ids_df: pd.DataFrame | None = None
    if require_ids or has_all_ids:
        ids_df = pairs_df[ID_COLUMNS].copy().reset_index(drop=True)
        for id_col in ID_COLUMNS:
            str_vals = ids_df[id_col].astype(str).str.strip()
            if (str_vals == "").any():
                raise ValueError(
                    f"La columna identificadora '{id_col}' contiene cadenas vacías."
                )
            ids_df[id_col] = str_vals

        id1 = ids_df[ID_COLUMNS[0]]
        id2 = ids_df[ID_COLUMNS[1]]

        # Rechazar pares con la misma materia en ambos extremos (auto-aristas)
        self_pairs_mask = id1 == id2
        if self_pairs_mask.any():
            bad_subject = id1[self_pairs_mask].iloc[0]
            raise ValueError(
                f"Se detectó un par inválido con la misma materia en ambos extremos: ({bad_subject!r}, {bad_subject!r})."
            )

        # Rechazar pares duplicados (directos (u, v) o simétricos (v, u))
        canonical_u = np.minimum(id1.to_numpy(), id2.to_numpy())
        canonical_v = np.maximum(id1.to_numpy(), id2.to_numpy())
        canonical_pairs = pd.DataFrame({"u": canonical_u, "v": canonical_v})
        dup_mask = canonical_pairs.duplicated()
        if dup_mask.any():
            dup_row = canonical_pairs[dup_mask].iloc[0]
            raise ValueError(
                f"Se detectaron pares de materias duplicados en la entrada: ({dup_row['u']!r}, {dup_row['v']!r})."
            )

    # Seleccionar y ordenar las 10 features en una copia limpia (sin mutar pairs_df)
    X_df = pairs_df[FEATURE_COLUMNS].copy().reset_index(drop=True)
    input_source = Path("<inference_input>")

    # Validar estrictamente las 5 variables booleanas/binarias
    for bcol in BOOLEAN_FEATURE_COLUMNS:
        X_df[bcol] = parse_strict_binary_column(X_df[bcol], bcol, input_source)

    # Validar que las 5 variables numéricas restantes no sean booleanas
    non_boolean_features = [c for c in FEATURE_COLUMNS if c not in BOOLEAN_FEATURE_COLUMNS]
    for ncol in non_boolean_features:
        if pd.api.types.is_bool_dtype(X_df[ncol]) or X_df[ncol].map(lambda v: isinstance(v, bool)).any():
            raise ValueError(
                f"La columna numérica '{ncol}' contiene valores booleanos no permitidos."
            )

    # Verificar conversión numérica de todas las features y ausencia de infinitos
    try:
        numeric_matrix = X_df[FEATURE_COLUMNS].to_numpy(dtype=float)
    except (ValueError, TypeError) as exc:
        raise ValueError(
            "Los datos de entrada contienen valores no numéricos en las features."
        ) from exc

    if np.isnan(numeric_matrix).any():
        raise ValueError("Los datos de entrada contienen valores NaN tras la conversión numérica.")

    inf_count = int(np.isinf(numeric_matrix).sum())
    if inf_count > 0:
        raise ValueError(
            f"Los datos de entrada contienen {inf_count} valores infinitos en las features."
        )

    # Validar dominio físico de los 4 conteos históricos (>= 0)
    for ccol in COUNT_FEATURE_COLUMNS:
        col_vals = X_df[ccol].to_numpy(dtype=float)
        if (col_vals < 0.0).any():
            raise ValueError(
                f"La columna de conteo histórico '{ccol}' contiene valores negativos no válidos."
            )

    # Validar dominio de jaccard_historico dentro de [0, 1]
    jaccard_vals = X_df[JACCARD_FEATURE_COLUMN].to_numpy(dtype=float)
    if (jaccard_vals < 0.0).any() or (jaccard_vals > 1.0).any():
        raise ValueError(
            f"La columna '{JACCARD_FEATURE_COLUMN}' debe pertenecer al intervalo [0, 1]."
        )

    # Garantizar el orden exacto de FEATURE_COLUMNS
    X_ordered = X_df[FEATURE_COLUMNS]
    return X_ordered, ids_df


def predict_conflict_pairs(
    pairs_df: pd.DataFrame,
    *,
    model: Any | None = None,
    model_path: Path | str = DEFAULT_MODEL_PATH,
    threshold: float | None = None,
    metrics_summary_path: Path | str = DEFAULT_METRICS_SUMMARY_PATH,
    model_name: str = DEFAULT_MODEL_NAME,
    require_ids: bool = True,
    include_features: bool = False,
) -> pd.DataFrame:
    """Predice la probabilidad y la existencia de conflicto potencial para cada par de materias.

    Parámetros:
    -----------
    pairs_df : pd.DataFrame
        DataFrame con los pares de materias (`materia_id_1`, `materia_id_2`) y las 10
        variables predictoras definidas en `FEATURE_COLUMNS` (en cualquier orden de
        columnas; la función las valida y ordena automáticamente).
    model : estimador opcional
        Pipeline ya cargado en memoria. Si es `None`, se carga desde `model_path`.
    model_path : Path | str
        Ruta al archivo `.joblib` (por defecto `results/models/knn_full_unweighted.joblib`).
    threshold : float | None
        Umbral de decisión en `(0, 1)`. Si es `None`, se carga el umbral oficial (`0.21`)
        desde `results/metrics/metrics_summary.json`.
    metrics_summary_path : Path | str
        Ruta al archivo `metrics_summary.json` de donde se lee el umbral por defecto.
    model_name : str
        Clave del modelo dentro de `calibrated_thresholds_from_validation` (`KNN_Full_Unweighted`).
    require_ids : bool
        Si es `True` (por defecto), exige y conserva `materia_id_1` y `materia_id_2`.
    include_features : bool
        Si es `True`, incluye también las 10 features validadas en el DataFrame de salida.

    Retorna:
    --------
    pd.DataFrame
        DataFrame con la misma cantidad de filas que `pairs_df` y las columnas:
        - `materia_id_1`, `materia_id_2` (si están presentes)
        - (opcionalmente las 10 `FEATURE_COLUMNS` si `include_features=True`)
        - `prob_conflicto` (float en `[0.0, 1.0]` calculado con `model.predict_proba(X)[:, 1]`)
        - `pred_conflicto` (int `0` o `1`, resultante de `prob_conflicto >= threshold`)
    """
    X_ordered, ids_df = validate_and_prepare_features(
        pairs_df, require_ids=require_ids
    )

    resolved_threshold = (
        load_calibrated_threshold(
            model_name=model_name, metrics_summary_path=metrics_summary_path
        )
        if threshold is None
        else _validate_threshold_value(threshold)
    )

    loaded_model = (
        load_conflict_model(model_path)
        if model is None
        else _validate_loaded_model(model, "<in_memory_model>")
    )

    # El Pipeline ya incluye StandardScaler; NO se debe volver a escalar X_ordered.
    # Nunca se usa model.predict(X) porque aplicaría el umbral por defecto 0.50.
    probabilities = np.asarray(
        loaded_model.predict_proba(X_ordered)[:, 1], dtype=float
    )
    predictions = (probabilities >= resolved_threshold).astype(int)

    output_parts = []
    if ids_df is not None:
        output_parts.append(ids_df)
    if include_features:
        output_parts.append(X_ordered)

    preds_df = pd.DataFrame(
        {
            PROBABILITY_COLUMN: probabilities,
            PREDICTION_COLUMN: predictions,
        }
    )
    output_parts.append(preds_df)

    return pd.concat(output_parts, axis=1)
