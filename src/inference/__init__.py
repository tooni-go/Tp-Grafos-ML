"""Módulo de inferencia de Machine Learning para predicción de conflictos entre pares de materias.

Expone la interfaz pública para que el módulo de construcción y coloreo del grafo
pueda predecir probabilidades y aristas de incompatibilidad utilizando el modelo
seleccionado en la Fase 3 (`KNN_Full_Unweighted`) y su umbral validado (`0.21`).
"""

from src.inference.predict_conflicts import (
    DEFAULT_METRICS_SUMMARY_PATH,
    DEFAULT_MODEL_NAME,
    DEFAULT_MODEL_PATH,
    FEATURE_COLUMNS,
    ID_COLUMNS,
    PREDICTION_COLUMN,
    PROBABILITY_COLUMN,
    load_calibrated_threshold,
    load_conflict_model,
    predict_conflict_pairs,
    validate_and_prepare_features,
)

__all__ = [
    "DEFAULT_METRICS_SUMMARY_PATH",
    "DEFAULT_MODEL_NAME",
    "DEFAULT_MODEL_PATH",
    "FEATURE_COLUMNS",
    "ID_COLUMNS",
    "PREDICTION_COLUMN",
    "PROBABILITY_COLUMN",
    "load_calibrated_threshold",
    "load_conflict_model",
    "predict_conflict_pairs",
    "validate_and_prepare_features",
]
