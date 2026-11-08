"""Probabilidades de conflicto para TODOS los pares de materias (insumo del grafo).

El modelo oficial (`results/models/decision_tree_full_unweighted.joblib`) se entrenó con las 714
materias de `ml_train_full.csv`. Para armar el grafo hacen falta los 319.600 pares de
las 800 materias, y los pares de entrenamiento serían *in-sample*.

Para que el cronograma se evalúe con predicciones honestas se ofrecen dos modos:
- `oof`  (por defecto): validación cruzada por materia (GroupKFold-like). Cada par se
  predice con un modelo que NUNCA vio ninguna de sus dos materias. Hiperparámetros
  clonados del `.joblib` oficial.
- `final`: usa el modelo oficial ya entrenado (es lo que se usaría para materias nuevas;
  los pares de train quedan in-sample).
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.base import clone

from src.common.schema import (
    FEATURE_COLUMNS,
    ID_COLUMNS,
    PROJECT_ROOT,
    RESULTS_PREDICTIONS_DIR,
    TARGET_CLASSIFICATION,
    parse_strict_binary_column,
    BOOLEAN_FEATURE_COLUMNS,
)
from src.inference.predict_conflicts import (
    DEFAULT_MODEL_NAME,
    load_calibrated_threshold,
    load_conflict_model,
)

PAIRS_CSV: Path = PROJECT_ROOT / "data" / "processed" / "subject_pairs_poc.csv"
DEFAULT_CACHE: Path = RESULTS_PREDICTIONS_DIR / "pair_probabilities_oof.csv"
PROB_COL = "prob_conflicto"


def load_all_pairs(path: Path | str = PAIRS_CSV) -> pd.DataFrame:
    """Lee `subject_pairs_poc.csv` (319.600 pares) con features binarias normalizadas."""
    df = pd.read_csv(path)
    for col in BOOLEAN_FEATURE_COLUMNS:
        df[col] = parse_strict_binary_column(df[col], col, path)
    df[TARGET_CLASSIFICATION] = parse_strict_binary_column(
        df[TARGET_CLASSIFICATION], TARGET_CLASSIFICATION, path
    )
    return df


def predict_all_pairs(
    pairs_df: pd.DataFrame,
    mode: Literal["oof", "final"] = "oof",
    n_folds: int = 5,
    seed: int = 42,
) -> pd.DataFrame:
    """Devuelve `materia_id_1, materia_id_2, prob_conflicto` para todos los pares."""
    model = load_conflict_model()
    X = pairs_df[FEATURE_COLUMNS].astype(float)
    out = pairs_df[ID_COLUMNS].copy().reset_index(drop=True)

    if mode == "final":
        out[PROB_COL] = model.predict_proba(X)[:, 1]
        return out

    if mode != "oof":
        raise ValueError(f"mode debe ser 'oof' o 'final'; se recibió {mode!r}")

    y = pairs_df[TARGET_CLASSIFICATION].astype(int).to_numpy()
    subjects = np.array(sorted(set(pairs_df[ID_COLUMNS[0]]) | set(pairs_df[ID_COLUMNS[1]])))
    rng = np.random.RandomState(seed)
    rng.shuffle(subjects)
    fold_of = {s: i % n_folds for i, s in enumerate(subjects)}
    f1 = pairs_df[ID_COLUMNS[0]].map(fold_of).to_numpy()
    f2 = pairs_df[ID_COLUMNS[1]].map(fold_of).to_numpy()

    probs = np.full(len(pairs_df), np.nan)
    for k in range(n_folds):
        train_mask = (f1 != k) & (f2 != k)               # ninguna materia del fold k
        todo = ((f1 == k) | (f2 == k)) & np.isnan(probs)  # pares aún sin predecir
        est = clone(model).fit(X[train_mask], y[train_mask])
        probs[todo] = est.predict_proba(X[todo])[:, 1]
    if np.isnan(probs).any():
        raise RuntimeError("Quedaron pares sin predecir en la validación cruzada.")
    out[PROB_COL] = probs
    return out


def get_pair_probabilities(
    mode: Literal["oof", "final"] = "oof",
    cache_path: Path | str | None = None,
    force: bool = False,
) -> pd.DataFrame:
    """Carga (o calcula y cachea) las probabilidades para todos los pares."""
    cache = Path(cache_path) if cache_path else (
        DEFAULT_CACHE if mode == "oof" else RESULTS_PREDICTIONS_DIR / "pair_probabilities_final.csv"
    )
    if cache.exists() and not force:
        return pd.read_csv(cache)
    pairs = load_all_pairs()
    out = predict_all_pairs(pairs, mode=mode)
    cache.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(cache, index=False)
    return out


def default_threshold() -> float:
    """Umbral calibrado en validación para el modelo oficial (0.20), desde metrics_summary.json."""
    return load_calibrated_threshold(DEFAULT_MODEL_NAME)
