"""Fase 3: Entrenamiento, validación y evaluación de modelos de clasificación.

Alcance exclusivo de esta fase: Clasificación binaria (`hubo_conflicto_potencial`).
No modifica ninguno de los archivos CSV originales de las Fases 1, 2 o 2.5.

Decisiones metodológicas implementadas:
1. Archivos de entrada:
   - `data/processed/ml_train_balanced.csv` (6.996 filas, 50/50, 714 materias)
   - `data/processed/ml_train_full.csv` (254.541 filas, 1,37% pos, 714 materias)
   - `data/processed/ml_test_full.csv` (65.059 filas, 1,49% pos, 86 materias de test)
2. Selección de 10 variables predictoras (`FEATURE_COLUMNS`):
   - Se incluye `anio_confirmado` y las variables históricas de tamaño e intersección.
   - Se excluye `tasa_compartidos_historico` por colinealidad exacta (r = 1.0) con
     `cant_estudiantes_compartidos_historico` (conteo / 4810).
   - Se excluyen siempre de X los identificadores (`materia_id_1`, `materia_id_2`)
     y los dos targets (`cant_estudiantes_afectados` y `hubo_conflicto_potencial`).
3. Separación interna de Validación por Materias (a partir de `ml_train_full.csv`):
   - Se reserva estratificadamente por `raw_level_char` un subconjunto de las 714
     materias de entrenamiento (`val_subjects`, 77 materias) y el resto queda en
     `subtrain_subjects` (637 materias) con semilla fija `RANDOM_SEED = 42`.
   - `subtrain_full`: pares donde ambas materias pertenecen a `subtrain_subjects`.
   - `subtrain_balanced`: undersampling 50/50 aplicado exclusivamente sobre `subtrain_full`.
   - `validation`: pares donde al menos una materia pertenece a `val_subjects`
     (conservando la distribución natural de clases, sin forzar 50/50).
4. Modelos comparados:
   - Baselines sencillos:
     * Baseline Mayoritario (`DummyClassifier(strategy="prior")`)
     * Baseline Heurístico Histórico (`cant_estudiantes_compartidos_historico > 0`)
   - Tres algoritmos clásicos de scikit-learn:
     * `DecisionTreeClassifier(max_depth=5, random_state=42)` (sin escalado)
     * `Pipeline([StandardScaler(), LogisticRegression(max_iter=1000, random_state=42)])`
     * `Pipeline([StandardScaler(), KNeighborsClassifier(n_neighbors=15, n_jobs=-1)])`
   - Se comparan el entrenamiento sobre `subtrain_balanced` vs. `subtrain_full`
     (incluyendo `class_weight="balanced"` en Árbol y Regresión Logística) sobre el
     mismo conjunto de validación interna.
"""

from __future__ import annotations

import csv
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


RANDOM_SEED = 42
VAL_SUBJECT_FRACTION = 0.10625  # ~77 materias de las 714 -> ~20,4% de pares en validación

PROJECT_ROOT = Path(__file__).resolve().parents[2]

EXAMS_CLEAN_PATH = PROJECT_ROOT / "data" / "interim" / "exams_clean.csv"
TRAIN_BALANCED_PATH = PROJECT_ROOT / "data" / "processed" / "ml_train_balanced.csv"
TRAIN_FULL_PATH = PROJECT_ROOT / "data" / "processed" / "ml_train_full.csv"
TEST_FULL_PATH = PROJECT_ROOT / "data" / "processed" / "ml_test_full.csv"

RESULTS_METRICS_DIR = PROJECT_ROOT / "results" / "metrics"
RESULTS_MODELS_DIR = PROJECT_ROOT / "results" / "models"
RESULTS_PREDICTIONS_DIR = PROJECT_ROOT / "results" / "predictions"

ID_COLUMNS: List[str] = ["materia_id_1", "materia_id_2"]
TARGET_COLUMN: str = "hubo_conflicto_potencial"
REGRESSION_TARGET_COLUMN: str = "cant_estudiantes_afectados"
COLLINEAR_EXCLUDED_COLUMN: str = "tasa_compartidos_historico"

# 10 variables predictoras aprobadas para la Fase 3
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

BOOLEAN_FEATURE_COLUMNS: List[str] = [
    "mismo_departamento",
    "mismo_prefijo_curricular",
    "anio_confirmado",
    "mismo_anio",
    "misma_carrera",
]


class HistoricalOverlapBaseline(BaseEstimator, ClassifierMixin):
    """Baseline heurístico de dominio: predice conflicto si hubo alumnos compartidos en 1989-1993.

    Nota metodológica sobre `predict_proba`:
    Divide el conteo histórico por el máximo del conjunto evaluado para acotar el
    score monótono en [0, 1]. Dado que PR-AUC y ROC-AUC son métricas basadas en
    rangos (invariantes ante reescalados lineales positivos dentro de un mismo
    conjunto), este score permite calcular ambas curvas sin distorsionar el orden
    relativo de los pares, aunque NO representa una probabilidad calibrada y su
    escala numérica depende del máximo del lote recibido. Por ello, `predict()`
    evalúa directamente la regla determinística `vals > 0`.
    """

    def __init__(self, feature_index: int = 8) -> None:
        self.feature_index = feature_index
        self.classes_ = np.array([0, 1])

    def fit(self, X: Any, y: Any = None) -> "HistoricalOverlapBaseline":
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        vals = (
            X["cant_estudiantes_compartidos_historico"].to_numpy(dtype=float)
            if isinstance(X, pd.DataFrame)
            else np.asarray(X)[:, self.feature_index].astype(float)
        )
        # Escala monótona acotada en [0, 1] para permitir cálculo de PR-AUC y ROC-AUC
        scores = np.clip(vals / (vals.max() if vals.max() > 0 else 1.0), 0.0, 1.0)
        return np.column_stack([1.0 - scores, scores])

    def predict(self, X: Any) -> np.ndarray:
        vals = (
            X["cant_estudiantes_compartidos_historico"].to_numpy(dtype=float)
            if isinstance(X, pd.DataFrame)
            else np.asarray(X)[:, self.feature_index].astype(float)
        )
        return (vals > 0).astype(int)


_STRICT_BINARY_MAP: Dict[Any, int] = {
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


def _parse_strict_binary_column(
    series: pd.Series, col_name: str, csv_path: Path
) -> pd.Series:
    """Convierte una columna booleana/binaria a int {0, 1} sin conversiones silenciosas."""
    if series.isna().any():
        raise ValueError(
            f"La columna '{col_name}' en {csv_path} contiene valores nulos."
        )
    if pd.api.types.is_bool_dtype(series):
        return series.astype(int)
    mapped = series.map(_STRICT_BINARY_MAP)
    if mapped.isna().any():
        invalid_vals = series[mapped.isna()].unique().tolist()[:5]
        raise ValueError(
            f"La columna '{col_name}' en {csv_path} contiene valores no binarios inesperados: {invalid_vals}"
        )
    return mapped.astype(int)


def load_ml_dataframe(csv_path: Path) -> pd.DataFrame:
    """Carga un CSV de la Fase 2.5 y valida columnas, tipos binarios y ausencia de nulos/infinitos."""
    if not csv_path.exists():
        raise FileNotFoundError(f"No se encontró el archivo requerido: {csv_path}")

    df = pd.read_csv(csv_path)

    required_cols = ID_COLUMNS + FEATURE_COLUMNS + [TARGET_COLUMN]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Al archivo {csv_path} le faltan las columnas requeridas: {missing_cols}"
        )

    # Verificar ausencia de nulos antes de cualquier conversión de tipos
    null_count = int(df[required_cols].isna().sum().sum())
    if null_count > 0:
        raise ValueError(f"El archivo {csv_path} contiene {null_count} valores nulos.")

    for bcol in BOOLEAN_FEATURE_COLUMNS:
        df[bcol] = _parse_strict_binary_column(df[bcol], bcol, csv_path)
    df[TARGET_COLUMN] = _parse_strict_binary_column(
        df[TARGET_COLUMN], TARGET_COLUMN, csv_path
    )

    # Verificar que las features numéricas sean numéricas y no existan infinitos
    check_cols = FEATURE_COLUMNS + [TARGET_COLUMN]
    try:
        numeric_matrix = df[check_cols].to_numpy(dtype=float)
    except (ValueError, TypeError) as exc:
        raise ValueError(
            f"El archivo {csv_path} contiene valores no numéricos en las features."
        ) from exc

    inf_count = int(np.isinf(numeric_matrix).sum())
    if inf_count > 0:
        raise ValueError(
            f"El archivo {csv_path} contiene {inf_count} valores infinitos."
        )
    return df


def build_internal_subject_validation_split(
    train_full_df: pd.DataFrame,
    exams_csv: Path = EXAMS_CLEAN_PATH,
    seed: int = RANDOM_SEED,
    val_fraction: float = VAL_SUBJECT_FRACTION,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Construye `subtrain_full`, `subtrain_balanced` y `validation` por materias.

    Respeta estrictamente la estrategia de la Fase 2.5:
    - Identifica las 714 materias presentes en `ml_train_full.csv`.
    - Selecciona un subconjunto estratificado por `raw_level_char` (`val_subjects`).
    - `subtrain_full`: pares donde ambas materias pertenecen a `subtrain_subjects`.
    - `validation`: pares donde al menos una materia pertenece a `val_subjects`
      (conservando la distribución natural de clases).
    - `subtrain_balanced`: undersampling 50/50 sin reemplazo aplicado únicamente
      sobre `subtrain_full`.
    """
    train_subjects: Set[str] = set(train_full_df["materia_id_1"]).union(
        set(train_full_df["materia_id_2"])
    )

    by_level: Dict[str, List[str]] = defaultdict(list)
    with exams_csv.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            ex_id = row["exam_id"]
            if ex_id in train_subjects:
                by_level[row["raw_level_char"]].append(ex_id)

    rng = random.Random(seed)
    val_subjects: Set[str] = set()
    for lvl in sorted(by_level.keys()):
        candidates = sorted(by_level[lvl])
        k = max(1, round(len(candidates) * val_fraction))
        val_subjects.update(rng.sample(candidates, k))

    subtrain_subjects = train_subjects - val_subjects

    is_val_mask = train_full_df["materia_id_1"].isin(val_subjects) | train_full_df[
        "materia_id_2"
    ].isin(val_subjects)

    val_df = train_full_df[is_val_mask].copy().reset_index(drop=True)
    subtrain_full_df = train_full_df[~is_val_mask].copy().reset_index(drop=True)

    # Undersampling 50/50 exclusivamente sobre subtrain_full_df
    pos_df = subtrain_full_df[subtrain_full_df[TARGET_COLUMN] == 1]
    neg_df = subtrain_full_df[subtrain_full_df[TARGET_COLUMN] == 0]
    sampled_neg_df = neg_df.sample(n=len(pos_df), random_state=seed, replace=False)
    subtrain_balanced_df = (
        pd.concat([pos_df, sampled_neg_df], ignore_index=True)
        .sample(frac=1.0, random_state=seed)
        .reset_index(drop=True)
    )

    split_info = {
        "total_train_subjects": len(train_subjects),
        "subtrain_subjects_count": len(subtrain_subjects),
        "val_subjects_count": len(val_subjects),
        "subtrain_full_rows": len(subtrain_full_df),
        "subtrain_full_positives": int(subtrain_full_df[TARGET_COLUMN].sum()),
        "subtrain_full_negatives": int((subtrain_full_df[TARGET_COLUMN] == 0).sum()),
        "subtrain_full_prevalence": round(
            float(subtrain_full_df[TARGET_COLUMN].mean()), 6
        ),
        "subtrain_balanced_rows": len(subtrain_balanced_df),
        "subtrain_balanced_positives": int(subtrain_balanced_df[TARGET_COLUMN].sum()),
        "subtrain_balanced_negatives": int(
            (subtrain_balanced_df[TARGET_COLUMN] == 0).sum()
        ),
        "subtrain_balanced_prevalence": round(
            float(subtrain_balanced_df[TARGET_COLUMN].mean()), 6
        ),
        "validation_rows": len(val_df),
        "validation_positives": int(val_df[TARGET_COLUMN].sum()),
        "validation_negatives": int((val_df[TARGET_COLUMN] == 0).sum()),
        "validation_prevalence": round(float(val_df[TARGET_COLUMN].mean()), 6),
    }
    return subtrain_full_df, subtrain_balanced_df, val_df, split_info


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None,
    model_name: str,
    train_regime: str,
    eval_split: str,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """Calcula todas las métricas requeridas y desglosa la matriz de confusión."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    acc = float(accuracy_score(y_true, y_pred))

    pr_auc: float | None = None
    roc_auc: float | None = None
    if y_prob is not None:
        pr_auc = float(average_precision_score(y_true, y_prob))
        roc_auc = float(roc_auc_score(y_true, y_prob))

    return {
        "model_name": model_name,
        "train_regime": train_regime,
        "eval_split": eval_split,
        "threshold": round(float(threshold), 4),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "precision": round(prec, 6),
        "recall": round(rec, 6),
        "f1_score": round(f1, 6),
        "accuracy": round(acc, 6),
        "pr_auc": round(pr_auc, 6) if pr_auc is not None else None,
        "roc_auc": round(roc_auc, 6) if roc_auc is not None else None,
    }


def find_best_f1_threshold_on_validation(
    y_val: np.ndarray, y_prob_val: np.ndarray
) -> float:
    """Encuentra en validación el umbral de probabilidad que maximiza F1 (sin tocar Test).

    Recorre la grilla [0.01, 0.95] con paso 0.01 en orden ascendente. En caso de
    empate exacto en F1 (por ejemplo, en modelos con probabilidades discretas como
    KNN con K=15), la desigualdad estricta `score > best_f1` conserva el primer
    umbral (el menor) del tramo óptimo.
    """
    best_th = 0.50
    best_f1 = -1.0
    for th in np.linspace(0.01, 0.95, 95):
        preds = (y_prob_val >= th).astype(int)
        score = float(f1_score(y_val, preds, zero_division=0))
        if score > best_f1:
            best_f1 = score
            best_th = float(th)
    return round(best_th, 4)


def build_models_catalog(seed: int = RANDOM_SEED) -> Dict[str, Tuple[Any, str]]:
    """Construye los modelos para ambos regímenes de entrenamiento."""
    return {
        # Baselines
        "Baseline_Mayoritario": (
            DummyClassifier(strategy="prior", random_state=seed),
            "subtrain_full",
        ),
        "Baseline_Heuristico_Historico": (
            HistoricalOverlapBaseline(),
            "subtrain_balanced",
        ),
        # Régimen Principal: Entrenados sobre set balanceado 50/50
        "DecisionTree_Balanced": (
            DecisionTreeClassifier(max_depth=5, random_state=seed),
            "subtrain_balanced",
        ),
        "LogisticRegression_Balanced": (
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("clf", LogisticRegression(max_iter=1000, random_state=seed)),
                ]
            ),
            "subtrain_balanced",
        ),
        "KNN_Balanced": (
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("clf", KNeighborsClassifier(n_neighbors=15, n_jobs=-1)),
                ]
            ),
            "subtrain_balanced",
        ),
        # Régimen Alternativo: Entrenados sobre set completo (distribución original)
        "DecisionTree_Full_Unweighted": (
            DecisionTreeClassifier(max_depth=5, random_state=seed),
            "subtrain_full",
        ),
        "DecisionTree_Full_ClassWeight": (
            DecisionTreeClassifier(
                max_depth=5, class_weight="balanced", random_state=seed
            ),
            "subtrain_full",
        ),
        "LogisticRegression_Full_Unweighted": (
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("clf", LogisticRegression(max_iter=1000, random_state=seed)),
                ]
            ),
            "subtrain_full",
        ),
        "LogisticRegression_Full_ClassWeight": (
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "clf",
                        LogisticRegression(
                            max_iter=1000,
                            class_weight="balanced",
                            random_state=seed,
                        ),
                    ),
                ]
            ),
            "subtrain_full",
        ),
        "KNN_Full_Unweighted": (
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("clf", KNeighborsClassifier(n_neighbors=15, n_jobs=-1)),
                ]
            ),
            "subtrain_full",
        ),
    }


def run_phase3_comparison() -> Dict[str, Any]:
    """Ejecuta el flujo completo de validación interna, entrenamiento final y test."""
    print("1. Cargando y auditando los 3 archivos CSV de la Fase 2.5...")
    train_bal_df = load_ml_dataframe(TRAIN_BALANCED_PATH)
    train_full_df = load_ml_dataframe(TRAIN_FULL_PATH)
    test_full_df = load_ml_dataframe(TEST_FULL_PATH)

    print("2. Construyendo partición interna de Validación por Materias...")
    subtrain_full_df, subtrain_bal_df, val_df, split_info = (
        build_internal_subject_validation_split(train_full_df)
    )

    X_sub_bal = subtrain_bal_df[FEATURE_COLUMNS]
    y_sub_bal = subtrain_bal_df[TARGET_COLUMN].to_numpy(dtype=int)

    X_sub_full = subtrain_full_df[FEATURE_COLUMNS]
    y_sub_full = subtrain_full_df[TARGET_COLUMN].to_numpy(dtype=int)

    X_val = val_df[FEATURE_COLUMNS]
    y_val = val_df[TARGET_COLUMN].to_numpy(dtype=int)

    X_train_bal_final = train_bal_df[FEATURE_COLUMNS]
    y_train_bal_final = train_bal_df[TARGET_COLUMN].to_numpy(dtype=int)

    X_train_full_final = train_full_df[FEATURE_COLUMNS]
    y_train_full_final = train_full_df[TARGET_COLUMN].to_numpy(dtype=int)

    X_test = test_full_df[FEATURE_COLUMNS]
    y_test = test_full_df[TARGET_COLUMN].to_numpy(dtype=int)

    validation_results: List[Dict[str, Any]] = []
    calibrated_thresholds: Dict[str, float] = {}

    print("3. Entrenando y evaluando en Validación Interna (51.975 pares)...")
    val_catalog = build_models_catalog(RANDOM_SEED)
    for name, (estimator, regime) in val_catalog.items():
        if regime == "subtrain_balanced":
            estimator.fit(X_sub_bal, y_sub_bal)
        else:
            estimator.fit(X_sub_full, y_sub_full)

        y_pred_val = estimator.predict(X_val)
        y_prob_val = (
            estimator.predict_proba(X_val)[:, 1]
            if hasattr(estimator, "predict_proba")
            else None
        )

        metrics_05 = compute_classification_metrics(
            y_true=y_val,
            y_pred=y_pred_val,
            y_prob=y_prob_val,
            model_name=name,
            train_regime=regime,
            eval_split="validation_natural",
            threshold=0.50,
        )
        validation_results.append(metrics_05)

        # Para todos los modelos entrenables (excluyendo los baselines),
        # buscar exclusivamente en validación el umbral que maximiza F1 (sin usar Test).
        if y_prob_val is not None and not name.startswith("Baseline_"):
            best_th = find_best_f1_threshold_on_validation(y_val, y_prob_val)
            calibrated_thresholds[name] = best_th
            y_pred_cal = (y_prob_val >= best_th).astype(int)
            metrics_cal = compute_classification_metrics(
                y_true=y_val,
                y_pred=y_pred_cal,
                y_prob=y_prob_val,
                model_name=f"{name}_UmbralVal",
                train_regime=regime,
                eval_split="validation_natural",
                threshold=best_th,
            )
            validation_results.append(metrics_cal)

    print("4. Entrenando modelos finales sobre Train oficial y evaluando en Test (65.059 pares)...")
    final_catalog = build_models_catalog(RANDOM_SEED)
    test_results: List[Dict[str, Any]] = []
    test_predictions_df = test_full_df[ID_COLUMNS + [TARGET_COLUMN]].copy()

    for directory in (RESULTS_METRICS_DIR, RESULTS_MODELS_DIR, RESULTS_PREDICTIONS_DIR):
        try:
            directory.mkdir(parents=True, exist_ok=True)
        except OSError:
            if not directory.is_dir():
                raise

    feature_importances_rows: List[Dict[str, Any]] = []

    for name, (estimator, regime) in final_catalog.items():
        train_label = (
            "ml_train_balanced (6.996)"
            if regime == "subtrain_balanced"
            else "ml_train_full (254.541)"
        )
        if regime == "subtrain_balanced":
            estimator.fit(X_train_bal_final, y_train_bal_final)
        else:
            estimator.fit(X_train_full_final, y_train_full_final)

        # Guardar modelo serializado (comprimido para evitar binarios pesados en KNN)
        model_path = RESULTS_MODELS_DIR / f"{name.lower()}.joblib"
        joblib.dump(estimator, model_path, compress=3)

        y_pred_test = estimator.predict(X_test)
        y_prob_test = (
            estimator.predict_proba(X_test)[:, 1]
            if hasattr(estimator, "predict_proba")
            else None
        )

        test_metrics_05 = compute_classification_metrics(
            y_true=y_test,
            y_pred=y_pred_test,
            y_prob=y_prob_test,
            model_name=name,
            train_regime=train_label,
            eval_split="ml_test_full",
            threshold=0.50,
        )
        test_results.append(test_metrics_05)

        if y_prob_test is not None:
            test_predictions_df[f"prob_{name}"] = np.round(y_prob_test, 6)
        test_predictions_df[f"pred_{name}_th050"] = y_pred_test

        if name in calibrated_thresholds and y_prob_test is not None:
            th_val = calibrated_thresholds[name]
            y_pred_test_cal = (y_prob_test >= th_val).astype(int)
            test_predictions_df[f"pred_{name}_thVal"] = y_pred_test_cal
            test_metrics_cal = compute_classification_metrics(
                y_true=y_test,
                y_pred=y_pred_test_cal,
                y_prob=y_prob_test,
                model_name=f"{name}_UmbralVal",
                train_regime=train_label,
                eval_split="ml_test_full",
                threshold=th_val,
            )
            test_results.append(test_metrics_cal)

        # Extraer feature_importances_ para DecisionTree
        if isinstance(estimator, DecisionTreeClassifier):
            importances = estimator.feature_importances_
            ranked = sorted(
                zip(FEATURE_COLUMNS, importances),
                key=lambda x: x[1],
                reverse=True,
            )
            for rank_idx, (feat_name, imp_val) in enumerate(ranked, start=1):
                feature_importances_rows.append(
                    {
                        "model_name": name,
                        "rank": rank_idx,
                        "feature": feat_name,
                        "importance": round(float(imp_val), 6),
                    }
                )

    # Guardar archivos de métricas, importancias y predicciones
    pd.DataFrame(validation_results).to_csv(
        RESULTS_METRICS_DIR / "validation_metrics.csv", index=False
    )
    pd.DataFrame(test_results).to_csv(
        RESULTS_METRICS_DIR / "test_metrics.csv", index=False
    )
    pd.DataFrame(feature_importances_rows).to_csv(
        RESULTS_METRICS_DIR / "decision_tree_feature_importances.csv", index=False
    )
    test_predictions_df.to_csv(
        RESULTS_PREDICTIONS_DIR / "test_predictions_all_models.csv", index=False
    )

    summary_payload = {
        "random_seed": RANDOM_SEED,
        "feature_columns": FEATURE_COLUMNS,
        "excluded_collinear_column": COLLINEAR_EXCLUDED_COLUMN,
        "internal_validation_split": split_info,
        "calibrated_thresholds_from_validation": calibrated_thresholds,
        "validation_metrics": validation_results,
        "test_metrics": test_results,
        "decision_tree_feature_importances": [
            r
            for r in feature_importances_rows
            if r["model_name"] == "DecisionTree_Balanced"
        ],
    }
    with (RESULTS_METRICS_DIR / "metrics_summary.json").open(
        "w", encoding="utf-8"
    ) as fh:
        json.dump(summary_payload, fh, indent=2, ensure_ascii=False)

    return summary_payload


def main() -> None:
    """Punto de entrada CLI para ejecutar `compare_models.py`."""
    summary = run_phase3_comparison()
    print("\n=== RESUMEN DE PARTICIÓN INTERNA DE VALIDACIÓN (POR MATERIAS) ===")
    for k, v in summary["internal_validation_split"].items():
        print(f"  {k}: {v}")

    print("\n=== RESULTADOS EN VALIDACIÓN INTERNA (51.975 PARES) ===")
    val_df = pd.DataFrame(summary["validation_metrics"])
    print(
        val_df[
            [
                "model_name",
                "threshold",
                "precision",
                "recall",
                "f1_score",
                "pr_auc",
                "tp",
                "fp",
                "fn",
                "tn",
            ]
        ].to_string(index=False)
    )

    print("\n=== RESULTADOS FINALES EN TEST (ml_test_full.csv: 65.059 PARES) ===")
    test_df = pd.DataFrame(summary["test_metrics"])
    print(
        test_df[
            [
                "model_name",
                "threshold",
                "precision",
                "recall",
                "f1_score",
                "pr_auc",
                "accuracy",
                "tp",
                "fp",
                "fn",
                "tn",
            ]
        ].to_string(index=False)
    )

    print("\n=== IMPORTANCIA DE VARIABLES (DecisionTree_Balanced, max_depth=5) ===")
    imp_df = pd.DataFrame(summary["decision_tree_feature_importances"])
    print(imp_df[["rank", "feature", "importance"]].to_string(index=False))


if __name__ == "__main__":
    main()
