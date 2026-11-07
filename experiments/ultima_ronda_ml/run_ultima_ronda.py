"""Última ronda de investigación y mejora del modelo ML (Investigaciones A, B, C y D).

Script 100% reproducible y aislado en `experiments/ultima_ronda_ml/`.
No modifica ningún archivo en `src/`, `data/` ni `results/`.

Cubre:
- Investigación A (Features): evaluación sistemática de representaciones de features
  (`F0_Actuales_10`, `F1_Solo_Estructurales_5`, `F2_Solo_Historicas_5`,
  `F3_Sin_Redundantes_8`, `F4_Simetrizadas_Log_10`, `F5_Ingenieria_10Cols_16`
  y prueba diagnóstica `D_Con_Nivel_Y_2Hop_Hist` para la Investigación D).
- Investigación B (Modelos permitidos por la consigna + referencias exploratorias):
  1. Regresión Lineal (clasificación por umbral sobre score continuo).
  2. Regresión Logística (L1/L2, grilla de C, ponderaciones de clase, submuestreo).
  3. KNN (K=15, 31, 51, 75, 101; distancias Euclídea y Manhattan; pesos uniformes y por distancia;
     con y sin escalado; distintas representaciones de features).
  4. Árbol de Decisión (profundidad, min_samples_leaf, min_samples_split, criterio gini/entropy/log_loss,
     ponderación de clases).
  5. SVM (kernel lineal y kernel RBF, valores de C, class_weight, escalado, costo de probability=True).
  6. Referencias exploratorias (RandomForest, ExtraTrees, HistGradientBoosting).
- Investigación C (Métricas y umbrales en Validación y Test):
  PR-AUC, Precision, Recall, F1, matriz de confusión (TP, FP, FN, TN), aristas predichas (TP+FP),
  ratio aristas/positivos y FP por cada TP. Selección estricta en Validación.
"""

from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.schema import FEATURE_COLUMNS, TARGET_CLASSIFICATION
from src.models.compare_models import (
    HistoricalOverlapBaseline,
    RANDOM_SEED,
    TEST_FULL_PATH,
    TRAIN_BALANCED_PATH,
    TRAIN_FULL_PATH,
    build_internal_subject_validation_split,
    find_best_f1_threshold_on_validation,
    load_ml_dataframe,
)

OUT_DIR = PROJECT_ROOT / "experiments" / "ultima_ronda_ml" / "results"

STRUCTURAL_COLS = [
    "mismo_departamento",
    "mismo_prefijo_curricular",
    "anio_confirmado",
    "mismo_anio",
    "misma_carrera",
]

HISTORICAL_COLS = [
    "cant_carreras_compartidas_hist",
    "cant_inscriptos_hist_1",
    "cant_inscriptos_hist_2",
    "cant_estudiantes_compartidos_historico",
    "jaccard_historico",
]

# Eliminando mismo_departamento (r=0.9088 con mismo_prefijo_curricular)
# y misma_carrera (r=0.8380 con cant_carreras_compartidas_hist)
NON_REDUNDANT_8_COLS = [
    "mismo_prefijo_curricular",
    "anio_confirmado",
    "mismo_anio",
    "cant_carreras_compartidas_hist",
    "cant_inscriptos_hist_1",
    "cant_inscriptos_hist_2",
    "cant_estudiantes_compartidos_historico",
    "jaccard_historico",
]


# =============================================================================
# Transformaciones de Features (100% libres de Data Leakage, operan sobre las 10 cols)
# =============================================================================


def build_f4_sym_log(df: pd.DataFrame) -> np.ndarray:
    """F4 (10 features): Simetriza (inscriptos_1, inscriptos_2) en (min, max) y aplica log1p a conteos."""
    n1 = df["cant_inscriptos_hist_1"].to_numpy(dtype=float)
    n2 = df["cant_inscriptos_hist_2"].to_numpy(dtype=float)
    min_h = np.minimum(n1, n2)
    max_h = np.maximum(n1, n2)
    shared_h = df["cant_estudiantes_compartidos_historico"].to_numpy(dtype=float)
    carr_h = df["cant_carreras_compartidas_hist"].to_numpy(dtype=float)

    return np.column_stack(
        [
            df["mismo_departamento"].to_numpy(dtype=float),
            df["mismo_prefijo_curricular"].to_numpy(dtype=float),
            df["anio_confirmado"].to_numpy(dtype=float),
            df["mismo_anio"].to_numpy(dtype=float),
            df["misma_carrera"].to_numpy(dtype=float),
            np.log1p(carr_h),
            np.log1p(min_h),
            np.log1p(max_h),
            np.log1p(shared_h),
            df["jaccard_historico"].to_numpy(dtype=float),
        ]
    )


def build_f5_engineered_10cols(df: pd.DataFrame) -> np.ndarray:
    """F5 (16 features): Ingeniería derivada exclusivamente de las 10 features oficiales."""
    depto = df["mismo_departamento"].to_numpy(dtype=float)
    pref = df["mismo_prefijo_curricular"].to_numpy(dtype=float)
    anio_conf = df["anio_confirmado"].to_numpy(dtype=float)
    mismo_anio = df["mismo_anio"].to_numpy(dtype=float)
    misma_carr = df["misma_carrera"].to_numpy(dtype=float)
    carr_h = df["cant_carreras_compartidas_hist"].to_numpy(dtype=float)
    n1 = df["cant_inscriptos_hist_1"].to_numpy(dtype=float)
    n2 = df["cant_inscriptos_hist_2"].to_numpy(dtype=float)
    shared_h = df["cant_estudiantes_compartidos_historico"].to_numpy(dtype=float)
    jacc_h = df["jaccard_historico"].to_numpy(dtype=float)

    min_h = np.minimum(n1, n2)
    max_h = np.maximum(n1, n2)
    log_min_h = np.log1p(min_h)
    log_max_h = np.log1p(max_h)
    diff_log_h = log_max_h - log_min_h

    # Indicadores de ausencia de historial (proxy de materias de 1.er año sin cohorte previa)
    both_zero_h = ((max_h == 0)).astype(float)
    one_zero_h = ((min_h == 0) & (max_h > 0)).astype(float)

    # Indicador binario y coeficiente de Szymkiewicz-Simpson histórico
    has_shared_h = (shared_h > 0).astype(float)
    overlap_min_coef = np.where(min_h > 0, shared_h / np.maximum(1.0, min_h), 0.0)

    # Interacciones estructurales
    inter_carr_anio = misma_carr * mismo_anio
    inter_pref_anio = pref * mismo_anio
    inter_no_anio_conf_carr = (1.0 - anio_conf) * misma_carr

    return np.column_stack(
        [
            depto,
            pref,
            anio_conf,
            mismo_anio,
            misma_carr,
            np.log1p(carr_h),
            log_min_h,
            log_max_h,
            diff_log_h,
            both_zero_h,
            one_zero_h,
            has_shared_h,
            np.log1p(shared_h),
            jacc_h,
            overlap_min_coef,
            inter_carr_anio + inter_pref_anio + 0.5 * inter_no_anio_conf_carr,
        ]
    )


def build_diagnostic_d_metadata_Lookup() -> Callable[[pd.DataFrame], np.ndarray]:
    """Construye transformador diagnóstico para Investigación D (evaluando qué aportaría
    el nivel '1'/'A' vs '2'/'B' vs '3'/'C' de `exams_clean.csv` y vecinos comunes a 2 saltos
    en `historico_1989_1993`).
    """
    exams_df = pd.read_csv(PROJECT_ROOT / "data" / "interim" / "exams_clean.csv")
    students_df = pd.read_csv(PROJECT_ROOT / "data" / "interim" / "students_clean.csv")
    enrol_df = pd.read_csv(PROJECT_ROOT / "data" / "interim" / "enrolments_clean.csv")

    level_map = {"0": 1, "P": 1, "1": 1, "A": 1, "2": 2, "B": 2, "3": 3, "C": 3, "4": 4}
    exam_equiv_lvl: Dict[str, int] = {}
    exam_raw_lvl: Dict[str, str] = {}
    for r in exams_df.itertuples(index=False):
        exam_raw_lvl[r.exam_id] = str(r.raw_level_char)
        exam_equiv_lvl[r.exam_id] = level_map.get(str(r.raw_level_char), 0)

    # Grafo histórico (solo cohortes 1989-1993) para calcular vecinos comunes a 2 saltos
    hist_students = set(
        students_df[students_df["cohort_group"] == "historico_1989_1993"]["student_id"]
    )
    hist_enrol = enrol_df[enrol_df["student_id"].isin(hist_students)]
    student_to_exams: Dict[str, List[str]] = defaultdict(list)
    for r in hist_enrol.itertuples(index=False):
        student_to_exams[r.student_id].append(r.exam_id)

    hist_adj: Dict[str, set] = defaultdict(set)
    for ex_list in student_to_exams.values():
        u_list = list(set(ex_list))
        for i, e1 in enumerate(u_list):
            for e2 in u_list[i + 1 :]:
                hist_adj[e1].add(e2)
                hist_adj[e2].add(e1)

    def transform_d(df: pd.DataFrame) -> np.ndarray:
        base_f5 = build_f5_engineered_10cols(df)
        m1_arr = df["materia_id_1"].to_numpy()
        m2_arr = df["materia_id_2"].to_numpy()
        n = len(df)
        same_equiv_lvl = np.zeros(n, dtype=float)
        both_lvl_1 = np.zeros(n, dtype=float)
        any_lvl_1 = np.zeros(n, dtype=float)
        both_upper_lvl = np.zeros(n, dtype=float)
        same_faculty_1c = np.zeros(n, dtype=float)
        common_neighbors_log = np.zeros(n, dtype=float)

        for idx in range(n):
            e1 = m1_arr[idx]
            e2 = m2_arr[idx]
            l1 = exam_equiv_lvl.get(e1, 0)
            l2 = exam_equiv_lvl.get(e2, 0)
            if l1 == l2 and l1 > 0:
                same_equiv_lvl[idx] = 1.0
            if l1 == 1 and l2 == 1:
                both_lvl_1[idx] = 1.0
            if l1 == 1 or l2 == 1:
                any_lvl_1[idx] = 1.0
            if l1 >= 2 and l2 >= 2:
                both_upper_lvl[idx] = 1.0
            if e1[0] == e2[0]:
                same_faculty_1c[idx] = 1.0
            cn = len(hist_adj.get(e1, set()) & hist_adj.get(e2, set()))
            if cn > 0:
                common_neighbors_log[idx] = np.log1p(cn)

        return np.column_stack(
            [
                base_f5,
                same_equiv_lvl,
                both_lvl_1,
                any_lvl_1,
                both_upper_lvl,
                same_faculty_1c,
                common_neighbors_log,
            ]
        )

    return transform_d


# =============================================================================
# Funciones auxiliares de evaluación y selección de umbral en Validación
# =============================================================================


def get_continuous_scores(model: Any, X: np.ndarray) -> np.ndarray:
    """Obtiene probabilidades en [0, 1] o scores continuos monotonizados en [0, 1]."""
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(X)[:, 1], dtype=float)
    if hasattr(model, "decision_function"):
        raw = np.asarray(model.decision_function(X), dtype=float)
        # Sigmoid para mapear a (0, 1) preservando estrictamente el orden de ranking
        return 1.0 / (1.0 + np.exp(-np.clip(raw, -50.0, 50.0)))
    raw = np.asarray(model.predict(X), dtype=float)
    return np.clip(raw, 0.0, 1.0)


def select_best_f1_threshold_val(y_val: np.ndarray, scores_val: np.ndarray) -> float:
    """Usa exactamente la función oficial de selección de umbral en Validación (grilla 0.05..0.95)."""
    best_th = find_best_f1_threshold_on_validation(y_val, scores_val)
    return round(float(best_th), 4)


def select_recall_target_threshold_val(
    y_val: np.ndarray, scores_val: np.ndarray, target_recall: float
) -> float:
    """Busca en Validación el umbral más alto que alcanza Recall >= target_recall."""
    candidates = np.sort(
        np.unique(
            np.concatenate(
                [
                    np.linspace(0.001, 0.99, 150),
                    np.quantile(scores_val, np.linspace(0.50, 0.999, 80)),
                ]
            )
        )
    )[::-1]
    chosen = float(candidates[-1])
    for th in candidates:
        preds = (scores_val >= th).astype(int)
        rec = float(recall_score(y_val, preds, zero_division=0))
        if rec >= target_recall:
            chosen = float(th)
            break
    return round(chosen, 4)


def compute_metrics_row(
    model_name: str,
    family: str,
    allowed_by_syllabus: bool,
    feature_set: str,
    split_name: str,
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    fit_seconds: float,
    notes: str = "",
) -> Dict[str, Any]:
    preds = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()
    prec = float(precision_score(y_true, preds, zero_division=0))
    rec = float(recall_score(y_true, preds, zero_division=0))
    f1 = float(f1_score(y_true, preds, zero_division=0))
    pr_auc = float(average_precision_score(y_true, scores))
    roc_auc = float(roc_auc_score(y_true, scores))
    acc = float(accuracy_score(y_true, preds))
    pred_edges = int(tp + fp)
    pos_total = int(tp + fn)
    return {
        "model_name": model_name,
        "family": family,
        "allowed_by_syllabus": allowed_by_syllabus,
        "feature_set": feature_set,
        "split": split_name,
        "threshold": round(float(threshold), 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "pr_auc": round(pr_auc, 4),
        "roc_auc": round(roc_auc, 4),
        "accuracy": round(acc, 4),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "predicted_edges": pred_edges,
        "positives_total": pos_total,
        "edges_to_positives_ratio": round(pred_edges / pos_total, 3) if pos_total > 0 else 0.0,
        "fp_per_tp": round(fp / tp, 3) if tp > 0 else None,
        "fit_seconds": round(fit_seconds, 2),
        "notes": notes,
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("1. Cargando datasets oficiales (solo lectura)...")
    train_full_df = load_ml_dataframe(TRAIN_FULL_PATH)
    train_bal_df = load_ml_dataframe(TRAIN_BALANCED_PATH)
    test_full_df = load_ml_dataframe(TEST_FULL_PATH)

    subtrain_full_df, subtrain_bal_df, val_df, _ = (
        build_internal_subject_validation_split(train_full_df)
    )

    y_sub_full = subtrain_full_df[TARGET_CLASSIFICATION].to_numpy(dtype=int)
    y_sub_bal = subtrain_bal_df[TARGET_CLASSIFICATION].to_numpy(dtype=int)
    y_val = val_df[TARGET_CLASSIFICATION].to_numpy(dtype=int)
    y_train_full = train_full_df[TARGET_CLASSIFICATION].to_numpy(dtype=int)
    y_train_bal = train_bal_df[TARGET_CLASSIFICATION].to_numpy(dtype=int)
    y_test = test_full_df[TARGET_CLASSIFICATION].to_numpy(dtype=int)

    # Pre-construir matrices de features para subtrain_full, subtrain_bal, val, train_full, train_bal y test
    print("2. Construyendo representaciones de features (F0..F5 y Diagnóstico D)...")
    transform_d = build_diagnostic_d_metadata_Lookup()

    feature_builders: Dict[str, Callable[[pd.DataFrame], np.ndarray]] = {
        "F0_Actuales_10": lambda df: df[FEATURE_COLUMNS].to_numpy(dtype=float),
        "F1_Solo_Estructurales_5": lambda df: df[STRUCTURAL_COLS].to_numpy(dtype=float),
        "F2_Solo_Historicas_5": lambda df: df[HISTORICAL_COLS].to_numpy(dtype=float),
        "F3_Sin_Redundantes_8": lambda df: df[NON_REDUNDANT_8_COLS].to_numpy(dtype=float),
        "F4_Simetrizadas_Log_10": build_f4_sym_log,
        "F5_Ingenieria_10Cols_16": build_f5_engineered_10cols,
        "D_Diagnostico_Nivel_Y_2Hop_22": transform_d,
    }

    mats_sub_full: Dict[str, np.ndarray] = {}
    mats_sub_bal: Dict[str, np.ndarray] = {}
    mats_val: Dict[str, np.ndarray] = {}
    mats_train_full: Dict[str, np.ndarray] = {}
    mats_train_bal: Dict[str, np.ndarray] = {}
    mats_test: Dict[str, np.ndarray] = {}

    for fname, fn in feature_builders.items():
        mats_sub_full[fname] = fn(subtrain_full_df)
        mats_sub_bal[fname] = fn(subtrain_bal_df)
        mats_val[fname] = fn(val_df)
        if fname != "D_Diagnostico_Nivel_Y_2Hop_22":
            mats_train_full[fname] = fn(train_full_df)
            mats_train_bal[fname] = fn(train_bal_df)
            mats_test[fname] = fn(test_full_df)

    # =========================================================================
    # INVESTIGACIÓN A: Experimentos de representación de features (en Validación)
    # =========================================================================
    print("3. Ejecutando Investigación A (Comparación de representaciones de features)...")
    feat_exp_rows: List[Dict[str, Any]] = []

    rep_models_for_features = [
        (
            "LogReg_L2_C1_Unw",
            "LogisticRegression",
            True,
            lambda: Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("clf", LogisticRegression(C=1.0, max_iter=1000, random_state=RANDOM_SEED)),
                ]
            ),
        ),
        (
            "KNN_k31_Eucl_Unif",
            "KNN",
            True,
            lambda: Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("clf", KNeighborsClassifier(n_neighbors=31, metric="euclidean", weights="uniform", n_jobs=-1)),
                ]
            ),
        ),
        (
            "DT_d6_leaf50_Unw",
            "DecisionTree",
            True,
            lambda: DecisionTreeClassifier(
                max_depth=6, min_samples_leaf=50, criterion="log_loss", random_state=RANDOM_SEED
            ),
        ),
        (
            "LinearSVM_Calib_C1_Unw",
            "SVM",
            True,
            lambda: Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "clf",
                        CalibratedClassifierCV(
                            LinearSVC(C=1.0, random_state=RANDOM_SEED, max_iter=3000),
                            cv=3,
                        ),
                    ),
                ]
            ),
        ),
        (
            "HGB_Ref_Unw",
            "HistGradientBoosting (Ref)",
            False,
            lambda: HistGradientBoostingClassifier(
                max_depth=6, learning_rate=0.05, max_iter=200, random_state=RANDOM_SEED
            ),
        ),
    ]

    for fset_name in [
        "F0_Actuales_10",
        "F1_Solo_Estructurales_5",
        "F2_Solo_Historicas_5",
        "F3_Sin_Redundantes_8",
        "F4_Simetrizadas_Log_10",
        "F5_Ingenieria_10Cols_16",
        "D_Diagnostico_Nivel_Y_2Hop_22",
    ]:
        X_tr = mats_sub_full[fset_name]
        X_va = mats_val[fset_name]
        for m_label, fam, allowed, builder in rep_models_for_features:
            # En KNN omitimos D_Diagnostico para ahorrar tiempo de cómputo innecesario
            if fam == "KNN" and fset_name == "D_Diagnostico_Nivel_Y_2Hop_22":
                continue
            model = builder()
            t0 = time.time()
            model.fit(X_tr, y_sub_full)
            fit_sec = time.time() - t0
            scores_va = get_continuous_scores(model, X_va)
            th_val = select_best_f1_threshold_val(y_val, scores_va)
            row = compute_metrics_row(
                model_name=f"{m_label}__{fset_name}",
                family=fam,
                allowed_by_syllabus=allowed,
                feature_set=fset_name,
                split_name="validation",
                y_true=y_val,
                scores=scores_va,
                threshold=th_val,
                fit_seconds=fit_sec,
                notes=f"n_features={X_tr.shape[1]}",
            )
            feat_exp_rows.append(row)

    pd.DataFrame(feat_exp_rows).to_csv(OUT_DIR / "feature_experiments.csv", index=False)

    # =========================================================================
    # INVESTIGACIÓN B: Comparación controlada de modelos permitidos (+ referencias)
    # =========================================================================
    print("4. Ejecutando Investigación B (Grilla controlada de modelos permitidos)...")

    # Muestra estratificada de 30.000 pares de subtrain_full (conservando los 2.836 positivos
    # y 27.164 negativos aleatorios con semilla fija) exclusivamente para evaluar SVC(kernel='rbf')
    # sin bloqueo cuadrático en libsvm sobre 202.566 filas, además de evaluar en subtrain_balanced (5.672 filas).
    rng = np.random.default_rng(RANDOM_SEED)
    pos_idx = np.where(y_sub_full == 1)[0]
    neg_idx = np.where(y_sub_full == 0)[0]
    sampled_neg_idx = rng.choice(neg_idx, size=27164, replace=False)
    svm_sub30k_idx = np.concatenate([pos_idx, sampled_neg_idx])
    rng.shuffle(svm_sub30k_idx)
    y_sub_30k = y_sub_full[svm_sub30k_idx]

    pos_tr_idx = np.where(y_train_full == 1)[0]
    neg_tr_idx = np.where(y_train_full == 0)[0]
    sampled_neg_tr_idx = rng.choice(neg_tr_idx, size=30000 - len(pos_tr_idx), replace=False)
    svm_train30k_idx = np.concatenate([pos_tr_idx, sampled_neg_tr_idx])
    rng.shuffle(svm_train30k_idx)
    y_train_30k = y_train_full[svm_train30k_idx]

    # Lista de especificaciones:
    # (name, family, allowed_by_syllabus, feature_set, train_regime, estimator_factory, notes)
    model_specs: List[Tuple[str, str, bool, str, str, Callable[[], Any], str]] = [
        # --- 0. Baseline Heurístico ---
        (
            "Baseline_Heuristico_Historico",
            "Baseline",
            True,
            "F0_Actuales_10",
            "full",
            lambda: HistoricalOverlapBaseline(),
            "Regla compartidos_hist > 0",
        ),
        # --- 1. Regresión Lineal (permitida por consigna) ---
        (
            "LinReg_Ridge_a1_F0",
            "LinearRegression",
            True,
            "F0_Actuales_10",
            "full",
            lambda: Pipeline([("scaler", StandardScaler()), ("reg", Ridge(alpha=1.0, random_state=RANDOM_SEED))]),
            "Regresión lineal L2 (alpha=1) sobre F0",
        ),
        (
            "LinReg_Ridge_a1_F4_SymLog",
            "LinearRegression",
            True,
            "F4_Simetrizadas_Log_10",
            "full",
            lambda: Pipeline([("scaler", StandardScaler()), ("reg", Ridge(alpha=1.0, random_state=RANDOM_SEED))]),
            "Regresión lineal L2 (alpha=1) sobre F4 (log + simetrización)",
        ),
        (
            "LinReg_Ridge_a10_F5_Eng",
            "LinearRegression",
            True,
            "F5_Ingenieria_10Cols_16",
            "full",
            lambda: Pipeline([("scaler", StandardScaler()), ("reg", Ridge(alpha=10.0, random_state=RANDOM_SEED))]),
            "Regresión lineal L2 (alpha=10) sobre F5 (ingeniería 16 vars)",
        ),
        # --- 2. Regresión Logística (permitida por consigna) ---
        (
            "LogReg_L2_C1_Unw_F0",
            "LogisticRegression",
            True,
            "F0_Actuales_10",
            "full",
            lambda: Pipeline(
                [("scaler", StandardScaler()), ("clf", LogisticRegression(C=1.0, max_iter=1000, random_state=RANDOM_SEED))]
            ),
            "Configuración actual (L2, C=1.0, sin pesos, F0)",
        ),
        (
            "LogReg_L2_C1_Balanced_F0",
            "LogisticRegression",
            True,
            "F0_Actuales_10",
            "full",
            lambda: Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("clf", LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000, random_state=RANDOM_SEED)),
                ]
            ),
            "L2, C=1.0, class_weight=balanced, F0",
        ),
        (
            "LogReg_L2_C1_Unw_F4_SymLog",
            "LogisticRegression",
            True,
            "F4_Simetrizadas_Log_10",
            "full",
            lambda: Pipeline(
                [("scaler", StandardScaler()), ("clf", LogisticRegression(C=1.0, max_iter=1000, random_state=RANDOM_SEED))]
            ),
            "L2, C=1.0, sin pesos, F4 (log + simetrización)",
        ),
        (
            "LogReg_L2_C0.1_Unw_F5_Eng",
            "LogisticRegression",
            True,
            "F5_Ingenieria_10Cols_16",
            "full",
            lambda: Pipeline(
                [("scaler", StandardScaler()), ("clf", LogisticRegression(C=0.1, max_iter=1000, random_state=RANDOM_SEED))]
            ),
            "L2, C=0.1, sin pesos, F5 (ingeniería 16 vars)",
        ),
        (
            "LogReg_L2_C1_Unw_F5_Eng",
            "LogisticRegression",
            True,
            "F5_Ingenieria_10Cols_16",
            "full",
            lambda: Pipeline(
                [("scaler", StandardScaler()), ("clf", LogisticRegression(C=1.0, max_iter=1000, random_state=RANDOM_SEED))]
            ),
            "L2, C=1.0, sin pesos, F5 (ingeniería 16 vars)",
        ),
        (
            "LogReg_L2_C10_Unw_F5_Eng",
            "LogisticRegression",
            True,
            "F5_Ingenieria_10Cols_16",
            "full",
            lambda: Pipeline(
                [("scaler", StandardScaler()), ("clf", LogisticRegression(C=10.0, max_iter=1000, random_state=RANDOM_SEED))]
            ),
            "L2, C=10.0, sin pesos, F5",
        ),
        (
            "LogReg_L1_C1_Unw_F5_Eng",
            "LogisticRegression",
            True,
            "F5_Ingenieria_10Cols_16",
            "full",
            lambda: Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "clf",
                        LogisticRegression(
                            C=1.0, penalty="l1", solver="saga", max_iter=1000, random_state=RANDOM_SEED
                        ),
                    ),
                ]
            ),
            "Penalización L1 (saga), C=1.0, sin pesos, F5",
        ),
        (
            "LogReg_L2_C1_Weight5_F5_Eng",
            "LogisticRegression",
            True,
            "F5_Ingenieria_10Cols_16",
            "full",
            lambda: Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "clf",
                        LogisticRegression(
                            C=1.0, class_weight={0: 1, 1: 5}, max_iter=1000, random_state=RANDOM_SEED
                        ),
                    ),
                ]
            ),
            "L2, C=1.0, class_weight={0:1, 1:5}, F5",
        ),
        (
            "LogReg_L2_C1_Weight10_F5_Eng",
            "LogisticRegression",
            True,
            "F5_Ingenieria_10Cols_16",
            "full",
            lambda: Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "clf",
                        LogisticRegression(
                            C=1.0, class_weight={0: 1, 1: 10}, max_iter=1000, random_state=RANDOM_SEED
                        ),
                    ),
                ]
            ),
            "L2, C=1.0, class_weight={0:1, 1:10}, F5",
        ),
        (
            "LogReg_L2_C1_Balanced_F5_Eng",
            "LogisticRegression",
            True,
            "F5_Ingenieria_10Cols_16",
            "full",
            lambda: Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "clf",
                        LogisticRegression(
                            C=1.0, class_weight="balanced", max_iter=1000, random_state=RANDOM_SEED
                        ),
                    ),
                ]
            ),
            "L2, C=1.0, class_weight=balanced, F5",
        ),
        (
            "LogReg_L2_C1_Undersamp_F5_Eng",
            "LogisticRegression",
            True,
            "F5_Ingenieria_10Cols_16",
            "balanced",
            lambda: Pipeline(
                [("scaler", StandardScaler()), ("clf", LogisticRegression(C=1.0, max_iter=1000, random_state=RANDOM_SEED))]
            ),
            "L2, C=1.0, entrenado en subtrain_balanced (50/50), F5",
        ),
        # --- 3. KNN (permitido por consigna) ---
        (
            "KNN_k15_Eucl_Unif_Scaled_F0",
            "KNN",
            True,
            "F0_Actuales_10",
            "full",
            lambda: Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("clf", KNeighborsClassifier(n_neighbors=15, metric="euclidean", weights="uniform", n_jobs=-1)),
                ]
            ),
            "Modelo Oficial Actual (K=15, Euclídea, Uniforme, StandardScaler, F0)",
        ),
        (
            "KNN_k15_Eucl_Unif_Unscaled_F0",
            "KNN",
            True,
            "F0_Actuales_10",
            "full",
            lambda: KNeighborsClassifier(n_neighbors=15, metric="euclidean", weights="uniform", n_jobs=-1),
            "K=15 sin escalar (dominado por conteos brutos)",
        ),
        (
            "KNN_k15_Eucl_Dist_Scaled_F0",
            "KNN",
            True,
            "F0_Actuales_10",
            "full",
            lambda: Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("clf", KNeighborsClassifier(n_neighbors=15, metric="euclidean", weights="distance", n_jobs=-1)),
                ]
            ),
            "K=15, Euclídea, weights=distance, F0",
        ),
        (
            "KNN_k31_Eucl_Unif_Scaled_F0",
            "KNN",
            True,
            "F0_Actuales_10",
            "full",
            lambda: Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("clf", KNeighborsClassifier(n_neighbors=31, metric="euclidean", weights="uniform", n_jobs=-1)),
                ]
            ),
            "K=31, Euclídea, Uniforme, F0",
        ),
        (
            "KNN_k31_Manh_Unif_Scaled_F0",
            "KNN",
            True,
            "F0_Actuales_10",
            "full",
            lambda: Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("clf", KNeighborsClassifier(n_neighbors=31, metric="manhattan", weights="uniform", n_jobs=-1)),
                ]
            ),
            "K=31, Manhattan (L1), Uniforme, F0",
        ),
        (
            "KNN_k51_Eucl_Unif_Scaled_F0",
            "KNN",
            True,
            "F0_Actuales_10",
            "full",
            lambda: Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("clf", KNeighborsClassifier(n_neighbors=51, metric="euclidean", weights="uniform", n_jobs=-1)),
                ]
            ),
            "K=51, Euclídea, Uniforme, F0",
        ),
        (
            "KNN_k51_Manh_Unif_Scaled_F0",
            "KNN",
            True,
            "F0_Actuales_10",
            "full",
            lambda: Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("clf", KNeighborsClassifier(n_neighbors=51, metric="manhattan", weights="uniform", n_jobs=-1)),
                ]
            ),
            "K=51, Manhattan (L1), Uniforme, F0",
        ),
        (
            "KNN_k51_Eucl_Dist_Scaled_F0",
            "KNN",
            True,
            "F0_Actuales_10",
            "full",
            lambda: Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("clf", KNeighborsClassifier(n_neighbors=51, metric="euclidean", weights="distance", n_jobs=-1)),
                ]
            ),
            "K=51, Euclídea, weights=distance, F0",
        ),
        (
            "KNN_k75_Eucl_Unif_Scaled_F0",
            "KNN",
            True,
            "F0_Actuales_10",
            "full",
            lambda: Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("clf", KNeighborsClassifier(n_neighbors=75, metric="euclidean", weights="uniform", n_jobs=-1)),
                ]
            ),
            "K=75, Euclídea, Uniforme, F0",
        ),
        (
            "KNN_k31_Eucl_Unif_Scaled_F4_SymLog",
            "KNN",
            True,
            "F4_Simetrizadas_Log_10",
            "full",
            lambda: Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("clf", KNeighborsClassifier(n_neighbors=31, metric="euclidean", weights="uniform", n_jobs=-1)),
                ]
            ),
            "K=31, Euclídea, Uniforme, sobre F4 (log + simetrización)",
        ),
        (
            "KNN_k51_Manh_Unif_Scaled_F4_SymLog",
            "KNN",
            True,
            "F4_Simetrizadas_Log_10",
            "full",
            lambda: Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("clf", KNeighborsClassifier(n_neighbors=51, metric="manhattan", weights="uniform", n_jobs=-1)),
                ]
            ),
            "K=51, Manhattan, Uniforme, sobre F4 (log + simetrización)",
        ),
        (
            "KNN_k75_Manh_Unif_Scaled_F4_SymLog",
            "KNN",
            True,
            "F4_Simetrizadas_Log_10",
            "full",
            lambda: Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("clf", KNeighborsClassifier(n_neighbors=75, metric="manhattan", weights="uniform", n_jobs=-1)),
                ]
            ),
            "K=75, Manhattan, Uniforme, sobre F4 (log + simetrización)",
        ),
        (
            "KNN_k51_Manh_Unif_Scaled_F5_Eng",
            "KNN",
            True,
            "F5_Ingenieria_10Cols_16",
            "full",
            lambda: Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("clf", KNeighborsClassifier(n_neighbors=51, metric="manhattan", weights="uniform", n_jobs=-1)),
                ]
            ),
            "K=51, Manhattan, Uniforme, sobre F5 (ingeniería 16 vars)",
        ),
        # --- 4. Árboles de Decisión (permitidos por consigna) ---
        (
            "DT_d5_leaf1_Gini_Unw_F0",
            "DecisionTree",
            True,
            "F0_Actuales_10",
            "full",
            lambda: DecisionTreeClassifier(max_depth=5, criterion="gini", random_state=RANDOM_SEED),
            "Árbol Fase 3 original (max_depth=5, gini, sin pesos, F0)",
        ),
        (
            "DT_d4_leaf50_Gini_Unw_F0",
            "DecisionTree",
            True,
            "F0_Actuales_10",
            "full",
            lambda: DecisionTreeClassifier(max_depth=4, min_samples_leaf=50, criterion="gini", random_state=RANDOM_SEED),
            "max_depth=4, min_samples_leaf=50, gini, F0",
        ),
        (
            "DT_d6_leaf50_Gini_Unw_F0",
            "DecisionTree",
            True,
            "F0_Actuales_10",
            "full",
            lambda: DecisionTreeClassifier(
                max_depth=6, min_samples_leaf=50, min_samples_split=100, criterion="gini", random_state=RANDOM_SEED
            ),
            "max_depth=6, min_samples_leaf=50, min_split=100, gini, F0",
        ),
        (
            "DT_d6_leaf50_LogLoss_Unw_F0",
            "DecisionTree",
            True,
            "F0_Actuales_10",
            "full",
            lambda: DecisionTreeClassifier(
                max_depth=6, min_samples_leaf=50, min_samples_split=100, criterion="log_loss", random_state=RANDOM_SEED
            ),
            "max_depth=6, min_samples_leaf=50, criterion=log_loss (entropy), F0",
        ),
        (
            "DT_d8_leaf100_LogLoss_Unw_F0",
            "DecisionTree",
            True,
            "F0_Actuales_10",
            "full",
            lambda: DecisionTreeClassifier(
                max_depth=8, min_samples_leaf=100, min_samples_split=200, criterion="log_loss", random_state=RANDOM_SEED
            ),
            "max_depth=8, min_samples_leaf=100, criterion=log_loss, F0",
        ),
        (
            "DT_d10_leaf100_LogLoss_Unw_F0",
            "DecisionTree",
            True,
            "F0_Actuales_10",
            "full",
            lambda: DecisionTreeClassifier(
                max_depth=10, min_samples_leaf=100, min_samples_split=200, criterion="log_loss", random_state=RANDOM_SEED
            ),
            "max_depth=10, min_samples_leaf=100, criterion=log_loss, F0",
        ),
        (
            "DT_d6_leaf50_LogLoss_Unw_F4_SymLog",
            "DecisionTree",
            True,
            "F4_Simetrizadas_Log_10",
            "full",
            lambda: DecisionTreeClassifier(
                max_depth=6, min_samples_leaf=50, min_samples_split=100, criterion="log_loss", random_state=RANDOM_SEED
            ),
            "max_depth=6, min_samples_leaf=50, log_loss, F4 (simetrizadas)",
        ),
        (
            "DT_d8_leaf100_LogLoss_Unw_F4_SymLog",
            "DecisionTree",
            True,
            "F4_Simetrizadas_Log_10",
            "full",
            lambda: DecisionTreeClassifier(
                max_depth=8, min_samples_leaf=100, min_samples_split=200, criterion="log_loss", random_state=RANDOM_SEED
            ),
            "max_depth=8, min_samples_leaf=100, log_loss, F4 (simetrizadas)",
        ),
        (
            "DT_d6_leaf50_LogLoss_Unw_F5_Eng",
            "DecisionTree",
            True,
            "F5_Ingenieria_10Cols_16",
            "full",
            lambda: DecisionTreeClassifier(
                max_depth=6, min_samples_leaf=50, min_samples_split=100, criterion="log_loss", random_state=RANDOM_SEED
            ),
            "max_depth=6, min_samples_leaf=50, log_loss, F5 (ingeniería 16 vars)",
        ),
        (
            "DT_d7_leaf75_LogLoss_Unw_F5_Eng",
            "DecisionTree",
            True,
            "F5_Ingenieria_10Cols_16",
            "full",
            lambda: DecisionTreeClassifier(
                max_depth=7, min_samples_leaf=75, min_samples_split=150, criterion="log_loss", random_state=RANDOM_SEED
            ),
            "max_depth=7, min_samples_leaf=75, log_loss, F5",
        ),
        (
            "DT_d8_leaf100_LogLoss_Unw_F5_Eng",
            "DecisionTree",
            True,
            "F5_Ingenieria_10Cols_16",
            "full",
            lambda: DecisionTreeClassifier(
                max_depth=8, min_samples_leaf=100, min_samples_split=200, criterion="log_loss", random_state=RANDOM_SEED
            ),
            "max_depth=8, min_samples_leaf=100, log_loss, F5",
        ),
        (
            "DT_d8_leaf100_Gini_Unw_F5_Eng",
            "DecisionTree",
            True,
            "F5_Ingenieria_10Cols_16",
            "full",
            lambda: DecisionTreeClassifier(
                max_depth=8, min_samples_leaf=100, min_samples_split=200, criterion="gini", random_state=RANDOM_SEED
            ),
            "max_depth=8, min_samples_leaf=100, gini, F5",
        ),
        (
            "DT_d7_leaf75_LogLoss_Weight5_F5_Eng",
            "DecisionTree",
            True,
            "F5_Ingenieria_10Cols_16",
            "full",
            lambda: DecisionTreeClassifier(
                max_depth=7,
                min_samples_leaf=75,
                min_samples_split=150,
                criterion="log_loss",
                class_weight={0: 1, 1: 5},
                random_state=RANDOM_SEED,
            ),
            "max_depth=7, min_samples_leaf=75, log_loss, class_weight={0:1, 1:5}, F5",
        ),
        (
            "DT_d7_leaf75_LogLoss_Balanced_F5_Eng",
            "DecisionTree",
            True,
            "F5_Ingenieria_10Cols_16",
            "full",
            lambda: DecisionTreeClassifier(
                max_depth=7,
                min_samples_leaf=75,
                min_samples_split=150,
                criterion="log_loss",
                class_weight="balanced",
                random_state=RANDOM_SEED,
            ),
            "max_depth=7, min_samples_leaf=75, log_loss, class_weight=balanced, F5",
        ),
        # --- 5. SVM (permitido por consigna) ---
        (
            "SVM_Linear_C0.1_Unw_F0",
            "SVM",
            True,
            "F0_Actuales_10",
            "full",
            lambda: Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("clf", CalibratedClassifierCV(LinearSVC(C=0.1, random_state=RANDOM_SEED, max_iter=3000), cv=3)),
                ]
            ),
            "SVM Lineal (Calibrated LinearSVC, C=0.1, sin pesos, F0)",
        ),
        (
            "SVM_Linear_C1_Unw_F0",
            "SVM",
            True,
            "F0_Actuales_10",
            "full",
            lambda: Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("clf", CalibratedClassifierCV(LinearSVC(C=1.0, random_state=RANDOM_SEED, max_iter=3000), cv=3)),
                ]
            ),
            "SVM Lineal (Calibrated LinearSVC, C=1.0, sin pesos, F0)",
        ),
        (
            "SVM_Linear_C1_Balanced_F0",
            "SVM",
            True,
            "F0_Actuales_10",
            "full",
            lambda: Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "clf",
                        CalibratedClassifierCV(
                            LinearSVC(C=1.0, class_weight="balanced", random_state=RANDOM_SEED, max_iter=3000), cv=3
                        ),
                    ),
                ]
            ),
            "SVM Lineal (Calibrated LinearSVC, C=1.0, class_weight=balanced, F0)",
        ),
        (
            "SVM_Linear_C1_Unw_F5_Eng",
            "SVM",
            True,
            "F5_Ingenieria_10Cols_16",
            "full",
            lambda: Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("clf", CalibratedClassifierCV(LinearSVC(C=1.0, random_state=RANDOM_SEED, max_iter=3000), cv=3)),
                ]
            ),
            "SVM Lineal (Calibrated LinearSVC, C=1.0, sin pesos, F5)",
        ),
        (
            "SVM_Linear_C1_Weight5_F5_Eng",
            "SVM",
            True,
            "F5_Ingenieria_10Cols_16",
            "full",
            lambda: Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "clf",
                        CalibratedClassifierCV(
                            LinearSVC(C=1.0, class_weight={0: 1, 1: 5}, random_state=RANDOM_SEED, max_iter=3000), cv=3
                        ),
                    ),
                ]
            ),
            "SVM Lineal (Calibrated LinearSVC, C=1.0, class_weight={0:1, 1:5}, F5)",
        ),
        (
            "SVM_Linear_C1_Balanced_F5_Eng",
            "SVM",
            True,
            "F5_Ingenieria_10Cols_16",
            "full",
            lambda: Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "clf",
                        CalibratedClassifierCV(
                            LinearSVC(C=1.0, class_weight="balanced", random_state=RANDOM_SEED, max_iter=3000), cv=3
                        ),
                    ),
                ]
            ),
            "SVM Lineal (Calibrated LinearSVC, C=1.0, class_weight=balanced, F5)",
        ),
        (
            "SVC_RBF_C1_BalUndersamp_NoProb_F0",
            "SVM",
            True,
            "F0_Actuales_10",
            "balanced",
            lambda: Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("clf", SVC(kernel="rbf", C=1.0, probability=False, cache_size=1000, random_state=RANDOM_SEED)),
                ]
            ),
            "SVC RBF (C=1.0, decision_function, subtrain_balanced, F0)",
        ),
        (
            "SVC_RBF_C1_BalUndersamp_ProbTrue_F0",
            "SVM",
            True,
            "F0_Actuales_10",
            "balanced",
            lambda: Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("clf", SVC(kernel="rbf", C=1.0, probability=True, cache_size=1000, random_state=RANDOM_SEED)),
                ]
            ),
            "SVC RBF (C=1.0, probability=True [5-fold Platt], subtrain_balanced, F0)",
        ),
        (
            "SVC_RBF_C1_BalUndersamp_ProbTrue_F5_Eng",
            "SVM",
            True,
            "F5_Ingenieria_10Cols_16",
            "balanced",
            lambda: Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("clf", SVC(kernel="rbf", C=1.0, probability=True, cache_size=1000, random_state=RANDOM_SEED)),
                ]
            ),
            "SVC RBF (C=1.0, probability=True, subtrain_balanced, F5)",
        ),
        (
            "SVC_RBF_C10_BalUndersamp_ProbTrue_F5_Eng",
            "SVM",
            True,
            "F5_Ingenieria_10Cols_16",
            "balanced",
            lambda: Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("clf", SVC(kernel="rbf", C=10.0, probability=True, cache_size=1000, random_state=RANDOM_SEED)),
                ]
            ),
            "SVC RBF (C=10.0, probability=True, subtrain_balanced, F5)",
        ),
        (
            "SVC_RBF_C1_Sub30k_Unw_NoProb_F5_Eng",
            "SVM",
            True,
            "F5_Ingenieria_10Cols_16",
            "sub30k",
            lambda: Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("clf", SVC(kernel="rbf", C=1.0, probability=False, cache_size=1000, random_state=RANDOM_SEED)),
                ]
            ),
            "SVC RBF (C=1.0, decision_function, muestra 30k sin pesos, F5)",
        ),
        (
            "SVC_RBF_C1_Sub30k_Unw_ProbTrue_F5_Eng",
            "SVM",
            True,
            "F5_Ingenieria_10Cols_16",
            "sub30k",
            lambda: Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("clf", SVC(kernel="rbf", C=1.0, probability=True, cache_size=1000, random_state=RANDOM_SEED)),
                ]
            ),
            "SVC RBF (C=1.0, probability=True [5x costo], muestra 30k sin pesos, F5)",
        ),
        (
            "SVC_RBF_C1_Sub30k_Weight5_NoProb_F5_Eng",
            "SVM",
            True,
            "F5_Ingenieria_10Cols_16",
            "sub30k",
            lambda: Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "clf",
                        SVC(
                            kernel="rbf",
                            C=1.0,
                            class_weight={0: 1, 1: 5},
                            probability=False,
                            cache_size=1000,
                            random_state=RANDOM_SEED,
                        ),
                    ),
                ]
            ),
            "SVC RBF (C=1.0, class_weight={0:1, 1:5}, muestra 30k, F5)",
        ),
        # --- 6. Referencias Exploratorias (NO candidatos principales salvo permiso explícito) ---
        (
            "Ref_RF_d6_Unw_F0",
            "RandomForest (Ref)",
            False,
            "F0_Actuales_10",
            "full",
            lambda: RandomForestClassifier(
                n_estimators=150, max_depth=6, min_samples_leaf=25, random_state=RANDOM_SEED, n_jobs=-1
            ),
            "Referencia exploratoria: RandomForest d=6 sobre F0",
        ),
        (
            "Ref_ET_d8_Unw_F0",
            "ExtraTrees (Ref)",
            False,
            "F0_Actuales_10",
            "full",
            lambda: ExtraTreesClassifier(
                n_estimators=150, max_depth=8, min_samples_leaf=25, random_state=RANDOM_SEED, n_jobs=-1
            ),
            "Referencia exploratoria: ExtraTrees d=8 sobre F0",
        ),
        (
            "Ref_HGB_d6_Unw_F0",
            "HistGradientBoosting (Ref)",
            False,
            "F0_Actuales_10",
            "full",
            lambda: HistGradientBoostingClassifier(
                max_depth=6, learning_rate=0.05, max_iter=200, random_state=RANDOM_SEED
            ),
            "Referencia exploratoria: HistGradientBoosting sobre F0",
        ),
        (
            "Ref_HGB_d6_Unw_F5_Eng",
            "HistGradientBoosting (Ref)",
            False,
            "F5_Ingenieria_10Cols_16",
            "full",
            lambda: HistGradientBoostingClassifier(
                max_depth=6, learning_rate=0.05, max_iter=200, random_state=RANDOM_SEED
            ),
            "Referencia exploratoria: HistGradientBoosting sobre F5",
        ),
    ]

    comparison_rows: List[Dict[str, Any]] = []
    val_scores_store: Dict[str, np.ndarray] = {}
    test_scores_store: Dict[str, np.ndarray] = {}
    val_best_th_store: Dict[str, float] = {}
    spec_meta_store: Dict[str, Tuple[str, bool, str, float]] = {}

    for m_name, fam, allowed, fset, regime, factory, notes in model_specs:
        print(f"   -> Entrenando {m_name} ({regime}, {fset})...")
        if regime == "full":
            X_tr_sub = mats_sub_full[fset]
            y_tr_sub = y_sub_full
            X_tr_final = mats_train_full[fset]
            y_tr_final = y_train_full
        elif regime == "balanced":
            X_tr_sub = mats_sub_bal[fset]
            y_tr_sub = y_sub_bal
            X_tr_final = mats_train_bal[fset]
            y_tr_final = y_train_bal
        elif regime == "sub30k":
            X_tr_sub = mats_sub_full[fset][svm_sub30k_idx]
            y_tr_sub = y_sub_30k
            X_tr_final = mats_train_full[fset][svm_train30k_idx]
            y_tr_final = y_train_30k
        else:
            raise ValueError(f"Régimen desconocido: {regime}")

        X_va = mats_val[fset]
        X_te = mats_test[fset]

        est_val = factory()
        t0 = time.time()
        if isinstance(est_val, HistoricalOverlapBaseline):
            est_val.fit(subtrain_full_df[FEATURE_COLUMNS], y_sub_full)
            fit_sec = time.time() - t0
            scores_va = est_val.predict_proba(val_df[FEATURE_COLUMNS])[:, 1]
            scores_te = est_val.predict_proba(test_full_df[FEATURE_COLUMNS])[:, 1]
            th_val = 0.0001
        else:
            est_val.fit(X_tr_sub, y_tr_sub)
            fit_sec = time.time() - t0
            scores_va = get_continuous_scores(est_val, X_va)
            th_val = select_best_f1_threshold_val(y_val, scores_va)

            # Re-entrenar en train_full / train_bal (protocolo oficial Fase 3) para evaluar en Test con th_val fijo
            est_test = factory()
            est_test.fit(X_tr_final, y_tr_final)
            scores_te = get_continuous_scores(est_test, X_te)

        val_scores_store[m_name] = scores_va
        test_scores_store[m_name] = scores_te
        val_best_th_store[m_name] = th_val
        spec_meta_store[m_name] = (fam, allowed, fset, fit_sec)

        # Registrar fila en Validación (umbral seleccionado en Validación)
        comparison_rows.append(
            compute_metrics_row(
                model_name=m_name,
                family=fam,
                allowed_by_syllabus=allowed,
                feature_set=fset,
                split_name="validation",
                y_true=y_val,
                scores=scores_va,
                threshold=th_val,
                fit_seconds=fit_sec,
                notes=notes,
            )
        )
        # Registrar evaluación correspondiente en Test con el MISMO umbral fijado en Validación
        comparison_rows.append(
            compute_metrics_row(
                model_name=m_name,
                family=fam,
                allowed_by_syllabus=allowed,
                feature_set=fset,
                split_name="test",
                y_true=y_test,
                scores=scores_te,
                threshold=th_val,
                fit_seconds=fit_sec,
                notes=f"{notes} [th fijado en val]",
            )
        )

    pd.DataFrame(comparison_rows).to_csv(
        OUT_DIR / "model_comparison_val_and_test.csv", index=False
    )

    # =========================================================================
    # INVESTIGACIÓN C: Umbrales y Trade-off de Aristas (Validación y Test)
    # =========================================================================
    print("5. Ejecutando Investigación C (Barrido de umbrales y costo de aristas)...")
    key_models_for_thresholds = [
        "KNN_k15_Eucl_Unif_Scaled_F0",
        "KNN_k51_Manh_Unif_Scaled_F0",
        "KNN_k51_Manh_Unif_Scaled_F4_SymLog",
        "KNN_k51_Eucl_Unif_Scaled_F0",
        "DT_d5_leaf1_Gini_Unw_F0",
        "DT_d6_leaf50_Gini_Unw_F0",
        "DT_d6_leaf50_LogLoss_Unw_F0",
        "DT_d7_leaf75_LogLoss_Unw_F5_Eng",
        "DT_d8_leaf100_LogLoss_Unw_F5_Eng",
        "LogReg_L2_C1_Unw_F5_Eng",
        "LogReg_L2_C1_Weight5_F5_Eng",
        "SVM_Linear_C1_Unw_F5_Eng",
        "SVC_RBF_C1_Sub30k_Unw_ProbTrue_F5_Eng",
        "LinReg_Ridge_a10_F5_Eng",
        "Ref_HGB_d6_Unw_F0",
        "Ref_HGB_d6_Unw_F5_Eng",
    ]

    threshold_rows: List[Dict[str, Any]] = []
    target_regimes = [
        ("Max_F1_Val", None),
        ("Recall_Val_ge_30pct", 0.30),
        ("Recall_Val_ge_40pct", 0.40),
        ("Recall_Val_ge_50pct", 0.50),
        ("Recall_Val_ge_60pct", 0.60),
    ]

    for m_name in key_models_for_thresholds:
        fam, allowed, fset, fit_sec = spec_meta_store[m_name]
        s_va = val_scores_store[m_name]
        s_te = test_scores_store[m_name]
        for reg_label, rec_target in target_regimes:
            if rec_target is None:
                th = val_best_th_store[m_name]
            else:
                th = select_recall_target_threshold_val(y_val, s_va, rec_target)
            threshold_rows.append(
                compute_metrics_row(
                    model_name=m_name,
                    family=fam,
                    allowed_by_syllabus=allowed,
                    feature_set=fset,
                    split_name="validation",
                    y_true=y_val,
                    scores=s_va,
                    threshold=th,
                    fit_seconds=fit_sec,
                    notes=reg_label,
                )
            )
            threshold_rows.append(
                compute_metrics_row(
                    model_name=m_name,
                    family=fam,
                    allowed_by_syllabus=allowed,
                    feature_set=fset,
                    split_name="test",
                    y_true=y_test,
                    scores=s_te,
                    threshold=th,
                    fit_seconds=fit_sec,
                    notes=f"{reg_label} (th de val)",
                )
            )

    pd.DataFrame(threshold_rows).to_csv(
        OUT_DIR / "thresholds_tradeoff.csv", index=False
    )

    # =========================================================================
    # GRÁFICO: Curvas Precision-Recall de las familias permitidas + Referencia
    # =========================================================================
    print("6. Generando curvas Precision-Recall...")
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5), sharey=True)

    plot_specs = [
        ("DT_d7_leaf75_LogLoss_Unw_F5_Eng", "Árbol Decisión d=7 leaf=75 (F5)", "#2ca02c", "-"),
        ("DT_d6_leaf50_LogLoss_Unw_F0", "Árbol Decisión d=6 leaf=50 (F0)", "#98df8a", "--"),
        ("KNN_k51_Manh_Unif_Scaled_F4_SymLog", "KNN K=51 Manhattan (F4 SymLog)", "#9467bd", "-"),
        ("KNN_k15_Eucl_Unif_Scaled_F0", "KNN K=15 Euclídea (Oficial F0)", "#d62728", "-"),
        ("LogReg_L2_C1_Unw_F5_Eng", "Reg. Logística L2 C=1 (F5)", "#ff7f0e", "-"),
        ("SVM_Linear_C1_Unw_F5_Eng", "SVM Lineal C=1 (F5)", "#8c564b", "-."),
        ("LinReg_Ridge_a10_F5_Eng", "Reg. Lineal Ridge (F5)", "#e377c2", ":"),
        ("Ref_HGB_d6_Unw_F5_Eng", "[Ref] HistGradientBoosting (F5)", "#1f77b4", "--"),
    ]

    for m_name, label, color, ls in plot_specs:
        s_va = val_scores_store[m_name]
        p_v, r_v, _ = precision_recall_curve(y_val, s_va)
        ap_v = average_precision_score(y_val, s_va)
        axes[0].plot(r_v, p_v, label=f"{label} (PR-AUC={ap_v:.4f})", color=color, linestyle=ls, linewidth=2)

        s_te = test_scores_store[m_name]
        p_t, r_t, _ = precision_recall_curve(y_test, s_te)
        ap_t = average_precision_score(y_test, s_te)
        axes[1].plot(r_t, p_t, label=f"{label} (PR-AUC={ap_t:.4f})", color=color, linestyle=ls, linewidth=2)

    axes[0].set_title("Curvas Precision-Recall — Validación (51.975 pares)")
    axes[0].set_xlabel("Recall (Sensibilidad)")
    axes[0].set_ylabel("Precision")
    axes[0].set_xlim(0.0, 1.0)
    axes[0].set_ylim(0.0, 1.0)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="upper right", fontsize=8.5)

    axes[1].set_title("Curvas Precision-Recall — Test (65.059 pares)")
    axes[1].set_xlabel("Recall (Sensibilidad)")
    axes[1].set_xlim(0.0, 1.0)
    axes[1].set_ylim(0.0, 1.0)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="upper right", fontsize=8.5)

    plt.tight_layout()
    plt.savefig(OUT_DIR / "pr_curves_ultima_ronda.png", dpi=160)
    plt.close(fig)

    print(f"Experimentos finalizados. Archivos guardados en: {OUT_DIR}")


if __name__ == "__main__":
    main()
