"""Diagnóstico experimental profundo sobre la predicción de conflictos entre materias.

Este script es 100% independiente del pipeline oficial (`src/`).
Lee los datasets en modo de solo lectura y guarda todos sus resultados en
`experiments/diagnostico_ml/results/` sin modificar ningún artefacto oficial.

Investigaciones ejecutadas:
- Investigación A: Aporte real de la información histórica vs. estructural.
- Investigación B: Comparación de familias de modelos (LogReg, DT, RF, ET, HGB, KNN)
  y estrategias de desbalance (Unweighted, ClassWeight, CustomWeight, Undersampling).
- Investigación C: Curvas de umbral y costo de aristas (Precision, Recall, F1, FN, TP+FP)
  seleccionadas exclusivamente en Validación.
- Investigación D: Desglose por tipo de par (ambas desconocidas vs. 1 conocida y 1 desconocida)
  y anatomía de falsos negativos / perfiles de indistinguibilidad en el espacio de features.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
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
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.schema import FEATURE_COLUMNS, ID_COLUMNS, TARGET_CLASSIFICATION
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

EXP_DIR = PROJECT_ROOT / "experiments" / "diagnostico_ml"
EXP_RESULTS_DIR = EXP_DIR / "results"

STRUCTURAL_FEATURES = [
    "mismo_departamento",
    "mismo_prefijo_curricular",
    "anio_confirmado",
    "mismo_anio",
    "misma_carrera",
]

HISTORICAL_COUNT_FEATURES = [
    "cant_carreras_compartidas_hist",
    "cant_inscriptos_hist_1",
    "cant_inscriptos_hist_2",
    "cant_estudiantes_compartidos_historico",
    "jaccard_historico",
]


def eval_predictions(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    model_name: str,
    split_name: str,
    notes: str = "",
) -> Dict[str, Any]:
    """Calcula métricas detalladas y conteos de aristas para un umbral dado."""
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    acc = float(accuracy_score(y_true, y_pred))
    pr_auc = float(average_precision_score(y_true, y_prob))
    roc_auc = float(roc_auc_score(y_true, y_prob))
    predicted_edges = int(tp + fp)
    pos_total = int(tp + fn)

    return {
        "model_name": model_name,
        "split": split_name,
        "threshold": round(float(threshold), 4),
        "precision": round(prec, 6),
        "recall": round(rec, 6),
        "f1_score": round(f1, 6),
        "pr_auc": round(pr_auc, 6),
        "roc_auc": round(roc_auc, 6),
        "accuracy": round(acc, 6),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "predicted_edges": predicted_edges,
        "positives_total": pos_total,
        "edges_to_positives_ratio": round(predicted_edges / pos_total, 3) if pos_total > 0 else 0.0,
        "fp_per_tp": round(fp / tp, 3) if tp > 0 else None,
        "notes": notes,
    }


def find_threshold_for_min_recall(
    y_val: np.ndarray, y_prob_val: np.ndarray, min_recall: float
) -> float:
    """Busca en validación el umbral más alto en [0.01, 0.95] que logra Recall >= min_recall."""
    chosen_th = 0.01
    for th in np.linspace(0.95, 0.01, 95):
        preds = (y_prob_val >= th).astype(int)
        rec = float(recall_score(y_val, preds, zero_division=0))
        if rec >= min_recall:
            chosen_th = float(th)
            break
    return round(chosen_th, 4)


def run_all_investigations() -> None:
    EXP_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("1. Cargando datasets oficiales (solo lectura)...")
    train_full_df = load_ml_dataframe(TRAIN_FULL_PATH)
    train_bal_df = load_ml_dataframe(TRAIN_BALANCED_PATH)
    test_full_df = load_ml_dataframe(TEST_FULL_PATH)

    subtrain_full_df, subtrain_bal_df, val_df, split_info = (
        build_internal_subject_validation_split(train_full_df)
    )

    # Identificar conjuntos de materias para desglosar subgrupos (1 conocida vs ambas desconocidas)
    subtrain_subjects = set(subtrain_full_df["materia_id_1"]).union(
        set(subtrain_full_df["materia_id_2"])
    )
    train_all_subjects = set(train_full_df["materia_id_1"]).union(
        set(train_full_df["materia_id_2"])
    )

    val_m1_known = val_df["materia_id_1"].isin(subtrain_subjects)
    val_m2_known = val_df["materia_id_2"].isin(subtrain_subjects)
    val_one_known_mask = (val_m1_known ^ val_m2_known).to_numpy()
    val_both_unknown_mask = (~val_m1_known & ~val_m2_known).to_numpy()

    test_m1_known = test_full_df["materia_id_1"].isin(train_all_subjects)
    test_m2_known = test_full_df["materia_id_2"].isin(train_all_subjects)
    test_one_known_mask = (test_m1_known ^ test_m2_known).to_numpy()
    test_both_unknown_mask = (~test_m1_known & ~test_m2_known).to_numpy()

    y_sub_full = subtrain_full_df[TARGET_CLASSIFICATION].to_numpy(dtype=int)
    y_sub_bal = subtrain_bal_df[TARGET_CLASSIFICATION].to_numpy(dtype=int)
    y_val = val_df[TARGET_CLASSIFICATION].to_numpy(dtype=int)
    y_train_full = train_full_df[TARGET_CLASSIFICATION].to_numpy(dtype=int)
    y_train_bal = train_bal_df[TARGET_CLASSIFICATION].to_numpy(dtype=int)
    y_test = test_full_df[TARGET_CLASSIFICATION].to_numpy(dtype=int)

    # =========================================================================
    # INVESTIGACIÓN A: ¿Cuánto aporta el solapamiento histórico vs features?
    # =========================================================================
    print("2. Ejecutando Investigación A (Aporte de información histórica vs estructural)...")
    inv_a_specs: List[Tuple[str, Any, List[str], str]] = [
        (
            "A1_Baseline_Regla_Hist_Positivo",
            HistoricalOverlapBaseline(feature_index=0),
            ["cant_estudiantes_compartidos_historico"],
            "subtrain_full",
        ),
        (
            "A2_LogReg_Solo_Compartidos_Hist_Unw",
            Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_SEED))]),
            ["cant_estudiantes_compartidos_historico"],
            "subtrain_full",
        ),
        (
            "A3_LogReg_Solo_Jaccard_Hist_Unw",
            Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_SEED))]),
            ["jaccard_historico"],
            "subtrain_full",
        ),
        (
            "A4_DT_Solo_Compartidos_Hist",
            DecisionTreeClassifier(max_depth=5, random_state=RANDOM_SEED),
            ["cant_estudiantes_compartidos_historico"],
            "subtrain_full",
        ),
        (
            "A5_DT_Solo_Jaccard_Hist",
            DecisionTreeClassifier(max_depth=5, random_state=RANDOM_SEED),
            ["jaccard_historico"],
            "subtrain_full",
        ),
        (
            "A6_HGB_Solo_5_Historicas",
            HistGradientBoostingClassifier(max_iter=150, max_depth=5, min_samples_leaf=50, learning_rate=0.05, random_state=RANDOM_SEED),
            HISTORICAL_COUNT_FEATURES,
            "subtrain_full",
        ),
        (
            "A7_HGB_Solo_5_Estructurales",
            HistGradientBoostingClassifier(max_iter=150, max_depth=5, min_samples_leaf=50, learning_rate=0.05, random_state=RANDOM_SEED),
            STRUCTURAL_FEATURES,
            "subtrain_full",
        ),
        (
            "A8_HGB_Todas_10_Features",
            HistGradientBoostingClassifier(max_iter=150, max_depth=5, min_samples_leaf=50, learning_rate=0.05, random_state=RANDOM_SEED),
            FEATURE_COLUMNS,
            "subtrain_full",
        ),
        (
            "A9_KNN15_Solo_5_Historicas",
            Pipeline([("scaler", StandardScaler()), ("clf", KNeighborsClassifier(n_neighbors=15, n_jobs=-1))]),
            HISTORICAL_COUNT_FEATURES,
            "subtrain_full",
        ),
        (
            "A10_KNN15_Todas_10_Features",
            Pipeline([("scaler", StandardScaler()), ("clf", KNeighborsClassifier(n_neighbors=15, n_jobs=-1))]),
            FEATURE_COLUMNS,
            "subtrain_full",
        ),
    ]

    inv_a_rows: List[Dict[str, Any]] = []
    prob_dist_rows: List[Dict[str, Any]] = []

    for name, est, feat_cols, regime in inv_a_specs:
        X_tr = subtrain_full_df[feat_cols]
        X_va = val_df[feat_cols]
        est.fit(X_tr, y_sub_full)

        prob_tr = est.predict_proba(X_tr)[:, 1]
        prob_va = est.predict_proba(X_va)[:, 1]

        if name.startswith("A1_"):
            th_val = 0.0001  # En HistoricalOverlapBaseline, prob > 0 equivale a conteo > 0
        else:
            th_val = find_best_f1_threshold_on_validation(y_val, prob_va)

        # Evaluar en Validation global, 1 conocida, ambas desconocidas, y referencia en subtrain (ambas conocidas)
        inv_a_rows.append(eval_predictions(y_val, prob_va, th_val, name, "val_global", f"features={len(feat_cols)}"))
        inv_a_rows.append(
            eval_predictions(
                y_val[val_one_known_mask],
                prob_va[val_one_known_mask],
                th_val,
                name,
                "val_1_conocida_1_desconocida",
                f"features={len(feat_cols)}",
            )
        )
        inv_a_rows.append(
            eval_predictions(
                y_val[val_both_unknown_mask],
                prob_va[val_both_unknown_mask],
                th_val,
                name,
                "val_ambas_desconocidas",
                f"features={len(feat_cols)}",
            )
        )
        inv_a_rows.append(
            eval_predictions(
                y_sub_full,
                prob_tr,
                th_val,
                name,
                "subtrain_ambas_conocidas (in-sample)",
                f"features={len(feat_cols)}",
            )
        )

        # Distribución de probabilidades en validación (positivos vs negativos)
        for cls_label, mask in [("positivos (y=1)", y_val == 1), ("negativos (y=0)", y_val == 0)]:
            p_sub = prob_va[mask]
            prob_dist_rows.append(
                {
                    "model_name": name,
                    "class_group": cls_label,
                    "min": round(float(np.min(p_sub)), 6),
                    "p25": round(float(np.percentile(p_sub, 25)), 6),
                    "median_p50": round(float(np.percentile(p_sub, 50)), 6),
                    "mean": round(float(np.mean(p_sub)), 6),
                    "p75": round(float(np.percentile(p_sub, 75)), 6),
                    "p90": round(float(np.percentile(p_sub, 90)), 6),
                    "p99": round(float(np.percentile(p_sub, 99)), 6),
                    "max": round(float(np.max(p_sub)), 6),
                }
            )

    pd.DataFrame(inv_a_rows).to_csv(EXP_RESULTS_DIR / "inv_a_historical_ablation.csv", index=False)
    pd.DataFrame(prob_dist_rows).to_csv(EXP_RESULTS_DIR / "inv_a_prob_distributions.csv", index=False)

    # =========================================================================
    # INVESTIGACIÓN B: Comparación de modelos y regímenes de desbalance en Validación
    # =========================================================================
    print("3. Ejecutando Investigación B (Comparación de algoritmos y desbalance en Validación)...")
    X_sub_full = subtrain_full_df[FEATURE_COLUMNS]
    X_sub_bal = subtrain_bal_df[FEATURE_COLUMNS]
    X_val = val_df[FEATURE_COLUMNS]

    inv_b_catalog: List[Tuple[str, Any, str]] = [
        # Baselines
        ("Baseline_Heuristico_Historico", HistoricalOverlapBaseline(feature_index=8), "subtrain_full"),
        # Logistic Regression
        (
            "LogReg_Full_Unweighted",
            Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_SEED))]),
            "subtrain_full",
        ),
        (
            "LogReg_Full_ClassWeight_Balanced",
            Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_SEED))]),
            "subtrain_full",
        ),
        (
            "LogReg_Full_CustomWeight_10",
            Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=1000, class_weight={0: 1, 1: 10}, random_state=RANDOM_SEED))]),
            "subtrain_full",
        ),
        (
            "LogReg_Balanced_Undersampling",
            Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_SEED))]),
            "subtrain_balanced",
        ),
        # Decision Trees
        (
            "DT_d5_Full_Unweighted",
            DecisionTreeClassifier(max_depth=5, random_state=RANDOM_SEED),
            "subtrain_full",
        ),
        (
            "DT_d5_Full_ClassWeight_Balanced",
            DecisionTreeClassifier(max_depth=5, class_weight="balanced", random_state=RANDOM_SEED),
            "subtrain_full",
        ),
        (
            "DT_d6_leaf50_Full_Unweighted",
            DecisionTreeClassifier(max_depth=6, min_samples_leaf=50, random_state=RANDOM_SEED),
            "subtrain_full",
        ),
        (
            "DT_d5_Balanced_Undersampling",
            DecisionTreeClassifier(max_depth=5, random_state=RANDOM_SEED),
            "subtrain_balanced",
        ),
        # Random Forest
        (
            "RF_d6_Full_Unweighted",
            RandomForestClassifier(n_estimators=200, max_depth=6, min_samples_leaf=25, random_state=RANDOM_SEED, n_jobs=-1),
            "subtrain_full",
        ),
        (
            "RF_d6_Full_ClassWeight_Balanced",
            RandomForestClassifier(n_estimators=200, max_depth=6, min_samples_leaf=25, class_weight="balanced", random_state=RANDOM_SEED, n_jobs=-1),
            "subtrain_full",
        ),
        (
            "RF_d8_Full_ClassWeight_Subsample",
            RandomForestClassifier(n_estimators=200, max_depth=8, min_samples_leaf=25, class_weight="balanced_subsample", random_state=RANDOM_SEED, n_jobs=-1),
            "subtrain_full",
        ),
        (
            "RF_d6_Full_CustomWeight_10",
            RandomForestClassifier(n_estimators=200, max_depth=6, min_samples_leaf=25, class_weight={0: 1, 1: 10}, random_state=RANDOM_SEED, n_jobs=-1),
            "subtrain_full",
        ),
        (
            "RF_d6_Balanced_Undersampling",
            RandomForestClassifier(n_estimators=200, max_depth=6, min_samples_leaf=25, random_state=RANDOM_SEED, n_jobs=-1),
            "subtrain_balanced",
        ),
        # Extra Trees
        (
            "ET_d8_Full_Unweighted",
            ExtraTreesClassifier(n_estimators=200, max_depth=8, min_samples_leaf=25, random_state=RANDOM_SEED, n_jobs=-1),
            "subtrain_full",
        ),
        (
            "ET_d8_Full_ClassWeight_Balanced",
            ExtraTreesClassifier(n_estimators=200, max_depth=8, min_samples_leaf=25, class_weight="balanced", random_state=RANDOM_SEED, n_jobs=-1),
            "subtrain_full",
        ),
        # HistGradientBoosting
        (
            "HGB_Full_Unweighted",
            HistGradientBoostingClassifier(max_iter=150, max_depth=5, min_samples_leaf=50, learning_rate=0.05, random_state=RANDOM_SEED),
            "subtrain_full",
        ),
        (
            "HGB_Full_ClassWeight_Balanced",
            HistGradientBoostingClassifier(max_iter=150, max_depth=5, min_samples_leaf=50, learning_rate=0.05, class_weight="balanced", random_state=RANDOM_SEED),
            "subtrain_full",
        ),
        (
            "HGB_Balanced_Undersampling",
            HistGradientBoostingClassifier(max_iter=150, max_depth=5, min_samples_leaf=50, learning_rate=0.05, random_state=RANDOM_SEED),
            "subtrain_balanced",
        ),
        # KNN con distintos K
        (
            "KNN_k15_Full_Unweighted",
            Pipeline([("scaler", StandardScaler()), ("clf", KNeighborsClassifier(n_neighbors=15, n_jobs=-1))]),
            "subtrain_full",
        ),
        (
            "KNN_k31_Full_Unweighted",
            Pipeline([("scaler", StandardScaler()), ("clf", KNeighborsClassifier(n_neighbors=31, n_jobs=-1))]),
            "subtrain_full",
        ),
        (
            "KNN_k51_Full_Unweighted",
            Pipeline([("scaler", StandardScaler()), ("clf", KNeighborsClassifier(n_neighbors=51, n_jobs=-1))]),
            "subtrain_full",
        ),
        (
            "KNN_k15_Balanced_Undersampling",
            Pipeline([("scaler", StandardScaler()), ("clf", KNeighborsClassifier(n_neighbors=15, n_jobs=-1))]),
            "subtrain_balanced",
        ),
    ]

    inv_b_val_rows: List[Dict[str, Any]] = []
    val_probs_by_model: Dict[str, np.ndarray] = {}
    val_best_f1_th: Dict[str, float] = {}

    for name, est, regime in inv_b_catalog:
        if regime == "subtrain_balanced":
            est.fit(X_sub_bal, y_sub_bal)
        else:
            est.fit(X_sub_full, y_sub_full)

        prob_val = est.predict_proba(X_val)[:, 1]
        val_probs_by_model[name] = prob_val

        # Evaluación en umbral por defecto 0.50 (o >0 para baseline heurístico)
        if name == "Baseline_Heuristico_Historico":
            inv_b_val_rows.append(
                eval_predictions(y_val, prob_val, 0.0001, name, "validation", f"regime={regime}, rule=hist>0")
            )
            val_best_f1_th[name] = 0.0001
        else:
            inv_b_val_rows.append(
                eval_predictions(y_val, prob_val, 0.50, f"{name}_th050", "validation", f"regime={regime}, default_th=0.50")
            )
            best_th = find_best_f1_threshold_on_validation(y_val, prob_val)
            val_best_f1_th[name] = best_th
            inv_b_val_rows.append(
                eval_predictions(y_val, prob_val, best_th, f"{name}_thValF1", "validation", f"regime={regime}, max_f1_val")
            )

            # Agregar también puntos operativos orientados a Recall (>= 0.40 y >= 0.60 en validación)
            for target_rec in (0.40, 0.60):
                th_rec = find_threshold_for_min_recall(y_val, prob_val, target_rec)
                inv_b_val_rows.append(
                    eval_predictions(
                        y_val,
                        prob_val,
                        th_rec,
                        f"{name}_thValRec{int(target_rec*100)}",
                        "validation",
                        f"regime={regime}, target_val_recall>={target_rec:.2f}",
                    )
                )

    pd.DataFrame(inv_b_val_rows).to_csv(EXP_RESULTS_DIR / "inv_b_validation_comparison.csv", index=False)

    # =========================================================================
    # INVESTIGACIÓN C: Análisis detallado de umbrales y costo de errores (en Validación)
    # =========================================================================
    print("4. Ejecutando Investigación C (Trade-off de umbrales y aristas en Validación)...")
    top_val_models = [
        "HGB_Full_Unweighted",
        "RF_d6_Full_Unweighted",
        "RF_d6_Full_ClassWeight_Balanced",
        "DT_d5_Full_Unweighted",
        "KNN_k15_Full_Unweighted",
        "KNN_k31_Full_Unweighted",
        "KNN_k51_Full_Unweighted",
        "LogReg_Full_ClassWeight_Balanced",
    ]

    inv_c_rows: List[Dict[str, Any]] = []
    for mname in top_val_models:
        p_val = val_probs_by_model[mname]
        # Evaluar umbrales específicos de la grilla y objetivos de Recall
        candidate_thresholds = sorted(
            set(
                [0.02, 0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.21, 0.25, 0.30, 0.40, 0.50, 0.70, 0.85, 0.90, 0.94]
                + [
                    val_best_f1_th[mname],
                    find_threshold_for_min_recall(y_val, p_val, 0.30),
                    find_threshold_for_min_recall(y_val, p_val, 0.40),
                    find_threshold_for_min_recall(y_val, p_val, 0.50),
                    find_threshold_for_min_recall(y_val, p_val, 0.60),
                    find_threshold_for_min_recall(y_val, p_val, 0.75),
                ]
            )
        )
        for th in candidate_thresholds:
            row = eval_predictions(y_val, p_val, th, mname, "validation", "threshold_sweep_val")
            inv_c_rows.append(row)

    pd.DataFrame(inv_c_rows).to_csv(EXP_RESULTS_DIR / "inv_c_validation_thresholds_tradeoff.csv", index=False)

    # =========================================================================
    # EVALUACIÓN EN TEST DE LOS CANDIDATOS SELECCIONADOS EN VALIDACIÓN + INVESTIGACIÓN D
    # =========================================================================
    print("5. Evaluando candidatos seleccionados en Validación sobre Test y ejecutando Investigación D...")
    X_train_full = train_full_df[FEATURE_COLUMNS]
    X_train_bal = train_bal_df[FEATURE_COLUMNS]
    X_test = test_full_df[FEATURE_COLUMNS]

    selected_for_test: List[Tuple[str, Any, str]] = [
        ("Baseline_Heuristico_Historico", HistoricalOverlapBaseline(feature_index=8), "train_full"),
        ("KNN_k15_Full_Unweighted", Pipeline([("scaler", StandardScaler()), ("clf", KNeighborsClassifier(n_neighbors=15, n_jobs=-1))]), "train_full"),
        ("KNN_k31_Full_Unweighted", Pipeline([("scaler", StandardScaler()), ("clf", KNeighborsClassifier(n_neighbors=31, n_jobs=-1))]), "train_full"),
        ("KNN_k51_Full_Unweighted", Pipeline([("scaler", StandardScaler()), ("clf", KNeighborsClassifier(n_neighbors=51, n_jobs=-1))]), "train_full"),
        ("DT_d5_Full_Unweighted", DecisionTreeClassifier(max_depth=5, random_state=RANDOM_SEED), "train_full"),
        ("RF_d6_Full_Unweighted", RandomForestClassifier(n_estimators=200, max_depth=6, min_samples_leaf=25, random_state=RANDOM_SEED, n_jobs=-1), "train_full"),
        ("RF_d6_Full_ClassWeight_Balanced", RandomForestClassifier(n_estimators=200, max_depth=6, min_samples_leaf=25, class_weight="balanced", random_state=RANDOM_SEED, n_jobs=-1), "train_full"),
        ("HGB_Full_Unweighted", HistGradientBoostingClassifier(max_iter=150, max_depth=5, min_samples_leaf=50, learning_rate=0.05, random_state=RANDOM_SEED), "train_full"),
        ("HGB_Full_ClassWeight_Balanced", HistGradientBoostingClassifier(max_iter=150, max_depth=5, min_samples_leaf=50, learning_rate=0.05, class_weight="balanced", random_state=RANDOM_SEED), "train_full"),
        ("LogReg_Full_ClassWeight_Balanced", Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_SEED))]), "train_full"),
    ]

    test_eval_rows: List[Dict[str, Any]] = []
    inv_d_subgroup_rows: List[Dict[str, Any]] = []
    test_probs_by_model: Dict[str, np.ndarray] = {}

    for mname, est, tr_regime in selected_for_test:
        if tr_regime == "train_bal":
            est.fit(X_train_bal, y_train_bal)
        else:
            est.fit(X_train_full, y_train_full)

        p_test = est.predict_proba(X_test)[:, 1]
        test_probs_by_model[mname] = p_test

        th_f1 = val_best_f1_th[mname]
        # Evaluar en Test con el umbral F1 elegido exclusivamente en Validación
        test_eval_rows.append(
            eval_predictions(y_test, p_test, th_f1, mname, "test_global", "th_selected_on_val_max_f1")
        )

        # Si no es el baseline heurístico, evaluar también los umbrales de validación para Recall>=0.40 y Recall>=0.60
        if mname != "Baseline_Heuristico_Historico":
            p_val_ref = val_probs_by_model[mname]
            for target_rec in (0.40, 0.50, 0.60):
                th_rec_val = find_threshold_for_min_recall(y_val, p_val_ref, target_rec)
                test_eval_rows.append(
                    eval_predictions(
                        y_test,
                        p_test,
                        th_rec_val,
                        f"{mname}_ValRec{int(target_rec*100)}",
                        "test_global",
                        f"th_selected_on_val_recall>={target_rec:.2f}",
                    )
                )

        # Desglose Investigación D (Validación y Test por subgrupos: 1 conocida vs ambas desconocidas)
        p_val_m = val_probs_by_model[mname]
        inv_d_subgroup_rows.extend(
            [
                eval_predictions(y_val, p_val_m, th_f1, mname, "val_global (51.975)", "th_val_f1"),
                eval_predictions(
                    y_val[val_one_known_mask],
                    p_val_m[val_one_known_mask],
                    th_f1,
                    mname,
                    "val_1_conocida_1_desconocida (49.049)",
                    "th_val_f1",
                ),
                eval_predictions(
                    y_val[val_both_unknown_mask],
                    p_val_m[val_both_unknown_mask],
                    th_f1,
                    mname,
                    "val_ambas_desconocidas (2.926)",
                    "th_val_f1",
                ),
                eval_predictions(y_test, p_test, th_f1, mname, "test_global (65.059)", "th_val_f1"),
                eval_predictions(
                    y_test[test_one_known_mask],
                    p_test[test_one_known_mask],
                    th_f1,
                    mname,
                    "test_1_conocida_1_desconocida (61.404)",
                    "th_val_f1",
                ),
                eval_predictions(
                    y_test[test_both_unknown_mask],
                    p_test[test_both_unknown_mask],
                    th_f1,
                    mname,
                    "test_ambas_desconocidas (3.655)",
                    "th_val_f1",
                ),
            ]
        )

    pd.DataFrame(test_eval_rows).to_csv(EXP_RESULTS_DIR / "test_candidates_evaluation.csv", index=False)
    pd.DataFrame(inv_d_subgroup_rows).to_csv(EXP_RESULTS_DIR / "inv_d_generalization_subgroups.csv", index=False)

    # =========================================================================
    # INVESTIGACIÓN D (Parte 2): Anatomía de los positivos y perfiles estructurales
    # =========================================================================
    print("6. Analizando perfiles estructurales de positivos y negativos (Investigación D)...")
    profile_rows: List[Dict[str, Any]] = []
    for split_label, df_split in [("validation (51.975)", val_df), ("test (65.059)", test_full_df)]:
        hist_pos = df_split["cant_estudiantes_compartidos_historico"] > 0
        same_car = df_split["misma_carrera"] == 1
        same_yr = df_split["mismo_anio"] == 1
        same_dep_or_pref = (df_split["mismo_departamento"] == 1) | (df_split["mismo_prefijo_curricular"] == 1)

        profiles = [
            ("P1: compartidos_hist > 0 (Solapamiento histórico directo)", hist_pos),
            ("P2: compartidos_hist == 0, misma_carrera == 1, mismo_anio == 1", (~hist_pos) & same_car & same_yr),
            ("P3: compartidos_hist == 0, misma_carrera == 1, mismo_anio == 0", (~hist_pos) & same_car & (~same_yr)),
            (
                "P4: compartidos_hist == 0, misma_carrera == 0, (mismo_anio==1 o mismo_depto/pref==1)",
                (~hist_pos) & (~same_car) & (same_yr | same_dep_or_pref),
            ),
            (
                "P5: compartidos_hist == 0, misma_carrera == 0, mismo_anio == 0, mismo_depto/pref == 0 (Sin señal relacional)",
                (~hist_pos) & (~same_car) & (~same_yr) & (~same_dep_or_pref),
            ),
        ]

        total_pos_split = int(df_split[TARGET_CLASSIFICATION].sum())
        total_neg_split = int((df_split[TARGET_CLASSIFICATION] == 0).sum())

        for p_name, mask in profiles:
            sub = df_split[mask]
            n_total = len(sub)
            n_pos = int(sub[TARGET_CLASSIFICATION].sum())
            n_neg = n_total - n_pos
            profile_rows.append(
                {
                    "split": split_label,
                    "profile": p_name,
                    "total_pairs": n_total,
                    "positives": n_pos,
                    "negatives": n_neg,
                    "share_of_all_positives_pct": round(100.0 * n_pos / total_pos_split, 2),
                    "share_of_all_negatives_pct": round(100.0 * n_neg / total_neg_split, 2),
                    "empirical_positive_rate_pct": round(100.0 * n_pos / n_total, 3) if n_total > 0 else 0.0,
                    "negatives_per_positive": round(n_neg / n_pos, 1) if n_pos > 0 else None,
                }
            )

    pd.DataFrame(profile_rows).to_csv(EXP_RESULTS_DIR / "inv_d_structural_profiles.csv", index=False)

    # =========================================================================
    # GRÁFICO: Curvas Precision-Recall en Validación y Test
    # =========================================================================
    print("7. Generando gráfico de curvas Precision-Recall...")
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)

    plot_models = [
        ("HGB_Full_Unweighted", "HistGradientBoosting (Full Unw)", "#1f77b4"),
        ("RF_d6_Full_Unweighted", "RandomForest d=6 (Full Unw)", "#2ca02c"),
        ("DT_d5_Full_Unweighted", "DecisionTree d=5 (Full Unw)", "#ff7f0e"),
        ("KNN_k31_Full_Unweighted", "KNN K=31 (Full Unw)", "#9467bd"),
        ("KNN_k15_Full_Unweighted", "KNN K=15 (Oficial Actual)", "#d62728"),
        ("LogReg_Full_ClassWeight_Balanced", "LogReg (ClassWeight)", "#8c564b"),
        ("Baseline_Heuristico_Historico", "Baseline Histórico (hist > 0)", "#7f7f7f"),
    ]

    for mname, label, color in plot_models:
        # Validación
        p_v = val_probs_by_model[mname]
        prec_v, rec_v, _ = precision_recall_curve(y_val, p_v)
        ap_v = average_precision_score(y_val, p_v)
        axes[0].plot(rec_v, prec_v, label=f"{label} (AP={ap_v:.4f})", color=color, linewidth=2)

        # Test
        p_t = test_probs_by_model[mname]
        prec_t, rec_t, _ = precision_recall_curve(y_test, p_t)
        ap_t = average_precision_score(y_test, p_t)
        axes[1].plot(rec_t, prec_t, label=f"{label} (AP={ap_t:.4f})", color=color, linewidth=2)

    axes[0].set_title("Curvas Precision-Recall — Validación Interna (51.975 pares)")
    axes[0].set_xlabel("Recall (Sensibilidad)")
    axes[0].set_ylabel("Precision")
    axes[0].set_xlim(0.0, 1.0)
    axes[0].set_ylim(0.0, 1.0)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="upper right", fontsize=9)

    axes[1].set_title("Curvas Precision-Recall — Test Final (65.059 pares)")
    axes[1].set_xlabel("Recall (Sensibilidad)")
    axes[1].set_xlim(0.0, 1.0)
    axes[1].set_ylim(0.0, 1.0)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="upper right", fontsize=9)

    plt.tight_layout()
    plt.savefig(EXP_RESULTS_DIR / "pr_curves_val_and_test.png", dpi=160)
    plt.close(fig)

    print("Experimentos finalizados. Resultados guardados en:", EXP_RESULTS_DIR)


if __name__ == "__main__":
    run_all_investigations()
