"""Inspección profunda de features y limitaciones estructurales de los datos (Investigaciones A y D).

Lee en modo solo lectura `data/interim/` y `data/processed/` para responder con
evidencia cuantitativa exacta todas las preguntas sobre:
1. Importancia individual, correlaciones, redundancias y asimetrías de las 10 features.
2. Qué ocurre en el estrato de pares con las 5 señales relacionales en cero.
3. Qué información contienen los códigos crudos de examen (`exam_id`), nivel (`raw_level_char`),
   duración y carreras (`course_code`), y cómo impacta la separación de cohortes
   (`1989-1993` vs `1994`).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.schema import (
    FEATURE_COLUMNS,
    TARGET_CLASSIFICATION,
)
from src.models.compare_models import (
    TEST_FULL_PATH,
    TRAIN_FULL_PATH,
    build_internal_subject_validation_split,
    load_ml_dataframe,
)

OUT_DIR = PROJECT_ROOT / "experiments" / "ultima_ronda_ml" / "results"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    exams_path = PROJECT_ROOT / "data" / "interim" / "exams_clean.csv"
    students_path = PROJECT_ROOT / "data" / "interim" / "students_clean.csv"
    enrolments_path = PROJECT_ROOT / "data" / "interim" / "enrolments_clean.csv"

    df_train_full = load_ml_dataframe(TRAIN_FULL_PATH)
    df_test_full = load_ml_dataframe(TEST_FULL_PATH)
    df_subtrain, df_subtrain_bal, df_val, split_info = (
        build_internal_subject_validation_split(df_train_full)
    )

    # 1. Análisis univariado de cada una de las 10 features en Subtrain y Validación
    univariate_rows = []
    y_sub = df_subtrain[TARGET_CLASSIFICATION].to_numpy(dtype=int)
    y_val = df_val[TARGET_CLASSIFICATION].to_numpy(dtype=int)

    for col in FEATURE_COLUMNS:
        x_sub = df_subtrain[col].to_numpy(dtype=float)
        x_val = df_val[col].to_numpy(dtype=float)

        corr_sub = float(np.corrcoef(x_sub, y_sub)[0, 1])
        roc_val = float(roc_auc_score(y_val, x_val))
        pr_val = float(average_precision_score(y_val, x_val))

        nonzero_sub_pct = float((x_sub > 0).mean() * 100)
        mean_pos_sub = float(x_sub[y_sub == 1].mean())
        mean_neg_sub = float(x_sub[y_sub == 0].mean())

        pos_rate_when_gt0 = (
            float(y_sub[x_sub > 0].mean() * 100) if (x_sub > 0).any() else 0.0
        )
        pos_rate_when_eq0 = (
            float(y_sub[x_sub == 0].mean() * 100) if (x_sub == 0).any() else 0.0
        )

        univariate_rows.append(
            {
                "feature": col,
                "nonzero_pct_subtrain": round(nonzero_sub_pct, 3),
                "mean_when_y1_subtrain": round(mean_pos_sub, 4),
                "mean_when_y0_subtrain": round(mean_neg_sub, 4),
                "pos_rate_when_gt0_pct": round(pos_rate_when_gt0, 3),
                "pos_rate_when_eq0_pct": round(pos_rate_when_eq0, 3),
                "pearson_corr_subtrain": round(corr_sub, 4),
                "univariate_roc_auc_val": round(roc_val, 4),
                "univariate_pr_auc_val": round(pr_val, 4),
            }
        )

    df_univ = pd.DataFrame(univariate_rows)
    df_univ.to_csv(OUT_DIR / "inv_a_univariate_features.csv", index=False)

    # 2. Matriz de correlación entre las 10 features en Subtrain
    corr_matrix = df_subtrain[FEATURE_COLUMNS].corr()
    corr_matrix.to_csv(OUT_DIR / "inv_a_correlation_matrix.csv")

    high_corrs = []
    for i, c1 in enumerate(FEATURE_COLUMNS):
        for c2 in FEATURE_COLUMNS[i + 1 :]:
            r = float(corr_matrix.loc[c1, c2])
            if abs(r) >= 0.20:
                high_corrs.append(
                    {"feature_1": c1, "feature_2": c2, "pearson_r": round(r, 4)}
                )
    high_corrs.sort(key=lambda x: abs(x["pearson_r"]), reverse=True)

    # 3. Investigación profunda de las limitaciones de los datos (Investigación D)
    exams_df = pd.read_csv(exams_path)
    students_df = pd.read_csv(students_path)
    enrol_df = pd.read_csv(enrolments_path)

    enrol_merged = enrol_df.merge(students_df, on="student_id", how="left").merge(
        exams_df[
            [
                "exam_id",
                "raw_level_char",
                "year_code",
                "department_code",
                "duration_minutes",
            ]
        ],
        on="exam_id",
        how="left",
    )

    # Distribución de inscripciones por raw_level_char y cohort_group
    level_by_cohort = (
        enrol_merged.groupby(["raw_level_char", "cohort_group"])
        .size()
        .unstack(fill_value=0)
    )
    level_by_cohort["pct_actual_1994"] = (
        level_by_cohort["actual_1994"]
        / (level_by_cohort["actual_1994"] + level_by_cohort["historico_1989_1993"])
        * 100
    ).round(2)

    exam_hist_counts = (
        enrol_merged[enrol_merged["cohort_group"] == "historico_1989_1993"]
        .groupby("exam_id")
        .size()
    )
    exam_curr_counts = (
        enrol_merged[enrol_merged["cohort_group"] == "actual_1994"]
        .groupby("exam_id")
        .size()
    )
    exams_df["n_hist"] = exams_df["exam_id"].map(exam_hist_counts).fillna(0).astype(int)
    exams_df["n_curr"] = exams_df["exam_id"].map(exam_curr_counts).fillna(0).astype(int)

    exams_by_level_summary = (
        exams_df.groupby("raw_level_char")
        .agg(
            total_exams=("exam_id", "count"),
            exams_with_zero_hist=("n_hist", lambda s: int((s == 0).sum())),
            exams_with_zero_curr=("n_curr", lambda s: int((s == 0).sum())),
            mean_n_hist=("n_hist", "mean"),
            mean_n_curr=("n_curr", "mean"),
        )
        .round(2)
    )

    exam_meta = exams_df.set_index("exam_id").to_dict(orient="index")
    level_equiv_map = {"1": 1, "A": 1, "2": 2, "B": 2, "3": 3, "C": 3, "4": 4}

    def analyze_split_limitations(df_split: pd.DataFrame, split_name: str) -> dict:
        mask_5zeros = (
            (df_split["cant_estudiantes_compartidos_historico"] == 0)
            & (df_split["misma_carrera"] == 0)
            & (df_split["mismo_anio"] == 0)
            & (df_split["mismo_departamento"] == 0)
            & (df_split["mismo_prefijo_curricular"] == 0)
        )
        sub = df_split[mask_5zeros].copy()
        pos = sub[sub[TARGET_CLASSIFICATION] == 1]
        neg = sub[sub[TARGET_CLASSIFICATION] == 0]
        y_5z = sub[TARGET_CLASSIFICATION].to_numpy(dtype=int)

        # Capacidad discriminativa de las 3 únicas features no nulas dentro del grupo 5-ceros
        min_h = np.minimum(
            sub["cant_inscriptos_hist_1"].to_numpy(),
            sub["cant_inscriptos_hist_2"].to_numpy(),
        )
        max_h = np.maximum(
            sub["cant_inscriptos_hist_1"].to_numpy(),
            sub["cant_inscriptos_hist_2"].to_numpy(),
        )
        anio_conf = sub["anio_confirmado"].to_numpy()

        roc_min_h = float(roc_auc_score(y_5z, -min_h))
        pr_min_h = float(average_precision_score(y_5z, -min_h))
        roc_max_h = float(roc_auc_score(y_5z, -max_h))
        roc_anio_conf = float(roc_auc_score(y_5z, anio_conf))

        pos_one_zero_hist = int(
            (
                (pos["cant_inscriptos_hist_1"] == 0)
                | (pos["cant_inscriptos_hist_2"] == 0)
            ).sum()
        )
        pos_both_zero_hist = int(
            (
                (pos["cant_inscriptos_hist_1"] == 0)
                & (pos["cant_inscriptos_hist_2"] == 0)
            ).sum()
        )
        neg_one_zero_hist = int(
            (
                (neg["cant_inscriptos_hist_1"] == 0)
                | (neg["cant_inscriptos_hist_2"] == 0)
            ).sum()
        )

        def check_code_features(df_part: pd.DataFrame) -> dict:
            same_raw_lvl = 0
            same_equiv_lvl = 0
            both_level_1_or_A = 0
            at_least_one_level_1_or_A = 0
            same_first_char_faculty = 0
            for r in df_part[["materia_id_1", "materia_id_2"]].itertuples(index=False):
                m1 = r.materia_id_1
                m2 = r.materia_id_2
                l1 = exam_meta[m1]["raw_level_char"]
                l2 = exam_meta[m2]["raw_level_char"]
                eq1 = level_equiv_map.get(l1, -1)
                eq2 = level_equiv_map.get(l2, -2)
                if l1 == l2:
                    same_raw_lvl += 1
                if eq1 == eq2:
                    same_equiv_lvl += 1
                if eq1 == 1 and eq2 == 1:
                    both_level_1_or_A += 1
                if eq1 == 1 or eq2 == 1:
                    at_least_one_level_1_or_A += 1
                if m1[0] == m2[0]:
                    same_first_char_faculty += 1
            n = max(1, len(df_part))
            return {
                "count": len(df_part),
                "same_raw_level_char": same_raw_lvl,
                "same_raw_level_char_pct": round(same_raw_lvl / n * 100, 2),
                "same_equiv_level_1A_2B_3C": same_equiv_lvl,
                "same_equiv_level_pct": round(same_equiv_lvl / n * 100, 2),
                "both_level_1_or_A": both_level_1_or_A,
                "both_level_1_or_A_pct": round(both_level_1_or_A / n * 100, 2),
                "at_least_one_level_1_or_A": at_least_one_level_1_or_A,
                "at_least_one_level_1_or_A_pct": round(
                    at_least_one_level_1_or_A / n * 100, 2
                ),
                "same_first_char_faculty": same_first_char_faculty,
                "same_first_char_faculty_pct": round(
                    same_first_char_faculty / n * 100, 2
                ),
            }

        all_pos = df_split[df_split[TARGET_CLASSIFICATION] == 1]
        all_neg = df_split[df_split[TARGET_CLASSIFICATION] == 0]
        all_pos_zero_hist_any = int(
            (
                (all_pos["cant_inscriptos_hist_1"] == 0)
                | (all_pos["cant_inscriptos_hist_2"] == 0)
            ).sum()
        )
        all_pos_zero_hist_both = int(
            (
                (all_pos["cant_inscriptos_hist_1"] == 0)
                & (all_pos["cant_inscriptos_hist_2"] == 0)
            ).sum()
        )

        return {
            "split": split_name,
            "total_pairs": len(df_split),
            "total_positives": len(all_pos),
            "all_positives_with_at_least_one_subject_zero_hist": all_pos_zero_hist_any,
            "all_positives_with_at_least_one_subject_zero_hist_pct": round(
                all_pos_zero_hist_any / len(all_pos) * 100, 2
            ),
            "all_positives_with_both_subjects_zero_hist": all_pos_zero_hist_both,
            "all_positives_code_breakdown": check_code_features(all_pos),
            "all_negatives_code_breakdown": check_code_features(all_neg),
            "group_5zeros_total": len(sub),
            "group_5zeros_positives": len(pos),
            "group_5zeros_negatives": len(neg),
            "group_5zeros_share_of_all_positives_pct": round(
                len(pos) / len(all_pos) * 100, 2
            ),
            "group_5zeros_positive_rate_pct": round(len(pos) / len(sub) * 100, 3),
            "group_5zeros_roc_auc_neg_min_hist": round(roc_min_h, 4),
            "group_5zeros_pr_auc_neg_min_hist": round(pr_min_h, 4),
            "group_5zeros_roc_auc_neg_max_hist": round(roc_max_h, 4),
            "group_5zeros_roc_auc_anio_confirmado": round(roc_anio_conf, 4),
            "group_5zeros_pos_with_zero_hist_any": pos_one_zero_hist,
            "group_5zeros_neg_with_zero_hist_any": neg_one_zero_hist,
            "group_5zeros_pos_both_zero_hist": pos_both_zero_hist,
            "group_5zeros_pos_code_breakdown": check_code_features(pos),
            "group_5zeros_neg_code_breakdown": check_code_features(neg),
        }

    zero_val = analyze_split_limitations(df_val, "validation")
    zero_test = analyze_split_limitations(df_test_full, "test")
    zero_subtrain = analyze_split_limitations(df_subtrain, "subtrain")

    summary_data = {
        "high_correlations_subtrain": high_corrs,
        "enrolments_by_level_and_cohort": level_by_cohort.to_dict(orient="index"),
        "exams_by_level_summary": exams_by_level_summary.to_dict(orient="index"),
        "zero_signal_analysis": {
            "subtrain": zero_subtrain,
            "validation": zero_val,
            "test": zero_test,
        },
    }

    with (OUT_DIR / "inv_a_and_d_data_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary_data, fh, indent=2, ensure_ascii=False)

    print("Inspección completada exitosamente.")


if __name__ == "__main__":
    main()
