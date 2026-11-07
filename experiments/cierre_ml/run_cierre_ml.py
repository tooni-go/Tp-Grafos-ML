"""Cierre definitivo del módulo ML: validación de features A/B/C, grafo histórico a 2 saltos,
comparación final de representaciones (F0, F4, F5, F6, F7) y recomendación.

Reglas metodológicas (aplicadas por código):
- Solo lectura sobre `data/`, `src/`, `results/`. Todas las salidas van a `experiments/cierre_ml/results/`.
- Split oficial: `ml_train_full.csv` / `ml_test_full.csv` y split interno por materias
  (`build_internal_subject_validation_split`, RANDOM_SEED=42).
- Selección de representación, modelo y umbral EXCLUSIVAMENTE con subtrain/validation.
  Las reglas de selección se fijan en este archivo antes de mirar test.
- Test se evalúa una única vez al final, reentrenando sobre train completo con el umbral de validación.
  (Test ya fue consultado en rondas anteriores; se documenta como limitación.)
- Modelos: KNN, Árbol de decisión, Regresión logística (permitidos por la consigna).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
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

from src.common.schema import FEATURE_COLUMNS, TARGET_CLASSIFICATION  # noqa: E402
from src.models.compare_models import (  # noqa: E402
    RANDOM_SEED,
    TEST_FULL_PATH,
    TRAIN_FULL_PATH,
    build_internal_subject_validation_split,
    find_best_f1_threshold_on_validation,
    load_ml_dataframe,
)

OUT_DIR = PROJECT_ROOT / "experiments" / "cierre_ml" / "results"
INTERIM = PROJECT_ROOT / "data" / "interim"

HIST_GROUP = "historico_1989_1993"
CURR_GROUP = "actual_1994"

# Hipótesis bajo prueba (Sección 1): solo A/B/C. '0' y 'P' NO se mapean (se mantienen como hoy).
NUMERIC_LEVEL = {"1": 1, "2": 2, "3": 3, "4": 4}
LETTER_HYPOTHESIS = {"A": 1, "B": 2, "C": 3}
# Cohorte esperada como dominante para cada año curricular en el semestre 1994-95
EXPECTED_COHORT = {1: 1994, 2: 1993, 3: 1992, 4: 1991}
# Criterio de aceptación pre-declarado para cada letra (evaluado sólo con materias de subtrain):
#  (i) la cohorte dominante agregada coincide con la del código numérico equivalente, y
#  (ii) >= 60 % de los exámenes de la letra tienen como cohorte dominante la esperada.
ACCEPT_MIN_EXAM_AGREEMENT = 0.60

# Regla de selección pre-declarada (Sección 3/8)
REP_TOLERANCE_PR_AUC = 0.01  # se elige la representación más simple dentro de esta tolerancia
REPRESENTATION_SIMPLICITY_ORDER = ["F0", "F4", "F6_sin_primer_anio", "F6", "F7", "F5"]
N_BOOTSTRAP = 300


# =============================================================================
# Utilidades
# =============================================================================


def log(msg: str) -> None:
    print(msg, flush=True)


def metrics_at(
    y: np.ndarray, s: np.ndarray, th: float, affected: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    p = (s >= th).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, p, labels=[0, 1]).ravel()
    out = {
        "threshold": round(float(th), 4),
        "precision": round(float(precision_score(y, p, zero_division=0)), 4),
        "recall": round(float(recall_score(y, p, zero_division=0)), 4),
        "f1": round(float(f1_score(y, p, zero_division=0)), 4),
        "pr_auc": round(float(average_precision_score(y, s)), 4),
        "roc_auc": round(float(roc_auc_score(y, s)), 4),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "aristas_predichas": int(tp + fp),
        "fp_por_tp": round(fp / tp, 3) if tp > 0 else None,
    }
    if affected is not None:
        tot = float(affected[y == 1].sum())
        out["recall_ponderado_estudiantes"] = (
            round(float(affected[(y == 1) & (p == 1)].sum()) / tot, 4) if tot > 0 else None
        )
    return out


# =============================================================================
# Sección 1 — Auditoría de niveles académicos
# =============================================================================


def audit_levels(
    exams: pd.DataFrame, students: pd.DataFrame, enrol: pd.DataFrame, subtrain_subjects: set
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, int], Dict[str, Any]]:
    m = enrol.merge(students, on="student_id").merge(
        exams[["exam_id", "raw_level_char", "department_code", "exam_name"]], on="exam_id"
    )
    m["cohort_year"] = m["cohort_year"].astype(int)

    # Cohorte dominante por examen (todas las inscripciones; describe a quién se dicta la materia)
    per_exam = (
        m.groupby(["exam_id", "cohort_year"]).size().unstack(fill_value=0)
    )
    dom = per_exam.idxmax(axis=1)
    hist_counts = m[m["cohort_group"] == HIST_GROUP].groupby("exam_id").size()

    rows = []
    accepted: Dict[str, int] = {}
    for code in sorted(exams["raw_level_char"].unique()):
        ex_ids = exams.loc[exams["raw_level_char"] == code, "exam_id"]
        sub_ids = [e for e in ex_ids if e in subtrain_subjects]
        mm = m[m["raw_level_char"] == code]
        cohort_share = mm["cohort_year"].value_counts(normalize=True).sort_index()
        hist_share = (
            mm[mm["cohort_group"] == HIST_GROUP]["cohort_year"].value_counts(normalize=True).sort_index()
        )
        year = NUMERIC_LEVEL.get(code, LETTER_HYPOTHESIS.get(code))
        exp = EXPECTED_COHORT.get(year) if year else None

        def agreement(ids: List[str]) -> Optional[float]:
            ids = [e for e in ids if e in dom.index]
            if not ids or exp is None:
                return None
            return round(float(np.mean([dom[e] == exp for e in ids])), 4)

        row = {
            "codigo": code,
            "n_examenes": int(len(ex_ids)),
            "n_examenes_subtrain": int(len(sub_ids)),
            "n_inscripciones": int(len(mm)),
            "n_examenes_sin_inscriptos_hist": int(sum(hist_counts.get(e, 0) == 0 for e in ex_ids)),
            "inscriptos_hist_promedio_por_examen": round(
                float(np.mean([hist_counts.get(e, 0) for e in ex_ids])), 2
            ),
            "anio_hipotesis": year,
            "cohorte_esperada": exp,
            "cohorte_dominante_agregada": int(cohort_share.idxmax()) if len(cohort_share) else None,
            "acuerdo_por_examen_todas": agreement(list(ex_ids)),
            "acuerdo_por_examen_subtrain": agreement(sub_ids),
        }
        for cy in range(1989, 1995):
            row[f"pct_inscr_{cy}"] = round(100 * float(cohort_share.get(cy, 0.0)), 1)
        for cy in range(1989, 1994):
            row[f"pct_inscr_hist_{cy}"] = round(100 * float(hist_share.get(cy, 0.0)), 1)
        rows.append(row)
    audit_df = pd.DataFrame(rows)

    # Criterio de aceptación para letras (con materias de subtrain)
    num_dom = {
        r["anio_hipotesis"]: r["cohorte_dominante_agregada"]
        for _, r in audit_df.iterrows()
        if r["codigo"] in NUMERIC_LEVEL
    }
    decisions = {}
    for code, year in LETTER_HYPOTHESIS.items():
        r = audit_df[audit_df["codigo"] == code].iloc[0]
        ok_i = r["cohorte_dominante_agregada"] == num_dom[year]
        ok_ii = (r["acuerdo_por_examen_subtrain"] or 0) >= ACCEPT_MIN_EXAM_AGREEMENT
        decisions[code] = {"cohorte_dominante_coincide": bool(ok_i), "acuerdo_subtrain_ok": bool(ok_ii)}
        if ok_i and ok_ii:
            accepted[code] = year

    # Mezcla de naturalezas: departamentos/prefijos que usan códigos numéricos y letras
    ex = exams.copy()
    ex["tipo"] = ex["raw_level_char"].map(
        lambda c: "numerico" if c in NUMERIC_LEVEL else ("letra_ABC" if c in LETTER_HYPOTHESIS else "otro")
    )
    dept_mix = pd.crosstab(ex["department_code"], ex["tipo"])
    n_dept_mixed = int(((dept_mix.get("numerico", 0) > 0) & (dept_mix.get("letra_ABC", 0) > 0)).sum())

    # Acuerdo por examen dentro de departamentos mixtos vs sólo-letras (¿cambia el significado?)
    mixed_depts = set(dept_mix[(dept_mix.get("numerico", 0) > 0) & (dept_mix.get("letra_ABC", 0) > 0)].index)
    letter_rows = []
    for code, year in LETTER_HYPOTHESIS.items():
        for label, depts in (("deptos_mixtos", True), ("deptos_solo_letras", False)):
            ids = ex[(ex["raw_level_char"] == code) & (ex["department_code"].isin(mixed_depts) == depts)]["exam_id"]
            ids = [e for e in ids if e in dom.index]
            if ids:
                letter_rows.append(
                    {
                        "codigo": code,
                        "grupo_departamentos": label,
                        "n_examenes": len(ids),
                        "acuerdo_cohorte_esperada": round(
                            float(np.mean([dom[e] == EXPECTED_COHORT[year] for e in ids])), 4
                        ),
                    }
                )
    mix_df = pd.DataFrame(letter_rows)

    special = exams[exams["raw_level_char"].isin(["0", "P", "4"])][
        ["exam_id", "exam_name", "department_code", "raw_level_char"]
    ].copy()
    special["cohorte_dominante"] = special["exam_id"].map(dom).astype("Int64")

    info = {
        "decisiones_letras": decisions,
        "letras_aceptadas": accepted,
        "departamentos_con_numericos_y_letras": n_dept_mixed,
        "total_departamentos": int(ex["department_code"].nunique()),
        "codigos_especiales_0_P_4": special.to_dict(orient="records"),
    }
    return audit_df, mix_df, accepted, info


# =============================================================================
# Construcción de features
# =============================================================================


class FeatureContext:
    """Metadatos por examen y grafo histórico, construidos SOLO con exams + cohortes 1989-1993."""

    def __init__(self, exams: pd.DataFrame, students: pd.DataFrame, enrol: pd.DataFrame, level_map: Dict[str, int]):
        self.level = {
            r.exam_id: level_map.get(str(r.raw_level_char)) for r in exams.itertuples(index=False)
        }
        hist_ids = set(students.loc[students["cohort_group"] == HIST_GROUP, "student_id"])
        hist_enrol = enrol[enrol["student_id"].isin(hist_ids)]
        self.n_hist_enrolments_used = int(len(hist_enrol))
        self.n_curr_students_in_graph = int(
            hist_enrol["student_id"].isin(
                set(students.loc[students["cohort_group"] == CURR_GROUP, "student_id"])
            ).sum()
        )
        self.exam_ids = sorted(exams["exam_id"])
        self.idx = {e: i for i, e in enumerate(self.exam_ids)}
        self.adj = self._build_adj(hist_enrol)

    def _build_adj(self, hist_enrol: pd.DataFrame) -> np.ndarray:
        n = len(self.exam_ids)
        adj = np.zeros((n, n), dtype=np.int32)
        for _, g in hist_enrol.groupby("student_id"):
            ids = sorted({self.idx[e] for e in g["exam_id"]})
            for a_i, a in enumerate(ids):
                for b in ids[a_i + 1:]:
                    adj[a, b] = 1
                    adj[b, a] = 1
        return adj

    def common_neighbors_matrix(self, allowed_intermediates: Optional[set] = None) -> np.ndarray:
        a = self.adj.astype(np.int64)
        if allowed_intermediates is not None:
            mask = np.array([e in allowed_intermediates for e in self.exam_ids], dtype=np.int64)
            return (a * mask[None, :]) @ a
        return a @ a


def level_columns(df: pd.DataFrame, ctx: FeatureContext) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    l1 = df["materia_id_1"].map(ctx.level)
    l2 = df["materia_id_2"].map(ctx.level)
    conf = (l1.notna() & l2.notna()).to_numpy()
    same = conf & (l1.fillna(-1).to_numpy() == l2.fillna(-2).to_numpy())
    both1 = ((l1 == 1) & (l2 == 1)).to_numpy()
    return conf.astype(float), same.astype(float), both1.astype(float)


def common_neighbors_col(df: pd.DataFrame, ctx: FeatureContext, cn: np.ndarray) -> np.ndarray:
    i1 = df["materia_id_1"].map(ctx.idx).to_numpy()
    i2 = df["materia_id_2"].map(ctx.idx).to_numpy()
    return cn[i1, i2].astype(float)


def f4_block(df: pd.DataFrame, anio_conf: np.ndarray, mismo_anio: np.ndarray) -> np.ndarray:
    n1 = df["cant_inscriptos_hist_1"].to_numpy(dtype=float)
    n2 = df["cant_inscriptos_hist_2"].to_numpy(dtype=float)
    return np.column_stack(
        [
            df["mismo_departamento"].to_numpy(dtype=float),
            df["mismo_prefijo_curricular"].to_numpy(dtype=float),
            anio_conf,
            mismo_anio,
            df["misma_carrera"].to_numpy(dtype=float),
            np.log1p(df["cant_carreras_compartidas_hist"].to_numpy(dtype=float)),
            np.log1p(np.minimum(n1, n2)),
            np.log1p(np.maximum(n1, n2)),
            np.log1p(df["cant_estudiantes_compartidos_historico"].to_numpy(dtype=float)),
            df["jaccard_historico"].to_numpy(dtype=float),
        ]
    )


F4_COLS = [
    "mismo_departamento",
    "mismo_prefijo_curricular",
    "anio_confirmado",
    "mismo_anio",
    "misma_carrera",
    "log1p(cant_carreras_compartidas_hist)",
    "log1p(min(cant_inscriptos_hist_1, cant_inscriptos_hist_2))",
    "log1p(max(cant_inscriptos_hist_1, cant_inscriptos_hist_2))",
    "log1p(cant_estudiantes_compartidos_historico)",
    "jaccard_historico",
]


def build_f5(df: pd.DataFrame) -> np.ndarray:
    """F5: copia exacta de `build_f5_engineered_10cols` de experiments/ultima_ronda_ml/run_ultima_ronda.py."""
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
    return np.column_stack(
        [
            depto, pref, anio_conf, mismo_anio, misma_carr, np.log1p(carr_h), log_min_h, log_max_h,
            log_max_h - log_min_h,
            (max_h == 0).astype(float),
            ((min_h == 0) & (max_h > 0)).astype(float),
            (shared_h > 0).astype(float),
            np.log1p(shared_h),
            jacc_h,
            np.where(min_h > 0, shared_h / np.maximum(1.0, min_h), 0.0),
            misma_carr * mismo_anio + pref * mismo_anio + 0.5 * (1.0 - anio_conf) * misma_carr,
        ]
    )


F5_COLS = [
    "mismo_departamento", "mismo_prefijo_curricular", "anio_confirmado", "mismo_anio", "misma_carrera",
    "log1p(cant_carreras_compartidas_hist)", "log1p(min_insc_hist)", "log1p(max_insc_hist)",
    "log1p(max_insc_hist)-log1p(min_insc_hist)", "ambas_sin_historial (max_insc_hist==0)",
    "una_sin_historial (min==0 & max>0)", "comparten_hist (compartidos>0)",
    "log1p(cant_estudiantes_compartidos_historico)", "jaccard_historico",
    "coef_solapamiento (compartidos/min_insc_hist)",
    "interaccion_ad_hoc (misma_carrera*mismo_anio + prefijo*mismo_anio + 0.5*(1-anio_conf)*misma_carrera)",
]

F6_SIN_COLS = [
    c if c not in ("anio_confirmado", "mismo_anio") else f"{c}_corregido_ABC" for c in F4_COLS
]
F6_COLS = F6_SIN_COLS + ["ambas_primer_anio (nivel 1 o A en ambas)"]
F7_COLS = F6_COLS + ["log1p(vecinos_comunes_hist_2saltos)"]


def make_builders(ctx: FeatureContext, cn_full: np.ndarray) -> Dict[str, Callable[[pd.DataFrame], np.ndarray]]:
    def f0(df):
        return df[FEATURE_COLUMNS].to_numpy(dtype=float)

    def f4(df):
        return f4_block(df, df["anio_confirmado"].to_numpy(float), df["mismo_anio"].to_numpy(float))

    def f6_sin(df):
        conf, same, _ = level_columns(df, ctx)
        return f4_block(df, conf, same)

    def f6(df):
        conf, same, both1 = level_columns(df, ctx)
        return np.column_stack([f4_block(df, conf, same), both1])

    def f7(df):
        return np.column_stack([f6(df), np.log1p(common_neighbors_col(df, ctx, cn_full))])

    return {"F0": f0, "F4": f4, "F5": build_f5, "F6_sin_primer_anio": f6_sin, "F6": f6, "F7": f7}


REP_COLUMNS = {
    "F0": FEATURE_COLUMNS,
    "F4": F4_COLS,
    "F5": F5_COLS,
    "F6_sin_primer_anio": F6_SIN_COLS,
    "F6": F6_COLS,
    "F7": F7_COLS,
}


def swap_pair_order(df: pd.DataFrame) -> pd.DataFrame:
    s = df.copy()
    s["materia_id_1"], s["materia_id_2"] = df["materia_id_2"].values, df["materia_id_1"].values
    s["cant_inscriptos_hist_1"], s["cant_inscriptos_hist_2"] = (
        df["cant_inscriptos_hist_2"].values,
        df["cant_inscriptos_hist_1"].values,
    )
    return s


# =============================================================================
# Modelos (configuraciones fijas, sin búsqueda)
# =============================================================================

MODELS: Dict[str, Tuple[str, Callable[[], Any]]] = {
    "KNN_k51_Manhattan_Uniform": (
        "KNN",
        lambda: Pipeline(
            [("scaler", StandardScaler()),
             ("clf", KNeighborsClassifier(n_neighbors=51, metric="manhattan", weights="uniform", n_jobs=-1))]
        ),
    ),
    "KNN_k51_Euclidean_Distance": (
        "KNN",
        lambda: Pipeline(
            [("scaler", StandardScaler()),
             ("clf", KNeighborsClassifier(n_neighbors=51, metric="euclidean", weights="distance", n_jobs=-1))]
        ),
    ),
    "DecisionTree_d6_leaf50_Gini": (
        "DecisionTree",
        lambda: DecisionTreeClassifier(max_depth=6, min_samples_leaf=50, criterion="gini", random_state=RANDOM_SEED),
    ),
    "LogisticRegression_L2_C1": (
        "LogisticRegression",
        lambda: Pipeline(
            [("scaler", StandardScaler()),
             ("clf", LogisticRegression(C=1.0, max_iter=1000, random_state=RANDOM_SEED))]
        ),
    ),
}
MAIN_MODELS = ["KNN_k51_Manhattan_Uniform", "DecisionTree_d6_leaf50_Gini", "LogisticRegression_L2_C1"]

REFERENCE_OFFICIAL = (
    "KNN_k15_Oficial",
    lambda: Pipeline([("scaler", StandardScaler()), ("clf", KNeighborsClassifier(n_neighbors=15, n_jobs=-1))]),
)


def fit_score(builder, X_tr, y_tr, X_ev) -> Tuple[np.ndarray, float]:
    model = builder()
    t0 = time.time()
    model.fit(X_tr, y_tr)
    fit_s = time.time() - t0
    return model.predict_proba(X_ev)[:, 1], fit_s


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary: Dict[str, Any] = {}

    log("1. Cargando datos (solo lectura)...")
    exams = pd.read_csv(INTERIM / "exams_clean.csv", dtype=str)
    students = pd.read_csv(INTERIM / "students_clean.csv", dtype=str)
    enrol = pd.read_csv(INTERIM / "enrolments_clean.csv", dtype=str)

    train_df = load_ml_dataframe(TRAIN_FULL_PATH)
    test_df = load_ml_dataframe(TEST_FULL_PATH)
    sub_df, _, val_df, split_info = build_internal_subject_validation_split(train_df)
    summary["split"] = split_info
    subtrain_subjects = set(sub_df["materia_id_1"]) | set(sub_df["materia_id_2"])
    val_subjects = (set(val_df["materia_id_1"]) | set(val_df["materia_id_2"])) - subtrain_subjects
    train_subjects = set(train_df["materia_id_1"]) | set(train_df["materia_id_2"])
    test_subjects = (set(test_df["materia_id_1"]) | set(test_df["materia_id_2"])) - train_subjects
    summary["materias"] = {
        "subtrain": len(subtrain_subjects), "validation": len(val_subjects),
        "train": len(train_subjects), "test": len(test_subjects),
    }

    y_sub = sub_df[TARGET_CLASSIFICATION].to_numpy(int)
    y_val = val_df[TARGET_CLASSIFICATION].to_numpy(int)
    y_tr = train_df[TARGET_CLASSIFICATION].to_numpy(int)
    y_te = test_df[TARGET_CLASSIFICATION].to_numpy(int)
    aff_val = val_df["cant_estudiantes_afectados"].to_numpy(float)
    aff_te = test_df["cant_estudiantes_afectados"].to_numpy(float)

    # ------------------------------------------------------------------ Sección 1
    log("2. Auditoría de niveles académicos A/B/C...")
    audit_df, mix_df, accepted, lvl_info = audit_levels(exams, students, enrol, subtrain_subjects)
    audit_df.to_csv(OUT_DIR / "auditoria_niveles_codigos.csv", index=False)
    mix_df.to_csv(OUT_DIR / "auditoria_niveles_deptos_mixtos.csv", index=False)
    summary["niveles"] = lvl_info
    level_map = dict(NUMERIC_LEVEL)
    level_map.update(accepted)
    summary["niveles"]["mapa_final_usado"] = level_map

    ctx = FeatureContext(exams, students, enrol, level_map)
    cn_full = ctx.common_neighbors_matrix()

    # Cuantificación de pares afectados (todos los 319.600 pares = train_full ∪ test_full)
    all_pairs = pd.concat([train_df, test_df], ignore_index=True)
    conf_new, same_new, both1 = level_columns(all_pairs, ctx)
    affected = {
        "pares_totales": int(len(all_pairs)),
        "anio_confirmado_original_1": int(all_pairs["anio_confirmado"].sum()),
        "anio_confirmado_corregido_1": int(conf_new.sum()),
        "pares_que_pasan_a_confirmado": int(((all_pairs["anio_confirmado"] == 0) & (conf_new == 1)).sum()),
        "mismo_anio_original_1": int(all_pairs["mismo_anio"].sum()),
        "mismo_anio_corregido_1": int(same_new.sum()),
        "pares_que_pasan_a_mismo_anio": int(((all_pairs["mismo_anio"] == 0) & (same_new == 1)).sum()),
        "pares_mismo_anio_que_se_pierden": int(((all_pairs["mismo_anio"] == 1) & (same_new == 0)).sum()),
        "pares_ambas_primer_anio": int(both1.sum()),
        "examenes_con_nivel_original": int(sum(1 for e in exams["raw_level_char"] if e in NUMERIC_LEVEL)),
        "examenes_con_nivel_corregido": int(sum(1 for v in ctx.level.values() if v is not None)),
    }
    # Positivos afectados: solo subtrain + validation (no se mira test para describir)
    sv = pd.concat([sub_df, val_df], ignore_index=True)
    c_sv, s_sv, b_sv = level_columns(sv, ctx)
    ysv = sv[TARGET_CLASSIFICATION].to_numpy(int)
    gained = (sv["mismo_anio"].to_numpy() == 0) & (s_sv == 1)
    affected["subtrain_val_positivos"] = int(ysv.sum())
    affected["subtrain_val_pares_ganan_mismo_anio"] = int(gained.sum())
    affected["subtrain_val_positivos_ganan_mismo_anio"] = int((gained & (ysv == 1)).sum())
    affected["subtrain_val_tasa_positiva_en_ganados"] = round(float(ysv[gained].mean()), 4)
    affected["subtrain_val_tasa_positiva_mismo_anio_original"] = round(
        float(ysv[sv["mismo_anio"].to_numpy() == 1].mean()), 4
    )
    affected["subtrain_val_tasa_positiva_ambas_primer_anio"] = round(float(ysv[b_sv == 1].mean()), 4)
    affected["subtrain_val_tasa_positiva_global"] = round(float(ysv.mean()), 4)
    summary["pares_afectados"] = affected

    # ------------------------------------------------------------------ Sección 2: verificación de afirmaciones previas
    log("3. Verificando afirmaciones de la ronda anterior (solo validation)...")
    zero5 = (
        (val_df["cant_estudiantes_compartidos_historico"] == 0)
        & (val_df["misma_carrera"] == 0)
        & (val_df["mismo_anio"] == 0)
        & (val_df["mismo_departamento"] == 0)
        & (val_df["mismo_prefijo_curricular"] == 0)
    ).to_numpy()
    _, _, b_val = level_columns(val_df, ctx)
    old_map = {"0": 1, "P": 1, "1": 1, "A": 1, "2": 2, "B": 2, "3": 3, "C": 3, "4": 4}
    l1o = val_df["materia_id_1"].map(dict(zip(exams["exam_id"], exams["raw_level_char"]))).map(old_map)
    l2o = val_df["materia_id_2"].map(dict(zip(exams["exam_id"], exams["raw_level_char"]))).map(old_map)
    b_val_old = ((l1o == 1) & (l2o == 1)).to_numpy()
    cn_val = common_neighbors_col(val_df, ctx, cn_full)
    pos5 = zero5 & (y_val == 1)
    neg5 = zero5 & (y_val == 0)
    summary["verificacion_afirmaciones_previas"] = {
        "val_pares_5_ceros": int(zero5.sum()),
        "val_positivos_5_ceros": int(pos5.sum()),
        "val_positivos_totales": int(y_val.sum()),
        "pct_positivos_en_5_ceros": round(100 * pos5.sum() / y_val.sum(), 2),
        "pos5_ambas_primer_anio_mapa_corregido": int((pos5 & (b_val == 1)).sum()),
        "neg5_ambas_primer_anio_mapa_corregido": int((neg5 & (b_val == 1)).sum()),
        "pos5_ambas_primer_anio_mapa_anterior_0P_como_1": int((pos5 & b_val_old).sum()),
        "neg5_ambas_primer_anio_mapa_anterior_0P_como_1": int((neg5 & b_val_old).sum()),
        "pct_pos5_con_vecinos_comunes_2saltos": round(100 * float((cn_val[pos5] > 0).mean()), 2),
        "pct_neg5_con_vecinos_comunes_2saltos": round(100 * float((cn_val[neg5] > 0).mean()), 2),
    }

    # ------------------------------------------------------------------ Auditoría de leakage
    log("4. Auditoría de leakage de features nuevas...")
    builders = make_builders(ctx, cn_full)
    # (a) el grafo histórico no contiene estudiantes de 1994
    # (b) invariancia: recalcular el grafo tras ELIMINAR todas las inscripciones de 1994 -> idéntico
    curr_ids = set(students.loc[students["cohort_group"] == CURR_GROUP, "student_id"])
    ctx_no94 = FeatureContext(exams, students, enrol[~enrol["student_id"].isin(curr_ids)], level_map)
    inv_no94 = bool(np.array_equal(ctx_no94.adj, ctx.adj))
    # (c) invariancia: permutar al azar las inscripciones de 1994 entre exámenes -> idéntico
    rng = np.random.default_rng(RANDOM_SEED)
    enrol_perm = enrol.copy()
    mask94 = enrol_perm["student_id"].isin(curr_ids).to_numpy()
    enrol_perm.loc[mask94, "exam_id"] = rng.permutation(enrol_perm.loc[mask94, "exam_id"].to_numpy())
    ctx_perm = FeatureContext(exams, students, enrol_perm, level_map)
    inv_perm = bool(np.array_equal(ctx_perm.adj, ctx.adj))
    # (d) simetría: intercambiar materia_1 y materia_2 no cambia la matriz
    sym = {}
    probe = val_df.sample(n=5000, random_state=RANDOM_SEED).reset_index(drop=True)
    probe_sw = swap_pair_order(probe)
    for name, fn in builders.items():
        sym[name] = bool(np.allclose(fn(probe), fn(probe_sw)))
    # (e) consistencia del grafo con la feature oficial de 1 salto: arista histórica <=> compartidos_hist > 0
    i1 = all_pairs["materia_id_1"].map(ctx.idx).to_numpy()
    i2 = all_pairs["materia_id_2"].map(ctx.idx).to_numpy()
    one_hop_consistent = bool(
        np.array_equal(ctx.adj[i1, i2] > 0, all_pairs["cant_estudiantes_compartidos_historico"].to_numpy() > 0)
    )
    # (f) los niveles salen sólo de exam_id: comprobar que no dependan de inscripciones
    summary["leakage_checks"] = {
        "inscripciones_historicas_usadas_en_grafo": ctx.n_hist_enrolments_used,
        "inscripciones_1994_en_grafo": ctx.n_curr_students_in_graph,
        "grafo_identico_sin_inscripciones_1994": inv_no94,
        "grafo_identico_con_1994_permutado": inv_perm,
        "arista_1salto_equivale_a_compartidos_hist_oficial": one_hop_consistent,
        "simetria_por_representacion": sym,
    }

    # ------------------------------------------------------------------ Matrices
    log("5. Construyendo matrices de features...")
    reps = ["F0", "F4", "F5", "F6_sin_primer_anio", "F6", "F7"]
    X = {r: {} for r in reps}
    for r in reps:
        fn = builders[r]
        X[r]["sub"], X[r]["val"] = fn(sub_df), fn(val_df)
        X[r]["train"], X[r]["test"] = fn(train_df), fn(test_df)

    # Auditoría extra F7: intermediarios restringidos a materias de subtrain (sin nodos held-out)
    cn_sub = ctx.common_neighbors_matrix(allowed_intermediates=subtrain_subjects)
    X_f7_restr_sub = np.column_stack([builders["F6"](sub_df), np.log1p(common_neighbors_col(sub_df, ctx, cn_sub))])
    X_f7_restr_val = np.column_stack([builders["F6"](val_df), np.log1p(common_neighbors_col(val_df, ctx, cn_sub))])

    # Bootstrap por materia de validación (clusters)
    val_cluster = np.where(
        val_df["materia_id_1"].isin(val_subjects), val_df["materia_id_1"], val_df["materia_id_2"]
    )
    clusters = np.unique(val_cluster)
    cl_index = {c: np.where(val_cluster == c)[0] for c in clusters}
    boot_idx = []
    rngb = np.random.default_rng(RANDOM_SEED)
    for _ in range(N_BOOTSTRAP):
        pick = rngb.choice(clusters, size=len(clusters), replace=True)
        boot_idx.append(np.concatenate([cl_index[c] for c in pick]))

    # ------------------------------------------------------------------ Sección 5: validación
    log("6. Entrenando en subtrain y seleccionando umbral en validation...")
    val_rows, val_scores, thresholds = [], {}, {}
    for r in reps:
        for mname, (fam, builder) in MODELS.items():
            s, fit_s = fit_score(builder, X[r]["sub"], y_sub, X[r]["val"])
            th = find_best_f1_threshold_on_validation(y_val, s)
            row = {"representacion": r, "n_features": X[r]["sub"].shape[1], "modelo": mname, "familia": fam,
                   "split": "validation", **metrics_at(y_val, s, th, aff_val), "fit_segundos": round(fit_s, 2)}
            val_rows.append(row)
            val_scores[(r, mname)] = s
            thresholds[(r, mname)] = th
            log(f"   {r:20s} {mname:30s} F1={row['f1']:.4f} PR-AUC={row['pr_auc']:.4f} th={th}")

    # Referencia: modelo oficial actual
    s_off, fit_off = fit_score(REFERENCE_OFFICIAL[1], X["F0"]["sub"], y_sub, X["F0"]["val"])
    th_off = find_best_f1_threshold_on_validation(y_val, s_off)
    val_rows.append({"representacion": "F0", "n_features": 10, "modelo": REFERENCE_OFFICIAL[0], "familia": "KNN (referencia)",
                     "split": "validation", **metrics_at(y_val, s_off, th_off, aff_val), "fit_segundos": round(fit_off, 2)})

    # Auditoría F7 restringido
    s_r, fit_r = fit_score(MODELS["DecisionTree_d6_leaf50_Gini"][1], X_f7_restr_sub, y_sub, X_f7_restr_val)
    s_r2, _ = fit_score(MODELS["LogisticRegression_L2_C1"][1], X_f7_restr_sub, y_sub, X_f7_restr_val)
    summary["leakage_checks"]["F7_intermediarios_solo_subtrain_val_pr_auc"] = {
        "DecisionTree": round(float(average_precision_score(y_val, s_r)), 4),
        "LogisticRegression": round(float(average_precision_score(y_val, s_r2)), 4),
    }

    # Bootstrap: PR-AUC y delta vs F0 del mismo modelo
    log("7. Bootstrap por materias de validación...")
    boot_rows = []
    for r in reps:
        for mname in MODELS:
            s = val_scores[(r, mname)]
            s0 = val_scores[("F0", mname)]
            vals, deltas = [], []
            for idx in boot_idx:
                yb = y_val[idx]
                if yb.sum() == 0:
                    continue
                a = average_precision_score(yb, s[idx])
                b = average_precision_score(yb, s0[idx])
                vals.append(a)
                deltas.append(a - b)
            boot_rows.append({
                "representacion": r, "modelo": mname,
                "pr_auc_val": round(float(average_precision_score(y_val, s)), 4),
                "pr_auc_ic95_inf": round(float(np.percentile(vals, 2.5)), 4),
                "pr_auc_ic95_sup": round(float(np.percentile(vals, 97.5)), 4),
                "delta_vs_F0": round(float(np.mean(deltas)), 4),
                "delta_ic95_inf": round(float(np.percentile(deltas, 2.5)), 4),
                "delta_ic95_sup": round(float(np.percentile(deltas, 97.5)), 4),
                "prob_delta_mayor_0": round(float(np.mean(np.array(deltas) > 0)), 3),
            })
    boot_df = pd.DataFrame(boot_rows)
    val_df_rows = pd.DataFrame(val_rows)
    rep_cmp = val_df_rows[val_df_rows["modelo"].isin(MODELS.keys())].merge(
        boot_df.drop(columns=["pr_auc_val"]), on=["representacion", "modelo"]
    )
    rep_cmp["columnas"] = rep_cmp["representacion"].map(lambda r: " | ".join(REP_COLUMNS[r]))
    rep_cmp.to_csv(OUT_DIR / "comparacion_representaciones_validation.csv", index=False)

    # ------------------------------------------------------------------ Selección (solo validation)
    log("8. Aplicando regla de selección pre-declarada (solo validation)...")
    leak_ok = {r: True for r in reps}
    if not (inv_no94 and inv_perm and ctx.n_curr_students_in_graph == 0 and one_hop_consistent):
        leak_ok["F7"] = False
    if not accepted:
        leak_ok["F6"] = leak_ok["F6_sin_primer_anio"] = leak_ok["F7"] = False
    main_only = val_df_rows[val_df_rows["modelo"].isin(MAIN_MODELS)]
    rep_score = main_only.groupby("representacion")["pr_auc"].mean().to_dict()
    rep_f1 = main_only.groupby("representacion")["f1"].mean().to_dict()
    valid_reps = [r for r in REPRESENTATION_SIMPLICITY_ORDER if leak_ok[r]]
    best_score = max(rep_score[r] for r in valid_reps)
    chosen_rep = next(r for r in valid_reps if rep_score[r] >= best_score - REP_TOLERANCE_PR_AUC)
    cand = val_df_rows[(val_df_rows["representacion"] == chosen_rep) & (val_df_rows["modelo"].isin(MODELS.keys()))]
    cand = cand.sort_values(["pr_auc", "f1"], ascending=False)
    main_cfg = cand.iloc[0]
    alt_cfg = cand[cand["familia"] != main_cfg["familia"]].iloc[0]
    summary["seleccion"] = {
        "pr_auc_medio_val_por_representacion": {k: round(v, 4) for k, v in rep_score.items()},
        "f1_medio_val_por_representacion": {k: round(v, 4) for k, v in rep_f1.items()},
        "representaciones_validas": valid_reps,
        "representacion_elegida": chosen_rep,
        "principal": {"modelo": main_cfg["modelo"], "umbral": float(main_cfg["threshold"])},
        "alternativa": {"modelo": alt_cfg["modelo"], "umbral": float(alt_cfg["threshold"])},
    }
    log(f"   Representación elegida: {chosen_rep}; principal={main_cfg['modelo']}; alternativa={alt_cfg['modelo']}")

    # ------------------------------------------------------------------ Test (una sola vez)
    log("9. Reentrenando en train completo y evaluando UNA vez en test...")
    test_rows, test_scores = [], {}
    for r in reps:
        for mname, (fam, builder) in MODELS.items():
            s, fit_s = fit_score(builder, X[r]["train"], y_tr, X[r]["test"])
            th = thresholds[(r, mname)]
            test_rows.append({"representacion": r, "n_features": X[r]["train"].shape[1], "modelo": mname, "familia": fam,
                              "split": "test", **metrics_at(y_te, s, th, aff_te), "fit_segundos": round(fit_s, 2)})
            test_scores[(r, mname)] = s
    s_off_te, fit_off_te = fit_score(REFERENCE_OFFICIAL[1], X["F0"]["train"], y_tr, X["F0"]["test"])
    test_rows.append({"representacion": "F0", "n_features": 10, "modelo": REFERENCE_OFFICIAL[0], "familia": "KNN (referencia)",
                      "split": "test", **metrics_at(y_te, s_off_te, th_off, aff_te), "fit_segundos": round(fit_off_te, 2)})
    models_cmp = pd.concat([val_df_rows, pd.DataFrame(test_rows)], ignore_index=True)
    models_cmp["seleccionable"] = models_cmp["representacion"].map(leak_ok)
    models_cmp.to_csv(OUT_DIR / "comparacion_modelos_val_test.csv", index=False)

    # ------------------------------------------------------------------ Sección 6: umbrales / grafo
    log("10. Barrido de umbrales para principal, alternativa y referencia...")
    thr_rows = []
    configs = [
        ("principal", chosen_rep, main_cfg["modelo"]),
        ("alternativa", chosen_rep, alt_cfg["modelo"]),
        ("referencia_F0", "F0", main_cfg["modelo"]),
    ]
    grid = np.round(np.arange(0.02, 0.61, 0.01), 2)
    recall_points = {}
    for role, r, mname in configs:
        sv_, ste_ = val_scores[(r, mname)], test_scores[(r, mname)]
        f1s = [f1_score(y_val, (sv_ >= t).astype(int), zero_division=0) for t in grid]
        fmax = max(f1s)
        # Punto "orientado a recall" pre-declarado: menor umbral con F1_val >= 95 % del máximo
        th_rec = float(min(t for t, f in zip(grid, f1s) if f >= 0.95 * fmax))
        recall_points[role] = th_rec
        for t in grid:
            for split, yy, ss, aa in (("validation", y_val, sv_, aff_val), ("test", y_te, ste_, aff_te)):
                thr_rows.append({"rol": role, "representacion": r, "modelo": mname, "split": split,
                                 **metrics_at(yy, ss, float(t), aa)})
    thr_df = pd.DataFrame(thr_rows)
    thr_df.to_csv(OUT_DIR / "umbrales_tradeoff.csv", index=False)

    graph_rows = []
    held_val, held_te = val_subjects, test_subjects
    for role, r, mname in configs:
        for label, th in (("umbral_F1_val", thresholds[(r, mname)]), ("umbral_orientado_recall", recall_points[role])):
            for split, dfx, yy, ss, aa, held in (
                ("validation", val_df, y_val, val_scores[(r, mname)], aff_val, held_val),
                ("test", test_df, y_te, test_scores[(r, mname)], aff_te, held_te),
            ):
                p = ss >= th
                deg_pred = pd.concat([dfx.loc[p, "materia_id_1"], dfx.loc[p, "materia_id_2"]]).value_counts()
                deg_real = pd.concat([dfx.loc[yy == 1, "materia_id_1"], dfx.loc[yy == 1, "materia_id_2"]]).value_counts()
                held_list = sorted(held)
                dp = np.array([deg_pred.get(h, 0) for h in held_list])
                dr = np.array([deg_real.get(h, 0) for h in held_list])
                graph_rows.append({
                    "rol": role, "representacion": r, "modelo": mname, "criterio_umbral": label, "split": split,
                    **metrics_at(yy, ss, th, aa),
                    "aristas_reales": int(yy.sum()),
                    "estudiantes_afectados_reales": int(aa[yy == 1].sum()),
                    "estudiantes_afectados_cubiertos": int(aa[(yy == 1) & p].sum()),
                    "grado_medio_predicho_materias_heldout": round(float(dp.mean()), 2),
                    "grado_medio_real_materias_heldout": round(float(dr.mean()), 2),
                    "grado_max_predicho_materias_heldout": int(dp.max()),
                    "materias_heldout_aisladas_predichas": int((dp == 0).sum()),
                    "materias_heldout_aisladas_reales": int((dr == 0).sum()),
                })
    # Referencia oficial
    for split, dfx, yy, ss, aa in (("validation", val_df, y_val, s_off, aff_val), ("test", test_df, y_te, s_off_te, aff_te)):
        graph_rows.append({"rol": "modelo_oficial_actual", "representacion": "F0", "modelo": "KNN_k15_Oficial",
                           "criterio_umbral": "umbral_F1_val", "split": split, **metrics_at(yy, ss, th_off, aa),
                           "aristas_reales": int(yy.sum()), "estudiantes_afectados_reales": int(aa[yy == 1].sum()),
                           "estudiantes_afectados_cubiertos": int(aa[(yy == 1) & (ss >= th_off)].sum())})
    pd.DataFrame(graph_rows).to_csv(OUT_DIR / "impacto_grafo.csv", index=False)

    # ------------------------------------------------------------------ Auditoría de leakage (tabla)
    lc = summary["leakage_checks"]
    leak_rows = [
        {"feature": "anio_confirmado_corregido_ABC", "representa": "Ambas materias tienen nivel curricular identificable (1-4 o A/B/C aceptadas)",
         "fuente": "exams (exam_id[2]) -> exams_clean.csv:raw_level_char", "solo_info_previa": "Sí (metadato del catálogo)",
         "usa_target": "No", "riesgo_leakage": "Bajo: regla fija; la equivalencia se validó con distribución de inscripciones por cohorte, no con etiquetas de pares",
         "simetrica": sym["F6_sin_primer_anio"], "sentido_academico": "Corrige la variable oficial existente", "veredicto": "Aprobada" if accepted else "Rechazada"},
        {"feature": "mismo_anio_corregido_ABC", "representa": "Ambas materias del mismo año curricular, con A=1, B=2, C=3",
         "fuente": "exams_clean.csv:raw_level_char", "solo_info_previa": "Sí", "usa_target": "No",
         "riesgo_leakage": "Bajo (idem)", "simetrica": sym["F6_sin_primer_anio"],
         "sentido_academico": "Es la definición pretendida de mismo_anio", "veredicto": "Aprobada" if accepted else "Rechazada"},
        {"feature": "ambas_primer_anio", "representa": "Ambas materias de nivel 1 o A",
         "fuente": "exams_clean.csv:raw_level_char", "solo_info_previa": "Sí", "usa_target": "No",
         "riesgo_leakage": "Sin leakage de etiquetas, pero es específica de la población objetivo (cohorte 1994 = ingresantes): no transferible si cambia la definición del target",
         "simetrica": sym["F6"], "sentido_academico": "Sí, bajo el target actual", "veredicto": "Aprobada con advertencia" if accepted else "Rechazada"},
        {"feature": "log1p(vecinos_comunes_hist_2saltos)", "representa": "Cantidad de materias w que compartieron alumnos históricos con ambas materias del par",
         "fuente": "enrolments_clean.csv + students_clean.csv (solo cohort_group=historico_1989_1993)",
         "solo_info_previa": f"Sí: {lc['inscripciones_historicas_usadas_en_grafo']} inscripciones históricas, {lc['inscripciones_1994_en_grafo']} de 1994",
         "usa_target": "No (grafo idéntico al eliminar o permutar inscripciones 1994: "
                       f"{lc['grafo_identico_sin_inscripciones_1994']}/{lc['grafo_identico_con_1994_permutado']})",
         "riesgo_leakage": "Usa materias held-out como intermediarias, igual que las features oficiales de 1 salto usan su historial; "
                           f"con intermediarios restringidos a subtrain PR-AUC val = {lc['F7_intermediarios_solo_subtrain_val_pr_auc']}",
         "simetrica": sym["F7"], "sentido_academico": "Proximidad curricular indirecta (materias que comparten 'vecinas')",
         "veredicto": "Aprobada" if leak_ok["F7"] else "Rechazada"},
    ]
    pd.DataFrame(leak_rows).to_csv(OUT_DIR / "auditoria_leakage_features.csv", index=False)

    # ------------------------------------------------------------------ PR curves
    log("11. Curvas Precision-Recall...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    mname = main_cfg["modelo"]
    for col, (split, yy, sc, ref) in enumerate(
        (("validation", y_val, val_scores, s_off), ("test", y_te, test_scores, s_off_te))
    ):
        ax = axes[0, col]
        for r in ["F0", "F4", "F5", "F6", "F7"]:
            p_, r_, _ = precision_recall_curve(yy, sc[(r, mname)])
            ax.plot(r_, p_, label=f"{r} (AP={average_precision_score(yy, sc[(r, mname)]):.3f})")
        p_, r_, _ = precision_recall_curve(yy, ref)
        ax.plot(r_, p_, "k--", label=f"KNN_k15 oficial F0 (AP={average_precision_score(yy, ref):.3f})")
        ax.set_title(f"{mname} por representación — {split}")
        ax2 = axes[1, col]
        for m2 in MODELS:
            p_, r_, _ = precision_recall_curve(yy, sc[(chosen_rep, m2)])
            ax2.plot(r_, p_, label=f"{m2} (AP={average_precision_score(yy, sc[(chosen_rep, m2)]):.3f})")
        ax2.set_title(f"Modelos sobre {chosen_rep} — {split}")
        for a in (ax, ax2):
            a.set_xlabel("Recall")
            a.set_ylabel("Precision")
            a.set_xlim(0, 1)
            a.set_ylim(0, 1)
            a.grid(alpha=0.3)
            a.legend(fontsize=8)
    fig.suptitle("Cierre ML — curvas PR (selección en validation; test evaluado una sola vez)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "pr_curves_cierre_ml.png", dpi=130)

    summary["umbral_orientado_recall"] = recall_points
    summary["rep_columns"] = REP_COLUMNS
    with (OUT_DIR / "resumen_cierre_ml.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2, default=str)
    log("Listo.")


if __name__ == "__main__":
    main()
