"""Fase 2.5: Preparación del dataset para Machine Learning (partición y balanceo).

Este script es independiente de la Fase 2 (`build_subject_pairs.py`) y mantiene
intacto `data/processed/subject_pairs_poc.csv`.

Decisiones técnicas implementadas:
1. Separación Entrenamiento / Prueba por Materias de Retención Estratificada
   (Stratified Subject-Level Holdout Split):
   - Se seleccionan 86 materias de prueba (~10,75% de las 800 materias) de forma
     estratificada según su nivel (`raw_level_char`) con semilla fija `RANDOM_SEED = 42`.
   - Entrenamiento (`ml_train_full.csv`, 254.541 pares, 79,64%): pares donde ambas
     materias pertenecen al grupo de 714 materias de entrenamiento.
   - Prueba (`ml_test_full.csv`, 65.059 pares, 20,36%): pares donde al menos una
     de las dos materias pertenece al grupo de 86 materias de prueba (nunca vista
     en ningún par de entrenamiento), conservando la distribución real de clases
     (1,491% positivos / 98,509% negativos).
2. Undersampling Aleatorio 50/50 exclusivamente sobre Entrenamiento:
   - En `ml_train_balanced.csv` (6.996 pares) se conservan todos los 3.498 pares
     positivos de entrenamiento y se seleccionan aleatoriamente sin reemplazo
     3.498 pares negativos de entrenamiento (`RANDOM_SEED = 42`).
3. Selección estricta de variables sin Data Leakage:
   - Se excluyen todas las variables de la cohorte 1994 que no son target
     (`cant_inscriptos_actual_1`, `cant_inscriptos_actual_2`, `tasa_afectados_actual`)
     y los metadatos descriptivos o con nulos estructurales (`materia_nombre_1`,
     `materia_nombre_2`, `departamento_1`, `departamento_2`, `anio_1`, `anio_2`).
   - Se conservan `materia_id_1` y `materia_id_2` únicamente como columnas de
     identificación (`ID_COLUMNS`, NO predictoras) para poder vincular las
     predicciones de los modelos con las aristas del grafo de conflictos, junto
     con las 11 variables predictoras limpias (`PREDICTOR_COLUMNS`) y las 2
     variables objetivo (`cant_estudiantes_afectados` y `hubo_conflicto_potencial`).
"""

from __future__ import annotations

import csv
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Set, Tuple


from src.common.schema import (
    COLLINEAR_EXCLUDED_COLUMN,
    FEATURE_COLUMNS,
    ID_COLUMNS,
    PROJECT_ROOT,
    TARGET_CLASSIFICATION,
    TARGET_REGRESSION,
)


RANDOM_SEED = 42
TEST_SUBJECT_FRACTION = 0.10625  # ~86 materias de 800 -> ~20,36% de los pares en test

DEFAULT_INTERIM_EXAMS_CSV = PROJECT_ROOT / "data" / "interim" / "exams_clean.csv"
DEFAULT_POC_PAIRS_CSV = PROJECT_ROOT / "data" / "processed" / "subject_pairs_poc.csv"
DEFAULT_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

TRAIN_BALANCED_FILENAME = "ml_train_balanced.csv"
TEST_FULL_FILENAME = "ml_test_full.csv"
TRAIN_FULL_FILENAME = "ml_train_full.csv"

# Subconjunto sin colinealidad perfecta (10 features aprobadas en src/common/schema.py)
PREDICTOR_COLUMNS_NO_COLLINEAR: List[str] = FEATURE_COLUMNS

# 11 variables predictoras limpias exportadas en los CSV de Fase 2.5 (incluye tasa_compartidos_historico)
PREDICTOR_COLUMNS: List[str] = (
    FEATURE_COLUMNS[:-1] + [COLLINEAR_EXCLUDED_COLUMN, FEATURE_COLUMNS[-1]]
)

# Variables objetivo (calculadas con cohorte 1994)
TARGET_COLUMNS: List[str] = [TARGET_REGRESSION, TARGET_CLASSIFICATION]

# Columnas excluidas por ser texto descriptivo, alta cardinalidad redundante o tener nulos estructurales
EXCLUDED_DESCRIPTIVE_COLUMNS: List[str] = [
    "materia_nombre_1",
    "materia_nombre_2",
    "departamento_1",
    "departamento_2",
    "anio_1",
    "anio_2",
]

# Columnas excluidas físicamente por riesgo directo de Data Leakage (derivadas de la cohorte 1994)
EXCLUDED_LEAKAGE_COLUMNS: List[str] = [
    "cant_inscriptos_actual_1",
    "cant_inscriptos_actual_2",
    "tasa_afectados_actual",
]

ML_OUTPUT_COLUMNS: List[str] = ID_COLUMNS + PREDICTOR_COLUMNS + TARGET_COLUMNS


@dataclass
class Phase25Summary:
    """Resumen de auditoría y conteos de la Fase 2.5."""

    random_seed: int
    total_subjects: int
    train_subjects_count: int
    test_subjects_count: int

    poc_total_rows: int
    poc_positives: int
    poc_negatives: int

    train_full_rows: int
    train_full_positives: int
    train_full_negatives: int

    train_balanced_rows: int
    train_balanced_positives: int
    train_balanced_negatives: int

    test_full_rows: int
    test_full_positives: int
    test_full_negatives: int

    duplicate_pairs_found: int
    train_test_pair_overlap: int


def select_stratified_test_subjects(
    exams_csv: Path = DEFAULT_INTERIM_EXAMS_CSV,
    seed: int = RANDOM_SEED,
    fraction: float = TEST_SUBJECT_FRACTION,
) -> Tuple[Set[str], Set[str]]:
    """Selecciona un conjunto de materias de prueba estratificado por `raw_level_char`."""
    by_level: Dict[str, List[str]] = defaultdict(list)
    all_exams: Set[str] = set()

    with exams_csv.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            exam_id = row["exam_id"]
            level_char = row["raw_level_char"]
            by_level[level_char].append(exam_id)
            all_exams.add(exam_id)

    rng = random.Random(seed)
    test_subjects: Set[str] = set()
    for level_char in sorted(by_level.keys()):
        candidates = sorted(by_level[level_char])
        k = max(1, round(len(candidates) * fraction))
        test_subjects.update(rng.sample(candidates, k))

    train_subjects = all_exams - test_subjects
    return train_subjects, test_subjects


def write_ml_csv(output_path: Path, rows: List[Dict[str, str]]) -> None:
    """Escribe un archivo CSV listo para ML solo con predictores limpios y targets."""
    with output_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(ML_OUTPUT_COLUMNS)
        for row in rows:
            writer.writerow([row[col] for col in ML_OUTPUT_COLUMNS])


def run_phase25_preparation(
    exams_csv: Path = DEFAULT_INTERIM_EXAMS_CSV,
    poc_pairs_csv: Path = DEFAULT_POC_PAIRS_CSV,
    processed_dir: Path = DEFAULT_PROCESSED_DIR,
    seed: int = RANDOM_SEED,
) -> Phase25Summary:
    """Ejecuta la separación Train/Test por materias y el undersampling en Train."""
    train_subjects, test_subjects = select_stratified_test_subjects(
        exams_csv=exams_csv,
        seed=seed,
    )

    train_pos_rows: List[Dict[str, str]] = []
    train_neg_rows: List[Dict[str, str]] = []
    test_rows: List[Dict[str, str]] = []

    seen_pairs: Set[Tuple[str, str]] = set()
    train_pairs_set: Set[Tuple[str, str]] = set()
    test_pairs_set: Set[Tuple[str, str]] = set()
    duplicate_pairs = 0

    poc_positives = 0
    poc_negatives = 0
    test_positives = 0
    test_negatives = 0

    with poc_pairs_csv.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            pair_key = (row["materia_id_1"], row["materia_id_2"])
            if pair_key in seen_pairs:
                duplicate_pairs += 1
                continue
            seen_pairs.add(pair_key)

            is_positive = row[TARGET_CLASSIFICATION] == "True"
            if is_positive:
                poc_positives += 1
            else:
                poc_negatives += 1

            if (row["materia_id_1"] in test_subjects) or (
                row["materia_id_2"] in test_subjects
            ):
                test_rows.append(row)
                test_pairs_set.add(pair_key)
                if is_positive:
                    test_positives += 1
                else:
                    test_negatives += 1
            else:
                train_pairs_set.add(pair_key)
                if is_positive:
                    train_pos_rows.append(row)
                else:
                    train_neg_rows.append(row)

    # Undersampling 50/50 sin reemplazo EXCLUSIVAMENTE sobre el conjunto de entrenamiento
    rng = random.Random(seed)
    sampled_train_neg_rows = rng.sample(train_neg_rows, len(train_pos_rows))
    train_balanced_rows = list(train_pos_rows) + sampled_train_neg_rows
    rng.shuffle(train_balanced_rows)

    train_full_rows = train_pos_rows + train_neg_rows
    # Mantener orden lexicográfico original en train_full_rows
    train_full_rows.sort(key=lambda r: (r["materia_id_1"], r["materia_id_2"]))

    try:
        processed_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass

    write_ml_csv(processed_dir / TRAIN_BALANCED_FILENAME, train_balanced_rows)
    write_ml_csv(processed_dir / TEST_FULL_FILENAME, test_rows)
    write_ml_csv(processed_dir / TRAIN_FULL_FILENAME, train_full_rows)

    overlap = len(train_pairs_set & test_pairs_set)

    return Phase25Summary(
        random_seed=seed,
        total_subjects=len(train_subjects) + len(test_subjects),
        train_subjects_count=len(train_subjects),
        test_subjects_count=len(test_subjects),
        poc_total_rows=poc_positives + poc_negatives,
        poc_positives=poc_positives,
        poc_negatives=poc_negatives,
        train_full_rows=len(train_full_rows),
        train_full_positives=len(train_pos_rows),
        train_full_negatives=len(train_neg_rows),
        train_balanced_rows=len(train_balanced_rows),
        train_balanced_positives=len(train_pos_rows),
        train_balanced_negatives=len(sampled_train_neg_rows),
        test_full_rows=len(test_rows),
        test_full_positives=test_positives,
        test_full_negatives=test_negatives,
        duplicate_pairs_found=duplicate_pairs,
        train_test_pair_overlap=overlap,
    )


def main() -> None:
    """Punto de entrada CLI para ejecutar la Fase 2.5."""
    s = run_phase25_preparation()
    print("=== FASE 2.5: PREPARACIÓN DEL DATASET PARA MACHINE LEARNING ===")
    print(f"Semilla aleatoria fija:          {s.random_seed}")
    print(
        f"Materias totales:                {s.total_subjects} "
        f"(Train: {s.train_subjects_count}, Test: {s.test_subjects_count})"
    )
    print(
        f"Dataset PoC original:            {s.poc_total_rows} filas "
        f"(Pos: {s.poc_positives} [{s.poc_positives/s.poc_total_rows:.3%}], "
        f"Neg: {s.poc_negatives} [{s.poc_negatives/s.poc_total_rows:.3%}])"
    )
    print(
        f"Train Full (ml_train_full.csv):  {s.train_full_rows} filas "
        f"(Pos: {s.train_full_positives} [{s.train_full_positives/s.train_full_rows:.3%}], "
        f"Neg: {s.train_full_negatives} [{s.train_full_negatives/s.train_full_rows:.3%}])"
    )
    print(
        f"Train Bal (ml_train_balanced.csv): {s.train_balanced_rows} filas "
        f"(Pos: {s.train_balanced_positives} [{s.train_balanced_positives/s.train_balanced_rows:.2%}], "
        f"Neg: {s.train_balanced_negatives} [{s.train_balanced_negatives/s.train_balanced_rows:.2%}])"
    )
    print(
        f"Test Full (ml_test_full.csv):    {s.test_full_rows} filas "
        f"(Pos: {s.test_full_positives} [{s.test_full_positives/s.test_full_rows:.3%}], "
        f"Neg: {s.test_full_negatives} [{s.test_full_negatives/s.test_full_rows:.3%}])"
    )
    print(f"Pares duplicados detectados:     {s.duplicate_pairs_found}")
    print(f"Solapamiento de pares Train/Test:{s.train_test_pair_overlap}")
    print(f"Columnas exportadas ({len(ML_OUTPUT_COLUMNS)}):      {ML_OUTPUT_COLUMNS}")


if __name__ == "__main__":
    main()
