"""Fase 2: Construcción de los 319.600 pares posibles de materias y variables por cohorte.

Este módulo lee exclusivamente las tablas intermedias validadas en la Fase 1
(`data/interim/exams_clean.csv`, `students_clean.csv`, `enrolments_clean.csv`)
y genera `data/processed/subject_pairs_poc.csv` con las 319.600 combinaciones
posibles de 800 materias (`materia_id_1 < materia_id_2`).

Reglas metodológicas aplicadas estrictamente:
1. Separación total de cohortes sin data leakage:
   - Grupo de referencia histórico (`historico_1989_1993`, 4.810 alumnos): se usa
     exclusivamente para calcular `cant_estudiantes_compartidos_historico`,
     `tasa_compartidos_historico`, `jaccard_historico`, `cant_inscriptos_hist_1`,
     `cant_inscriptos_hist_2`, `cant_carreras_compartidas_hist` y `misma_carrera`.
   - Grupo actual (`actual_1994`, 3.086 alumnos): se usa exclusivamente para
     calcular las variables objetivo `cant_estudiantes_afectados`,
     `tasa_afectados_actual` y `hubo_conflicto_potencial` (además de los conteos
     informativos `cant_inscriptos_actual_1` y `cant_inscriptos_actual_2`).
2. Inclusión de los 319.600 pares posibles (incluidos aquellos con 0 estudiantes
   compartidos) para disponer del universo completo y de ejemplos negativos en
   clasificación sin sesgo de muestreo.
3. Distinción explícita entre conflicto potencial de demanda (`hubo_conflicto_potencial`)
   y conflicto efectivo de horario (no existe asignación horaria definitiva en `data`).
"""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


from src.common.schema import PROJECT_ROOT


DEFAULT_INTERIM_DIR = PROJECT_ROOT / "data" / "interim"
DEFAULT_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DEFAULT_OUTPUT_FILENAME = "subject_pairs_poc.csv"

HISTORICAL_GROUP = "historico_1989_1993"
CURRENT_GROUP = "actual_1994"


@dataclass
class ExamMetadata:
    """Metadatos limpios de un examen provenientes de `exams_clean.csv`."""

    exam_id: str
    exam_name: str
    duration: str
    duration_minutes: int
    department_code: str
    raw_level_char: str
    year_code: Optional[int]


@dataclass
class Phase2Summary:
    """Resumen estadístico de la construcción de los 319.600 pares en Fase 2."""

    total_exams: int = 0
    total_possible_pairs: int = 0
    historical_students_count: int = 0
    current_students_count: int = 0

    pairs_with_hist_shared_ge_1: int = 0
    pairs_with_hist_shared_ge_5: int = 0
    pairs_with_hist_shared_ge_10: int = 0
    max_hist_shared: int = 0

    pairs_with_curr_affected_ge_1: int = 0
    pairs_with_curr_affected_ge_5: int = 0
    pairs_with_curr_affected_ge_10: int = 0
    max_curr_affected: int = 0

    pairs_with_any_shared_ge_1: int = 0
    pairs_with_both_hist_and_curr_ge_1: int = 0
    pairs_hist_only_ge_1: int = 0
    pairs_curr_only_ge_1: int = 0
    pairs_zero_in_both: int = 0

    pairs_same_department: int = 0
    pairs_same_prefix: int = 0
    pairs_same_career_hist: int = 0
    pairs_year_confirmed: int = 0
    pairs_same_year: int = 0


def load_exams_clean(exams_csv: Path) -> Dict[str, ExamMetadata]:
    """Carga `exams_clean.csv` indexado por `exam_id`."""
    exams: Dict[str, ExamMetadata] = {}
    with exams_csv.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            yc_str = row["year_code"].strip()
            year_code = int(yc_str) if yc_str else None
            exams[row["exam_id"]] = ExamMetadata(
                exam_id=row["exam_id"],
                exam_name=row["exam_name"],
                duration=row["duration"],
                duration_minutes=int(row["duration_minutes"]),
                department_code=row["department_code"],
                raw_level_char=row["raw_level_char"],
                year_code=year_code,
            )
    return exams


def load_students_clean(students_csv: Path) -> Dict[str, Tuple[str, str]]:
    """Carga `students_clean.csv` devolviendo `student_id -> (course_code, cohort_group)`."""
    students: Dict[str, Tuple[str, str]] = {}
    with students_csv.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            students[row["student_id"]] = (row["course_code"], row["cohort_group"])
    return students


def build_subject_pairs(
    interim_dir: Path = DEFAULT_INTERIM_DIR,
    processed_dir: Path = DEFAULT_PROCESSED_DIR,
    output_filename: str = DEFAULT_OUTPUT_FILENAME,
) -> Phase2Summary:
    """Construye los 319.600 pares posibles con separación estricta de cohortes."""
    exams_map = load_exams_clean(interim_dir / "exams_clean.csv")
    students_map = load_students_clean(interim_dir / "students_clean.csv")

    hist_students_total = sum(
        1 for _, (_, grp) in students_map.items() if grp == HISTORICAL_GROUP
    )
    curr_students_total = sum(
        1 for _, (_, grp) in students_map.items() if grp == CURRENT_GROUP
    )

    # Inscripciones por estudiante separadas estrictamente por grupo de cohorte
    hist_student_exams: Dict[str, List[str]] = defaultdict(list)
    curr_student_exams: Dict[str, List[str]] = defaultdict(list)

    # Conteos de inscriptos por examen en cada grupo de cohorte
    hist_exam_enrol_count: Counter[str] = Counter()
    curr_exam_enrol_count: Counter[str] = Counter()

    # Conjunto de carreras (course_code) presentes en cada examen SOLO en el grupo histórico
    # (Evita cualquier filtración/leakage de la cohorte 1994 hacia features curriculares)
    hist_exam_courses: Dict[str, Set[str]] = defaultdict(set)

    enrolments_csv = interim_dir / "enrolments_clean.csv"
    with enrolments_csv.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            s_id = row["student_id"]
            e_id = row["exam_id"]
            course_code, cohort_group = students_map[s_id]

            if cohort_group == HISTORICAL_GROUP:
                hist_student_exams[s_id].append(e_id)
                hist_exam_enrol_count[e_id] += 1
                hist_exam_courses[e_id].add(course_code)
            elif cohort_group == CURRENT_GROUP:
                curr_student_exams[s_id].append(e_id)
                curr_exam_enrol_count[e_id] += 1

    # Conteo de intersecciones en el grupo histórico (1989-1993)
    hist_pair_counts: Counter[Tuple[str, str]] = Counter()
    for e_list in hist_student_exams.values():
        sorted_unique = sorted(set(e_list))
        for e1, e2 in combinations(sorted_unique, 2):
            hist_pair_counts[(e1, e2)] += 1

    # Conteo de intersecciones en el grupo actual (1994)
    curr_pair_counts: Counter[Tuple[str, str]] = Counter()
    for e_list in curr_student_exams.values():
        sorted_unique = sorted(set(e_list))
        for e1, e2 in combinations(sorted_unique, 2):
            curr_pair_counts[(e1, e2)] += 1

    try:
        processed_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass

    output_path = processed_dir / output_filename
    sorted_exam_ids = sorted(exams_map.keys())
    summary = Phase2Summary(
        total_exams=len(sorted_exam_ids),
        total_possible_pairs=(len(sorted_exam_ids) * (len(sorted_exam_ids) - 1)) // 2,
        historical_students_count=hist_students_total,
        current_students_count=curr_students_total,
    )

    fieldnames = [
        "materia_id_1",
        "materia_id_2",
        "materia_nombre_1",
        "materia_nombre_2",
        "departamento_1",
        "departamento_2",
        "mismo_departamento",
        "mismo_prefijo_curricular",
        "anio_1",
        "anio_2",
        "anio_confirmado",
        "mismo_anio",
        "misma_carrera",
        "cant_carreras_compartidas_hist",
        "cant_inscriptos_hist_1",
        "cant_inscriptos_hist_2",
        "cant_estudiantes_compartidos_historico",
        "tasa_compartidos_historico",
        "jaccard_historico",
        "cant_inscriptos_actual_1",
        "cant_inscriptos_actual_2",
        "cant_estudiantes_afectados",
        "tasa_afectados_actual",
        "hubo_conflicto_potencial",
    ]

    with output_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(fieldnames)

        for e1_id, e2_id in combinations(sorted_exam_ids, 2):
            ex1 = exams_map[e1_id]
            ex2 = exams_map[e2_id]

            mismo_depto = ex1.department_code == ex2.department_code
            mismo_prefijo = e1_id[:2] == e2_id[:2]

            anio_confirmado = (ex1.year_code is not None) and (ex2.year_code is not None)
            mismo_anio = anio_confirmado and (ex1.year_code == ex2.year_code)

            # Carreras compartidas calculadas EXCLUSIVAMENTE con cohortes 1989-1993
            courses_1 = hist_exam_courses.get(e1_id, set())
            courses_2 = hist_exam_courses.get(e2_id, set())
            shared_courses_hist = len(courses_1 & courses_2)
            misma_carrera = shared_courses_hist > 0

            n_hist_1 = hist_exam_enrol_count[e1_id]
            n_hist_2 = hist_exam_enrol_count[e2_id]
            shared_hist = hist_pair_counts.get((e1_id, e2_id), 0)
            tasa_hist = (
                round(shared_hist / hist_students_total, 6)
                if hist_students_total > 0
                else 0.0
            )
            union_hist = n_hist_1 + n_hist_2 - shared_hist
            jaccard_hist = (
                round(shared_hist / union_hist, 6) if union_hist > 0 else 0.0
            )

            n_curr_1 = curr_exam_enrol_count[e1_id]
            n_curr_2 = curr_exam_enrol_count[e2_id]
            affected_curr = curr_pair_counts.get((e1_id, e2_id), 0)
            tasa_curr = (
                round(affected_curr / curr_students_total, 6)
                if curr_students_total > 0
                else 0.0
            )
            hubo_conflicto_pot = affected_curr > 0

            # Actualizar estadísticas del resumen
            if shared_hist >= 1:
                summary.pairs_with_hist_shared_ge_1 += 1
            if shared_hist >= 5:
                summary.pairs_with_hist_shared_ge_5 += 1
            if shared_hist >= 10:
                summary.pairs_with_hist_shared_ge_10 += 1
            if shared_hist > summary.max_hist_shared:
                summary.max_hist_shared = shared_hist

            if affected_curr >= 1:
                summary.pairs_with_curr_affected_ge_1 += 1
            if affected_curr >= 5:
                summary.pairs_with_curr_affected_ge_5 += 1
            if affected_curr >= 10:
                summary.pairs_with_curr_affected_ge_10 += 1
            if affected_curr > summary.max_curr_affected:
                summary.max_curr_affected = affected_curr

            if shared_hist >= 1 or affected_curr >= 1:
                summary.pairs_with_any_shared_ge_1 += 1
                if shared_hist >= 1 and affected_curr >= 1:
                    summary.pairs_with_both_hist_and_curr_ge_1 += 1
                elif shared_hist >= 1:
                    summary.pairs_hist_only_ge_1 += 1
                else:
                    summary.pairs_curr_only_ge_1 += 1
            else:
                summary.pairs_zero_in_both += 1

            if mismo_depto:
                summary.pairs_same_department += 1
            if mismo_prefijo:
                summary.pairs_same_prefix += 1
            if misma_carrera:
                summary.pairs_same_career_hist += 1
            if anio_confirmado:
                summary.pairs_year_confirmed += 1
            if mismo_anio:
                summary.pairs_same_year += 1

            writer.writerow(
                [
                    e1_id,
                    e2_id,
                    ex1.exam_name,
                    ex2.exam_name,
                    ex1.department_code,
                    ex2.department_code,
                    mismo_depto,
                    mismo_prefijo,
                    "" if ex1.year_code is None else ex1.year_code,
                    "" if ex2.year_code is None else ex2.year_code,
                    anio_confirmado,
                    mismo_anio,
                    misma_carrera,
                    shared_courses_hist,
                    n_hist_1,
                    n_hist_2,
                    shared_hist,
                    f"{tasa_hist:.6f}",
                    f"{jaccard_hist:.6f}",
                    n_curr_1,
                    n_curr_2,
                    affected_curr,
                    f"{tasa_curr:.6f}",
                    hubo_conflicto_pot,
                ]
            )

    return summary


def main() -> None:
    """Punto de entrada CLI para ejecutar la Fase 2 y mostrar estadísticas."""
    summary = build_subject_pairs()
    print("=== FASE 2: CONSTRUCCIÓN DE PARES DE MATERIAS (319.600 PARES) ===")
    print(f"Total de exámenes:                           {summary.total_exams}")
    print(f"Total de pares generados:                    {summary.total_possible_pairs}")
    print(f"Estudiantes grupo histórico (1989-1993):     {summary.historical_students_count}")
    print(f"Estudiantes grupo actual (1994):             {summary.current_students_count}")
    print("--- Grupo Histórico (Cohortes 1989-1993) ---")
    print(f"  Pares con >= 1 alumno compartido:          {summary.pairs_with_hist_shared_ge_1}")
    print(f"  Pares con >= 5 alumnos compartidos:        {summary.pairs_with_hist_shared_ge_5}")
    print(f"  Pares con >= 10 alumnos compartidos:       {summary.pairs_with_hist_shared_ge_10}")
    print(f"  Máximo alumnos compartidos (histórico):    {summary.max_hist_shared}")
    print("--- Grupo Actual (Cohorte 1994 - Objetivo) ---")
    print(f"  Pares con >= 1 alumno afectado (conflicto):{summary.pairs_with_curr_affected_ge_1}")
    print(f"  Pares con >= 5 alumnos afectados:          {summary.pairs_with_curr_affected_ge_5}")
    print(f"  Pares con >= 10 alumnos afectados:         {summary.pairs_with_curr_affected_ge_10}")
    print(f"  Máximo alumnos afectados (1994):           {summary.max_curr_affected}")
    print("--- Cruce Histórico vs. Actual ---")
    print(f"  Pares con >= 1 en alguno de los grupos:    {summary.pairs_with_any_shared_ge_1}")
    print(f"  Pares con >= 1 en AMBOS grupos:            {summary.pairs_with_both_hist_and_curr_ge_1}")
    print(f"  Pares con >= 1 SOLO en histórico (89-93):  {summary.pairs_hist_only_ge_1}")
    print(f"  Pares con >= 1 SOLO en actual (1994):      {summary.pairs_curr_only_ge_1}")
    print(f"  Pares con 0 en ambos grupos:               {summary.pairs_zero_in_both}")
    print("--- Atributos Académicos ---")
    print(f"  Pares con mismo_departamento:              {summary.pairs_same_department}")
    print(f"  Pares con mismo_prefijo_curricular:        {summary.pairs_same_prefix}")
    print(f"  Pares con misma_carrera (hist 89-93):      {summary.pairs_same_career_hist}")
    print(f"  Pares con anio_confirmado (ambos 1..4):    {summary.pairs_year_confirmed}")
    print(f"  Pares con mismo_anio:                      {summary.pairs_same_year}")


if __name__ == "__main__":
    main()
