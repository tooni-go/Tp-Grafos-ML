"""Fase 1: Lectura, parsing y validación inicial del dataset University of Nottingham.

Este módulo lee los archivos originales de texto plano de ancho fijo ubicados en
`Nottingham Dataset/` (`README`, `exams`, `students`, `enrolements`, `data`),
valida su formato e integridad referencial sin modificar los datos crudos y
genera las tres tablas intermedias en `data/interim/`:
- `exams_clean.csv`
- `students_clean.csv`
- `enrolments_clean.csv`
"""

from __future__ import annotations

import csv
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


from src.common.schema import PROJECT_ROOT


DEFAULT_RAW_DIR = PROJECT_ROOT / "Nottingham Dataset"
DEFAULT_INTERIM_DIR = PROJECT_ROOT / "data" / "interim"

VALID_NUMERIC_YEAR_CODES = {"1", "2", "3", "4"}
HISTORICAL_COHORT_YEARS = {1989, 1990, 1991, 1992, 1993}
CURRENT_COHORT_YEAR = 1994


@dataclass
class ExamRecord:
    """Representa un examen leído del archivo original `exams`."""

    exam_id: str
    exam_name: str
    duration: str
    duration_minutes: int
    department_code: str
    raw_level_char: str
    year_code: Optional[int]


@dataclass
class StudentRecord:
    """Representa un estudiante leído del archivo original `students`."""

    student_id: str
    course_code: str
    cohort_year: Optional[int]
    cohort_group: str


@dataclass
class EnrolmentRecord:
    """Representa una inscripción individual leída del archivo `enrolements`."""

    student_id: str
    exam_id: str


@dataclass
class DataConstraintsSummary:
    """Resumen estructural del archivo `data` de Nottingham."""

    date_range: str = ""
    times_lines: List[str] = field(default_factory=list)
    rooms_count: int = 0
    room_assignments_count: int = 0
    coincidences_count: int = 0
    earliness_priority_count: int = 0
    misc_lines_count: int = 0
    has_final_schedule_assignment: bool = False


@dataclass
class Phase1ValidationSummary:
    """Resultado de las validaciones de lectura e integridad referencial de Fase 1."""

    exams_read: int = 0
    exams_malformed: List[Tuple[int, str]] = field(default_factory=list)
    exams_duplicates: List[str] = field(default_factory=list)
    exams_empty_ids: int = 0
    exam_level_char_counts: Dict[str, int] = field(default_factory=dict)
    exams_with_numeric_year: int = 0
    exams_with_non_numeric_year: int = 0

    students_read: int = 0
    students_malformed: List[Tuple[int, str]] = field(default_factory=list)
    students_duplicates: List[str] = field(default_factory=list)
    students_empty_ids: int = 0
    students_by_cohort_year: Dict[int, int] = field(default_factory=dict)
    students_by_cohort_group: Dict[str, int] = field(default_factory=dict)

    enrolments_read: int = 0
    enrolments_malformed: List[Tuple[int, str]] = field(default_factory=list)
    enrolments_duplicates: List[Tuple[str, str]] = field(default_factory=list)
    enrolments_empty_ids: int = 0
    enrolments_missing_student: List[Tuple[str, str]] = field(default_factory=list)
    enrolments_missing_exam: List[Tuple[str, str]] = field(default_factory=list)
    enrolments_by_cohort_year: Dict[int, int] = field(default_factory=dict)
    enrolments_by_cohort_group: Dict[str, int] = field(default_factory=dict)
    exams_without_enrolments: int = 0

    data_summary: DataConstraintsSummary = field(default_factory=DataConstraintsSummary)


def parse_duration_to_minutes(duration_str: str) -> int:
    """Convierte una duración en formato 'H:MM' a minutos enteros."""
    parts = duration_str.strip().split(":")
    if len(parts) != 2 or not parts[0].isdigit() or not parts[1].isdigit():
        raise ValueError(f"Formato de duración inválido: {duration_str!r}")
    return int(parts[0]) * 60 + int(parts[1])


def parse_exams_file(
    exams_path: Path,
) -> Tuple[List[ExamRecord], List[Tuple[int, str]], List[str], int]:
    """Parsea el archivo `exams` respetando los anchos fijos documentados.

    Formato verificado en `README` y en las 800 líneas de 57 caracteres:
    - Posiciones 0..8 (8 chars): `exam code`
    - Posición 8 (1 char): espacio separador
    - Posiciones 9..49 (40 chars): `description`
    - Posición 49 (1 char): espacio separador
    - Posiciones 50..54 (4 chars): `duration` (`H:MM`)
    - Posición 54 (1 char): espacio separador
    - Posiciones 55..57 (2 chars): `department code`
    """
    records: List[ExamRecord] = []
    malformed: List[Tuple[int, str]] = []
    duplicates: List[str] = []
    empty_ids = 0
    seen_ids: Set[str] = set()

    with exams_path.open("r", encoding="utf-8") as fh:
        for line_num, raw_line in enumerate(fh, start=1):
            line = raw_line.rstrip("\r\n")
            if not line:
                continue

            if len(line) < 56 or line[8] != " " or line[49] != " " or line[54] != " ":
                malformed.append((line_num, line))
                continue

            exam_id = line[0:8].strip()
            exam_name = line[9:49].strip()
            duration = line[50:54].strip()
            department_code = line[55:57].strip()

            if not exam_id or not exam_name or not duration or not department_code:
                empty_ids += 1
                malformed.append((line_num, line))
                continue

            try:
                duration_minutes = parse_duration_to_minutes(duration)
            except ValueError:
                malformed.append((line_num, line))
                continue

            if exam_id in seen_ids:
                duplicates.append(exam_id)
                continue
            seen_ids.add(exam_id)

            raw_level_char = exam_id[2] if len(exam_id) >= 3 else ""
            year_code = (
                int(raw_level_char)
                if raw_level_char in VALID_NUMERIC_YEAR_CODES
                else None
            )

            records.append(
                ExamRecord(
                    exam_id=exam_id,
                    exam_name=exam_name,
                    duration=duration,
                    duration_minutes=duration_minutes,
                    department_code=department_code,
                    raw_level_char=raw_level_char,
                    year_code=year_code,
                )
            )

    return records, malformed, duplicates, empty_ids


def parse_students_file(
    students_path: Path,
) -> Tuple[List[StudentRecord], List[Tuple[int, str]], List[str], int]:
    """Parsea el archivo `students` respetando los anchos fijos documentados.

    Formato verificado en `README` y en las 7.896 líneas de 15 caracteres:
    - Posiciones 0..10 (10 chars): `student code` (ej. `A890186790`)
    - Posición 10 (1 char): espacio separador
    - Posiciones 11..15 (4 chars): `course code` (ej. `R100`)

    Además extrae `cohort_year` de `student_id[1:3]` (`89`..`94` -> `1989`..`1994`)
    y clasifica el grupo en `historico_1989_1993` o `actual_1994`.
    """
    records: List[StudentRecord] = []
    malformed: List[Tuple[int, str]] = []
    duplicates: List[str] = []
    empty_ids = 0
    seen_ids: Set[str] = set()

    with students_path.open("r", encoding="utf-8") as fh:
        for line_num, raw_line in enumerate(fh, start=1):
            line = raw_line.rstrip("\r\n")
            if not line:
                continue

            if len(line) < 12 or line[10] != " ":
                malformed.append((line_num, line))
                continue

            student_id = line[0:10].strip()
            course_code = line[11:15].strip()

            if not student_id or not course_code:
                empty_ids += 1
                malformed.append((line_num, line))
                continue

            if student_id in seen_ids:
                duplicates.append(student_id)
                continue
            seen_ids.add(student_id)

            cohort_digits = student_id[1:3] if len(student_id) >= 3 else ""
            if cohort_digits.isdigit():
                cohort_year: Optional[int] = 1900 + int(cohort_digits)
            else:
                cohort_year = None

            if cohort_year in HISTORICAL_COHORT_YEARS:
                cohort_group = "historico_1989_1993"
            elif cohort_year == CURRENT_COHORT_YEAR:
                cohort_group = "actual_1994"
            else:
                cohort_group = "desconocido"

            records.append(
                StudentRecord(
                    student_id=student_id,
                    course_code=course_code,
                    cohort_year=cohort_year,
                    cohort_group=cohort_group,
                )
            )

    return records, malformed, duplicates, empty_ids


def parse_enrolments_file(
    enrolments_path: Path,
) -> Tuple[List[EnrolmentRecord], List[Tuple[int, str]], List[Tuple[str, str]], int]:
    """Parsea el archivo `enrolements` respetando los anchos fijos documentados.

    Formato verificado en `README` y en las 33.997 líneas de 19 caracteres:
    - Posiciones 0..10 (10 chars): `student code`
    - Posición 10 (1 char): espacio separador
    - Posiciones 11..19 (8 chars): `exam code`
    """
    records: List[EnrolmentRecord] = []
    malformed: List[Tuple[int, str]] = []
    duplicates: List[Tuple[str, str]] = []
    empty_ids = 0
    seen_pairs: Set[Tuple[str, str]] = set()

    with enrolments_path.open("r", encoding="utf-8") as fh:
        for line_num, raw_line in enumerate(fh, start=1):
            line = raw_line.rstrip("\r\n")
            if not line:
                continue

            if len(line) < 12 or line[10] != " ":
                malformed.append((line_num, line))
                continue

            student_id = line[0:10].strip()
            exam_id = line[11:19].strip()

            if not student_id or not exam_id:
                empty_ids += 1
                malformed.append((line_num, line))
                continue

            pair = (student_id, exam_id)
            if pair in seen_pairs:
                duplicates.append(pair)
                continue
            seen_pairs.add(pair)

            records.append(EnrolmentRecord(student_id=student_id, exam_id=exam_id))

    return records, malformed, duplicates, empty_ids


def inspect_data_file(data_path: Path) -> DataConstraintsSummary:
    """Inspecciona y resume las secciones del archivo `data` de Nottingham."""
    summary = DataConstraintsSummary()
    current_section: Optional[str] = None
    known_headers = {
        "DATES",
        "TIMES",
        "ROOMS",
        "ROOM ASSIGNMENTS",
        "COINCIDENCES",
        "EARLINESS PRIORITY (1-100)",
        "MISC",
    }

    with data_path.open("r", encoding="utf-8") as fh:
        lines = [l.rstrip("\r\n") for l in fh]

    for idx, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or set(stripped) == {"-"}:
            continue
        if stripped in known_headers:
            current_section = stripped
            continue

        if current_section == "DATES":
            summary.date_range = stripped
        elif current_section == "TIMES":
            summary.times_lines.append(stripped)
        elif current_section == "ROOMS":
            if not stripped.startswith("|"):
                summary.rooms_count += 1
        elif current_section == "ROOM ASSIGNMENTS":
            if not stripped.startswith("then"):
                summary.room_assignments_count += 1
        elif current_section == "COINCIDENCES":
            summary.coincidences_count += 1
        elif current_section == "EARLINESS PRIORITY (1-100)":
            summary.earliness_priority_count += 1
        elif current_section == "MISC":
            summary.misc_lines_count += 1

    summary.has_final_schedule_assignment = False
    return summary


def write_interim_csvs(
    interim_dir: Path,
    exams: List[ExamRecord],
    students: List[StudentRecord],
    enrolments: List[EnrolmentRecord],
) -> None:
    """Escribe los tres archivos CSV intermedios limpios en `data/interim/`."""
    try:
        interim_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass

    exams_csv = interim_dir / "exams_clean.csv"
    with exams_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "exam_id",
                "exam_name",
                "duration",
                "duration_minutes",
                "department_code",
                "raw_level_char",
                "year_code",
            ]
        )
        for ex in exams:
            writer.writerow(
                [
                    ex.exam_id,
                    ex.exam_name,
                    ex.duration,
                    ex.duration_minutes,
                    ex.department_code,
                    ex.raw_level_char,
                    "" if ex.year_code is None else ex.year_code,
                ]
            )

    students_csv = interim_dir / "students_clean.csv"
    with students_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["student_id", "course_code", "cohort_year", "cohort_group"])
        for st in students:
            writer.writerow(
                [
                    st.student_id,
                    st.course_code,
                    "" if st.cohort_year is None else st.cohort_year,
                    st.cohort_group,
                ]
            )

    enrolments_csv = interim_dir / "enrolments_clean.csv"
    with enrolments_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["student_id", "exam_id"])
        for en in enrolments:
            writer.writerow([en.student_id, en.exam_id])


def run_phase1_parsing(
    raw_dir: Path = DEFAULT_RAW_DIR,
    interim_dir: Path = DEFAULT_INTERIM_DIR,
) -> Phase1ValidationSummary:
    """Ejecuta el parsing de Fase 1, valida integridad referencial y guarda CSVs."""
    exams_path = raw_dir / "exams"
    students_path = raw_dir / "students"
    enrolments_path = raw_dir / "enrolements"
    data_path = raw_dir / "data"

    exams, ex_malformed, ex_dups, ex_empty = parse_exams_file(exams_path)
    students, st_malformed, st_dups, st_empty = parse_students_file(students_path)
    enrolments, en_malformed, en_dups, en_empty = parse_enrolments_file(enrolments_path)
    data_summary = inspect_data_file(data_path)

    exam_ids = {e.exam_id for e in exams}
    student_map = {s.student_id: s for s in students}

    missing_student: List[Tuple[str, str]] = []
    missing_exam: List[Tuple[str, str]] = []
    enrol_by_year: Counter[int] = Counter()
    enrol_by_group: Counter[str] = Counter()
    exams_with_enrolments: Set[str] = set()

    for en in enrolments:
        st = student_map.get(en.student_id)
        if st is None:
            missing_student.append((en.student_id, en.exam_id))
        else:
            if st.cohort_year is not None:
                enrol_by_year[st.cohort_year] += 1
            enrol_by_group[st.cohort_group] += 1

        if en.exam_id not in exam_ids:
            missing_exam.append((en.student_id, en.exam_id))
        else:
            exams_with_enrolments.add(en.exam_id)

    write_interim_csvs(interim_dir, exams, students, enrolments)

    level_counts = Counter(e.raw_level_char for e in exams)
    numeric_years = sum(1 for e in exams if e.year_code is not None)
    non_numeric_years = len(exams) - numeric_years

    st_by_year = Counter(s.cohort_year for s in students if s.cohort_year is not None)
    st_by_group = Counter(s.cohort_group for s in students)

    return Phase1ValidationSummary(
        exams_read=len(exams),
        exams_malformed=ex_malformed,
        exams_duplicates=ex_dups,
        exams_empty_ids=ex_empty,
        exam_level_char_counts=dict(sorted(level_counts.items())),
        exams_with_numeric_year=numeric_years,
        exams_with_non_numeric_year=non_numeric_years,
        students_read=len(students),
        students_malformed=st_malformed,
        students_duplicates=st_dups,
        students_empty_ids=st_empty,
        students_by_cohort_year=dict(sorted(st_by_year.items())),
        students_by_cohort_group=dict(sorted(st_by_group.items())),
        enrolments_read=len(enrolments),
        enrolments_malformed=en_malformed,
        enrolments_duplicates=en_dups,
        enrolments_empty_ids=en_empty,
        enrolments_missing_student=missing_student,
        enrolments_missing_exam=missing_exam,
        enrolments_by_cohort_year=dict(sorted(enrol_by_year.items())),
        enrolments_by_cohort_group=dict(sorted(enrol_by_group.items())),
        exams_without_enrolments=len(exam_ids - exams_with_enrolments),
        data_summary=data_summary,
    )


def main() -> None:
    """Punto de entrada CLI para ejecutar la Fase 1 y mostrar el reporte en consola."""
    summary = run_phase1_parsing()
    print("=== FASE 1: PARSING Y VALIDACIÓN INICIAL (NOTTINGHAM DATASET) ===")
    print(f"Exámenes leídos:            {summary.exams_read}")
    print(f"  - Filas mal formadas:     {len(summary.exams_malformed)}")
    print(f"  - IDs duplicados:         {len(summary.exams_duplicates)}")
    print(f"  - Con year_code numérico: {summary.exams_with_numeric_year}")
    print(f"  - Con código especial:    {summary.exams_with_non_numeric_year}")
    print(f"  - 3.er carácter (nivel):  {summary.exam_level_char_counts}")
    print(f"Estudiantes leídos:         {summary.students_read}")
    print(f"  - Filas mal formadas:     {len(summary.students_malformed)}")
    print(f"  - IDs duplicados:         {len(summary.students_duplicates)}")
    print(f"  - Por cohorte (año):      {summary.students_by_cohort_year}")
    print(f"  - Por grupo de cohorte:   {summary.students_by_cohort_group}")
    print(f"Inscripciones leídas:       {summary.enrolments_read}")
    print(f"  - Filas mal formadas:     {len(summary.enrolments_malformed)}")
    print(f"  - Duplicadas:             {len(summary.enrolments_duplicates)}")
    print(f"  - Estudiante inexistente: {len(summary.enrolments_missing_student)}")
    print(f"  - Examen inexistente:     {len(summary.enrolments_missing_exam)}")
    print(f"  - Exámenes sin alumnos:   {summary.exams_without_enrolments}")
    print(f"  - Inscripciones por año:  {summary.enrolments_by_cohort_year}")
    print(f"  - Inscripciones por grp:  {summary.enrolments_by_cohort_group}")
    print(f"Archivo data (fechas):      {summary.data_summary.date_range}")
    print(
        f"Archivo data (tiene cronograma resuelto): "
        f"{summary.data_summary.has_final_schedule_assignment}"
    )


if __name__ == "__main__":
    main()
