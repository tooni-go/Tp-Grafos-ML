# Plan de prueba de concepto y construcción del dataset

**Proyecto:** Planificación de mesas de examen universitarias  
**Fuente principal:** University of Nottingham Exam Timetabling Dataset (`Nottingham Dataset/`)  
**Objetivo de este documento:** definir una prueba de concepto (PoC) reproducible y organizada en 4 fases (commits incrementales) que transforme los archivos originales en un dataset de pares de materias, valide los resultados y deje preparada la base para Machine Learning y teoría de grafos.

---

## 1. Objetivo, alcance y organización por fases

La prueba de concepto debe demostrar que podemos pasar de los archivos crudos de Nottingham a una tabla estructurada donde cada fila represente un par de materias y sus características, incorporando una **aproximación histórica por partición de cohortes**.

La PoC **no consiste todavía en entrenar modelos ni en construir la interfaz**. Primero debe resolver y validar la extracción, transformación y construcción de variables en **4 Fases incrementales**:

- **Fase 1 (Etapa 1):** Inspección, parsing estricto de los archivos originales y validación de integridad de las tablas intermedias (`data/interim/`). **En la Fase 1 no se construyen todavía los pares ni las variables por cohortes.**
- **Fase 2 (Etapas 2, 3, 4, 5 y 6):** Construcción de pares de materias, atributos académicos y variables basadas en la partición por cohortes (`1989–1993` como referencia histórica vs. `1994` como grupo actual), generando `data/processed/subject_pairs_poc.csv`.
- **Fase 3 (Etapa 7):** Auditoría integral de consistencia, conteos por cohorte, evaluación de tasas/proporciones y generación de `reports/poc_validation_report.md`.
- **Fase 4 (Etapa 7 y cierre de PoC):** Suite de pruebas automatizadas (`tests/`) con casos sintéticos controlados y datos reales.

### Resultado esperado al finalizar la PoC

1. Un proceso ETL reproducible que lea los archivos originales sin alterarlos.
2. Un dataset procesado de pares de materias con partición estricta entre cohortes de referencia (`1989–1993`) y cohorte actual (`1994`).
3. Un informe de validación que permita comprobar que los conteos, proporciones y relaciones son coherentes.
4. Documentación metodológica explícita sobre la aproximación histórica por cohortes, sus alcances y limitaciones.

> **Importante:** No inventar ni simular datos artificiales durante la PoC. No mezclar estudiantes del grupo histórico (`1989–1993`) con los del grupo actual (`1994`) al calcular las variables.

---

## 2. Fuente de datos y archivos

El análisis realizado sobre los archivos descargados en `Nottingham Dataset/` identificó esta estructura:

| Archivo | Contenido | Formato / Anchos de campo | Uso previsto |
|---|---|---|---|
| `README` | Documentación original del dataset | Texto plano (26 líneas) | Referencia oficial de formato |
| `exams` | Código, nombre, duración y departamento de cada examen | Fijo (57 caracteres): `exam_code` (8), espacio (1), `description` (40), espacio (1), `duration` (4), espacio (1), `department_code` (2) | Catálogo de materias y metadatos |
| `students` | Código de estudiante y código de carrera (`course code`) | Fijo (15 caracteres): `student_code` (10), espacio (1), `course_code` (4) | Identificación de estudiantes, cohorte de ingreso y carrera |
| `enrolements` | Relación estudiante–examen | Fijo (19 caracteres): `student_code` (10), espacio (1), `exam_code` (8) | Inscripciones individuales |
| `data` | Fechas, turnos, aulas y restricciones | Semi-estructurado (182 líneas) | Contexto de planificación (32 turnos posibles y restricciones) |

El conjunto contiene **800 exámenes**, **7.896 estudiantes** y **33.997 inscripciones** correspondientes al primer semestre de 1994–95.

### Reglas para los datos originales

- No modificar los archivos descargados en `Nottingham Dataset/`.
- No sobrescribirlos con versiones transformadas.
- Mantener una copia intacta de los datos fuente.
- Documentar cualquier limpieza, normalización o exclusión.
- No asumir que dos códigos representan la misma entidad sin evidencia en los archivos.

---

## 3. Estructura propuesta del proyecto

Organizar el trabajo para separar datos originales, documentación, código y resultados:

```text
Tp-Grafos-ML/
├── Nottingham Dataset/          # Archivos crudos originales (intactos)
│   ├── README
│   ├── exams
│   ├── students
│   ├── enrolements
│   └── data
├── docs/
│   └── plan_poc_dataset_nottingham.md
├── data/
│   ├── interim/
│   │   ├── exams_clean.csv
│   │   ├── students_clean.csv
│   │   └── enrolments_clean.csv
│   └── processed/
│       ├── subject_pairs_poc.csv
│       └── subject_pairs_dataset.csv
├── src/
│   ├── parsing/
│   │   └── parse_nott.py
│   ├── transformation/
│   │   └── build_subject_pairs.py
│   ├── validation/
│   │   └── validate_dataset.py
│   └── main.py
├── reports/
│   └── poc_validation_report.md
├── tests/
│   ├── test_parsing.py
│   ├── test_pairs.py
│   └── test_validation.py
└── README.md
```

---

## 4. Fase 1 (Etapa 1) — Inspección, parsing y validación de archivos intermedios

### Objetivo

Leer correctamente los archivos originales de `Nottingham Dataset/`, validar su integridad referencial y convertirlos a estructuras tabulares intermedias en `data/interim/`. **Esta Fase 1 es exclusivamente de parsing y validación de tablas intermedias; la construcción de pares y variables por cohortes se realiza recién en la Fase 2.**

### Tareas

1. Leer el `README` del dataset y confirmar los anchos y posiciones exactas de cada campo.
2. Implementar un parser para `exams`.
3. Implementar un parser para `students`, extrayendo además el año de cohorte de ingreso codificado en los caracteres 2 y 3 del `student_id` (`1989` a `1994`) para dejarlo validado y documentado de cara a la Fase 2.
4. Implementar un parser para `enrolements` (manteniendo la lectura del archivo original `enrolements` y generando `enrolments_clean.csv`).
5. Inspeccionar y documentar el archivo `data` (fechas, 32 turnos posibles, aulas y restricciones).
6. Normalizar espacios y tipos sin alterar los identificadores originales.

### Tablas intermedias esperadas (`data/interim/`)

**`exams_clean.csv`**

| exam_id | exam_name | duration | duration_minutes | department_code | raw_level_char | year_code |
|---|---|---|---|---|---|---|

*Nota:* `raw_level_char` conserva intacto el tercer carácter del código de examen (`1`, `2`, `3`, `4`, `A`, `B`, `C`, `P`, `0`). `year_code` solo asigna un entero (`1` a `4`) cuando el carácter es un dígito numérico explícito, dejando vacío (`null`/`NA`) los códigos alfabéticos o especiales para no inventar años no confirmados en la documentación original.

**`students_clean.csv`**

| student_id | course_code | cohort_year | cohort_group |
|---|---|---|---|

*Nota:* `cohort_year` documenta el año de ingreso identificado en `student_id[1:3]` (`1989` a `1994`) y `cohort_group` identifica la pertenencia a `historico_1989_1993` o `actual_1994` para su uso posterior en la Fase 2.

**`enrolments_clean.csv`**

| student_id | exam_id |
|---|---|

### Validaciones iniciales (Fase 1)

- Los identificadores no deben quedar vacíos.
- Comprobar duplicados en cada archivo (`exams`, `students`, `enrolements`).
- Toda inscripción debe referenciar un estudiante existente en `students`.
- Toda inscripción debe referenciar un examen existente en `exams`.
- Contabilizar registros leídos, descartados y duplicados.
- Documentar la cantidad exacta de estudiantes e inscripciones por cada cohorte (`1989` a `1994`).
- Informar filas que no cumplen el formato esperado; no descartarlas silenciosamente.

**Criterio de aceptación de la Fase 1:** los tres archivos intermedios se generan correctamente en `data/interim/` y las relaciones entre estudiantes, exámenes e inscripciones pasan el 100% de las validaciones de integridad referencial.

---

## 5. Fase 2 (Etapas 2, 3, 4, 5 y 6) — Construcción de pares, atributos académicos y variables por cohortes

### 5.1. Construcción de pares de materias (Etapa 2)

Una vez validados los archivos intermedios en la Fase 1, la Fase 2 genera una fila por cada par no ordenado de materias `(materia_id_1, materia_id_2)` con `materia_id_1 < materia_id_2`, garantizando:

- Cada par aparece una sola vez (`min(id1, id2)`, `max(id1, id2)`).
- No se generan pares de una materia consigo misma `(A, A)`.
- Se calculan de forma estrictamente separada las intersecciones de estudiantes del **grupo de referencia histórico (cohortes 1989–1993)** y del **grupo actual (cohorte 1994)**, sin mezclar estudiantes de ambos grupos.

#### Inclusión de los 319.600 pares posibles (incluidos pares con cero estudiantes compartidos)

Con 800 materias existen exactamente $\binom{800}{2} = 319.600$ pares posibles.
- La salida `data/processed/subject_pairs_poc.csv` **contiene los 319.600 pares posibles**, incluidos los pares sin estudiantes compartidos (`0` en ambos grupos), por tres razones fundamentales:
  1. Para entrenar modelos de clasificación (`hubo_conflicto_potencial`), es indispensable contar con **ejemplos negativos** (`cant_estudiantes_afectados == 0` en la cohorte 1994).
  2. Existen pares con actividad en el grupo histórico (`1989–1993`) pero `0` en `1994` (5.645 pares) y viceversa (3.481 pares), además de 987 pares activos en ambos y 309.487 con `0` en ambos. Conservar los 319.600 pares evita cualquier sesgo de selección en la tabla base.
  3. Cualquier submuestreo o balanceo de clases negativo para el entrenamiento posterior de ML puede aplicarse de forma parametrizada y reproducible sobre este universo completo.

### 5.2. Incorporación de atributos académicos (Etapa 3)

| Variable | Cómo obtenerla | Observaciones |
|---|---|---|
| `materia_id_1`, `materia_id_2` | `exams_clean.csv` | Códigos originales ordenados (`id_1 < id_2`) |
| `materia_nombre_1`, `materia_nombre_2` | `exams_clean.csv` | Nombres descriptivos para visualización |
| `mismo_departamento` | `exams_clean.csv` | Booleano: `department_code_1 == department_code_2` |
| `misma_carrera` | `students_clean.csv` + `enrolments_clean.csv` | Regla explícita y documentada (ver precaución abajo) |
| `mismo_anio` | `exams_clean.csv` (`year_code`) | Booleano cuando ambos exámenes tienen año numérico confirmado (`1..4`); documentar códigos especiales (`A, B, C, P, 0`) |
| `cant_inscriptos_hist_1`, `cant_inscriptos_hist_2` | `enrolments_clean.csv` (cohortes `1989–1993`) | Tamaño de cada materia en el grupo de referencia histórico |
| `cant_inscriptos_actual_1`, `cant_inscriptos_actual_2` | `enrolments_clean.csv` (cohorte `1994`) | Tamaño de cada materia en la cohorte actual (o tamaños totales sin leakage) |
| `cant_carreras_compartidas_hist` | `students_clean.csv` (`1989–1993`) | Cantidad de `course_code` con alumnos en ambas materias en el grupo histórico |

#### Precaución con `misma_carrera` y `mismo_departamento`
- No asumir que `department_code` equivale a carrera (`course_code`).
- Guardar `mismo_departamento` como variable separada (`department_code_1 == department_code_2`).
- Definir `misma_carrera` mediante reglas transparentes y sin *data leakage* (por ejemplo, coincidencia de prefijo curricular `exam_id[:2]` o existencia de carreras `course_code` compartidas calculadas sobre el grupo de referencia histórico `1989–1993`).

#### Precaución con `mismo_anio`
- En `exams`, 542 exámenes tienen un dígito numérico (`1`, `2`, `3`, `4`) en su 3.er carácter y 258 tienen un código alfabético o especial (`A`, `B`, `C`, `P`, `0`).
- Documentar ambos casos: calcular `mismo_anio` sobre los códigos confirmados y mantener una bandera `anio_confirmado` sin inventar años para códigos desconocidos.

### 5.3. Definición de variables por cohortes y distinción de conflictos (Etapas 4 y 5)

#### Decisión metodológica adoptada: Partición por cohortes de ingreso
Dado que el dataset de Nottingham corresponde a **un único semestre académico real (1994–95, semestre 1)**, se adopta como aproximación histórica la **partición estricta por cohortes de ingreso** identificables en `student_id[1:3]`:

- **Grupo de referencia histórico — Cohortes 1989–1993 (`student_id[1:3]` en `{'89', '90', '91', '92', '93'}`):**
  - Se utiliza **exclusivamente** para calcular `cant_estudiantes_compartidos_historico` (intersección de inscripciones entre `materia_id_1` y `materia_id_2` considerando únicamente estudiantes de las cohortes 1989–1993).
- **Grupo actual — Cohorte 1994 (`student_id[1:3] == '94'`):**
  - Se utiliza **exclusivamente** para construir las variables objetivo:
    - `cant_estudiantes_afectados`: intersección de inscripciones entre `materia_id_1` y `materia_id_2` utilizando únicamente estudiantes de la cohorte 1994.
    - `hubo_conflicto_potencial`: booleano definido como `cant_estudiantes_afectados > 0`.

#### Advertencias metodológicas obligatorias (Limitaciones y Diferencias de Tamaño / Tasas)
1. **Aproximación transversal dentro de un único semestre:**
   - Debe quedar explícitamente documentado que las cohortes `1989–1993` y `1994` fueron observadas rindiendo exámenes durante el **mismo semestre (1994–95, Semestre 1)**. No se trata de múltiples semestres históricos reales en el tiempo, sino de una partición por cohorte de ingreso dentro de un único corte transversal.
2. **Prohibición de mezclar grupos:**
   - Ningún estudiante de la cohorte 1994 puede intervenir en el cálculo de `cant_estudiantes_compartidos_historico` ni de features históricas, y ningún estudiante de 1989–1993 puede intervenir en `cant_estudiantes_afectados` ni en `hubo_conflicto_potencial`.
3. **Diferencias de tamaño y estructura curricular entre cohortes (Evaluación de Tasas y Proporciones):**
   - El grupo histórico `1989–1993` agrupa 5 cohortes (**4.810 estudiantes** y **19.798 inscripciones**), mientras que el grupo actual `1994` comprende 1 cohorte (**3.086 estudiantes** y **14.199 inscripciones**).
   - Además, por progresión académica dentro del semestre 1994–95, los ingresantes 1994 cursan mayoritariamente materias iniciales (1.er año), mientras que las cohortes 1989–1993 cursan mayoritariamente materias de 2.do y 3.er año (aunque existen 467 exámenes con inscripciones en ambos grupos).
   - Por ello, en la Fase 2 y Fase 3 **se deben documentar los conteos por cohorte y calcular también tasas/proporciones normalizadas** (por ejemplo: proporción de estudiantes compartidos respecto al total de la cohorte o coeficiente de Jaccard / solapamiento relativo $\frac{|E_1 \cap E_2|}{\min(|E_1|, |E_2|)}$ en cada grupo) para contemplar las diferencias de tamaño entre `1989–1993` y `1994`.
4. **Distinción entre conflicto potencial y conflicto efectivo de horario:**
   - El archivo `data` de Nottingham define los 32 turnos del calendario, aulas y restricciones previas, pero **no contiene un cronograma definitivo de exámenes asignados**.
   - Por lo tanto, `cant_estudiantes_afectados` y `hubo_conflicto_potencial` miden el **conflicto potencial de demanda** (estudiantes de la cohorte 1994 que rinden ambas materias y se verían perjudicados si ambas mesas se asignaran al mismo turno), el cual constituye el peso de las aristas para el algoritmo de coloreo de grafos.

### 5.4. Salida de la Fase 2 (`data/processed/subject_pairs_poc.csv`)

El archivo `data/processed/subject_pairs_poc.csv` contiene las **319.600 filas** ($\binom{800}{2}$ pares posibles) con las siguientes 24 columnas documentadas:

```text
materia_id_1
materia_id_2
materia_nombre_1
materia_nombre_2
departamento_1
departamento_2
mismo_departamento
mismo_prefijo_curricular
anio_1
anio_2
anio_confirmado
mismo_anio
misma_carrera
cant_carreras_compartidas_hist
cant_inscriptos_hist_1
cant_inscriptos_hist_2
cant_estudiantes_compartidos_historico
tasa_compartidos_historico
jaccard_historico
cant_inscriptos_actual_1
cant_inscriptos_actual_2
cant_estudiantes_afectados
tasa_afectados_actual
hubo_conflicto_potencial
```

---

## 6. Fase 3 (Etapa 7) — Informe de validación (`reports/poc_validation_report.md`)

Crear `reports/poc_validation_report.md` con:

### Estadísticas generales y por cohorte
- Cantidad de exámenes (800), estudiantes (7.896) e inscripciones (33.997) leídos.
- Desglose exacto de estudiantes e inscripciones por cada cohorte (`1989`, `1990`, `1991`, `1992`, `1993` y `1994`) y comparación entre el bloque `1989–1993` y `1994`.
- Cantidad de pares con $\ge 1$, $\ge 5$ y $\ge 10$ estudiantes compartidos en el total, en el grupo histórico (`1989–1993`) y en el grupo actual (`1994`).
- Evaluación comparativa entre conteos absolutos y tasas/proporciones entre ambos grupos.
- Cantidad de materias sin inscripciones (en general y por grupo de cohorte).
- Cantidad de pares por año curricular, carrera y departamento.

### Integridad y pruebas de consistencia
- Verificación de 0 inscripciones huérfanas y 0 duplicados.
- Desglose de códigos curriculares interpretados (`1..4`) vs. no numéricos (`A, B, C, P, 0`).
- Comprobación manual sobre muestra de estudiantes y pares de que las combinaciones de `1989–1993` y `1994` no se mezclan y coinciden exactamente con la intersección directa de sus listas de alumnos.
- Confirmación de ausencia de pares `(B, A)` duplicados y de auto-pares `(A, A)`.
- Verificación de que `hubo_conflicto_potencial` coincide al 100% con `cant_estudiantes_afectados > 0`.

---

## 7. Fase 4 (Etapas 7 y 12) — Pruebas automatizadas y cierre de la PoC

Implementar `tests/test_parsing.py`, `tests/test_pairs.py` y `tests/test_validation.py` incluyendo casos sintéticos controlados (con separación de cohortes `1989–1993` y `1994`) y verificaciones sobre los archivos procesados.

### Checklist de cierre de la PoC

- [ ] Los archivos originales de `Nottingham Dataset/` se leen correctamente sin ser modificados.
- [ ] Fase 1 completada y validada: se extraen los 800 exámenes, 7.896 estudiantes y 33.997 inscripciones en `data/interim/`, documentando las cantidades por cohorte.
- [ ] Fase 2 completada: se generan pares únicos ordenados separando estrictamente las cohortes `1989–1993` (`cant_estudiantes_compartidos_historico`) y la cohorte `1994` (`cant_estudiantes_afectados` y `hubo_conflicto_potencial`).
- [ ] Se evalúan y documentan tasas/proporciones para contemplar diferencias de tamaño entre cohortes.
- [ ] Se distingue explícitamente conflicto potencial de demanda vs. conflicto efectivo de horario.
- [ ] Se evalúa e implementa una estrategia reproducible para pares sin estudiantes compartidos (ejemplos negativos para clasificación).
- [ ] Fase 3 completada: se genera `reports/poc_validation_report.md`.
- [ ] Fase 4 completada: pasan todos los tests automatizados en `tests/`.
