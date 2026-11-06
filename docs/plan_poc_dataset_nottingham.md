# Plan de prueba de concepto, construcción del dataset y preparación para ML

**Proyecto:** Planificación de mesas de examen universitarias  
**Fuente principal:** University of Nottingham Exam Timetabling Dataset (`Nottingham Dataset/`)  
**Objetivo de este documento:** definir y estandarizar el flujo reproducible por fases que transforma los archivos originales en un dataset de pares de materias, aplica la separación metodológica por cohortes y deja listos los conjuntos de datos y lineamientos para Machine Learning y teoría de grafos.

---

## 1. Objetivo, alcance y organización por fases

El flujo de datos pasa de los archivos crudos de Nottingham a una tabla estructurada de pares de materias (`319.600` pares posibles) y luego a particiones limpias de entrenamiento y prueba listas para los experimentos de Machine Learning.

El trabajo se estructura en fases incrementales:

- **Fase 1 (Completada):** Inspección, parsing estricto de los archivos originales y validación de integridad de las tablas intermedias (`data/interim/`).
- **Fase 2 (Completada):** Construcción de los `319.600` pares posibles de materias, atributos académicos y variables basadas en la partición por cohortes (`1989–1993` como referencia histórica vs. `1994` como grupo objetivo), generando `data/processed/subject_pairs_poc.csv`.
- **Fase 2.5 (Completada):** Preparación del dataset para Machine Learning mediante partición inductiva por materias (`train` / `test`), undersampling aleatorio 50/50 sobre entrenamiento para clasificación y filtrado de columnas con riesgo de *data leakage* (`ml_train_balanced.csv`, `ml_train_full.csv`, `ml_test_full.csv`).
- **Fase 3 (Siguiente etapa):** Entrenamiento, validación y comparación de modelos de Machine Learning (Clasificación y Regresión) y generación de predicciones identificadas por par para construir el grafo de conflictos.

> **Principios metodológicos rectores:**
> 1. No inventar ni simular datos artificiales.
> 2. No mezclar estudiantes del grupo histórico (`1989–1993`) con los del grupo actual (`1994`) al calcular variables basadas en inscripciones.
> 3. Mantener diferenciados los conflictos potenciales de demanda (`hubo_conflicto_potencial`) de los conflictos efectivos de horario.

---

## 2. Fuente de datos y archivos

El conjunto original en `Nottingham Dataset/` contiene **800 exámenes**, **7.896 estudiantes** y **33.997 inscripciones** correspondientes al primer semestre de 1994–95:

| Archivo | Contenido | Formato / Anchos de campo | Uso en el proyecto |
|---|---|---|---|
| `README` | Documentación original del dataset | Texto plano (26 líneas) | Referencia oficial de formato |
| `exams` | Código, nombre, duración y departamento de cada examen | Fijo (57 caracteres): `exam_code` (8), espacio (1), `description` (40), espacio (1), `duration` (4), espacio (1), `department_code` (2) | Catálogo de materias y metadatos |
| `students` | Código de estudiante y código de carrera (`course code`) | Fijo (15 caracteres): `student_code` (10), espacio (1), `course_code` (4) | Identificación de estudiantes, cohorte de ingreso y carrera |
| `enrolements` | Relación estudiante–examen | Fijo (19 caracteres): `student_code` (10), espacio (1), `exam_code` (8) | Inscripciones individuales |
| `data` | Fechas, turnos, aulas y restricciones | Semi-estructurado (182 líneas) | Contexto de planificación (32 turnos posibles y restricciones) |

---

## 3. Estructura estandarizada del proyecto

```text
Tp-Grafos-ML/
├── Nottingham Dataset/                  # Archivos crudos originales (intactos)
│   ├── README
│   ├── exams
│   ├── students
│   ├── enrolements
│   └── data
├── docs/
│   ├── plan_poc_dataset_nottingham.md   # Este documento maestro estandarizado
│   └── fase_2_5_preparacion_ml.md       # Documentación detallada de la Fase 2.5 y guía ML
├── data/
│   ├── interim/
│   │   ├── exams_clean.csv              # 800 exámenes limpios
│   │   ├── students_clean.csv           # 7.896 estudiantes con cohorte asignada
│   │   └── enrolments_clean.csv         # 33.997 inscripciones validadas
│   └── processed/
│       ├── subject_pairs_poc.csv        # Universo completo: 319.600 pares (24 columnas)
│       ├── ml_train_balanced.csv        # Train balanceado 50/50: 6.996 pares (15 columnas)
│       ├── ml_train_full.csv            # Train completo: 254.541 pares (15 columnas)
│       └── ml_test_full.csv             # Test distribución real: 65.059 pares (15 columnas)
├── src/
│   ├── parsing/
│   │   └── parse_nott.py                # Script de Fase 1
│   └── transformation/
│       ├── build_subject_pairs.py       # Script de Fase 2
│       └── prepare_ml_dataset.py        # Script de Fase 2.5
└── README.md
```

---

## 4. Fase 1 — Inspección, parsing y validación de archivos intermedios

Implementada en [`src/parsing/parse_nott.py`](file:///w:/home/anton/Tp-Grafos-ML/src/parsing/parse_nott.py). Genera en `data/interim/`:

1. **`exams_clean.csv` (800 filas):**
   - Columnas: `exam_id, exam_name, duration, duration_minutes, department_code, raw_level_char, year_code`.
   - `year_code` solo asigna un entero (`1` a `4`) en los **542 exámenes** cuyo 3.er carácter es numérico (`'1'`: 163, `'2'`: 162, `'3'`: 213, `'4'`: 4), dejando vacío los **258 exámenes** con carácter especial (`'A'`: 63, `'B'`: 107, `'C'`: 73, `'P'`: 10, `'0'`: 5).
2. **`students_clean.csv` (7.896 filas):**
   - Columnas: `student_id, course_code, cohort_year, cohort_group`.
   - Clasifica a cada estudiante según su año de ingreso en `student_id[1:3]`:
     - `historico_1989_1993`: **4.810 estudiantes (60,92%)** (`1989`: 15, `1990`: 31, `1991`: 381, `1992`: 1.908, `1993`: 2.475).
     - `actual_1994`: **3.086 estudiantes (39,08%)** (`1994`: 3.086).
3. **`enrolments_clean.csv` (33.997 filas):**
   - Columnas: `student_id, exam_id`.
   - Integridad referencial 100% verificada (0 huérfanos, 0 duplicados, 0 exámenes sin alumnos).
   - Distribución de inscripciones: **19.798 (58,23%)** en `historico_1989_1993` y **14.199 (41,77%)** en `actual_1994`.

---

## 5. Fase 2 — Construcción de los 319.600 pares y separación por cohortes

Implementada en [`src/transformation/build_subject_pairs.py`](file:///w:/home/anton/Tp-Grafos-ML/src/transformation/build_subject_pairs.py). Genera [`data/processed/subject_pairs_poc.csv`](file:///w:/home/anton/Tp-Grafos-ML/data/processed/subject_pairs_poc.csv) con los $\binom{800}{2} = \mathbf{319.600}$ pares posibles ordenados (`materia_id_1 < materia_id_2`), incluidos los pares sin estudiantes compartidos.

### Reglas aplicadas en la Fase 2
1. **Separación estricta de cohortes (sin *Data Leakage*):**
   - Todas las variables históricas e inscripciones por carrera (`cant_inscriptos_hist_1`, `cant_inscriptos_hist_2`, `cant_estudiantes_compartidos_historico`, `tasa_compartidos_historico`, `jaccard_historico`, `cant_carreras_compartidas_hist`, `misma_carrera`) se calculan **exclusivamente con las cohortes 1989–1993** (4.810 alumnos).
   - Las variables objetivo (`cant_estudiantes_afectados` y `hubo_conflicto_potencial`) se calculan **exclusivamente con la cohorte 1994** (3.086 alumnos).
2. **Diferenciación entre `mismo_departamento` y `misma_carrera`:**
   - `mismo_departamento`: `department_code_1 == department_code_2` (8.641 pares).
   - `mismo_prefijo_curricular`: `materia_id_1[:2] == materia_id_2[:2]` (8.735 pares).
   - `misma_carrera`: existencia de al menos una carrera (`course_code`) compartida entre los inscriptos de las cohortes 1989–1993 de ambas materias (28.355 pares).
3. **Distinción entre conflicto potencial y conflicto efectivo:**
   - `cant_estudiantes_afectados` (4.468 pares con valor $\ge 1$) y `hubo_conflicto_potencial` miden el **conflicto potencial de demanda** en la cohorte 1994, no choques de un cronograma ya asignado.

---

## 6. Fase 2.5 — Preparación del dataset para Machine Learning

Implementada en [`src/transformation/prepare_ml_dataset.py`](file:///w:/home/anton/Tp-Grafos-ML/src/transformation/prepare_ml_dataset.py) y documentada en detalle en [`docs/fase_2_5_preparacion_ml.md`](file:///w:/home/anton/Tp-Grafos-ML/docs/fase_2_5_preparacion_ml.md) (`RANDOM_SEED = 42`).

### Resumen del procedimiento
1. **Partición Entrenamiento / Prueba por Materias de Retención Estratificada:**
   - Para evitar que una división aleatoria de filas permita memorizar nodos individuales vistos en otros pares, se retienen estratificadamente **86 materias para Test (~10,75%)** y **714 materias para Train (~89,25%)**.
   - **Entrenamiento Completo (`ml_train_full.csv`, 254.541 pares — 79,64%):** Pares donde ambas materias pertenecen a las 714 materias de entrenamiento (**3.498 positivos [1,374%]** y **251.043 negativos [98,626%]**).
   - **Prueba Distribución Real (`ml_test_full.csv`, 65.059 pares — 20,36%):** Pares donde al menos una materia pertenece a las 86 materias reservadas para prueba (**970 positivos [1,491%]** y **64.089 negativos [98,509%]**).
2. **Undersampling 50/50 exclusivamente sobre Entrenamiento (`ml_train_balanced.csv`, 6.996 pares):**
   - Conserva los **3.498 positivos** de entrenamiento y selecciona aleatoriamente sin reemplazo **3.498 negativos** de entrenamiento (`50,00%` / `50,00%`).
3. **Esquema de 15 columnas exportado en los archivos de ML:**
   - **Identificación del par (`ID_COLUMNS`, no usar en `X`):** `materia_id_1`, `materia_id_2` (conservadas para reconstruir el grafo de conflictos con las predicciones).
   - **Variables predictoras limpias (`PREDICTOR_COLUMNS`, 11 columnas):** `mismo_departamento`, `mismo_prefijo_curricular`, `anio_confirmado`, `mismo_anio`, `misma_carrera`, `cant_carreras_compartidas_hist`, `cant_inscriptos_hist_1`, `cant_inscriptos_hist_2`, `cant_estudiantes_compartidos_historico`, `tasa_compartidos_historico`, `jaccard_historico`.
   - **Variables objetivo (`TARGET_COLUMNS`, 2 columnas):** `cant_estudiantes_afectados` (Regresión) y `hubo_conflicto_potencial` (Clasificación).
   - **Excluidas físicamente por riesgo de *leakage*:** `cant_inscriptos_actual_1`, `cant_inscriptos_actual_2`, `tasa_afectados_actual`.

---

## 7. Indicaciones para la Fase 3 — Entrenamiento y comparación de modelos

El dataset queda preparado para que el integrante responsable de Machine Learning experimente con los modelos exigidos por el TP.

Se deberán evaluar dos enfoques:

**Clasificación:** predecir `hubo_conflicto_potencial`.

- Regresión Logística.
- KNeighborsClassifier.
- DecisionTreeClassifier.

**Regresión:** predecir `cant_estudiantes_afectados`.

- Regresión Lineal.
- KNeighborsRegressor.
- DecisionTreeRegressor.

### Consideraciones para los experimentos

- Para clasificación, comparar el entrenamiento con `ml_train_balanced.csv` y `ml_train_full.csv`, evaluando alternativas de ponderación de clases cuando estén disponibles.
- Para regresión, utilizar inicialmente `ml_train_full.csv`, ya que el undersampling se realizó en función del target de clasificación y modifica la distribución de cantidades de estudiantes afectados.
- Utilizar `ml_test_full.csv` para la evaluación final de ambos enfoques.
- Aplicar escalado de variables en KNN, ajustándolo exclusivamente con los datos de entrenamiento.
- Seleccionar hiperparámetros y umbrales mediante validación, sin utilizar el conjunto de test para tomar decisiones.
- Al construir la matriz de predictores `X`, retirar siempre las columnas identificadoras (`materia_id_1`, `materia_id_2`) y ambas columnas objetivo (`cant_estudiantes_afectados`, `hubo_conflicto_potencial`). En Regresión Lineal/Logística y KNN se recomienda usar `PREDICTOR_COLUMNS_NO_COLLINEAR` (omitiendo `tasa_compartidos_historico` por ser múltiplo exacto de `cant_estudiantes_compartidos_historico`).

### Selección del modelo

No se establece de antemano un algoritmo ganador. La elección deberá justificarse mediante métricas adecuadas, comparación de resultados y análisis de sus limitaciones.

Para clasificación se priorizarán Precision, Recall, F1-score de la clase positiva, PR-AUC y matriz de confusión.

Para regresión se utilizarán MAE y RMSE, complementados con un análisis del comportamiento de las predicciones sobre valores cero y cantidades elevadas de estudiantes afectados.

El resultado deberá conservar la identificación de cada par de materias (`materia_id_1`, `materia_id_2`) y generar las predicciones correspondientes, para su posterior utilización en la construcción del grafo de conflictos.

---

## 8. Estado de Avance del Plan (Checklist)

- [x] Los archivos originales de `Nottingham Dataset/` se leen correctamente sin ser modificados (Fase 1).
- [x] Se extraen y validan los 800 exámenes, 7.896 estudiantes y 33.997 inscripciones en `data/interim/`, documentando las cantidades por cohorte (Fase 1).
- [x] Se generan los 319.600 pares únicos ordenados separando estrictamente las cohortes `1989–1993` (`cant_estudiantes_compartidos_historico`) y la cohorte `1994` (`cant_estudiantes_afectados` y `hubo_conflicto_potencial`) en `data/processed/subject_pairs_poc.csv` (Fase 2).
- [x] Se calculan tasas y proporciones (`tasa_compartidos_historico`, `jaccard_historico`, `tasa_afectados_actual`) para contemplar diferencias de tamaño entre cohortes (Fase 2).
- [x] Se distingue explícitamente conflicto potencial de demanda vs. conflicto efectivo de horario (Fases 1 y 2).
- [x] Se generan las particiones libres de *data leakage* para Machine Learning (`ml_train_balanced.csv`, `ml_train_full.csv`, `ml_test_full.csv`) conservando `materia_id_1` y `materia_id_2` para la reconstrucción del grafo (Fase 2.5).
- [ ] Entrenamiento, validación y comparación de los modelos de Clasificación y Regresión (Fase 3).
- [ ] Construcción del grafo ponderado de conflictos a partir de las predicciones y coloreo greedy de turnos (Fase 4).
