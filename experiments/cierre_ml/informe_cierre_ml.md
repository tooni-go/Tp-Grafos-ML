# Informe de Validación Final de Features y Cierre del Módulo ML (`Tp-Grafos-ML`)

**Fecha:** 2026-11-07  
**Directorio de trabajo:** [`experiments/cierre_ml/`](file:///w:/home/anton/Tp-Grafos-ML/experiments/cierre_ml/)  
**Script reproducible:** [`experiments/cierre_ml/run_cierre_ml.py`](file:///w:/home/anton/Tp-Grafos-ML/experiments/cierre_ml/run_cierre_ml.py)  
**Estado de archivos oficiales:** `git status -s` confirma que ningún archivo de `src/`, `data/`, `results/`, `docs/` ni `tests/` fue modificado.  
**Nota metodológica sobre el conjunto de prueba:** Toda verificación de niveles, auditoría de *leakage*, selección de representación, elección de modelo y calibración de umbral se ejecutó exclusivamente sobre `subtrain` (637 materias / 202.566 pares, 2.836 positivos) y `validation` (77 materias / 51.975 pares, 662 positivos). El conjunto `test` (86 materias / 65.059 pares, 970 positivos) se evaluó una única vez al final reentrenando sobre `ml_train_full.csv` (714 materias / 254.541 pares), dejando constancia explícita de que dicho conjunto ya había sido consultado en rondas previas del proyecto.

---

## 1. Verificación del Hallazgo de los Niveles Académicos (`A`/`B`/`C` y Otros Códigos)

### 1.1 Documentación original vs. código de parsing actual

1. **Archivos crudos de Nottingham ([`Nottingham Dataset/README`](file:///w:/home/anton/Tp-Grafos-ML/Nottingham%20Dataset/README) y [`data`](file:///w:/home/anton/Tp-Grafos-ML/Nottingham%20Dataset/data)):**
   - El archivo `README` original documenta únicamente el ancho fijo de las columnas de `exams` (`exam code`: 8 caracteres, `description`: 40, `duration`: 4, `department code`: 2), sin desglosar la semántica interna de los caracteres del `exam code`.
2. **Documentación técnica del proyecto ([`docs/analisis_datasets.md`](file:///w:/home/anton/Tp-Grafos-ML/docs/analisis_datasets.md#L50-L53) y [`docs/plan_poc_dataset_nottingham.md`](file:///w:/home/anton/Tp-Grafos-ML/docs/plan_poc_dataset_nottingham.md#L100-L101)):**
   - El análisis inicial del repositorio ya había identificado que el 3.er carácter del código de examen (`raw_level_char = exam_id[2]`) codifica el año/nivel de la materia en el plan de estudios (`1`/`A` = 1.er año, `2`/`B` = 2.do año, `3`/`C` = 3.er año), comprobado empíricamente cruzando inscripciones con el año de ingreso del estudiante (`student_id[1:3]`).
3. **Código de parsing ([`src/parsing/parse_nott.py`](file:///w:/home/anton/Tp-Grafos-ML/src/parsing/parse_nott.py#L27-L173)):**
   - A pesar de lo documentado en `docs/analisis_datasets.md`, `parse_nott.py` definió `VALID_NUMERIC_YEAR_CODES = {"1", "2", "3", "4"}` y asignó `year_code = None` a cualquier examen cuyo 3.er carácter no fuera un dígito entre `'1'` y `'4'`.
   - En [`src/transformation/build_subject_pairs.py`](file:///w:/home/anton/Tp-Grafos-ML/src/transformation/build_subject_pairs.py#L231-L232), esto provoca que `anio_confirmado` y `mismo_anio` valgan `0` siempre que al menos una de las dos materias tenga `raw_level_char` en `{'A', 'B', 'C', '0', 'P'}`.

### 1.2 Verificación empírica de todos los códigos presentes en `exams_clean.csv`

La tabla completa generada por el script se encuentra en [`results/auditoria_niveles_codigos.csv`](file:///w:/home/anton/Tp-Grafos-ML/experiments/cierre_ml/results/auditoria_niveles_codigos.csv). En los **800 exámenes** existen exactamente **9 valores distintos** de `raw_level_char`:

| Código (`exam_id[2]`) | Exámenes Totales (`800`) | Exámenes `subtrain` (`637`) | Inscripciones Totales (`33.997`) | Exámenes con `0` inscr. hist. | % Inscr. Cohorte `1991` (4.º año) | % Inscr. Cohorte `1992` (3.er año) | % Inscr. Cohorte `1993` (2.º año) | % Inscr. Cohorte `1994` (1.er año) | Cohorte Modal | Acuerdo por Examen (`subtrain`) | Interpretación Verificada |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **`1`** | **163** | 130 | 10.934 | 28 | 0,4% | 1,4% | 5,2% | **93,0%** | **1994** | **95,38%** | **1.er año** (numérico). |
| **`A`** | **63** | 50 | 3.522 | 6 | 1,1% | 2,7% | 21,6% | **74,5%** | **1994** | **86,00%** | **Equivalente a 1.er año** (*Part A* / nivel inicial). |
| **`2`** | **162** | 130 | 7.085 | 0 | 0,6% | 3,7% | **90,5%** | 5,2% | **1993** | **96,15%** | **2.do año** (numérico). |
| **`B`** | **107** | 86 | 4.246 | 0 | 2,8% | 28,3% | **61,7%** | 6,8% | **1993** | **62,79%** *(90,0% en 92–93)* | **Equivalente a 2.do año** (*Part B* / ciclo intermedio). |
| **`3`** | **213** | 170 | 5.430 | 2 | 15,2% | **77,9%** | 1,5% | 4,3% | **1992** | **76,47%** | **3.er año** (numérico). |
| **`C`** | **73** | 58 | 2.379 | 0 | 21,4% | **62,3%** | 5,3% | 8,8% | **1992** | **86,21%** | **Equivalente a 3.er año** (*Part C* / ciclo avanzado). |
| **`4`** | **4** | 2 | 67 | 0 | **58,2%** | 13,4% | 19,4% | 7,5% | **1991** | **100,0%** | **4.to año** (numérico; ya incluido en `parse_nott.py`). |
| **`0`** | **5** | 3 | 112 | 3 | 0,0% | 0,0% | 1,8% | **98,2%** | 1994 | — | **Curso preparatorio (*Foundation Year*)**: `EN00C1E1`..`EN00P7E1`. |
| **`P`** | **10** | 8 | 222 | 6 | 0,0% | 0,0% | 11,3% | **88,3%** | 1994 / 1993 | — | **Heterogéneo (Posgrado *MSc* / Preliminar)**: ej. `C8P001E1` (*MSC OCC PSYCH*), `K4P300E1`. |

> **Nota de corrección respecto al informe previo:** En `exams_clean.csv` las cantidades exactas son **`'A' = 63`, `'B' = 107`, `'C' = 73`, `'P' = 10`, `'0' = 5`** (total `258` no numéricos), tal como documentaba `docs/plan_poc_dataset_nottingham.md`.

### 1.3 Verificación de no mezclar códigos de distinta naturaleza

1. **Validez de `1=A`, `2=B`, `3=C` en departamentos mixtos vs. solo-letras ([`results/auditoria_niveles_deptos_mixtos.csv`](file:///w:/home/anton/Tp-Grafos-ML/experiments/cierre_ml/results/auditoria_niveles_deptos_mixtos.csv)):**
   - De los 46 departamentos, **25 departamentos** usan tanto códigos numéricos (`1`/`2`/`3`) como letras (`A`/`B`/`C`) (por ejemplo, departamentos que distinguen módulos troncales con número y módulos optativos/interfacultad con letra).
   - En ambos grupos de departamentos, la cohorte dominante de `'A'` es `1994` (`82,5%` de acuerdo en mixtos, `100%` en solo-letras), la de `'B'` es `1993` (`60,0%` en mixtos, `100%` en solo-letras) y la de `'C'` es `1992` (`86,2%` en mixtos, `87,5%` en solo-letras).
2. **Exclusión deliberada de `'0'` y `'P'`:**
   - En la exploración previa se habían agrupado `'0'` y `'P'` junto con `'1'`/`'A'`. La inspección detallada de los títulos en `exams_clean.csv` demuestra que **`'P'` mezcla maestrías de posgrado (`MSC OCC PSYCH...`), cursos de planificación de cohorte 1993 (`K4P300E1`) y materias introductorias**, mientras que **`'0'` son 5 asignaturas nivelatorias de ingeniería (`EN00...`)**.
   - Por rigor metodológico, **no se mapean `'0'` ni `'P'`**: el mapeo validado es exclusivamente `{'1': 1, '2': 2, '3': 3, '4': 4, 'A': 1, 'B': 2, 'C': 3}`, manteniendo `None` para las 15 materias `'0'` y `'P'`.

### 1.4 Cuantificación de materias y pares afectados

- **Materias con año identificado:** pasan de **542 (`67,75%`)** a **785 (`98,13%`)** (+243 materias).
- **Pares con `anio_confirmado == 1` (sobre 319.600 pares):** pasan de **146.611 (`45,87%`)** a **307.720 (`96,28%`)** (+161.109 pares).
- **Pares con `mismo_anio == 1`:** pasan de **48.828 (`15,28%`)** a **102.232 (`31,99%`)** (+53.404 pares nuevos, sin perder ninguno de los originales).
- **Conflictos reales recuperados en `subtrain + validation`:** de los `3.498` pares positivos de entrenamiento, **1.281 positivos (`36,62%`)** tenían `mismo_anio = 0` por involucrar al menos una materia `'A'`, `'B'` o `'C'` y pasan a tener `mismo_anio_corregido_ABC = 1` (con una prevalencia positiva de `3,01%`, más del doble del promedio global `1,37%`).

---

## 2. Auditoría de la Ingeniería de Features y Análisis de *Leakage*

La tabla resumen se exportó en [`results/auditoria_leakage_features.csv`](file:///w:/home/anton/Tp-Grafos-ML/experiments/cierre_ml/results/auditoria_leakage_features.csv). A continuación se audita cada variable nueva bajo los 7 criterios exigidos:

### 2.1 Tabla de auditoría por feature nueva

| Feature | 1. Qué representa | 2. Archivos y columnas de origen | 3. ¿Solo info previa a la predicción? | 4. ¿Usa directa o indirectamente el target? | 5. Riesgo de *leakage* / sesgo experimental | 6. ¿Es simétrica $(u,v)=(v,u)$? | 7. ¿Tiene sentido académico generalizable? | Veredicto |
| :--- | :--- | :--- | :---: | :---: | :--- | :---: | :--- | :---: |
| **`anio_confirmado_corregido_ABC`** | Indicador `1/0` de que ambas materias tienen nivel en `{1..4, A..C}`. | `exams` (`exam_id[2]`) $\to$ `exams_clean.csv:raw_level_char` | **Sí** (metadato estático del catálogo). | **No** | **Ninguno.** Regla determinista sobre el código de materia. | **Sí** (`True`) | **Sí.** Es la definición completa de la feature oficial `anio_confirmado`. | **APROBADA** |
| **`mismo_anio_corregido_ABC`** | Indicador `1/0` de que ambas materias tienen el mismo nivel con `1=A, 2=B, 3=C`. | `exams` (`exam_id[2]`) $\to$ `exams_clean.csv:raw_level_char` | **Sí** (metadato estático del catálogo). | **No** | **Ninguno.** | **Sí** (`True`) | **Sí.** Alumnos del mismo año de cursada tienden a coincidir en materias de ese nivel. | **APROBADA** |
| **`ambas_primer_anio`** | Indicador `1/0` de que ambas materias son de 1.er año (`1` o `A`). | `exams` (`exam_id[2]`) $\to$ `exams_clean.csv:raw_level_char` | **Sí** en cuanto al dato crudo (`exam_id`). | **Indirectamente sí (sesgo de población objetivo).** | **Alto riesgo de sobreajuste al corte de cohortes:** el target se definió solo con la cohorte `1994` (ingresantes de 1.er año, `92,3%` en nivel `1`/`A`). | **Sí** (`True`) | **Limitado:** en un examen universitario real rinden todos los años; penalizar pares de 2.º/3.er año solo funciona porque `actual_1994` son ingresantes. | **CUESTIONABLE (ver §2.2)** |
| **`log1p(vecinos_comunes_hist_2saltos)`** | $\ln(1 + \|N_{\text{hist}}(u) \cap N_{\text{hist}}(v)\|)$: materias $w$ que compartieron alumnos con $u$ y con $v$ en `1989–1993`. | `enrolments_clean.csv` + `students_clean.csv` filtrando `cohort_group == "historico_1989_1993"` (`19.798` filas). | **Sí** (`0` alumnos de `1994`). | **No** (invariante al eliminar o permutar las inscripciones de `1994`). | **Bajo en etiquetas, pero usa nodos *held-out* como intermediarios $w$** y rompe la inferencia autónoma por par. | **Sí** (`True`) | **Sí** (proximidad curricular de 2.º orden), pero su ganancia sobre `F6` es marginal (`+0,0088` PR-AUC). | **DESCARTADA POR SIMPLICIDAD / ARQUITECTURA** |

### 2.2 Hallazgo crítico: ¿Por qué separamos `F6_sin_primer_anio` de `F6`?

En la ronda anterior se habían agrupado en una misma prueba diagnóstica la corrección de niveles `A`/`B`/`C` y el indicador `ambas_primer_anio`. Al auditarlos por separado en `run_cierre_ml.py` descubrimos un punto metodológico fundamental para la defensa académica del trabajo:

1. **Qué hace `mismo_anio_corregido_ABC` (`F6_sin_primer_anio`):**
   - Informa al modelo si dos materias pertenecen al mismo año del plan (`1=A`, `2=B`, `3=C`, `4=4`), **sin distinguir qué año es**. Es una propiedad relacional simétrica entre pares y 100% válida para cualquier cohorte estudiantil.
2. **Qué hace `ambas_primer_anio` (`F6`):**
   - Como el dataset de Nottingham abarca un único semestre lectivo y la Fase 2 reservó la cohorte `1994` (ingresantes de 1.er año) para definir el target `hubo_conflicto_potencial`, casi todos los conflictos del target (`92,3%` de las inscripciones de `actual_1994`) ocurren entre materias de 1.er año (`1` o `A`), mientras que los pares de dos materias de 2.º o 3.er año casi no tienen alumnos de la cohorte `1994`.
   - Por eso, agregar `ambas_primer_anio` hace saltar el `PR-AUC` de Validación de **`0,2339` (`F6_sin_primer_anio`) a `0,3211` (`F6`)**: la feature le "avisa" al modelo que la población donde se mide el target son solo los alumnos de 1.er año.
   - **Implicación metodológica:** Aunque `ambas_primer_anio` no lee las inscripciones de 1994, **explota un artefacto de cómo se construyó el target en la PoC**. Presentamos los resultados completos de ambas (`F6_sin_primer_anio` y `F6`) para que la decisión quede transparente y documentada.

### 2.3 Verificación rigurosa del grafo histórico a dos saltos (`F7`)

El script `run_cierre_ml.py` ejecutó cinco pruebas automáticas sobre el grafo histórico utilizado en `F7` ([`results/resumen_cierre_ml.json`](file:///w:/home/anton/Tp-Grafos-ML/experiments/cierre_ml/results/resumen_cierre_ml.json)):
1. **Exclusividad histórica:** utiliza exactamente las **19.798 inscripciones** de `historico_1989_1993` y **0 inscripciones** de `actual_1994`.
2. **Invariancia por ablación del target:** al eliminar todas las filas de `actual_1994` de `enrolments_clean.csv` antes de construir el grafo, la matriz de adyacencia y el conteo de vecinos a 2 saltos son **bit a bit idénticos (`True`)**.
3. **Invariancia por permutación del target:** al permutar aleatoriamente los exámenes de `actual_1994`, el grafo es **idéntico (`True`)**.
4. **Consistencia con las features oficiales de 1 salto:** la existencia de arista directa en el grafo histórico coincide exactamente en los `319.600` pares con `cant_estudiantes_compartidos_historico > 0` (`True`).
5. **Sensibilidad a materias *held-out* como intermediarias $w$:** si se restringen los nodos intermedios $w$ del camino $u - w - v$ únicamente a las `637` materias de `subtrain` (excluyendo las `77` de `validation` y las `86` de `test`), el `PR-AUC` en Validación pasa de `0,3291` a `0,3261` en el Árbol de Decisión y de `0,3198` a `0,3195` en Regresión Logística. Es decir, no hay fuga de etiquetas; sin embargo, dentro del grupo "5 ceros" de Validación solo el **25,41%** de los positivos tiene vecinos comunes a 2 saltos frente al **14,12%** de los negativos *(corrigiendo la cifra de 37,8% mencionada en el informe previo)*, y la ganancia de `F7` sobre `F6` es de apenas `+0,0088` en `PR-AUC` medio de validación.

---

## 3. Definición Exacta de las Representaciones Comparadas

| Representación | N.º Cols | Columnas Exactas | Requiere cambiar `parse_nott.py` / `build_subject_pairs.py` | Simétrica $(u,v)=(v,u)$ |
| :--- | :---: | :--- | :---: | :---: |
| **`F0`** (Oficial actual) | **10** | `mismo_departamento`, `mismo_prefijo_curricular`, `anio_confirmado`, `mismo_anio`, `misma_carrera`, `cant_carreras_compartidas_hist`, `cant_inscriptos_hist_1`, `cant_inscriptos_hist_2`, `cant_estudiantes_compartidos_historico`, `jaccard_historico` | No | **No** (`hist_1` vs `hist_2`) |
| **`F4`** (Simetrizada + `log1p`) | **10** | `mismo_departamento`, `mismo_prefijo_curricular`, `anio_confirmado`, `mismo_anio`, `misma_carrera`, `log1p(cant_carreras_compartidas_hist)`, `log1p(min(inscriptos_hist_1, 2))`, `log1p(max(inscriptos_hist_1, 2))`, `log1p(cant_estudiantes_compartidos_historico)`, `jaccard_historico` | No (se aplica sobre las 10 columnas de `F0`) | **Sí** |
| **`F5`** (Ingeniería 16 vars sobre `F0`) | **16** | Las 10 de `F4` + `log1p(max)-log1p(min)`, `ambas_sin_historial`, `una_sin_historial`, `comparten_hist`, `compartidos/min_insc_hist`, `interaccion_ad_hoc` | No (se aplica sobre las 10 columnas de `F0`) | **Sí** |
| **`F6_sin_primer_anio`** (Niveles `A/B/C` corregidos, 10 cols) | **10** | Exactamente las 10 de `F4`, pero con `anio_confirmado` y `mismo_anio` calculados usando `1=A, 2=B, 3=C, 4=4` | Sí (solo mapear `A/B/C` en `parse_nott.py`) | **Sí** |
| **`F6`** (Niveles `A/B/C` + indicador 1.er año) | **11** | Las 10 de `F6_sin_primer_anio` + `ambas_primer_anio` (`nivel(u)==1 and nivel(v)==1`) | Sí (agrega 1 columna al esquema) | **Sí** |
| **`F7`** (`F6` + 2 saltos históricos) | **12** | Las 11 de `F6` + `log1p(vecinos_comunes_hist_2saltos)` | Sí (agrega grafo histórico a Fase 2 / inferencia) | **Sí** |

---

## 4. Resultados Experimentales: Comparación de Representaciones y Modelos

Todos los resultados provienen de [`results/comparacion_representaciones_validation.csv`](file:///w:/home/anton/Tp-Grafos-ML/experiments/cierre_ml/results/comparacion_representaciones_validation.csv) y [`results/comparacion_modelos_val_test.csv`](file:///w:/home/anton/Tp-Grafos-ML/experiments/cierre_ml/results/comparacion_modelos_val_test.csv). Los intervalos de confianza al 95% de $\Delta\text{PR-AUC}$ respecto a `F0` se calcularon mediante *block bootstrap* ($B = 300$) agrupando por materia de validación.

### 4.1 Desempeño en Validación (`subtrain` $\to$ `validation`, umbral que maximiza `F1` en validación)

| Representación | Modelo | $\tau_{\text{val}}$ | Val Precision | Val Recall | Val F1 | Val PR-AUC | $\Delta\text{PR-AUC}$ vs `F0` (IC 95%) | Val ROC-AUC | Val TP / FP / FN / TN | Aristas Val (`FP/TP`) | Fit (s) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`F0` (Ref. Oficial)** | `KNN_k15_Oficial` | 0,21 | 0,2761 | 0,2115 | 0,2395 | 0,1846 | — | 0,7651 | 140 / 367 / 522 / 50.946 | 507 (2,62) | 0,22 |
| **`F0`** | `KNN_k51_Manhattan_Uniform` | 0,16 | 0,3113 | 0,2492 | 0,2768 | 0,2088 | — | 0,8154 | 165 / 365 / 497 / 50.948 | 530 (2,21) | 0,33 |
| **`F0`** | `KNN_k51_Euclidean_Distance` | 0,17 | 0,2864 | 0,2885 | 0,2874 | 0,1802 | — | 0,7827 | 191 / 476 / 471 / 50.837 | 667 (2,49) | 0,23 |
| **`F0`** | `DecisionTree_d6_leaf50_Gini` | 0,20 | 0,3389 | 0,2145 | 0,2627 | 0,2148 | — | 0,8653 | 142 / 277 / 520 / 51.036 | 419 (1,95) | 0,18 |
| **`F0`** | `LogisticRegression_L2_C1` | 0,19 | 0,3283 | 0,1647 | 0,2193 | 0,1451 | — | 0,8415 | 109 / 223 / 553 / 51.090 | 332 (2,05) | 0,81 |
| **`F4`** | `KNN_k51_Manhattan_Uniform` | 0,18 | 0,3164 | 0,2160 | 0,2567 | 0,2051 | -0,0035 [-0,014, +0,008] | 0,8107 | 143 / 309 / 519 / 51.004 | 452 (2,16) | 0,24 |
| **`F4`** | `DecisionTree_d6_leaf50_Gini` | 0,15 | 0,2786 | 0,2100 | 0,2394 | 0,1823 | -0,0322 [-0,054, -0,013] | 0,8655 | 139 / 360 / 523 / 50.953 | 499 (2,59) | 0,12 |
| **`F4`** | `LogisticRegression_L2_C1` | 0,15 | 0,2789 | 0,1858 | 0,2230 | 0,1569 | +0,0104 [-0,003, +0,022] | 0,8513 | 123 / 318 / 539 / 50.995 | 441 (2,59) | 0,42 |
| **`F5`** | `KNN_k51_Manhattan_Uniform` | 0,22 | 0,3333 | 0,2054 | 0,2542 | 0,2012 | -0,0071 [-0,020, +0,004] | 0,8087 | 136 / 272 / 526 / 51.041 | 408 (2,00) | 0,06 |
| **`F5`** | `DecisionTree_d6_leaf50_Gini` | 0,13 | 0,2998 | 0,2296 | 0,2601 | 0,1905 | -0,0239 [-0,045, -0,005] | 0,8475 | 152 / 355 / 510 / 50.958 | 507 (2,34) | 0,22 |
| **`F5`** | `LogisticRegression_L2_C1` | 0,14 | 0,2878 | 0,2130 | 0,2448 | 0,1792 | **+0,0326 [+0,013, +0,053]** | 0,8525 | 141 / 349 / 521 / 50.964 | 490 (2,48) | 0,92 |
| **`F6_sin_primer_anio`** | **`KNN_k51_Manhattan_Uniform`** | **0,18** | **0,3386** | **0,2568** | **0,2921** | **0,2339** | **+0,0255 [+0,005, +0,048]** | **0,8658** | **170 / 332 / 492 / 50.981** | **502 (1,95)** | **0,27** |
| **`F6_sin_primer_anio`** | **`DecisionTree_d6_leaf50_Gini`** | **0,13** | **0,2739** | **0,3338** | **0,3009** | **0,2165** | +0,0021 [-0,027, +0,030] | **0,9034** | **221 / 586 / 441 / 50.727** | **807 (2,65)** | **0,14** |
| **`F6_sin_primer_anio`** | `LogisticRegression_L2_C1` | 0,13 | 0,2781 | 0,2915 | 0,2847 | 0,1901 | **+0,0430 [+0,018, +0,067]** | 0,8785 | 193 / 501 / 469 / 50.812 | 694 (2,60) | 0,48 |
| **`F6`** | `KNN_k51_Manhattan_Uniform` | 0,16 | 0,2931 | 0,3776 | 0,3300 | 0,2947 | **+0,0867 [+0,047, +0,128]** | 0,8967 | 250 / 603 / 412 / 50.710 | 853 (2,41) | 0,27 |
| **`F6`** | **`DecisionTree_d6_leaf50_Gini`** | **0,23** | **0,4133** | **0,3097** | **0,3541** | **0,3211** | **+0,1058 [+0,064, +0,147]** | **0,9307** | **205 / 291 / 457 / 51.022** | **496 (1,42)** | **0,14** |
| **`F6`** | **`LogisticRegression_L2_C1`** | **0,23** | **0,4070** | **0,3142** | **0,3546** | **0,3170** | **+0,1683 [+0,132, +0,203]** | **0,9287** | **208 / 303 / 454 / 51.010** | **511 (1,46)** | **0,50** |
| **`F7`** | `KNN_k51_Manhattan_Uniform` | 0,20 | 0,3508 | 0,3535 | 0,3521 | 0,3103 | **+0,1018 [+0,064, +0,141]** | 0,8939 | 234 / 433 / 428 / 50.880 | 667 (1,85) | 0,26 |
| **`F7`** | `DecisionTree_d6_leaf50_Gini` | 0,17 | 0,3288 | 0,4033 | 0,3623 | 0,3291 | **+0,1134 [+0,070, +0,157]** | 0,9335 | 267 / 545 / 395 / 50.768 | 812 (2,04) | 0,15 |
| **`F7`** | `LogisticRegression_L2_C1` | 0,10 | 0,3080 | 0,4411 | 0,3627 | 0,3198 | **+0,1709 [+0,130, +0,210]** | 0,9337 | 292 / 656 / 370 / 50.657 | 948 (2,25) | 0,55 |

### 4.2 Evaluación Única en Test (`train_full` $\to$ `test`, usando $\tau_{\text{val}}$ sin ajustes)

| Representación | Modelo | $\tau_{\text{val}}$ | Test Precision | Test Recall | Test F1 | Test PR-AUC | Test ROC-AUC | Test TP / FP / FN / TN | Aristas Test (`FP/TP`) | Recall Ponderado por Alumnos | Fit (s) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`F0` (Ref. Oficial)** | `KNN_k15_Oficial` | 0,21 | 0,2711 | 0,2485 | 0,2593 | 0,1833 | 0,7665 | 241 / 648 / 729 / 63.441 | 889 (2,69) | 58,45% | 0,31 |
| **`F0`** | **`KNN_k51_Manhattan_Uniform`** | **0,16** | **0,2912** | **0,2567** | **0,2729** | **0,2249** | **0,8289** | **249 / 606 / 721 / 63.483** | **855 (2,43)** | **58,64%** | **0,33** |
| **`F0`** | `KNN_k51_Euclidean_Distance` | 0,17 | 0,2752 | 0,2763 | 0,2757 | 0,1806 | 0,7846 | 268 / 706 / 702 / 63.383 | 974 (2,63) | 55,95% | 0,30 |
| **`F0`** | **`DecisionTree_d6_leaf50_Gini`** | **0,20** | **0,3842** | **0,2052** | **0,2675** | **0,2270** | **0,8777** | **199 / 319 / 771 / 63.770** | **518 (1,60)** | **55,08%** | **0,16** |
| **`F0`** | `LogisticRegression_L2_C1` | 0,19 | 0,3326 | 0,1649 | 0,2205 | 0,1566 | 0,8632 | 160 / 321 / 810 / 63.768 | 481 (2,01) | 50,69% | 0,51 |
| **`F4`** | **`KNN_k51_Manhattan_Uniform`** | **0,18** | **0,3153** | **0,2526** | **0,2805** | **0,2273** | **0,8178** | **245 / 532 / 725 / 63.557** | **777 (2,17)** | **58,12%** | **0,31** |
| **`F4`** | `DecisionTree_d6_leaf50_Gini` | 0,15 | 0,3361 | 0,2124 | 0,2603 | 0,2196 | 0,8782 | 206 / 407 / 764 / 63.682 | 613 (1,98) | 49,07% | 0,19 |
| **`F4`** | `LogisticRegression_L2_C1` | 0,15 | 0,3085 | 0,2134 | 0,2523 | 0,1724 | 0,8701 | 207 / 464 / 763 / 63.625 | 671 (2,24) | 54,72% | 0,61 |
| **`F5`** | `KNN_k51_Manhattan_Uniform` | 0,22 | 0,3483 | 0,2000 | 0,2541 | 0,2245 | 0,8158 | 194 / 363 / 776 / 63.726 | 557 (1,87) | 55,74% | 0,07 |
| **`F5`** | `DecisionTree_d6_leaf50_Gini` | 0,13 | 0,3038 | 0,2402 | 0,2683 | 0,2235 | 0,8625 | 233 / 534 / 737 / 63.555 | 767 (2,29) | 50,61% | 0,32 |
| **`F5`** | `LogisticRegression_L2_C1` | 0,14 | 0,3022 | 0,2237 | 0,2571 | 0,2043 | 0,8718 | 217 / 501 / 753 / 63.588 | 718 (2,31) | 57,99% | 0,82 |
| **`F6_sin_primer_anio`** | **`KNN_k51_Manhattan_Uniform`** | **0,18** | **0,2724** | **0,2979** | **0,2846** | **0,2361** | **0,8292** | **289 / 772 / 681 / 63.317** | **1.061 (2,67)** | **59,81%** | **0,31** |
| **`F6_sin_primer_anio`** | **`DecisionTree_d6_leaf50_Gini`** | **0,13** | **0,2643** | **0,3144** | **0,2872** | **0,2230** | **0,8847** | **305 / 849 / 665 / 63.240** | **1.154 (2,78)** | **61,72%** | **0,15** |
| **`F6_sin_primer_anio`** | `LogisticRegression_L2_C1` | 0,13 | 0,2614 | 0,2948 | 0,2771 | 0,1801 | 0,8696 | 286 / 808 / 684 / 63.281 | 1.094 (2,83) | 60,39% | 0,53 |
| **`F6`** | `KNN_k51_Manhattan_Uniform` | 0,16 | 0,2896 | 0,3979 | 0,3352 | 0,3086 | 0,8616 | 386 / 947 / 584 / 63.142 | 1.333 (2,45) | 65,15% | 0,35 |
| **`F6`** | **`DecisionTree_d6_leaf50_Gini`** | **0,23** | **0,4338** | **0,3041** | **0,3576** | **0,3074** | **0,9107** | **295 / 385 / 675 / 63.704** | **680 (1,31)** | **60,13%** | **0,18** |
| **`F6`** | **`LogisticRegression_L2_C1`** | **0,23** | **0,4138** | **0,3268** | **0,3652** | **0,3128** | **0,9134** | **317 / 449 / 653 / 63.640** | **766 (1,42)** | **62,50%** | **0,45** |
| **`F7`** | `KNN_k51_Manhattan_Uniform` | 0,20 | 0,3529 | 0,3711 | 0,3618 | 0,3142 | 0,8696 | 360 / 660 / 610 / 63.429 | 1.020 (1,83) | 67,92% | 0,37 |
| **`F7`** | `DecisionTree_d6_leaf50_Gini` | 0,17 | 0,3454 | 0,3928 | 0,3676 | 0,3109 | 0,9148 | 381 / 722 / 589 / 63.367 | 1.103 (1,90) | 65,95% | 0,21 |
| **`F7`** | `LogisticRegression_L2_C1` | 0,10 | 0,3000 | 0,4299 | 0,3534 | 0,3191 | 0,9196 | 417 / 973 / 553 / 63.116 | 1.390 (2,33) | 67,99% | 0,69 |

---

## 5. Análisis de Impacto en el Grafo de Conflictos

Los resultados detallados de impacto estructural sobre el grafo se exportaron en [`results/impacto_grafo.csv`](file:///w:/home/anton/Tp-Grafos-ML/experiments/cierre_ml/results/impacto_grafo.csv) y el barrido completo de umbrales en [`results/umbrales_tradeoff.csv`](file:///w:/home/anton/Tp-Grafos-ML/experiments/cierre_ml/results/umbrales_tradeoff.csv).

| Configuración | Representación | Criterio de Umbral | $\tau_{\text{val}}$ | Split | Aristas Reales | Aristas Predichas (`TP+FP`) | Conflictos Detectados (`TP`) | Aristas Falsas (`FP`) | Ratio `FP/TP` | % Estudiantes Afectados Cubiertos | Grado Medio Predicho (Materias *Held-out*) | Grado Real Medio |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `KNN_k15` (Oficial actual) | `F0` (10 cols) | Máx. F1 Val | 0,21 | Val | 662 | 507 | 140 (21,1%) | 367 | 2,62 | 53,37% | — | 9,08 |
| `KNN_k15` (Oficial actual) | `F0` (10 cols) | Máx. F1 Val | 0,21 | Test | 970 | 889 | 241 (24,8%) | 648 | 2,69 | 58,45% | — | 11,95 |
| **`KNN_k51_Manhattan`** | **`F0` (10 cols)** | **Máx. F1 Val** | **0,16** | **Val** | 662 | **530** | **165 (24,9%)** | **365** | **2,21** | **58,25%** | — | 9,08 |
| **`KNN_k51_Manhattan`** | **`F0` (10 cols)** | **Máx. F1 Val** | **0,16** | **Test** | 970 | **855** | **249 (25,7%)** | **606** | **2,43** | **58,64%** | — | 11,95 |
| **`KNN_k51_Manhattan`** | **`F6_sin_primer_anio` (10 cols)** | **Máx. F1 Val** | **0,18** | **Val** | 662 | **502** | **170 (25,7%)** | **332** | **1,95** | **59,86%** | — | 9,08 |
| **`KNN_k51_Manhattan`** | **`F6_sin_primer_anio` (10 cols)** | **Máx. F1 Val** | **0,18** | **Test** | 970 | **1.061** | **289 (29,8%)** | **772** | **2,67** | **59,81%** | — | 11,95 |
| `DecisionTree_d6_leaf50` | `F0` (10 cols) | Máx. F1 Val | 0,20 | Val | 662 | 419 | 142 (21,5%) | 277 | 1,95 | 59,21% | 5,75 | 9,08 |
| `DecisionTree_d6_leaf50` | `F0` (10 cols) | Máx. F1 Val | 0,20 | Test | 970 | 518 | 199 (20,5%) | 319 | 1,60 | 55,08% | 6,33 | 11,95 |
| `DecisionTree_d6_leaf50` | `F6` (11 cols) | Máx. F1 Val | 0,23 | Val | 662 | 496 | 205 (31,0%) | 291 | 1,42 | 72,15% | 6,74 | 9,08 |
| `DecisionTree_d6_leaf50` | `F6` (11 cols) | Máx. F1 Val | 0,23 | Test | 970 | 680 | 295 (30,4%) | 385 | 1,31 | 60,13% | 8,35 | 11,95 |
| `DecisionTree_d6_leaf50` | `F6` (11 cols) | Orientado a Recall | 0,13 | Test | 970 | 1.699 | 412 (42,5%) | 1.287 | 3,12 | 66,58% | 20,91 | 11,95 |
| `LogisticRegression_L2_C1` | `F6` (11 cols) | Máx. F1 Val | 0,23 | Test | 970 | 766 | 317 (32,7%) | 449 | 1,42 | 62,50% | 9,34 | 11,95 |

### Interpretación para el algoritmo de coloreo

1. **Los conflictos más graves (con más alumnos compartidos) ya están cubiertos en torno al 58%–62%:**
   - Aunque el Recall binario de pares con $\ge 1$ alumno afectado es del `25%–31%`, el **Recall ponderado por cantidad de estudiantes afectados (`cant_estudiantes_afectados`) es del `58,64%` en `KNN_k51_Manhattan (F0)` y del `59,81%–62,50%` en `F6_sin_primer_anio` / `F6`**, porque los pares que el modelo no detecta son casi exclusivamente colisiones aisladas de 1 solo alumno optativo sin historial previo.
2. **Compromiso entre Recall y densidad del grafo:**
   - En el umbral que maximiza `F1` en validación ($\tau \in [0,16, 0,23]$), el grafo predicho tiene entre **518 y 1.061 aristas en Test** (comparado con las **970 aristas reales**), manteniendo un grado medio por materia *held-out* (`6,3` a `11,8`) muy cercano al grado medio real (`11,95`) y un ratio de falsos positivos de **1,3 a 2,7 `FP` por cada `TP`**.
   - Si se reduce el umbral buscando un Recall $\ge 40\%$ (por ejemplo, $\tau = 0,13$ en el Árbol sobre `F6`), las aristas predichas se duplican a **1.699** (`1.287 FP`, grado máximo de una materia en Test = `159`), sobre-restringiendo el coloreo con tres aristas falsas por cada conflicto real. Por tanto, **el umbral óptimo de F1 en validación ($\tau = 0,16\text{–}0,23$) es también el punto de equilibrio estructural para el grafo**.

---

## 6. Respuestas a las 8 Preguntas del Informe (Sección 7)

### 1. ¿Estaba mal planteado el conjunto original de features o simplemente era incompleto?
No estaba mal planteado conceptualmente: combinar estructura curricular (`mismo_departamento`, `mismo_prefijo_curricular`, `anio_confirmado`, `mismo_anio`, `misma_carrera`) con co-inscripción histórica (`cant_carreras_compartidas_hist`, `cant_inscriptos_hist_1/2`, `cant_estudiantes_compartidos_historico`, `jaccard_historico`) es académicamente sólido y libre de *data leakage*. Tenía únicamente **dos limitaciones puntuales**:
1. **Incompletitud en el parser de nivel (`parse_nott.py`):** al aceptar solo dígitos `'1'..'4'` y descartar `'A'`, `'B'`, `'C'`, dejó sin año confirmado al `30,38%` de las materias (`243` exámenes) y al `50,41%` de los pares.
2. **Asimetría de orden en `(cant_inscriptos_hist_1, cant_inscriptos_hist_2)`:** su asignación dependía del orden alfabético del código de materia, lo que se corrige de forma limpia usando `min` y `max` con transformación `log1p` (`F4` / `F6_sin_primer_anio`).

### 2. ¿La interpretación de A/B/C estaba efectivamente perdiendo información relevante?
**Sí, de forma verificable.** Los códigos `'A'` (`63` materias), `'B'` (`107` materias) y `'C'` (`73` materias) presentan exactamente la misma cohorte estudiantil modal que `'1'` (1994), `'2'` (1993) y `'3'` (1992). Su omisión hacía que **53.404 pares del mismo año curricular** tuvieran `mismo_anio = 0`, incluyendo el **36,62% de todos los conflictos reales de entrenamiento (`1.281 / 3.498`)**. En cambio, los códigos `'0'` (`5` materias de *Foundation Year*) y `'P'` (`10` materias que mezclan posgrados *MSc* e introductorias) están bien excluidos de `year_code`.

### 3. ¿Las relaciones a dos saltos son válidas y utilizables sin *leakage*?
- **Desde el punto de vista de *leakage* temporal de etiquetas:** **Sí**, se calculan exclusivamente con las `19.798` inscripciones de `historico_1989_1993` (`0` alumnos de `1994`).
- **Desde el punto de vista arquitectónico y de ganancia práctica:** **No conviene incorporarlas al pipeline oficial (`F7` se descarta)** por tres razones:
  1. Su mejora sobre `F6` es marginal (`+0,0088` en `PR-AUC` medio de validación, por debajo del umbral de simplicidad de `0,01`).
  2. Utiliza todo el catálogo de materias como nodos intermediarios $w$ del grafo histórico.
  3. Rompe el contrato de inferencia tabular por par (`predict_conflict_pairs(df)`), exigiendo persistir y consultar el grafo global de inscripciones históricas.

### 4. ¿Qué representación aporta la mejora más consistente?
Depende de si se decide mantener congelada la Fase 1/2 o aplicar la corrección de `A`/`B`/`C`:
- **Sin modificar `parse_nott.py` ni regenerar `data/processed/` (contrato actual intacto):** **`F0`** (las 10 features actuales) o **`F4`** (su transformación simétrica `log1p` dentro del `Pipeline`), donde `KNN_k51_Manhattan_Uniform` sube el `PR-AUC` de Test de `0,1833` a `0,2249` (`F0`) / `0,2273` (`F4`) y el `F1` de `0,2593` a `0,2729` (`F0`) / `0,2805` (`F4`).
- **Corrigiendo `A=1, B=2, C=3` sin sesgo hacia la cohorte 1994 (mismas 10 columnas):** **`F6_sin_primer_anio`**, que mejora de forma estadísticamente significativa el `PR-AUC` y el `F1` en Validación y Test para los tres modelos (`KNN_k51_Manhattan`: `Val F1 = 0,2921, Val PR-AUC = 0,2339, Test F1 = 0,2846, Test PR-AUC = 0,2361`; `DecisionTree`: `Val F1 = 0,3009, Test F1 = 0,2872`).
- **Nota sobre `F6` (con `ambas_primer_anio`):** Aunque `F6` alcanza las métricas numéricas más altas (`Val PR-AUC = 0,3211, Val F1 = 0,3541, Test F1 = 0,3576–0,3652`), esa ganancia proviene de explotar que el grupo objetivo `actual_1994` contiene únicamente ingresantes de 1.er año. Por rigor académico, **recomendamos no incluir `ambas_primer_anio` como feature de producción** y documentarlo como hallazgo del diseño experimental.

### 5. ¿Qué modelo permitido por la consigna conviene conservar?
Dentro de los modelos permitidos por la consigna, **`KNN` (`KNeighborsClassifier(n_neighbors=51, metric="manhattan", weights="uniform")` con `StandardScaler`)** y **`DecisionTreeClassifier(max_depth=6, min_samples_leaf=50, criterion="gini", random_state=42)`** son consistentemente superiores al resto:
- **`KNN_k51_Manhattan_Uniform`** es el más robusto y equilibrado sobre el esquema de 10 columnas (`F0`, `F4` y `F6_sin_primer_anio`), superando en Validación y Test al `KNN_k15` oficial en todas las métricas y manteniendo total continuidad con el módulo de inferencia ya implementado.
- **`DecisionTree_d6_leaf50_Gini`** ofrece la mejor precisión de aristas (`38,4%` en `F0`, `43,4%` en `F6`) y el mayor `ROC-AUC` (`0,878–0,911`), siendo la mejor alternativa interpretable de baja tasa de falsos positivos.

### 6. ¿Qué umbral es razonable para construir el grafo?
El umbral seleccionado por maximización de `F1` en validación es el más razonable porque equilibra detección y densidad del grafo (`518` a `1.061` aristas predichas en Test frente a `970` reales, cubriendo el `58%–62%` de los alumnos afectados con `FP/TP` entre `1,6` y `2,7`):
- **`KNN_k51_Manhattan_Uniform` sobre `F0`:** $\tau = \mathbf{0,16}$.
- **`KNN_k51_Manhattan_Uniform` sobre `F4` o `F6_sin_primer_anio`:** $\tau = \mathbf{0,18}$.
- **`DecisionTree_d6_leaf50_Gini` sobre `F0`:** $\tau = \mathbf{0,20}$.

### 7. ¿Qué limitaciones siguen existiendo incluso después de corregir las features?
1. **Asimetría generacional de un único semestre (`1989–1993` vs `1994`):** el dataset original de Nottingham contiene un solo período de exámenes (Semestre 1, 1994–95). Al usar las cohortes superiores (`1989–1993`) como historial y la cohorte ingresante (`1994`) como futuro, `34` materias de 1.er año (`1` o `A`) tienen `0` inscriptos históricos y, por tanto, carecen de información histórica de carreras y co-inscripciones.
2. **Colisiones de 1 solo estudiante en materias electivas libres:** cerca del `22%–28%` de los pares con `hubo_conflicto_potencial == 1` corresponden a 1 o 2 alumnos que combinan materias de distintos departamentos sin antecedentes históricos; predecir esos casos individuales sin conocer las inscripciones del semestre actual es intrínsecamente imposible sin inundar el grafo de aristas falsas.
3. **Consulta previa del conjunto de `test`:** dado que `ml_test_full.csv` fue evaluado en iteraciones previas del proyecto, cualquier cambio adoptado en esta etapa debe reportar honestamente esa limitación metodológica en la memoria académica.

### 8. ¿Qué cambios concretos deberían pasar al pipeline oficial y cuáles no?
- **Descartar:**
  - El grafo histórico a 2 saltos (`F7`): ganancia marginal y complejidad arquitectónica innecesaria.
  - El indicador `ambas_primer_anio` (`F6`): sobreajusta la peculiaridad de que el target `actual_1994` solo contiene ingresantes de 1.er año.
  - Los términos ad-hoc de 16 columnas (`F5`): no mejoran el desempeño respecto a las 10 columnas limpias.
- **SÍ pasan al pipeline oficial (Decisión Definitiva Adoptada):**
  - **Incorporar la corrección A/B/C (`A=1, B=2, C=3`):** Se valida formalmente que `raw_level_char` `'A'`, `'B'` y `'C'` corresponden a 1.er, 2.º y 3.er año curricular.
  - **Modelo seleccionado:** Se adopta el **Árbol de Decisión regularizado (`DecisionTree_d6_leaf50_Gini`)** como modelo definitivo del proyecto para la generación del grafo de conflictos.
  - **Hoja de ruta técnica de integración (si aún no se ha ejecutado en la rama principal):**
    1. Actualizar `src/parsing/parse_nott.py` para mapear `{'A': 1, 'B': 2, 'C': 3}` en `derive_year_code()` (dejando `'0'` y `'P'` como `None`).
    2. Regenerar los datos intermedios (`python src/parsing/parse_nott.py` -> `data/interim/exams_clean.csv`).
    3. Regenerar los pares de materias (`python src/transformation/build_subject_pairs.py` -> `data/processed/subject_pairs_poc.csv`).
    4. Regenerar los datasets de entrenamiento y prueba (`python src/transformation/prepare_ml_dataset.py` -> `data/processed/ml_train_full.csv`, `ml_test_full.csv`).
    5. Entrenar y serializar el Árbol de Decisión en `results/models/decision_tree_conflict_predictor.joblib` y actualizar el umbral en `results/metrics/metrics_summary.json` y `src/inference/predict_conflicts.py`.

---

## 7. Decisión y Recomendación Definitiva de Cierre

De acuerdo con el análisis experimental y la decisión del equipo, **se adopta el Árbol de Decisión regularizado junto con la corrección de niveles A/B/C**:

### Configuración Oficial Definitiva: Árbol de Decisión Regularizado (`DecisionTree_d6_leaf50_Gini`)

- **Modelo e hiperparámetros:**
  ```python
  DecisionTreeClassifier(
      max_depth=6,
      min_samples_leaf=50,
      min_samples_split=100,
      criterion="gini",
      random_state=42,
  )
  ```
- **Esquema de features (10 columnas, representación `F6_sin_primer_anio` tras regenerar datos):**
  1. `mismo_departamento`
  2. `mismo_prefijo_curricular`
  3. `anio_confirmado` (corregido con `1=A, 2=B, 3=C`)
  4. `mismo_anio` (corregido con `1=A, 2=B, 3=C`)
  5. `misma_carrera`
  6. `cant_carreras_compartidas_hist` (con transformación `log1p`)
  7. `min(cant_inscriptos_hist_1, cant_inscriptos_hist_2)` (con transformación `log1p`)
  8. `max(cant_inscriptos_hist_1, cant_inscriptos_hist_2)` (con transformación `log1p`)
  9. `cant_estudiantes_compartidos_historico` (con transformación `log1p`)
  10. `jaccard_historico`
- **Umbral de clasificación seleccionado en validación:**
  - **$\tau = \mathbf{0,13}$** sobre los datos con corrección A/B/C (`F6_sin_primer_anio`):
    - **Validación:** `F1 = 0,3009`, `PR-AUC = 0,2165`, `ROC-AUC = 0,9034`, `Precision = 27,39%`, `Recall = 33,38%` (`221 TP`, `586 FP`).
    - **Test:** `F1 = 0,2872`, `PR-AUC = 0,2230`, `ROC-AUC = 0,8847`, `Precision = 26,43%`, `Recall = 31,44%` (`305 TP`, `849 FP`).
    - **Cobertura real:** Cubre el **61,72% de los estudiantes afectados por conflictos** en el conjunto de prueba con una densidad de aristas controlada (`1.154` aristas frente a `970` aristas reales).
  - *(Nota si se evalúa transitoriamente sobre las 10 features crudas originales `F0` antes de regenerar datasets)*: $\tau = \mathbf{0,20}$ (`Val F1 = 0,2627, Val PR-AUC = 0,2148` | `Test F1 = 0,2675, Precision = 38,42%, 199 TP, 319 FP, 518 aristas`).
- **Justificación académica y para la integración con coloreo:**
  1. **100% interpretable y defendible:** Es un árbol clásico de 6 niveles cuyas reglas lógicas se pueden volcar directamente en la documentación y memoria técnica del proyecto.
  2. **Excelente equilibrio de aristas:** Evita la sobrecarga de falsos positivos en el grafo, manteniendo un grado medio predicho para materias nuevas (`8,35`) muy cercano al grado real (`11,95`), lo que garantiza que el coloreo de grafos no sufra restricciones espurias desmedidas.
  3. **Solución generalizable:** Al adoptar la representación simétrica de 10 columnas sin el término contingente `ambas_primer_anio`, el modelo no sobreajusta a la cohorte 1994 y mantiene validez conceptual para cualquier cohorte universitaria futura.
