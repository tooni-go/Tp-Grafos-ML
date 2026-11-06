# Fase 2.5 — Preparación del Dataset para Machine Learning

**Proyecto:** Planificación de Mesas de Examen Universitarias  
**Dataset Base:** `data/processed/subject_pairs_poc.csv` (319.600 pares de 800 materias)  
**Script de Implementación:** [`src/transformation/prepare_ml_dataset.py`](file:///w:/home/anton/Tp-Grafos-ML/src/transformation/prepare_ml_dataset.py)  
**Semilla Aleatoria Fija:** `RANDOM_SEED = 42`

---

## 1. Problema del Desbalance de Clases

En el universo completo de $\binom{800}{2} = 319.600$ pares posibles de materias generado en la Fase 2 (`data/processed/subject_pairs_poc.csv`), la distribución de la variable objetivo `hubo_conflicto_potencial` (calculada sobre la cohorte 1994) presenta un **desbalance severo de clases**:

* **Pares positivos (`hubo_conflicto_potencial == True`):** **4.468 pares (1,398%)**
* **Pares negativos (`hubo_conflicto_potencial == False`):** **315.132 pares (98,602%)**
* **Relación de desbalance:** aproximadamente **1 positivo por cada 70,5 negativos** (`1 : 70,5`).

### Consecuencias si no se trata adecuadamente
1. **Colapso hacia la clase mayoritaria:** Un clasificador trivial que prediga siempre `False` (sin conflicto) alcanzaría un **98,60% de *Accuracy***, pero tendría **Recall = 0%** (no detectaría ninguna arista del grafo de conflictos y asignaría todas las materias al mismo turno).
2. **Sesgo en Regresión Lineal:** Al predecir `cant_estudiantes_afectados` sobre un conjunto donde el 98,6% de las filas vale `0`, el ajuste por mínimos cuadrados ordinarios se aplasta hacia cero ($\hat{y} \approx 0$).
3. **Asimetría en K-Nearest Neighbors (KNN):** En un vecindario de tamaño $k$ con prevalencia del 1,4%, los ejemplos negativos dominan localmente el voto por mayoría salvo en regiones de densidad extrema, además de encarecer el cómputo de distancias ($O(N \cdot d)$ sobre más de 250.000 filas).

---

## 2. Evaluación de Alternativas

Se analizaron cuatro estrategias considerando los modelos exigidos por la cátedra (**Regresión Lineal / Logística, KNN y Árboles de Decisión**):

| Estrategia | Ventajas | Limitaciones | Adecuación al TP |
| :--- | :--- | :--- | :---: |
| **1. Undersampling aleatorio 50/50 (solo en Train)** | Funciona de manera uniforme en **todos** los algoritmos (incluyendo KNN y Regresión, que no tienen `class_weight`). Reduce el set de entrenamiento a 6.996 filas, permitiendo validación cruzada y ajuste de hiperparámetros instantáneos. | Al entrenar con prevalencia 50/50 y evaluar en el conjunto de prueba real (1,49% positivos), las probabilidades predichas se desplazan hacia arriba, aumentando los Falsos Positivos si se usa el umbral por defecto de 0,5. | **Alta (Estrategia Principal para entrenamiento)** |
| **2. Dataset completo + ajuste de umbral de decisión** | Conserva los 254.541 pares de entrenamiento y mantiene las probabilidades calibradas a la prevalencia real. | Lento en KNN ($254.541 \times 65.059$ distancias); en Regresión Lineal el 98,6% de ceros distorsiona la pendiente. | **Media (Útil como línea base comparativa)** |
| **3. Ponderación de clases (`class_weight='balanced'`)** | Soportado nativamente en `LogisticRegression` y `DecisionTreeClassifier` de `scikit-learn` sin descartar filas negativas. | **No existe `class_weight` en KNN** (`KNeighborsClassifier` solo pondera por distancia, no por prevalencia de clase) ni en Regresión Lineal estándar, rompiendo la comparabilidad directa bajo un mismo esquema. | **Media (Complementaria sobre `ml_train_full.csv`)** |
| **4. Sobremuestreo sintético (SMOTE)** | Aumenta la clase minoritaria sin descartar negativos. | Genera más de 500.000 filas e inventa puntos interpolados sobre variables booleanas discretas (`mismo_anio`, `mismo_departamento`), produciendo artefactos irreales. | **Baja (Descartada)** |

### Estrategia Seleccionada y Justificación
Se eligió como estrategia principal el **Undersampling Aleatorio 50/50 aplicado exclusivamente sobre el conjunto de entrenamiento (`ml_train_balanced.csv`, 6.996 filas)**, acompañado de la exportación del **conjunto de entrenamiento completo (`ml_train_full.csv`, 254.541 filas)** y del **conjunto de prueba intacto con la distribución original (`ml_test_full.csv`, 65.059 filas)**.

**Justificación técnica:**
1. **Comparabilidad justa entre modelos:** Permite entrenar Regresión, KNN y Árboles de Decisión exactamente sobre la misma matriz de entrenamiento balanceada (`6.996` filas), sin que KNN quede en desventaja estructural por carecer de `class_weight`.
2. **Agilidad experimental:** Con 6.996 registros, cualquier búsqueda de hiperparámetros (`GridSearchCV` para $k$ en KNN o `max_depth` en Árboles) tarda menos de 1 segundo.
3. **Evaluación honesta y sin trampas:** Al mantener intacto `ml_test_full.csv` con la distribución real de clases (**1,491% positivos / 98,509% negativos**), todos los modelos se evalúan sobre el escenario real del problema. Además, al entregar también `ml_train_full.csv`, el encargado de ML puede comparar en el informe qué ocurre al entrenar un Árbol o Regresión Logística en el set balanceado vs. el set completo con `class_weight='balanced'`.

---

## 3. Procedimiento de Separación Entrenamiento / Prueba

### ¿Por qué una partición aleatoria convencional de filas puede producir fuga de información?
En este problema, cada fila es un **par de materias** $(i, j)$ (una díada o arista potencial de un grafo de 800 nodos). Cada materia individual $i$ aparece en **799 filas**.
* Si se realizara un `train_test_split` aleatorio convencional sobre las 319.600 filas (80% / 20%), **las 800 materias aparecerían tanto en entrenamiento (~640 veces cada una) como en prueba (~160 veces cada una)**.
* Dado que el dataset incluye atributos marginales de cada materia (`cant_inscriptos_hist_1`, `cant_inscriptos_hist_2`), un Árbol de Decisión profundo o KNN podría **memorizar la identidad implícita de materias específicas** (por ejemplo, identificar por su tamaño exacto de inscripción histórica qué materia tuvo un salto de inscriptos en 1994) y transferir esa memorización entre el par $(i, j) \in \text{Train}$ y el par $(i, k) \in \text{Test}$.

### Estrategia de Partición Implementada: Retención Estratificada por Materias (Node-Holdout Split)
Para evitar una evaluación artificial sin descartar ningún par del dataset:
1. Se agrupan las **800 materias** según su carácter de nivel curricular (`raw_level_char` $\in \{\text{'0', '1', '2', '3', '4', 'A', 'B', 'C', 'P'}\}$ en `data/interim/exams_clean.csv`).
2. Utilizando la semilla fija `RANDOM_SEED = 42`, se selecciona de forma estratificada el **~10,75% de las materias de cada nivel**, obteniendo **86 materias de prueba (`test_subjects`)** y **714 materias de entrenamiento (`train_subjects`)**.
3. **Regla de asignación de pares:**
   * **Conjunto de Entrenamiento (`ml_train_full.csv`):** Incluye únicamente los pares $(i, j)$ donde **ambas materias** pertenecen a `train_subjects`:
     $$\binom{714}{2} = 254.541 \text{ pares (79,64\% del total)}$$
   * **Conjunto de Prueba (`ml_test_full.csv`):** Incluye todos los pares $(i, j)$ donde **al menos una de las dos materias** pertenece a `test_subjects`:
     $$319.600 - 254.541 = 65.059 \text{ pares (20,36\% del total)}$$
4. **Propiedades garantizadas:**
   * **0 pares descartados:** $254.541 + 65.059 = 319.600$.
   * **0 solapamiento entre entrenamiento y prueba.**
   * **Evaluación inductiva genuina:** Todo par del conjunto de prueba involucra al menos una materia que **jamás fue vista en ningún par durante el entrenamiento**, obligando al modelo a aprender reglas relacionales generalizables entre materias.
   * **Preservación de la prevalencia por estratificación:** La proporción de conflictos positivos es consistente entre entrenamiento (**1,374%**) y prueba (**1,491%**).

---

## 4. Selección de Variables y Prevención de *Data Leakage*

El objetivo del modelo es predecir los conflictos de una **cohorte nueva (1994)** cuya inscripción individual aún no conocemos al momento de predecir. Por ello, se auditaron las 24 columnas de `subject_pairs_poc.csv` y se clasificaron en cuatro grupos estrictos:

### 4.1. Variables EXCLUIDAS por riesgo de *Data Leakage* (Derivadas de la cohorte 1994)
Estas columnas fueron eliminadas de los archivos de ML para impedir que se utilicen accidentalmente como predictores:
* `cant_inscriptos_actual_1`: cantidad de alumnos de la cohorte 1994 en la materia 1. *(Usarla revelaría, por ejemplo, qué materias tienen 0 alumnos en 1994, haciendo trivial predecir que no tienen conflicto).*
* `cant_inscriptos_actual_2`: cantidad de alumnos de la cohorte 1994 en la materia 2.
* `tasa_afectados_actual`: es igual a `cant_estudiantes_afectados / 3086` (transformación directa de la etiqueta objetivo).

### 4.2. Variables EXCLUIDAS por ser Identificadores, Texto o Metadatos Crudos
* `materia_id_1`, `materia_id_2`: identificadores alfanuméricos de los nodos.
* `materia_nombre_1`, `materia_nombre_2`: nombres descriptivos en texto libre.
* `departamento_1`, `departamento_2`: códigos categóricos nominales (46 valores); su relación ya está capturada en el booleano `mismo_departamento`.
* `anio_1`, `anio_2`: contienen valores vacíos (`null`) en los 258 exámenes con código especial (`A, B, C, P, 0`); su información limpia y sin nulos ya está capturada en `anio_confirmado` y `mismo_anio`.

### 4.3. Variables Predictoras Incluidas (11 Features limpias — Cohortes 1989–1993 y Catálogo)
Todas las variables predictoras incluidas en los archivos de ML conservan sus nombres originales y están libres de valores nulos:

| # | Columna Predictora | Tipo | Origen | Observación Técnica / Colinealidad |
| :---: | :--- | :--- | :--- | :--- |
| 1 | `mismo_departamento` | Booleano | Catálogo `exams` | `True` si ambas materias pertenecen al mismo departamento (46 deptos). |
| 2 | `mismo_prefijo_curricular` | Booleano | Catálogo `exams` | `True` si coinciden los primeros 2 caracteres del código (`materia_id[:2]`). |
| 3 | `anio_confirmado` | Booleano | Catálogo `exams` | `True` si ambas materias tienen año numérico confirmado (`1..4`). |
| 4 | `mismo_anio` | Booleano | Catálogo `exams` | `True` si `anio_confirmado` es `True` y coinciden en año (`1..4`). |
| 5 | `misma_carrera` | Booleano | Cohortes 1989–1993 | `True` si comparten $\ge 1$ carrera (`course_code`) en el grupo histórico. |
| 6 | `cant_carreras_compartidas_hist` | Entero $\ge 0$ | Cohortes 1989–1993 | Cantidad de carreras compartidas en el grupo histórico. |
| 7 | `cant_inscriptos_hist_1` | Entero $\ge 0$ | Cohortes 1989–1993 | Total de inscriptos históricos en la materia 1. |
| 8 | `cant_inscriptos_hist_2` | Entero $\ge 0$ | Cohortes 1989–1993 | Total de inscriptos históricos en la materia 2. |
| 9 | `cant_estudiantes_compartidos_historico` | Entero $\ge 0$ | Cohortes 1989–1993 | Alumnos de 1989–1993 que rindieron ambas materias. |
| 10 | `tasa_compartidos_historico` | Float $[0,1]$ | Cohortes 1989–1993 | Igual a `cant_estudiantes_compartidos_historico / 4810`. **Colineal con la #9** (ver nota abajo). |
| 11 | `jaccard_historico` | Float $[0,1]$ | Cohortes 1989–1993 | Intersección sobre unión en 1989–1993: $\frac{|E_1 \cap E_2|}{|E_1 \cup E_2|}$. |

> **Advertencia sobre colinealidad para Regresión y KNN:**  
> La columna `tasa_compartidos_historico` es un múltiplo escalar exacto ($\times \frac{1}{4810}$) de `cant_estudiantes_compartidos_historico`. Mientras que en Árboles de Decisión esto no afecta las particiones, en **Regresión Lineal / Logística** introduce colinealidad perfecta y en **KNN** (tras estandarizar con `StandardScaler`) duplica el peso de esa variable en la distancia euclídea. Por ello, en `prepare_ml_dataset.py` se dejó definida la constante `PREDICTOR_COLUMNS_NO_COLLINEAR` (que omite `tasa_compartidos_historico` y conserva `cant_estudiantes_compartidos_historico` y `jaccard_historico`).

### 4.4. Variables Objetivo Incluidas (Targets — Cohorte 1994)
* `hubo_conflicto_potencial` (Booleano): **Variable objetivo para Clasificación.**
* `cant_estudiantes_afectados` (Entero $\ge 0$): **Variable objetivo para Regresión.**
* **Regla crítica de uso:** Al entrenar un modelo de **Clasificación** (`y = df['hubo_conflicto_potencial']`), se debe excluir `cant_estudiantes_afectados` de `X`. Al entrenar un modelo de **Regresión** (`y = df['cant_estudiantes_afectados']`), se debe excluir `hubo_conflicto_potencial` de `X`.

---

## 5. Archivos Generados y Estadísticas Antes y Después

Todos los archivos fueron generados por [`src/transformation/prepare_ml_dataset.py`](file:///w:/home/anton/Tp-Grafos-ML/src/transformation/prepare_ml_dataset.py) con semilla `RANDOM_SEED = 42`, verificando **0 duplicados** y **0 solapamiento de pares** entre entrenamiento y prueba:

| Archivo | Ruta | Filas Totales | Positivos (`True`) | Negativos (`False`) | % Positivos | % Negativos | Finalidad |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| **Dataset PoC (Intacto)** | `data/processed/subject_pairs_poc.csv` | 319.600 | 4.468 | 315.132 | 1,398% | 98,602% | Universo completo de pares y columnas de auditoría (Fase 2). |
| **Entrenamiento Balanceado** | `data/processed/ml_train_balanced.csv` | **6.996** | **3.498** | **3.498** | **50,00%** | **50,00%** | **Archivo principal de entrenamiento** (undersampling 50/50 sobre Train). |
| **Prueba Distribución Real** | `data/processed/ml_test_full.csv` | **65.059** | **970** | **64.089** | **1,491%** | **98,509%** | **Archivo único de evaluación (Test)** con distribución original de clases. |
| **Entrenamiento Completo** | `data/processed/ml_train_full.csv` | **254.541** | **3.498** | **251.043** | **1,374%** | **98,626%** | Entrenamiento sin submuestrear (opcional para comparar `class_weight`). |

---

## 6. Limitaciones Metodológicas a Tener en Cuenta en la Evaluación de Modelos

El integrante encargado de entrenar y evaluar los modelos de Machine Learning debe tener presentes cinco consideraciones metodológicas clave:

1. **No evaluar jamás sobre un conjunto balanceado artificialmente:** Todas las métricas de test deben reportarse sobre `ml_test_full.csv` (distribución real 1,491% positivos).
2. **Insuficiencia del *Accuracy* en `ml_test_full.csv`:** Como el 98,51% de `ml_test_full.csv` es negativo, el *Accuracy* es engañoso. Deben priorizarse **Precision, Recall (Sensibilidad), F1-Score de la clase positiva, PR-AUC (Área bajo la curva Precision-Recall) y ROC-AUC**, además de la matriz de confusión.
3. **Efecto del entrenamiento 50/50 sobre la precisión en Test (Desplazamiento de Probabilidad Previa):** Un modelo entrenado en `ml_train_balanced.csv` asume una probabilidad a priori del 50%. Al predecir sobre `ml_test_full.csv` (donde la probabilidad a priori real es 1,49%), producirá un **Recall alto** (detectará la mayoría de los conflictos reales), pero con un umbral de probabilidad estándar de $0,5$ generará bastantes **Falsos Positivos** (baja Precision). En el contexto de coloreo de grafos, **elevar el umbral de decisión** (por ejemplo a $0,7$ o $0,8$, tal como pide el selector de umbral del frontend en el enunciado) permite controlar el equilibrio entre conflictos evitados y cantidad de turnos (colores) requeridos.
4. **Brecha entre materias de 1.er año y años superiores (Aproximación por Cohortes):** Dado que `historico_1989_1993` contiene alumnos ingresantes entre 1989 y 1993 (que en el semestre 1994–95 cursan mayoritariamente 2.do y 3.er año) y `actual_1994` contiene ingresantes 1994 (que cursan mayoritariamente 1.er año), existen pares de materias de 1.er año donde `cant_estudiantes_compartidos_historico == 0` pero `hubo_conflicto_potencial == True`. En esos casos, el modelo depende de las variables estructurales (`mismo_departamento`, `mismo_prefijo_curricular`, `mismo_anio`, `misma_carrera`) para predecir el conflicto.
5. **Estandarización para KNN y Regresión:** Antes de entrenar KNN o Regresión Lineal/Logística, se deben escalar las variables numéricas de conteo (`cant_inscriptos_hist_1`, `cant_inscriptos_hist_2`, `cant_estudiantes_compartidos_historico`, `cant_carreras_compartidas_hist`) ajustando el `StandardScaler` **únicamente sobre el conjunto de entrenamiento** (`fit_transform` en Train, `transform` en Test).
