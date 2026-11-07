# Fase 3 — Entrenamiento y Evaluación de Modelos de Clasificación

**Proyecto:** Planificación de Mesas de Examen Universitarias  
**Alcance de esta fase:** Clasificación binaria (`hubo_conflicto_potencial`)  
**Script Ejecutable:** [`src/models/compare_models.py`](file:///w:/home/anton/Tp-Grafos-ML/src/models/compare_models.py)  
**Semilla Aleatoria Fija:** `RANDOM_SEED = 42`

---

## 1. Objetivo de Clasificación

El objetivo de la Fase 3 es entrenar, validar y comparar clasificadores supervisados capaces de predecir si un par de materias $(u, v)$ tendrá **conflicto potencial de demanda** (`hubo_conflicto_potencial == 1`) en la cohorte entrante (`1994`), utilizando exclusivamente características curriculares de las materias e información de inscripciones de las cohortes anteriores (`1989–1993`).

En términos del problema de negocio de la secretaría académica:
* El proyecto adopta **coloreo clásico de grafos no ponderado**. La probabilidad predicha $\hat{p}_{uv}$ se utiliza únicamente para decidir si se agrega o no una arista de incompatibilidad $\{u, v\}$ en función del umbral ($\hat{p}_{uv} \ge \tau$), y el algoritmo de coloreo greedy opera exclusivamente sobre la estructura de adyacencia resultante, sin utilizar pesos en las aristas.

---

## 2. Dataset Utilizado y Auditoría Previa

Se utilizaron sin modificaciones los tres archivos generados en la Fase 2.5 en `data/processed/`:

| Archivo | Registros | Positivos (`1`) | Negativos (`0`) | Prevalencia Positiva | Materias Únicas | Rol en la Fase 3 |
| :--- | ---: | ---: | ---: | ---: | :---: | :--- |
| [`ml_train_balanced.csv`](file:///w:/home/anton/Tp-Grafos-ML/data/processed/ml_train_balanced.csv) | **6.996** | 3.498 | 3.498 | **50,00%** | 714 | Conjunto principal de entrenamiento balanceado (50/50). |
| [`ml_train_full.csv`](file:///w:/home/anton/Tp-Grafos-ML/data/processed/ml_train_full.csv) | **254.541** | 3.498 | 251.043 | **1,374%** | 714 | Conjunto de entrenamiento completo (distribución natural) y fuente de la validación interna. |
| [`ml_test_full.csv`](file:///w:/home/anton/Tp-Grafos-ML/data/processed/ml_test_full.csv) | **65.059** | 970 | 64.089 | **1,491%** | 800 (86 de test) | Conjunto de evaluación final (intacto hasta el final). |

### Nota de auditoría: ¿Por qué `ml_train_balanced.csv` tiene 6.996 filas y no ~8.800?
En el universo completo de `319.600` pares (`subject_pairs_poc.csv`) existen `4.468` pares positivos. Si se hubiera aplicado *undersampling* 50/50 sobre **todo** el dataset antes de separar entrenamiento y prueba, habrían resultado $4.468 \times 2 = 8.936$ filas (~8.800 filas), pero eso habría introducido una **evaluación artificial** (un conjunto de test falsamente balanceado 50/50). En cambio, en la Fase 2.5 se separó primero Train (`3.498` positivos) y Test (`970` positivos), y el *undersampling* se aplicó **exclusivamente sobre Train**, dando origen a las $3.498 \times 2 = \mathbf{6.996}$ filas de `ml_train_balanced.csv`.

---

## 3. Selección de Features y Target

* **Variable Objetivo (`y`):** `hubo_conflicto_potencial` (`1` si ambas materias comparten al menos un estudiante en la cohorte 1994; `0` en caso contrario).
* **Columnas Excluidas de `X`:**
  * `materia_id_1`, `materia_id_2`: conservadas únicamente como identificadores del par (`ID_COLUMNS`) para vincular las predicciones con los nodos del grafo en la Fase 4.
  * `cant_estudiantes_afectados`: variable objetivo de regresión (cohorte 1994); excluida para evitar *data leakage*.
  * `tasa_compartidos_historico`: excluida explícitamente porque es una transformación lineal exacta de `cant_estudiantes_compartidos_historico` ($\text{tasa} = \frac{\text{conteo}}{4810}$, correlación $r = 1,0000$). Tras aplicar `StandardScaler`, ambas producirían vectores idénticos, duplicando artificialmente el peso de esa variable en la distancia euclídea de KNN y dividiendo su coeficiente en Regresión Logística.
* **10 Variables Predictoras Seleccionadas (`FEATURE_COLUMNS`):**
  1. `mismo_departamento` (binaria `0/1`)
  2. `mismo_prefijo_curricular` (binaria `0/1`)
  3. `anio_confirmado` (binaria `0/1`: incluida porque `mismo_anio` vale `0` tanto cuando dos materias son de distinto año numérico confirmado como cuando alguna tiene código especial `A, B, C, P, 0`; `anio_confirmado` permite distinguir ambos casos).
  4. `mismo_anio` (binaria `0/1`)
  5. `misma_carrera` (binaria `0/1`, calculada sobre cohortes 1989–1993)
  6. `cant_carreras_compartidas_hist` (numérica $\ge 0$, cohortes 1989–1993)
  7. `cant_inscriptos_hist_1` (numérica $\ge 0$, cohortes 1989–1993)
  8. `cant_inscriptos_hist_2` (numérica $\ge 0$, cohortes 1989–1993)
  9. `cant_estudiantes_compartidos_historico` (numérica $\ge 0$, cohortes 1989–1993)
  10. `jaccard_historico` (continua $\in [0, 1]$, cohortes 1989–1993)

---

## 4. Estrategia de Separación Interna de Datos (Subtrain / Validación)

Para seleccionar umbrales y comparar regímenes sin tocar `ml_test_full.csv`, se construyó una partición interna a partir de `ml_train_full.csv` respetando la misma estrategia de **Retención Estratificada por Materias** de la Fase 2.5 (`RANDOM_SEED = 42`):

1. De las **714 materias** presentes en entrenamiento, se reservó estratificadamente por `raw_level_char` un subconjunto de **77 materias para Validación Interna (`val_subjects`)** y **637 materias para Sub-entrenamiento (`subtrain_subjects`)**.
2. **`subtrain_full`:** Pares de `ml_train_full.csv` donde **ambas** materias pertenecen a `subtrain_subjects` ($\binom{637}{2} = \mathbf{202.566}$ pares: **2.836 positivos [1,400%]** y **199.730 negativos [98,600%]**).
3. **`subtrain_balanced`:** *Undersampling* 50/50 sin reemplazo aplicado exclusivamente sobre `subtrain_full` (**5.672 pares**: **2.836 positivos [50,00%]** y **2.836 negativos [50,00%]**).
4. **`validation` (distribución natural):** Pares de `ml_train_full.csv` donde **al menos una** materia pertenece a `val_subjects` ($\mathbf{51.975}$ pares: **662 positivos [1,274%]** y **51.313 negativos [98,726%]**).

---

## 5. Preprocesamiento

* **Verificación de integridad:** Se comprobó programáticamente la ausencia total de valores nulos (`0`) e infinitos (`0`) en los tres conjuntos de datos.
* **Escalado (`StandardScaler`):**
  * Para **Logistic Regression** y **KNN**, se encapsuló `StandardScaler()` dentro de un `sklearn.pipeline.Pipeline`, garantizando que la media y desviación estándar se ajusten **exclusivamente sobre el conjunto de entrenamiento** de cada experimento (`subtrain` o `train`) y nunca sobre validación ni test.
  * Para **Decision Tree**, no se aplicó escalado dado que las particiones ortogonales por umbral sobre árboles de decisión son invariantes a transformaciones monótonas de escala.

---

## 6. Modelos Comparados

Se compararon **dos líneas base (baselines)** y **tres algoritmos clásicos** bajo dos regímenes de entrenamiento (balanceado 50/50 vs. distribución completa):

1. **Baselines:**
   * `Baseline_Mayoritario`: `DummyClassifier(strategy="prior")`, que predice siempre la clase mayoritaria (`0`).
   * `Baseline_Heuristico_Historico`: Regla determinística del dominio académico que predice conflicto en 1994 si y solo si el par ya tuvo estudiantes compartidos en 1989–1993 (`cant_estudiantes_compartidos_historico > 0`).
2. **Clasificadores entrenados sobre el conjunto balanceado 50/50 (`subtrain_balanced` / `ml_train_balanced.csv`):**
   * `DecisionTree_Balanced`: `DecisionTreeClassifier(max_depth=5, random_state=42)`.
   * `LogisticRegression_Balanced`: `Pipeline([StandardScaler(), LogisticRegression(max_iter=1000, random_state=42)])`.
   * `KNN_Balanced`: `Pipeline([StandardScaler(), KNeighborsClassifier(n_neighbors=15)])`.
3. **Clasificadores entrenados sobre la distribución original (`subtrain_full` / `ml_train_full.csv`):**
   * `DecisionTree_Full_Unweighted` y `DecisionTree_Full_ClassWeight` (`class_weight="balanced"`).
   * `LogisticRegression_Full_Unweighted` y `LogisticRegression_Full_ClassWeight` (`class_weight="balanced"`).
   * `KNN_Full_Unweighted` (`n_neighbors=15`).
4. **Selección del umbral de decisión en Validación Interna (`_UmbralVal`):**
   * El *undersampling* 50/50 y el hiperparámetro `class_weight="balanced"` no garantizan probabilidades calibradas en sentido estadístico, sino que modifican la distribución de clases o la ponderación de la función de pérdida durante el entrenamiento, desplazando los puntajes predichos hacia valores altos frente a la prevalencia natural (~1,37%). De manera inversa, los modelos no ponderados entrenados sobre la distribución original (`_Full_Unweighted`) concentran sus probabilidades estimadas en rangos bajos cercanos a la prevalencia base, por lo que el corte fijo $\tau = 0,50$ resulta excesivamente restrictivo para ellos y excesivamente permisivo para los modelos balanceados/ponderados.
   * Por este motivo —y completando la comparación para la totalidad de los modelos entrenables, incluidos los tres modelos `_Full_Unweighted` que inicialmente solo se habían evaluado en $\tau = 0,50$—, se buscó para cada modelo el umbral $\tau^*_{\text{val}} \in [0,01; 0,95]$ que **maximiza el F1-score exclusivamente sobre el conjunto de Validación Interna (`validation`, con prevalencia natural del 1,274% y sin tocar Test)**:
     * En los modelos entrenados con balance 50/50 o `class_weight="balanced"`, el umbral que maximiza F1 en validación se ubicó en $\tau^*_{\text{val}} = 0,90$ (`DecisionTree_Balanced`) y $\tau^*_{\text{val}} = 0,94$ (restantes).
     * En los modelos entrenados sin ponderación sobre la distribución natural (`_Full_Unweighted`), el umbral que maximiza F1 en validación se ubicó en $\tau^*_{\text{val}} = 0,20$ (`DecisionTree_Full_Unweighted`), $\tau^*_{\text{val}} = 0,19$ (`LogisticRegression_Full_Unweighted`) y $\tau^*_{\text{val}} = 0,21$ (`KNN_Full_Unweighted`).
   * Posteriormente, esos umbrales fijos seleccionados en validación se evaluaron directamente sobre `ml_test_full.csv` sin modificarlos a partir de los resultados de prueba.

---

## 7. Métricas y Resultados Obtenidos

### 7.1. Resultados en Validación Interna (`51.975` pares: `662` positivos [1,274%] y `51.313` negativos)

| Modelo / Régimen | Umbral | Precision | Recall | F1-Score | PR-AUC | TP | FP | FN | TN |
| :--- | :---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `Baseline_Mayoritario` | 0,50 | 0,0000 | 0,0000 | 0,0000 | 0,0127 | 0 | 0 | 662 | 51.313 |
| `Baseline_Heuristico_Historico` (`hist > 0`) | 0,50 | 0,1731 | 0,2432 | 0,2023 | 0,1025 | 161 | 769 | 501 | 50.544 |
| **`DecisionTree_Balanced`** | **0,50** | 0,0365 | **0,8867** | 0,0702 | 0,1350 | **587** | 15.475 | **75** | 35.838 |
| `DecisionTree_Balanced_UmbralVal` | 0,90 | 0,2735 | 0,1979 | 0,2296 | 0,1350 | 131 | 348 | 531 | 50.965 |
| **`LogisticRegression_Balanced`** | **0,50** | 0,0503 | **0,6888** | 0,0938 | 0,1484 | 456 | 8.606 | 206 | 42.707 |
| `LogisticRegression_Balanced_UmbralVal` | 0,94 | 0,2465 | 0,2145 | 0,2294 | 0,1484 | 142 | 434 | 520 | 50.879 |
| **`KNN_Balanced`** | **0,50** | 0,0502 | **0,7991** | 0,0945 | 0,0993 | 529 | 10.006 | 133 | 41.307 |
| `KNN_Balanced_UmbralVal` | 0,94 | 0,2120 | 0,1767 | 0,1928 | 0,0993 | 117 | 435 | 545 | 50.878 |
| `DecisionTree_Full_Unweighted` | 0,50 | **0,7640** | 0,1027 | 0,1811 | **0,1963** | 68 | 21 | 594 | 51.292 |
| **`DecisionTree_Full_Unweighted_UmbralVal`** | **0,20** | 0,3853 | 0,1979 | **0,2615** | **0,1963** | 131 | 209 | 531 | 51.104 |
| `DecisionTree_Full_ClassWeight` | 0,50 | 0,0448 | 0,7855 | 0,0847 | 0,1650 | 520 | 11.094 | 142 | 40.219 |
| `DecisionTree_Full_ClassWeight_UmbralVal` | 0,94 | 0,5517 | 0,1450 | 0,2297 | 0,1650 | 96 | 78 | 566 | 51.235 |
| `LogisticRegression_Full_Unweighted` | 0,50 | 0,5510 | 0,0408 | 0,0759 | 0,1451 | 27 | 22 | 635 | 51.291 |
| `LogisticRegression_Full_Unweighted_UmbralVal` | 0,19 | 0,3283 | 0,1647 | 0,2193 | 0,1451 | 109 | 223 | 553 | 51.090 |
| `LogisticRegression_Full_ClassWeight` | 0,50 | 0,0510 | 0,6858 | 0,0949 | 0,1555 | 454 | 8.447 | 208 | 42.866 |
| `LogisticRegression_Full_ClassWeight_UmbralVal` | 0,94 | 0,2541 | 0,2115 | 0,2308 | 0,1555 | 140 | 411 | 522 | 50.902 |
| `KNN_Full_Unweighted` | 0,50 | 0,6306 | 0,1057 | 0,1811 | 0,1846 | 70 | 41 | 592 | 51.272 |
| **`KNN_Full_Unweighted_UmbralVal`** | **0,21** | 0,2761 | 0,2115 | **0,2395** | 0,1846 | 140 | 367 | 522 | 50.946 |

---

### 7.2. Resultados Finales en Test (`ml_test_full.csv`: `65.059` pares: `970` positivos [1,491%] y `64.089` negativos)

| Modelo / Régimen | Umbral | Precision | Recall | F1-Score | PR-AUC | Accuracy | TP | FP | FN | TN |
| :--- | :---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `Baseline_Mayoritario` | 0,50 | 0,0000 | 0,0000 | 0,0000 | 0,0149 | 0,9851 | 0 | 0 | 970 | 64.089 |
| `Baseline_Heuristico_Historico` (`hist > 0`) | 0,50 | 0,1428 | 0,2082 | 0,1694 | 0,0750 | 0,9696 | 202 | 1.213 | 768 | 62.876 |
| **`DecisionTree_Balanced`** | **0,50** | 0,0456 | **0,8526** | 0,0866 | 0,1439 | 0,7318 | **827** | 17.303 | **143** | 46.786 |
| `DecisionTree_Balanced_UmbralVal` | 0,90 | 0,1431 | **0,3680** | 0,2061 | 0,1439 | 0,9577 | 357 | 2.138 | 613 | 61.951 |
| **`LogisticRegression_Balanced`** | **0,50** | 0,0636 | **0,7588** | 0,1173 | 0,1384 | 0,8297 | 736 | 10.843 | 234 | 53.246 |
| `LogisticRegression_Balanced_UmbralVal` | 0,94 | 0,2442 | 0,2485 | **0,2463** | 0,1384 | 0,9773 | 241 | 746 | 729 | 63.343 |
| **`KNN_Balanced`** | **0,50** | 0,0623 | **0,8134** | 0,1157 | 0,1183 | 0,8147 | 789 | 11.876 | 181 | 52.213 |
| `KNN_Balanced_UmbralVal` | 0,94 | 0,2141 | 0,2309 | 0,2222 | 0,1183 | 0,9759 | 224 | 822 | 746 | 63.267 |
| `DecisionTree_Full_Unweighted` | 0,50 | **0,9531** | 0,0629 | 0,1180 | **0,2082** | **0,9860** | 61 | **3** | 909 | 64.086 |
| **`DecisionTree_Full_Unweighted_UmbralVal`** | **0,20** | **0,4399** | 0,1546 | **0,2288** | **0,2082** | 0,9845 | 150 | 191 | 820 | 63.898 |
| `DecisionTree_Full_ClassWeight` | 0,50 | 0,0472 | **0,8598** | 0,0895 | 0,1767 | 0,7392 | **834** | 16.829 | **136** | 47.260 |
| `DecisionTree_Full_ClassWeight_UmbralVal` | 0,94 | 0,3956 | 0,1485 | 0,2159 | 0,1767 | 0,9839 | 144 | 220 | 826 | 63.869 |
| `LogisticRegression_Full_Unweighted` | 0,50 | 0,3333 | 0,0227 | 0,0425 | 0,1566 | 0,9848 | 22 | 44 | 948 | 64.045 |
| `LogisticRegression_Full_Unweighted_UmbralVal` | 0,19 | 0,3326 | 0,1649 | 0,2205 | 0,1566 | 0,9826 | 160 | 321 | 810 | 63.768 |
| `LogisticRegression_Full_ClassWeight` | 0,50 | 0,0642 | 0,7598 | 0,1183 | 0,1519 | 0,8312 | 737 | 10.750 | 233 | 53.339 |
| **`LogisticRegression_Full_ClassWeight_UmbralVal`** | **0,94** | 0,2646 | 0,2381 | **0,2507** | 0,1519 | 0,9788 | 231 | 642 | 739 | 63.447 |
| `KNN_Full_Unweighted` | 0,50 | 0,5597 | 0,0918 | 0,1577 | 0,1833 | 0,9854 | 89 | 70 | 881 | 64.019 |
| **`KNN_Full_Unweighted_UmbralVal`** | **0,21** | **0,2711** | **0,2485** | **0,2593** | 0,1833 | 0,9788 | 241 | 648 | 729 | 63.441 |

### Hallazgos clave de la comparación experimental
1. **Superioridad de los modelos de ML frente a los Baselines:**
   * El `Baseline_Heuristico_Historico` (`cant_estudiantes_compartidos_historico > 0`) solo alcanza un **Recall de 20,82%** (`F1 = 0,1694`, `PR-AUC = 0,0750`) porque el 79% de los conflictos de la cohorte 1994 ocurre en pares de materias de 1.er año donde las cohortes 1989–1993 no tuvieron alumnos compartidos en 1994–95.
   * Todos los modelos de Machine Learning superan ampliamente el PR-AUC del baseline histórico (`0,0750`), llegando a **`0,1439` en `DecisionTree_Balanced`**, **`0,1767` en `DecisionTree_Full_ClassWeight`**, **`0,1833` en `KNN_Full_Unweighted`** y **`0,2082` en `DecisionTree_Full_Unweighted`**.
2. **Importancia de evaluar el umbral seleccionado en validación (`_UmbralVal`) en todos los regímenes:**
   * En los modelos no ponderados (`_Full_Unweighted`), el umbral por defecto $\tau = 0,50$ produce un Recall muy bajo (`2,27%` a `9,18%`) debido a la baja prevalencia de entrenamiento (`1,37%`). Al aplicar el umbral que maximiza F1 en validación ($\tau^*_{\text{val}} \in [0,19; 0,21]$), el desempeño en Test mejora drásticamente: `DecisionTree_Full_Unweighted_UmbralVal` duplica su F1 (`0,1180` $\to$ `0,2288`, con `Precision = 43,99%`), `LogisticRegression_Full_Unweighted_UmbralVal` quintuplica su F1 (`0,0425` $\to$ `0,2205`) y **`KNN_Full_Unweighted_UmbralVal` ($\tau^*_{\text{val}} = 0,21$) alcanza el mayor F1-Score de todo el conjunto de Test (`0,2593`)**.
   * En **Logistic Regression**, entrenar sobre los 6.996 pares de `ml_train_balanced.csv` produce resultados muy similares a entrenar sobre los 254.541 pares de `ml_train_full.csv` con `class_weight="balanced"` (`F1 = 0,2463` vs. `0,2507` con umbral seleccionado en validación, y `Recall = 75,88%` vs. `75,98%` con umbral $0,50$), entrenando 36 veces más rápido.
   * En **Decision Tree** y **KNN**, disponer de las 254.541 filas de `ml_train_full.csv` mejora el ordenamiento de probabilidades en regiones poco densas, elevando el PR-AUC en Test de `0,1439` a `0,2082` en Árboles y de `0,1183` a `0,1833` en KNN.

---

## 8. Interpretación de Falsos Positivos y Falsos Negativos en la Planificación de Mesas

En el problema de asignación de turnos mediante coloreo de grafos, los dos tipos de error tienen consecuencias operativas completamente distintas:

* **Falso Negativo (`FN` — Conflicto real no detectado):**
  * **Qué significa:** Dos materias tienen alumnos en común en la cohorte 1994 (`y = 1`), pero el modelo predice que no hay conflicto ($\hat{y} = 0$).
  * **Consecuencia en el grafo:** Falta una arista en el grafo de conflictos. Por lo tanto, el algoritmo de coloreo puede asignarles el **mismo color (mismo día y turno)**, provocando un **choque de horario real** para los estudiantes que debían rendir ambas materias.
* **Falso Positivo (`FP` — Falso conflicto agregado):**
  * **Qué significa:** Dos materias no comparten ningún alumno en la cohorte 1994 (`y = 0`), pero el modelo predice conflicto ($\hat{y} = 1$).
  * **Consecuencia en el grafo:** Se agrega una arista extra innecesaria al grafo. Ningún estudiante sale perjudicado por superposición horaria, pero el grafo se vuelve **más denso** y el algoritmo de coloreo greedy necesitará **más colores (más turnos de examen)** para colorearlo.

> **Implicación práctica para la Fase 4 y el Frontend:**  
> Esta asimetría justifica directamente por qué el sistema incluye un **selector de umbral de probabilidad $\tau$**:
> * Si la secretaría académica prioriza **evitar choques a toda costa (minimizar `FN` / maximizar Recall)** y dispone de suficientes turnos, elegirá un umbral bajo ($\tau \approx 0,50$ en los modelos balanceados, que detecta entre el **76% y el 86%** de todos los conflictos reales).
> * Si la secretaría necesita **comprimir el calendario en pocos turnos (minimizar `FP` / maximizar Precision)** asegurando que solo se separen los conflictos más probables, subirá el umbral ($\tau \ge 0,90$), reduciendo drásticamente la densidad del grafo.

---

## 9. Importancia de Variables (`DecisionTree_Balanced`, `max_depth=5`)

Se extrajo el atributo `feature_importances_` del árbol de decisión entrenado sobre `ml_train_balanced.csv` (archivo exportado en [`results/metrics/decision_tree_feature_importances.csv`](file:///w:/home/anton/Tp-Grafos-ML/results/metrics/decision_tree_feature_importances.csv)):

| Rango | Variable Predictora | Importancia de Gini (`feature_importances_`) | Porcentaje |
| :---: | :--- | ---: | ---: |
| 1 | `misma_carrera` | 0,389986 | **39,00%** |
| 2 | `mismo_anio` | 0,231968 | **23,20%** |
| 3 | `cant_inscriptos_hist_1` | 0,184994 | **18,50%** |
| 4 | `cant_inscriptos_hist_2` | 0,067737 | **6,77%** |
| 5 | `anio_confirmado` | 0,066350 | **6,64%** |
| 6 | `mismo_prefijo_curricular` | 0,038459 | **3,85%** |
| 7 | `jaccard_historico` | 0,016619 | **1,66%** |
| 8 | `mismo_departamento` | 0,002774 | **0,28%** |
| 9 | `cant_estudiantes_compartidos_historico` | 0,001113 | **0,11%** |
| 10 | `cant_carreras_compartidas_hist` | 0,000000 | **0,00%** |

### Interpretación técnica y limitaciones (No es prueba de causalidad)
1. **Qué mide `feature_importances_`:** Mide la reducción acumulada y normalizada de la impureza de Gini ponderada por la cantidad de muestras que pasan por cada nodo donde se utiliza esa variable. **No mide causalidad** ni el efecto marginal aislado de una variable.
2. **Por qué dominan las variables curriculares y de tamaño (`misma_carrera`, `mismo_anio`, `cant_inscriptos_hist`):** En la cohorte 1994 (ingresantes de 1.er año), muchos conflictos ocurren entre materias de un mismo año y plan de estudios. El árbol descubre que compartir carrera histórica (`39,0%`), pertenecer al mismo año (`23,2%` + `6,64%` de `anio_confirmado`) y tener alta matrícula histórica permiten detectar conflictos incluso cuando `cant_estudiantes_compartidos_historico` es cero.
3. **Efecto de enmascaramiento entre variables correlacionadas:** Que `cant_carreras_compartidas_hist` tenga importancia `0,0000` y `mismo_departamento` tenga `0,0028` **no significa que no sean informativas**: una vez que el árbol realiza sus primeras particiones usando `misma_carrera` (`cant_carreras_compartidas_hist > 0`) y `mismo_prefijo_curricular`, la ganancia adicional de volver a cortar por `cant_carreras_compartidas_hist` o `mismo_departamento` hasta profundidad 5 queda absorbida por las primeras. Lo mismo ocurre entre `jaccard_historico` (`1,66%`) y `cant_estudiantes_compartidos_historico` (`0,11%`).

---

## 10. Limitaciones Metodológicas

1. **Desfase generacional entre cohortes (`1989–1993` vs. `1994`):** Al tratarse de cohortes de ingreso dentro de un único semestre (`1994–95, Semestre 1`), las materias exclusivas de 1.er año tienen baja inscripción en las cohortes `1989–1993` (solo recursantes), lo que limita el alcance de `cant_estudiantes_compartidos_historico` y obliga a los modelos a apoyarse en atributos estructurales (`misma_carrera`, `mismo_anio`), acotando el techo de PR-AUC alcanzable (~`0,21`).
2. **Sensibilidad al umbral bajo prevalencia del 1,49%:** Tanto el *undersampling* 50/50 como `class_weight="balanced"` alteran la distribución o ponderación empleada en el entrenamiento y desplazan los puntajes predichos respecto a la prevalencia natural del test (1,49%), mientras que los modelos no ponderados concentran sus probabilidades en valores bajos. Por ello resulta indispensable seleccionar el umbral de decisión $\tau$ sobre un conjunto de validación con prevalencia natural según el compromiso deseado entre Falsos Positivos y Falsos Negativos.
3. **Dependencia residual en pares semi-inductivos:** En la partición por materias, los pares de validación y test donde una materia pertenece al conjunto reservado y la otra fue vista en entrenamiento son semi-inductivos ($714 \times 86$ pares), mientras que los pares donde ambas materias pertenecen al conjunto reservado ($\binom{86}{2} = 3.655$ pares) son puramente inductivos.

---

## 11. Cómo Ejecutar el Código

Desde la raíz del proyecto en WSL / terminal:

```bash
# 1. Crear el entorno virtual e instalar dependencias (si no existe venv/)
python3 -m venv venv
./venv/bin/pip install -r requirements.txt

# 2. Ejecutar el pipeline de entrenamiento, validación y evaluación de Fase 3
./venv/bin/python -m src.models.compare_models
```

### Archivos de salida generados por `compare_models.py`
* [`results/metrics/validation_metrics.csv`](file:///w:/home/anton/Tp-Grafos-ML/results/metrics/validation_metrics.csv): métricas completas sobre los 51.975 pares de validación interna (18 filas: 2 baselines y los 8 modelos entrenables evaluados en $\tau = 0,50$ y en $\tau^*_{\text{val}}$).
* [`results/metrics/test_metrics.csv`](file:///w:/home/anton/Tp-Grafos-ML/results/metrics/test_metrics.csv): métricas finales sobre los 65.059 pares de `ml_test_full.csv` (18 filas: 2 baselines y los 8 modelos entrenables evaluados en $\tau = 0,50$ y en $\tau^*_{\text{val}}$).
* [`results/metrics/decision_tree_feature_importances.csv`](file:///w:/home/anton/Tp-Grafos-ML/results/metrics/decision_tree_feature_importances.csv): importancia de variables ordenada para cada árbol de decisión.
* [`results/metrics/metrics_summary.json`](file:///w:/home/anton/Tp-Grafos-ML/results/metrics/metrics_summary.json): resumen consolidado en formato JSON (incluye los 8 umbrales seleccionados en validación, conteos de validación y matrices de confusión).
* `results/models/*.joblib`: los 10 modelos entrenados serializados con `joblib`.
* [`results/predictions/test_predictions_all_models.csv`](file:///w:/home/anton/Tp-Grafos-ML/results/predictions/test_predictions_all_models.csv): tabla de 65.059 filas que conserva `materia_id_1`, `materia_id_2`, `hubo_conflicto_potencial`, las probabilidades `prob_<Modelo>` y las predicciones binarias `pred_<Modelo>_th050` y `pred_<Modelo>_thVal`.

---

## 12. Cómo se Utilizarán las Probabilidades en la Construcción del Grafo (Fase 4)

La decisión metodológica del proyecto para la **Fase 4 (Teoría de Grafos — Coloreo de Turnos)** es utilizar **coloreo clásico no ponderado** sobre el grafo de incompatibilidades $G(\tau) = (V, E(\tau))$:
1. **Nodos ($V$):** Las materias a planificar (`materia_id`).
2. **Aristas de incompatibilidad parametrizadas por umbral ($E(\tau)$):** La probabilidad predicha $\hat{p}_{uv}$ se utiliza **únicamente** para decidir si se agrega o no la arista $\{u, v\}$ según el umbral seleccionado ($\hat{p}_{uv} \ge \tau$). Cada arista incluida representa una incompatibilidad horaria entre ambas materias (no pueden asignarse al mismo turno).
3. **Coloreo clásico no ponderado:** El algoritmo de coloreo greedy trabaja exclusivamente sobre la estructura de adyacencia de $G(\tau)$ (los grados y vecindarios de cada vértice), **sin utilizar pesos en las aristas**, permitiendo analizar cómo varía la densidad del grafo y la cantidad de colores (turnos de examen necesarios) al modificar el umbral $\tau$.
