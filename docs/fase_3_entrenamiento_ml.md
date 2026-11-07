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

## 3. Selección de Features, Exclusiones y Prevención de *Data Leakage*

* **Variable Objetivo (`y`):** `hubo_conflicto_potencial` (`1` si ambas materias comparten al menos un estudiante en la cohorte 1994; `0` en caso contrario).
* **Justificación de las Columnas Excluidas de `X`:**
  * **Identificadores del par (`materia_id_1`, `materia_id_2`):** Se conservan únicamente como metadato de enlace (`ID_COLUMNS`) para asociar las predicciones con las aristas $\{u, v\}$ del grafo en la Fase 4. Se excluyen de `X` porque un clasificador inductivo debe aprender patrones relacionales generalizables entre atributos académicos e históricos, no memorizar códigos nominales de asignaturas.
  * **Variables objetivo (`hubo_conflicto_potencial`, `cant_estudiantes_afectados`):** Ambas se calculan sobre las inscripciones de la cohorte objetivo (`1994`). Incluir `cant_estudiantes_afectados` al predecir `hubo_conflicto_potencial` constituiría *data leakage* directo ($\text{hubo\_conflicto\_potencial} \iff \text{cant\_estudiantes\_afectados} > 0$).
  * **Colinealidad exacta (`tasa_compartidos_historico`):** Se excluye explícitamente porque es una transformación lineal exacta de `cant_estudiantes_compartidos_historico` ($\text{tasa} = \frac{\text{conteo}}{4810}$, correlación $r = 1,0000$). Tras aplicar `StandardScaler`, ambas columnas producirían exactamente el mismo vector estandarizado, duplicando artificialmente el peso de esa dimensión en la distancia euclídea de KNN y provocando colinealidad perfecta en los coeficientes de Regresión Logística.
  * **Prevención estricta de *data leakage* entre cohortes:** Todas las variables basadas en inscripciones dentro de `X` fueron calculadas exclusivamente sobre los **4.810 estudiantes de las cohortes históricas (`1989–1993`)**, sin intervenir jamás ningún registro de los **3.086 estudiantes de la cohorte objetivo (`1994`)** (cuyos conteos individuales `cant_inscriptos_actual_1`, `cant_inscriptos_actual_2` y `tasa_afectados_actual` ya fueron removidos físicamente en la Fase 2.5).
* **10 Variables Predictoras Seleccionadas (`FEATURE_COLUMNS`, en su orden exacto de entrenamiento e inferencia):**
  1. `mismo_departamento` (binaria `0/1`)
  2. `mismo_prefijo_curricular` (binaria `0/1`)
  3. `anio_confirmado` (binaria `0/1`: incluida porque `mismo_anio` vale `0` tanto cuando dos materias son de distinto año numérico confirmado como cuando alguna tiene código especial `A, B, C, P, 0`; `anio_confirmado` permite distinguir ambos casos).
  4. `mismo_anio` (binaria `0/1`)
  5. `misma_carrera` (binaria `0/1`, calculada exclusivamente sobre cohortes 1989–1993)
  6. `cant_carreras_compartidas_hist` (entera $\ge 0$, cohortes 1989–1993)
  7. `cant_inscriptos_hist_1` (entera $\ge 0$, cohortes 1989–1993)
  8. `cant_inscriptos_hist_2` (entera $\ge 0$, cohortes 1989–1993)
  9. `cant_estudiantes_compartidos_historico` (entera $\ge 0$, cohortes 1989–1993)
  10. `jaccard_historico` (continua $\in [0, 1]$, cohortes 1989–1993)

---

## 4. Auditoría de la Separación Entrenamiento, Validación y Test (por Materias)

### 4.1. Implementación de `build_internal_subject_validation_split()`
Para seleccionar umbrales y comparar regímenes sin tocar `ml_test_full.csv`, se construyó una partición interna a partir de `ml_train_full.csv` respetando la misma estrategia de **Retención Estratificada por Materias** de la Fase 2.5 (`RANDOM_SEED = 42`):

1. **Separación por materias estratificada por `raw_level_char`:** De las **714 materias** presentes en `ml_train_full.csv`, se seleccionó de forma estratificada según el carácter de nivel curricular (`raw_level_char` en `data/interim/exams_clean.csv`) con semilla fija `42` (`VAL_SUBJECT_FRACTION = 0.10625`) un subconjunto de **77 materias reservadas para Validación Interna (`val_subjects`)** y **637 materias para Sub-entrenamiento (`subtrain_subjects`)**.
2. **`subtrain_full`:** Pares de `ml_train_full.csv` donde **ambas** materias pertenecen a `subtrain_subjects` ($\binom{637}{2} = \mathbf{202.566}$ pares: **2.836 positivos [1,400%]** y **199.730 negativos [98,600%]**).
3. **`subtrain_balanced`:** *Undersampling* 50/50 sin reemplazo (`random_state=42`) aplicado **exclusivamente sobre `subtrain_full`** (**5.672 pares**: **2.836 positivos [50,00%]** y **2.836 negativos [50,00%]**). Ningún par de validación participa de este submuestreo.
4. **`validation` (distribución natural):** Pares de `ml_train_full.csv` donde **al menos una** materia pertenece a `val_subjects` ($\mathbf{51.975}$ pares: **662 positivos [1,274%]** y **51.313 negativos [98,726%]**), conservando la prevalencia natural de clases.
5. **`test` (`ml_test_full.csv`, evaluación final):** Compuesto por los $\mathbf{65.059}$ pares (970 positivos [1,491%] y 64.089 negativos) que involucran al menos una de las **86 materias reservadas para test** en la Fase 2.5. Los umbrales se seleccionan exclusivamente en `validation` y el conjunto de `test` se utiliza una única vez para reportar la evaluación final.

### 4.2. Justificación y limitaciones de la partición por materias frente a la división aleatoria de filas
* **Por qué es más apropiada que un `train_test_split` aleatorio de filas:** En un dataset diádico de $\binom{800}{2} = 319.600$ pares, cada materia participa en $799$ filas. Una división aleatoria de filas haría que las 800 materias aparezcan ~640 veces en entrenamiento y ~160 veces en prueba, permitiendo que modelos no lineales (Árboles o KNN) memoricen la identidad implícita de una materia a través de sus conteos marginales (`cant_inscriptos_hist_1`, `cant_inscriptos_hist_2`). Reservar materias completas fuerza al modelo a generalizar hacia asignaturas no vistas en entrenamiento.
* **Limitación metodológica (pares semi-inductivos vs. puramente inductivos):** Al definir test (y validación) como los pares que contienen *al menos una* materia reservada (para no descartar pares cruzados del universo de 319.600 filas), los pares de `ml_test_full.csv` se dividen en $\binom{86}{2} = 3.655$ pares **puramente inductivos** (ninguna de las dos materias fue vista en entrenamiento) y $714 \times 86 = 61.404$ pares **semi-inductivos** (una materia pertenece al grupo reservado de test y la otra participó en pares de entrenamiento, aunque el par como tal es 100% inédito).

---

## 5. Preprocesamiento y Robustez en la Carga de Datos

* **Validación explícita en [`load_ml_dataframe()`](file:///w:/home/anton/Tp-Grafos-ML/src/models/compare_models.py#L168-L208):**
  * Verifica la existencia física del archivo CSV (`FileNotFoundError`) y la presencia de todas las columnas requeridas (`ID_COLUMNS + FEATURE_COLUMNS + [TARGET_COLUMN]`).
  * Comprueba la ausencia de valores nulos (`isna()`) **antes** de realizar conversiones de tipos.
  * Convierte las 5 features booleanas y el target binario mediante `_parse_strict_binary_column()`, que valida explícitamente que los valores pertenezcan al dominio binario esperado (`True/False`, `1/0`) y evita que una conversión ingenua `astype(bool).astype(int)` transforme silenciosamente cadenas como `"False"` en `1`.
  * Comprueba que la matriz numérica de features y target sea finita (`0` valores infinitos con `np.isinf()`).
* **Escalado sin fuga de información (`StandardScaler`):**
  * Para **Logistic Regression** y **KNN**, `StandardScaler()` está encapsulado dentro de un `sklearn.pipeline.Pipeline`, garantizando que la media y desviación estándar se ajusten **exclusivamente sobre el conjunto de entrenamiento** (`subtrain` o `train`) y se apliquen mediante `transform` en validación e inferencia.
  * Para **Decision Tree**, no se aplica escalado dado que las particiones ortogonales por umbral de Gini son invariantes ante transformaciones lineales monótonas.

---

## 6. Decisiones Metodológicas: Algoritmos, Desbalance, Variantes y Umbrales

### 6.1. Algoritmos clásicos comparados
Se evaluaron los tres algoritmos supervisados clásicos exigidos por el trabajo práctico:

1. **Decision Tree (`DecisionTreeClassifier`, `max_depth=5`, `random_state=42`):**
   * **Cómo funciona y qué patrones aprende:** Divide recursivamente el espacio de features mediante cortes ortogonales (`feature <= valor`) que maximizan la reducción de impureza de Gini. Captura interacciones no lineales y condicionales entre variables (por ejemplo, combinar `misma_carrera == 1`, `mismo_anio == 1` y umbrales sobre `cant_inscriptos_hist_1/2`).
   * **Por qué es razonable en este problema:** La compatibilidad curricular sigue reglas jerárquicas naturales (compartir plan de estudios y año de cursada) y el acotamiento `max_depth=5` previene la memorización de hojas con pocos ejemplos, aportando además interpretabilidad directa mediante `feature_importances_`.
2. **Logistic Regression (`Pipeline([StandardScaler(), LogisticRegression(max_iter=1000, random_state=42)])`):**
   * **Cómo funciona y qué patrones aprende:** Modela el logaritmo de las chances (*log-odds*) de conflicto como un hiperplano lineal $\beta_0 + \sum_{j=1}^{10} \beta_j z_j$ sobre las features estandarizadas, proyectado al intervalo $(0, 1)$ mediante la función logística sigmoide. Aprende el efecto aditivo y monótono global de cada predictor.
   * **Por qué es razonable en este problema:** Constituye el modelo paramétrico lineal de referencia; tiene baja varianza, inferencia inmediata y permite contrastar cuánto aportan las fronteras no lineales frente a una combinación lineal ponderada de señales históricas y curriculares.
3. **K-Nearest Neighbors (`Pipeline([StandardScaler(), KNeighborsClassifier(n_neighbors=15, weights="uniform")])`):**
   * **Cómo funciona y qué patrones aprende:** Clasificador no paramétrico basado en instancias que, para cada par consultado $(u, v)$, busca en un `KDTree` los $K = 15$ pares de entrenamiento más cercanos según la distancia euclídea en el espacio estandarizado de 10 dimensiones y estima la probabilidad de conflicto como la proporción local de vecinos positivos ($\hat{p}_{uv} = k_{\text{pos}} / 15$).
   * **Por qué es razonable en este problema:** Permite capturar vecindarios locales de pares de materias con perfil curricular y tamaño de cátedra similares sin imponer una forma funcional global (lineal ni rectangular). Con $K = 15$ suaviza el ruido de pares aislados y logra un compromiso sólido entre Precision y Recall sobre la distribución natural.

### 6.2. Estrategias de tratamiento del desbalance (`1,40%` positivos vs. `98,60%` negativos)
Ante la relación aproximada de `1` par conflictivo cada `70,5` pares sin conflicto, se compararon tres regímenes de entrenamiento:

| Estrategia | Descripción | Ventajas | Limitaciones |
| :--- | :--- | :--- | :--- |
| **`Balanced`** | *Undersampling* aleatorio 50/50 de la clase negativa aplicado exclusivamente en entrenamiento (`5.672` filas en `subtrain_balanced`, `6.996` en `ml_train_balanced.csv`). | Aplicable de forma idéntica a los **tres** algoritmos (incluido KNN); entrenamiento e inferencia muy rápidos; maximiza el Recall en $\tau = 0,50$. | Descarta el ~98,6% de los ejemplos negativos de entrenamiento, perdiendo detalle de densidad local y desplazando los puntajes hacia arriba respecto a la prevalencia natural. |
| **`Full_Unweighted`** | Entrenamiento con el 100% de los pares de entrenamiento (`202.566` en `subtrain_full`, `254.541` en `ml_train_full.csv`) sin ponderación artificial de clases. | Aprovecha toda la información de los 254.541 pares y preserva la densidad real del espacio de features, logrando los mejores **PR-AUC** (`0,2082` en Tree y `0,1833` en KNN). | Con el umbral estándar $\tau = 0,50$ es demasiado conservador (bajo Recall); requiere seleccionar un umbral menor ($\tau^*_{\text{val}} \approx 0,19\text{–}0,21$) en validación y aumenta el costo de consulta en KNN (`KDTree` de 254.541 filas). |
| **`Full_ClassWeight`** | Entrenamiento con el 100% de los datos aplicando `class_weight="balanced"` (pondera la pérdida de cada clase de forma inversamente proporcional a su frecuencia: $w_1 \approx 36,4$, $w_0 \approx 0,51$). | No descarta pares negativos y compensa el desbalance directamente en la función objetivo del algoritmo. | Desplaza los puntajes predichos de manera similar al *undersampling* y **no está disponible en todos los algoritmos** (ver aclaración sobre KNN debajo). |

> **Aclaración técnica sobre por qué no existe `KNN_Full_ClassWeight`:**  
> En `scikit-learn`, `DecisionTreeClassifier` y `LogisticRegression` poseen el hiperparámetro `class_weight="balanced"` para ponderar la función de pérdida según la prevalencia de cada clase. Por el contrario, **`KNeighborsClassifier` no posee el parámetro `class_weight`** (su parámetro `weights` solo permite ponderar a los $K$ vecinos por su distancia geométrica `'uniform'` vs. `'distance'`, nunca por frecuencia de clase). Por esta razón metodológica e instrumental no se implementó una variante `KNN_Full_ClassWeight`.

### 6.3. Glosario explícito del significado de las variantes
Para evitar ambigüedades en la lectura de tablas, gráficos y archivos CSV/JSON, la nomenclatura sigue esta convención estricta:

* `DecisionTree` (o `Tree`): Árbol de Decisión (`DecisionTreeClassifier(max_depth=5, random_state=42)`).
* `LogisticRegression` (o `LR`): Regresión Logística (`Pipeline([StandardScaler(), LogisticRegression(max_iter=1000, random_state=42)])`).
* `KNN`: K Vecinos Más Cercanos (`Pipeline([StandardScaler(), KNeighborsClassifier(n_neighbors=15)])`).
* `Balanced`: Modelo entrenado sobre el conjunto balanceado 50/50 mediante *undersampling* de negativos (`subtrain_balanced` en validación; `ml_train_balanced.csv` en test).
* `Full`: Modelo entrenado sobre el **conjunto de entrenamiento completo** (`subtrain_full` de 202.566 filas en validación; `ml_train_full.csv` de 254.541 filas en test).
  > **Importante:** El término **`Full` refiere exclusivamente al uso del dataset de entrenamiento completo** (sin *undersampling*), y **NO significa que el árbol de decisión esté completamente desarrollado sin límite de profundidad**. Todos los árboles de decisión del proyecto (`DecisionTree_Balanced`, `DecisionTree_Full_Unweighted` y `DecisionTree_Full_ClassWeight`) tienen su profundidad acotada en `max_depth=5`.
* `Unweighted`: Entrenamiento sin compensación mediante pesos de clase (`class_weight=None`).
* `ClassWeight`: Entrenamiento con compensación mediante pesos de clase (`class_weight="balanced"`).
* `_UmbralVal`: Evaluación del modelo aplicando el umbral de probabilidad $\tau^*_{\text{val}}$ seleccionado exclusivamente en validación interna para maximizar F1 (en lugar del umbral por defecto $\tau = 0,50$).

### 6.4. Documentación de los Baselines y análisis de `HistoricalOverlapBaseline.predict_proba()`
Se implementaron dos líneas base de referencia:
1. **`Baseline_Mayoritario` (`DummyClassifier(strategy="prior")`):** Predice siempre la clase mayoritaria (`0`, ausencia de conflicto) y asigna probabilidad constante igual a la prevalencia previa de entrenamiento ($\approx 0,0137$). Sirve para demostrar por qué el *Accuracy* (`98,51%` en test) es engañoso bajo desbalance severo (`F1 = 0,0000`, `Recall = 0,0000`).
2. **`Baseline_Heuristico_Historico` (`HistoricalOverlapBaseline`):** Regla determinística de dominio académico que predice conflicto en 1994 (`predict() == 1`) si y solo si el par de materias ya registró alumnos compartidos en las cohortes 1989–1993 (`cant_estudiantes_compartidos_historico > 0`).
   * **Análisis metodológico de `HistoricalOverlapBaseline.predict_proba()`:**  
     Para poder calcular métricas de área bajo la curva (`average_precision_score` [PR-AUC] y `roc_auc_score` [ROC-AUC]), el método `predict_proba(X)` devuelve `scores = vals / vals.max()` (acotado en $[0, 1]$).
     * **¿Es apropiado para calcular PR-AUC y ROC-AUC en un conjunto de evaluación?** **Sí**, porque tanto `average_precision_score` como `roc_auc_score` son métricas **basadas exclusivamente en el ordenamiento por rangos (*rank-based*)** de los puntajes dentro del conjunto evaluado. Cualquier transformación lineal estrictamente creciente $s_i = c \cdot x_i$ con $c > 0$ preserva intactos todos los pares de orden ($s_i > s_j \iff x_i > x_j$) y todos los empates, arrojando exactamente el mismo PR-AUC (`0,1025` en validación y `0,0750` en test) y el mismo ROC-AUC (`0,6146` en validación y `0,5952` en test) que si se empleara el conteo bruto o se normalizara por el máximo de entrenamiento.
     * **Limitación importante del score:** Estos valores **no son probabilidades calibradas** y su escala numérica depende del máximo del lote $X$ recibido en la llamada. Por ello, **no deben umbralizarse en $0,50$**; la clasificación binaria de este baseline en `predict(X)` no usa `predict_proba(X) >= 0.50`, sino directamente la condición exacta `vals > 0`. Se conserva esta implementación sin cambios para mantener intactos todos los archivos de resultados y métricas del repositorio.

### 6.5. Selección de umbrales en Validación (`find_best_f1_threshold_on_validation`) y caso particular de `KNN_Full_Unweighted`
* **Qué representa el umbral $\tau$ y por qué $0,50$ no es adecuado por defecto:** El umbral $\tau \in (0, 1)$ es el valor de corte a partir del cual la probabilidad estimada $\hat{p}_{uv}$ se convierte en una decisión binaria $\hat{y}_{uv} = \mathbb{I}(\hat{p}_{uv} \ge \tau)$ (agregar o no la arista $\{u, v\}$ al grafo). Con prevalencia positiva natural de ~1,40%, los modelos `_Full_Unweighted` producen probabilidades concentradas por debajo de $0,50$ (haciendo que $\tau = 0,50$ tenga un Recall muy bajo), mientras que los modelos `_Balanced` y `_Full_ClassWeight` desplazan sus puntajes hacia arriba (haciendo que $\tau = 0,50$ genere más de $10.000$ falsos positivos).
* **Procedimiento de selección y prohibición de usar Test:** La función [`find_best_f1_threshold_on_validation()`](file:///w:/home/anton/Tp-Grafos-ML/src/models/compare_models.py#L347-L366) evalúa la grilla $\tau \in \{0,01; 0,02; \dots; 0,95\}$ **exclusivamente sobre `validation` (`51.975` pares con prevalencia natural 1,274%)** y elige el valor $\tau^*_{\text{val}}$ que maximiza el F1-score de la clase positiva. Nunca se utiliza `ml_test_full.csv` para elegir o ajustar el umbral, ya que optimizar un hiperparámetro de decisión sobre el conjunto de prueba introduciría sobreajuste optimista (*test leakage*).
* **Caso particular de `KNN_Full_Unweighted` ($\tau^*_{\text{val}} = 0,21$) y manejo de empates:**
  * El modelo `KNN_Full_Unweighted` es un `Pipeline` integrado por `StandardScaler()` y `KNeighborsClassifier(n_neighbors=15, weights="uniform")`.
  * Al utilizar $K = 15$ vecinos con pesos uniformes, las probabilidades devueltas por `predict_proba(X)[:, 1]` son **discretas** y solo pueden tomar los $16$ valores $\{0, \frac{1}{15}, \frac{2}{15}, \frac{3}{15}, \frac{4}{15}, \dots, 1\}$, es decir:
    $$0,0000;\; 0,0667;\; 0,1333;\; 0,2000;\; 0,2667;\; 0,3333;\; \dots$$
  * En validación interna, exigir al menos $3$ vecinos positivos ($\tau \in [0,14; 0,20]$) arroja menor F1 que exigir **al menos $4$ vecinos positivos** ($\hat{p}_{uv} \ge \frac{4}{15} \approx 0,2667$), donde se alcanza el máximo F1 de validación (**`0,239521`**, con `TP = 140`, `FP = 367`, `Precision = 0,2761`, `Recall = 0,2115`).
  * Como no existe ningún valor posible de probabilidad de KNN entre $\frac{3}{15} = 0,20$ y $\frac{4}{15} \approx 0,2667$, **todos los umbrales de la grilla en el intervalo $[0,21; 0,26]$ (`0.21, 0.22, 0.23, 0.24, 0.25, 0.26`) producen exactamente las mismas clasificaciones y el mismo F1-score**.
  * **Manejo de empates:** Dado que `find_best_f1_threshold_on_validation()` recorre la grilla en orden ascendente desde `0.01` hasta `0.95` y actualiza el óptimo mediante la desigualdad estricta `if score > best_f1:`, ante un empate exacto conserva determinísticamente el **primer (menor) umbral del tramo óptimo**, que para `KNN_Full_Unweighted` es **`0.21`**.
  * Este valor `0.21` queda registrado en [`results/metrics/metrics_summary.json`](file:///w:/home/anton/Tp-Grafos-ML/results/metrics/metrics_summary.json#L41) (`calibrated_thresholds_from_validation["KNN_Full_Unweighted"] = 0.21`) y en [`results/metrics/test_metrics.csv`](file:///w:/home/anton/Tp-Grafos-ML/results/metrics/test_metrics.csv#L19), pero **no está almacenado dentro del archivo binario `knn_full_unweighted.joblib`**.

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

### 7.3. Evaluación detallada del modelo candidato principal (`KNN_Full_Unweighted_UmbralVal`) y por qué *Accuracy* no es suficiente
El modelo con mejor desempeño equilibrado en evaluación final (`ml_test_full.csv`, `65.059` pares) es **`KNN_Full_Unweighted_UmbralVal`**:
* **Umbral seleccionado en validación ($\tau^*_{\text{val}}$):** `0.21` (equivalente a $\ge 4$ vecinos conflictivos de $K=15$).
* **Precision:** `0.271091` (~**`0,2711`**).
* **Recall (Sensibilidad):** `0.248454` (~**`0,2485`**).
* **F1-Score (clase positiva):** `0.259279` (~**`0,2593`**, el más alto de todos los modelos evaluados en Test).
* **PR-AUC (`average_precision_score`):** `0.183295` (~**`0,1833`**, más del doble del baseline histórico `0,0750` y `12,3` veces el azar `0,0149`).
* **ROC-AUC:** `0.766514` (~`0,7665`).
* **Matriz de confusión en Test (`65.059` pares):**
  * **Verdaderos Positivos (`TP`):** `241` conflictos reales detectados.
  * **Falsos Positivos (`FP`):** `648` aristas preventivas adicionales.
  * **Falsos Negativos (`FN`):** `729` conflictos no detectados.
  * **Verdaderos Negativos (`TN`):** `63.441` pares compatibles correctamente descartados.
  * **Total de aristas predichas en Test (`TP + FP`):** `889` aristas.

> **¿Por qué *Accuracy* no es una métrica suficiente en este problema?**  
> En `ml_test_full.csv`, el **98,509%** de los pares (`64.089` de `65.059`) pertenece a la clase negativa (`sin conflicto`). Como consecuencia, el clasificador trivial `Baseline_Mayoritario` —que predice siempre `0` y no detecta ni un solo conflicto (`TP = 0`, `Recall = 0%`, `F1 = 0`)— obtiene un **Accuracy de `98,51%`**, superior incluso al de modelos útiles que sí detectan conflictos. Evaluar por *Accuracy* premiaría generar un grafo vacío sin aristas, asignando todas las materias al mismo turno. Por ello la evaluación se apoya en **F1-score de la clase positiva, Precision, Recall, PR-AUC y la matriz de confusión**.

### 7.4. Exploración complementaria: Sensibilidad al umbral en `KNN_Full_Unweighted` (`notebooks/Test_modelo_ML.ipynb`)
En el notebook de auditoría [`notebooks/Test_modelo_ML.ipynb`](file:///w:/home/anton/Tp-Grafos-ML/notebooks/Test_modelo_ML.ipynb) se realizó una exploración descriptiva de cómo varían las métricas y la cantidad de aristas predichas (`TP + FP`) sobre `ml_test_full.csv` al desplazar el umbral $\tau \in [0,05; 0,80]$:

| Umbral ($\tau$) | Vecinos positivos requeridos ($k / 15$) | Precision | Recall | F1-Score | Aristas Predichas (`TP + FP`) |
| :---: | :---: | ---: | ---: | ---: | ---: |
| `0,05` | $\ge 1 / 15$ (`0,0667`) | 0,0912 | 0,6010 | 0,1584 | 6.393 |
| `0,10` | $\ge 2 / 15$ (`0,1333`) | 0,1527 | 0,4155 | 0,2233 | 2.639 |
| `0,15` – `0,20` | $\ge 3 / 15$ (`0,2000`) | 0,2029 | 0,3361 | 0,2530 | 1.607 |
| **`0,21` – `0,25`** | **$\ge 4 / 15$ (`0,2667`)** | **0,2711** | **0,2485** | **0,2593** | **889** |
| `0,30` | $\ge 5 / 15$ (`0,3333`) | 0,3378 | 0,1804 | 0,2352 | 518 |
| `0,35` – `0,40` | $\ge 6 / 15$ (`0,4000`) | 0,4747 | 0,1454 | 0,2226 | 297 |
| `0,45` | $\ge 7 / 15$ (`0,4667`) | 0,5263 | 0,1237 | 0,2003 | 228 |
| `0,50` | $\ge 8 / 15$ (`0,5333`) | 0,5597 | 0,0918 | 0,1577 | 159 |
| `0,55` | $\ge 9 / 15$ (`0,6000`) | 0,8986 | 0,0639 | 0,1193 | 69 |
| `0,60` – `0,65` | $\ge 10 / 15$ (`0,6667`) | 0,9091 | 0,0412 | 0,0789 | 44 |
| `0,70` | $\ge 11 / 15$ (`0,7333`) | 0,8857 | 0,0320 | 0,0617 | 35 |
| `0,75` – `0,80` | $\ge 12 / 15$ (`0,8000`) | 0,8571 | 0,0247 | 0,0481 | 28 |

> **Aclaración metodológica fundamental:**  
> Esta tabla constituye únicamente un **análisis descriptivo de sensibilidad** para comprender cómo la discreción de probabilidades de KNN (escalones de $\frac{1}{15} \approx 0,0667$) impacta en la densidad del grafo. **No se utiliza en ningún caso para volver a seleccionar el umbral mirando Test**; se conserva estrictamente el umbral **`0.21`** elegido a ciegas en validación interna.

### 7.5. Hallazgos generales de la comparación entre modelos y regímenes
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
> * Si la secretaría académica prioriza **evitar choques a toda costa (minimizar `FN` / maximizar Recall)** y dispone de suficientes turnos, elegirá un umbral bajo ($\tau \approx 0,50$ en los modelos balanceados, que detecta entre el **76% y el 86%** de todos los conflictos reales, o $\tau \in [0,05; 0,15]$ en `KNN_Full_Unweighted`).
> * Si la secretaría necesita **comprimir el calendario en pocos turnos (minimizar `FP` / maximizar Precision)** asegurando que solo se separen los conflictos más probables, subirá el umbral, reduciendo drásticamente la densidad del grafo.

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

## 10. Limitaciones Metodológicas y Alcance de las Métricas

1. **El F1-score mide clasificación de pares, NO calidad del cronograma de exámenes:** Un F1 de `0,2593` (o cualquier métrica de esta fase) evalúa exclusivamente la capacidad de predecir aristas individuales $(u, v)$ en `ml_test_full.csv`. **No mide la calidad final de la grilla de exámenes** (cantidad de turnos/colores requeridos por el algoritmo de coloreo ni cantidad de choques efectivos en el calendario), dado que incluso si un par conflictivo no es predicho como arista (`FN`), dos materias no adyacentes pueden recibir colores distintos durante el coloreo si sus respectivos vecindarios así lo fuerzan.
2. **Desfase generacional entre cohortes (`1989–1993` vs. `1994`):** Al tratarse de cohortes de ingreso dentro de un único semestre (`1994–95, Semestre 1`), las materias exclusivas de 1.er año tienen baja inscripción en las cohortes `1989–1993` (solo recursantes), lo que limita el alcance de `cant_estudiantes_compartidos_historico` y obliga a los modelos a apoyarse en atributos estructurales (`misma_carrera`, `mismo_anio`), acotando el techo de PR-AUC alcanzable (~`0,21`).
3. **Sensibilidad al umbral bajo prevalencia del 1,49%:** Tanto el *undersampling* 50/50 como `class_weight="balanced"` alteran la distribución o ponderación empleada en el entrenamiento y desplazan los puntajes predichos respecto a la prevalencia natural del test (1,49%), mientras que los modelos no ponderados concentran sus probabilidades en valores bajos. Por ello resulta indispensable seleccionar el umbral de decisión $\tau$ sobre un conjunto de validación con prevalencia natural según el compromiso deseado entre Falsos Positivos y Falsos Negativos.
4. **Dependencia residual en pares semi-inductivos:** En la partición por materias, los pares de validación y test donde una materia pertenece al conjunto reservado y la otra fue vista en entrenamiento son semi-inductivos ($714 \times 86 = 61.404$ pares en test), mientras que los pares donde ambas materias pertenecen al conjunto reservado ($\binom{86}{2} = 3.655$ pares en test) son puramente inductivos.

---

## 11. Persistencia, Inferencia y Especificación Técnica para Integración (`joblib`)

El modelo candidato principal se encuentra serializado en `results/models/knn_full_unweighted.joblib` (y verificado de forma independiente en [`notebooks/Test_modelo_ML.ipynb`](file:///w:/home/anton/Tp-Grafos-ML/notebooks/Test_modelo_ML.ipynb)):

1. **Contenido del archivo `.joblib`:** Contiene un objeto `sklearn.pipeline.Pipeline` con dos pasos ya ajustados sobre las `254.541` filas de `ml_train_full.csv`:
   * `('scaler', StandardScaler())`: conserva los vectores `mean_` y `scale_` de las 10 features.
   * `('clf', KNeighborsClassifier(n_neighbors=15, n_jobs=-1, weights='uniform'))`: contiene el `KDTree` entrenado.
2. **Orden estricto de las features y escalado interno:**
   * El pipeline exige un `pandas.DataFrame` con las **10 columnas en el orden exacto de `model.feature_names_in_`**:
     `['mismo_departamento', 'mismo_prefijo_curricular', 'anio_confirmado', 'mismo_anio', 'misma_carrera', 'cant_carreras_compartidas_hist', 'cant_inscriptos_hist_1', 'cant_inscriptos_hist_2', 'cant_estudiantes_compartidos_historico', 'jaccard_historico']`.
   * Si las columnas se entregan desordenadas, `scikit-learn` arroja `ValueError`.
   * **El escalado lo realiza internamente el pipeline** al invocar `predict_proba(X)`; no debe aplicarse ningún escalado manual externo.
3. **Aplicación externa del umbral `0.21` (No usar `model.predict()`):**
   * El archivo `.joblib` **no almacena el umbral `0.21`** (el cual está registrado en `results/metrics/metrics_summary.json`).
   * El método `model.predict(X)` aplica siempre el corte estándar `0.50` (`KNN_Full_Unweighted`). Para obtener las clasificaciones de `KNN_Full_Unweighted_UmbralVal`, se debe invocar siempre:
     ```python
     probabilidades = model.predict_proba(X[list(model.feature_names_in_)])[:, 1]
     predicciones = (probabilidades >= 0.21).astype(int)
     ```
   * *Nota sobre precisión numérica en `test_predictions_all_models.csv`:* En el CSV exportado, las probabilidades `prob_<Modelo>` se guardan redondeadas a 6 decimales (`np.round(..., 6)`), por lo que $\frac{1}{15} = 0,066666666667\dots$ figura como `0.066667` (diferencia máxima de redondeo de $3,33 \times 10^{-7}$), mientras que las clasificaciones binarias con $\tau = 0,21$ coinciden exactamente en el **100% de los 65.059 pares**.

---

## 12. Cómo Ejecutar el Código y Artefactos Generados

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
* `results/models/*.joblib`: los 10 modelos entrenados serializados con `joblib` (ignorados en Git mediante `.gitignore`, conservados localmente).
* [`results/predictions/test_predictions_all_models.csv`](file:///w:/home/anton/Tp-Grafos-ML/results/predictions/test_predictions_all_models.csv): tabla de 65.059 filas que conserva `materia_id_1`, `materia_id_2`, `hubo_conflicto_potencial`, las probabilidades `prob_<Modelo>` y las predicciones binarias `pred_<Modelo>_th050` y `pred_<Modelo>_thVal`.

---

## 13. Cómo se Utilizarán las Probabilidades en la Construcción del Grafo (Fase 4)

La decisión metodológica del proyecto para la **Fase 4 (Teoría de Grafos — Coloreo de Turnos)** es utilizar **coloreo clásico no ponderado** sobre el grafo de incompatibilidades $G(\tau) = (V, E(\tau))$:
1. **Nodos ($V$):** Las materias a planificar (`materia_id`).
2. **Aristas de incompatibilidad parametrizadas por umbral ($E(\tau)$):** La probabilidad predicha $\hat{p}_{uv}$ se utiliza **únicamente** para decidir si se agrega o no la arista $\{u, v\}$ según el umbral seleccionado ($\hat{p}_{uv} \ge \tau$). Cada arista incluida representa una incompatibilidad horaria entre ambas materias (no pueden asignarse al mismo turno).
3. **Coloreo clásico no ponderado:** El algoritmo de coloreo greedy trabaja exclusivamente sobre la estructura de adyacencia de $G(\tau)$ (los grados y vecindarios de cada vértice), **sin utilizar pesos en las aristas**, permitiendo analizar cómo varía la densidad del grafo y la cantidad de colores (turnos de examen necesarios) al modificar el umbral $\tau$.
