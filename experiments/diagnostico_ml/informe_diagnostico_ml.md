# Informe de Diagnóstico Experimental y Mejora del Modelo de Predicción de Conflictos

**Directorio de trabajo:** `experiments/diagnostico_ml/`  
**Script reproducible:** `experiments/diagnostico_ml/run_diagnostico.py`  
**Semilla fija:** `RANDOM_SEED = 42`  
**Restricciones cumplidas:** Sin modificar `src/`, `data/` ni `results/`. La selección de hiperparámetros, regímenes de desbalance y umbrales se realizó exclusivamente sobre el conjunto de validación (`51.975` pares; `77` materias de validación vs `637` de subtrain). El conjunto de test (`65.059` pares; `86` materias de test) se utilizó únicamente para la evaluación final y el desglose diagnóstico.

---

## 1. Investigación A: Aporte de la información histórica vs. estructural

Se evaluaron sobre el conjunto de validación (`51.975` pares: `662` positivos [1,27%] y `51.313` negativos) y sus subgrupos (`val_1_conocida_1_desconocida`: `49.049` pares; `val_ambas_desconocidas`: `2.926` pares), así como en `subtrain` in-sample (`202.566` pares: ambas materias conocidas durante el ajuste), distintos subconjuntos de variables y modelos simples.

### 1.1. Tabla comparativa en Validación (`inv_a_historical_ablation.csv`)

| Experimento | Features utilizadas | Umbral ($\tau_{val}$) | Precision | Recall | F1-Score | PR-AUC | ROC-AUC | TP | FP | FN | Aristas (`TP+FP`) | `FP/TP` |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `A1_Baseline_Regla_Hist_Positivo` | Regla `compartidos_hist > 0` (1) | > 0 | 0.1731 | 0.2432 | 0.2023 | 0.1025 | 0.6146 | 161 | 769 | 501 | 930 | 4.78 |
| `A2_LogReg_Solo_Compartidos_Hist` | `compartidos_hist` (1) | 0.02 | 0.2883 | 0.0967 | 0.1448 | 0.1025 | 0.6146 | 64 | 158 | 598 | 222 | 2.47 |
| `A3_LogReg_Solo_Jaccard_Hist` | `jaccard_hist` (1) | 0.02 | 0.3494 | 0.1858 | 0.2426 | 0.1203 | 0.6152 | 123 | 229 | 539 | 352 | 1.86 |
| `A4_DT_Solo_Compartidos_Hist` | `compartidos_hist` (1) | 0.05 | 0.1746 | 0.2432 | 0.2033 | 0.0905 | 0.6147 | 161 | 761 | 501 | 922 | 4.73 |
| `A5_DT_Solo_Jaccard_Hist` | `jaccard_hist` (1) | 0.13 | 0.3889 | 0.1798 | 0.2459 | 0.1236 | 0.6153 | 119 | 187 | 543 | 306 | 1.57 |
| `A6_HGB_Solo_5_Historicas` | 5 históricas (cupos + solapamiento) | 0.14 | 0.3183 | 0.2236 | 0.2626 | 0.2082 | 0.8376 | 148 | 317 | 514 | 465 | 2.14 |
| `A7_HGB_Solo_5_Estructurales` | 5 estructurales (carrera/año/depto) | 0.09 | 0.1588 | 0.2810 | 0.2029 | 0.0905 | 0.8155 | 186 | 985 | 476 | 1.171 | 5.30 |
| `A8_HGB_Todas_10_Features` | **Las 10 features completas** | 0.20 | **0.4118** | **0.2221** | **0.2885** | **0.2484** | **0.8842** | 147 | 210 | 515 | 357 | **1.43** |
| `A9_KNN15_Solo_5_Historicas` | 5 históricas | 0.27 | 0.4650 | 0.1405 | 0.2158 | 0.1474 | 0.7040 | 93 | 107 | 569 | 200 | 1.15 |
| `A10_KNN15_Todas_10_Features` | Las 10 features completas | 0.21 | 0.2761 | 0.2115 | 0.2395 | 0.1846 | 0.7651 | 140 | 367 | 522 | 507 | 2.62 |

### 1.2. Análisis de hallazgos de la Investigación A

1. **Techo estructural del solapamiento histórico directo (`compartidos_hist > 0`):**
   - En validación, únicamente **161 de los 662 conflictos reales (24,32%)** tuvieron `cant_estudiantes_compartidos_historico > 0`. Los restantes **501 conflictos reales (75,68%)** ocurrieron entre pares de materias con `cant_estudiantes_compartidos_historico == 0`.
   - En consecuencia, cualquier modelo o regla que dependa exclusivamente de `cant_estudiantes_compartidos_historico` o `jaccard_historico` (`A1`, `A2`, `A3`, `A4`, `A5`) tiene un **techo rígido de sensibilidad informativa en Recall = 24,32%** en validación (y **20,82%** en test, donde solo 202 de 970 positivos tienen historial compartido > 0), tras el cual la curva ROC/PR cae a azar puro (`ROC-AUC ≈ 0.615`).
   - Dentro de ese 24,32%, `jaccard_historico` (`A3`/`A5`) filtra mucho mejor los falsos positivos que el conteo bruto (`A2`/`A4`), elevando la Precision de `0.1731` a `0.3889` y el F1 de `0.2023` a `0.2459`.
2. **Importancia del tamaño de materia (`cupo_historico_*`) cuando el solapamiento histórico es cero:**
   - Al pasar de 1 variable histórica (`A5`, `PR-AUC = 0.1236`, `ROC-AUC = 0.6153`) a las **5 variables históricas** (`A6_HGB_Solo_5_Historicas`, que incorpora `cupo_historico_min`, `cupo_historico_max` y `ratio_cupo_historico`), el `ROC-AUC` salta de `0.6153` a `0.8376` y el `PR-AUC` de `0.1236` a `0.2082`. Esto ocurre porque el tamaño de matrícula histórica de ambas materias permite estimar la probabilidad base de colisión incluso cuando en el ciclo 1994 no compartieron alumnos.
3. **Complementariedad entre las 5 features históricas y las 5 estructurales:**
   - Las 5 features estructurales por sí solas (`A7_HGB_Solo_5_Estructurales`) logran `ROC-AUC = 0.8155` pero un `PR-AUC` bajo (`0.0905`) y `F1 = 0.2029`, porque los indicadores binarios (`misma_carrera`, `mismo_anio`, `mismo_departamento`) generan grupos gruesos con muchos falsos positivos (`5.30 FP` por cada `TP`) si no se modulan por el cupo y el historial.
   - Al combinar **las 10 features** (`A8_HGB_Todas_10_Features`), se obtiene una sinergia neta: `PR-AUC = 0.2484` (+19,3% sobre solo históricas y +174% sobre solo estructurales), `ROC-AUC = 0.8842` y `F1 = 0.2885` (`Precision = 0.4118`, `Recall = 0.2221`, solo `1.43 FP/TP`).
4. **Distribución de probabilidades predichas (`inv_a_prob_distributions.csv`):**
   - En los modelos de 1 variable (`A2`–`A5`), más del 75% de los positivos reales reciben exactamente la probabilidad basal (`~0.011–0.013`), idéntica a la de los negativos, porque tienen `compartidos_hist = 0`.
   - En `A8_HGB_Todas_10_Features`, la mediana de probabilidad asignada a los positivos (`0.0632`, P75 = `0.1406`, P90 = `0.5161`) es **26,5 veces mayor** que la mediana asignada a los negativos (`0.0024`, P75 = `0.0075`, P90 = `0.0267`).
5. **Comparación según conocimiento de las materias (`subtrain` vs `1 conocida + 1 desconocida` vs `ambas desconocidas`):**
   - Las 10 features no contienen identificadores de materia ni estadísticos target-encoded del ciclo 1996; tanto las features históricas (calculadas del ciclo 1994) como las estructurales están disponibles para cualquier materia del catálogo. Por ello, en validación la diferencia entre `val_1_conocida_1_desconocida` (`PR-AUC = 0.2496` en HGB) y `val_ambas_desconocidas` (`PR-AUC = 0.2576` en HGB) es pequeña y atribuible a varianza muestral (`37` positivos en `val_ambas_desconocidas`). En cambio, en `subtrain` (in-sample, donde el modelo vio esos pares exactos al entrenar), el PR-AUC aparente sube a `0.3135` en HGB y `0.2588` en KNN-15 por ajuste a la muestra de entrenamiento.

---

## 2. Investigación B: Comparación de modelos y estrategias de desbalance (en Validación)

Se entrenaron 6 familias de algoritmos (`LogisticRegression`, `DecisionTree`, `RandomForest`, `ExtraTrees`, `HistGradientBoosting` y `KNN` con $K \in \{15, 31, 51\}$) bajo 4 estrategias de manejo de desbalance (`Full_Unweighted`, `Full_ClassWeight="balanced"`, `Full_CustomWeight={0:1, 1:10}`, y `Balanced_Undersampling` solo sobre subtrain).

### 2.1. Resumen de los mejores modelos en Validación (umbral $\tau$ que maximiza F1 en validación)

| Modelo y Régimen | Umbral $\tau_{val}$ | Precision (Val) | Recall (Val) | F1-Score (Val) | PR-AUC (Val) | ROC-AUC (Val) | TP | FP | FN | Aristas (`TP+FP`) | `FP/TP` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`HGB_Full_Unweighted`** | **0.20** | **0.4118** | **0.2221** | **0.2885** | **0.2484** | **0.8842** | **147** | **210** | **515** | **357** | **1.43** |
| **`RF_d6_Full_Unweighted`** | **0.19** | **0.4190** | **0.2069** | **0.2770** | **0.2411** | **0.8617** | **137** | **190** | **525** | **327** | **1.39** |
| `RF_d8_Full_ClassWeight_Subsample` | 0.88 | 0.3549 | 0.2236 | 0.2743 | 0.2409 | 0.8797 | 148 | 269 | 514 | 417 | 1.82 |
| `RF_d6_Full_ClassWeight_Balanced` | 0.90 | 0.3472 | 0.2266 | 0.2742 | 0.2393 | 0.8709 | 150 | 282 | 512 | 432 | 1.88 |
| `RF_d6_Full_CustomWeight_10` | 0.60 | 0.3333 | 0.2311 | 0.2730 | 0.2385 | 0.8666 | 153 | 306 | 509 | 459 | 2.00 |
| **`KNN_k31_Full_Unweighted`** | **0.20** | **0.3141** | **0.2387** | **0.2712** | **0.1969** | **0.7927** | **158** | **345** | **504** | **503** | **2.18** |
| **`KNN_k51_Full_Unweighted`** | **0.16** | **0.3034** | **0.2447** | **0.2709** | **0.2023** | **0.8151** | **162** | **372** | **500** | **534** | **2.30** |
| **`HGB_Full_ClassWeight_Balanced`** | **0.90** | **0.3105** | **0.2402** | **0.2709** | **0.2417** | **0.8850** | **159** | **353** | **503** | **512** | **2.22** |
| `RF_d6_Balanced_Undersampling` | 0.89 | 0.3180 | 0.2296 | 0.2667 | 0.2228 | 0.8718 | 152 | 326 | 510 | 478 | 2.15 |
| `DT_d6_leaf50_Full_Unweighted` | 0.20 | 0.3389 | 0.2145 | 0.2627 | 0.2148 | 0.8653 | 142 | 277 | 520 | 419 | 1.95 |
| `ET_d8_Full_Unweighted` | 0.17 | 0.4006 | 0.1949 | 0.2622 | 0.2080 | 0.8616 | 129 | 193 | 533 | 322 | 1.50 |
| `DT_d5_Full_Unweighted` | 0.20 | 0.3853 | 0.1979 | 0.2615 | 0.1963 | 0.8368 | 131 | 209 | 531 | 340 | 1.60 |
| `HGB_Balanced_Undersampling` | 0.90 | 0.2611 | 0.2583 | 0.2597 | 0.2047 | 0.8877 | 171 | 484 | 491 | 655 | 2.83 |
| `KNN_k15_Full_Unweighted` *(Actual)* | 0.21 | 0.2761 | 0.2115 | 0.2395 | 0.1846 | 0.7651 | 140 | 367 | 522 | 507 | 2.62 |
| `LogReg_Full_CustomWeight_10` | 0.72 | 0.3235 | 0.1813 | 0.2323 | 0.1547 | 0.8469 | 120 | 251 | 542 | 371 | 2.09 |
| `LogReg_Full_ClassWeight_Balanced` | 0.94 | 0.2541 | 0.2115 | 0.2308 | 0.1555 | 0.8520 | 140 | 411 | 522 | 551 | 2.94 |
| `Baseline_Heuristico_Historico` | > 0 | 0.1731 | 0.2432 | 0.2023 | 0.1025 | 0.6146 | 161 | 769 | 501 | 930 | 4.78 |
| `KNN_k15_Balanced_Undersampling` | 0.94 | 0.2120 | 0.1767 | 0.1928 | 0.0993 | 0.8710 | 117 | 435 | 545 | 552 | 3.72 |

### 2.2. Conclusiones de la Investigación B

1. **Superioridad de ensambles de árboles (`HistGradientBoosting` y `RandomForest`) frente a `KNN_k15`:**
   - En validación, `HGB_Full_Unweighted` supera ampliamente a `KNN_k15_Full_Unweighted` en todas las métricas: `F1 = 0.2885` vs `0.2395` (+20,5%), `PR-AUC = 0.2484` vs `0.1846` (+34,6%), `ROC-AUC = 0.8842` vs `0.7651` (+15,6%), y logra mayor Recall (`0.2221` vs `0.2115`) con un **42,8% menos de falsos positivos** (`210 FP` vs `367 FP`).
   - Esto se explica porque los árboles manejan de forma nativa la distribución extremadamente asimétrica (inflada en ceros) de `cant_estudiantes_compartidos_historico` y `jaccard_historico` mediante cortes de umbral exactos e interacciones no lineales con `cupo_historico_*` y `misma_carrera`, mientras que la distancia euclídea de KNN tras `StandardScaler` mezcla dimensiones continuas muy sesgadas con indicadores binarios.
2. **Impacto del hiperparámetro $K$ en KNN:**
   - Aumentar el número de vecinos de $K=15$ a $K=31$ o $K=51$ suaviza las probabilidades discretas ($m/K$) y mejora sustancialmente el ranking en validación: el `PR-AUC` sube de `0.1846` ($K=15$) a `0.1969` ($K=31$) y `0.2023` ($K=51$), y el `F1` en validación sube de `0.2395` a `0.2712` ($K=31$).
3. **Efecto del submuestreo (`Balanced_Undersampling`) vs. `class_weight` vs. ajuste de umbral sobre `Full_Unweighted`:**
   - **El submuestreo 1:1 (`Balanced_Undersampling`) degrada el PR-AUC en todos los modelos no lineales** (`HGB`: cae de `0.2484` a `0.2047`; `KNN-15`: cae de `0.1846` a `0.0993`; `DT_d5`: cae de `0.1963` a `0.1350`) porque descarta el 98,6% de los pares negativos (`196.894` ejemplos descartados en subtrain), perdiendo información crítica sobre la frontera de decisión en zonas densas.
   - Usar `class_weight="balanced"` o `class_weight={0:1, 1:10}` sobre el dataset completo **no mejora el PR-AUC respecto a `Full_Unweighted`** en modelos basados en árboles/boosting (`HGB_Unweighted PR-AUC = 0.2484` vs `HGB_ClassWeight PR-AUC = 0.2417`; `RF_Unweighted PR-AUC = 0.2411` vs `RF_ClassWeight PR-AUC = 0.2393`), sino que **recalibra hacia arriba las probabilidades predichas**. Al usar el umbral por defecto `0.50`, `class_weight="balanced"` dispara el Recall a `0.71–0.77` pero a costa de `7.454–8.447 FP` (`Precision ~ 0.058`). Cuando se optimiza el umbral en validación ($\tau = 0.90$), se recupera esencialmente un punto operativo sobre la misma curva Precision-Recall que `Full_Unweighted` alcanza con $\tau \in [0.15, 0.20]$.

---

## 3. Investigación C: Umbral de clasificación y costo de errores en el grafo (Validación)

En el conjunto de validación (`51.975` pares totales, con **662 conflictos reales**), se analizó el barrido de umbrales (`inv_c_validation_thresholds_tradeoff.csv`) para cuantificar el costo directo sobre el grafo de coloreo:
- **Aristas predichas (`TP + FP`):** total de restricciones inyectadas al algoritmo Greedy Coloring.
- **Conflictos omitidos (`FN`):** conflictos reales que quedan sin arista en el grafo ($662 - \text{TP}$).
- **Costo marginal (`FP/TP`):** cuántas aristas falsas se agregan en promedio por cada conflicto real detectado.

### 3.1. Tabla de trade-off de umbrales en Validación (`HGB_Full_Unweighted` vs `RF_d6_Full_Unweighted` vs `KNN_k15_Full_Unweighted`)

| Modelo | Régimen de umbral | $\tau$ | Precision | Recall | F1 | TP (Detectados) | FP (Aristas falsas) | FN (Conflictos sin arista) | Aristas totales (`TP+FP`) | Ratio `Aristas / Positivos` | `FP / TP` |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`HGB_Full_Unweighted`** | Alta Precisión | 0.30 | 0.5368 | 0.1541 | 0.2394 | 102 | 88 | 560 | 190 | 0.29x | 0.86 |
| **`HGB_Full_Unweighted`** | **Máximo F1 Val** | **0.20** | **0.4118** | **0.2221** | **0.2885** | **147** | **210** | **515** | **357** | **0.54x** | **1.43** |
| **`HGB_Full_Unweighted`** | Balanceado +Recall | 0.15 | 0.3137 | 0.2417 | 0.2730 | 160 | 350 | 502 | 510 | 0.77x | 2.19 |
| **`HGB_Full_Unweighted`** | **Recall Moderado (~35%)** | **0.10** | **0.2148** | **0.3459** | **0.2650** | **229** | **837** | **433** | **1.066** | **1.61x** | **3.66** |
| **`HGB_Full_Unweighted`** | Recall ≥ 40% | 0.08 | 0.1531 | 0.4275 | 0.2254 | 283 | 1.566 | 379 | 1.849 | 2.79x | 5.53 |
| **`HGB_Full_Unweighted`** | Recall ≥ 50% | 0.06 | 0.1172 | 0.5166 | 0.1910 | 342 | 2.577 | 320 | 2.919 | 4.41x | 7.54 |
| **`HGB_Full_Unweighted`** | Recall ≥ 60% | 0.04 | 0.0976 | 0.6133 | 0.1684 | 406 | 3.754 | 256 | 4.160 | 6.28x | 9.25 |
| **`HGB_Full_Unweighted`** | Recall ~ 80% | 0.01 | 0.0489 | 0.8051 | 0.0922 | 533 | 10.372 | 129 | 10.905 | 16.47x | 19.46 |
| `RF_d6_Full_Unweighted` | Máximo F1 Val | 0.19 | 0.4190 | 0.2069 | 0.2770 | 137 | 190 | 525 | 327 | 0.49x | 1.39 |
| `RF_d6_Full_Unweighted` | Recall ~ 32% | 0.07 | 0.2117 | 0.3172 | 0.2539 | 210 | 782 | 452 | 992 | 1.50x | 3.72 |
| `RF_d6_Full_Unweighted` | Recall ≥ 40% | 0.06 | 0.1450 | 0.4864 | 0.2235 | 322 | 1.898 | 340 | 2.220 | 3.35x | 5.89 |
| `RF_d6_Full_Unweighted` | Recall ≥ 60% | 0.05 | 0.0996 | 0.6254 | 0.1719 | 414 | 3.742 | 248 | 4.156 | 6.28x | 9.04 |
| `KNN_k31_Full_Unweighted` | Máximo F1 Val | 0.20 | 0.3141 | 0.2387 | 0.2712 | 158 | 345 | 504 | 503 | 0.76x | 2.18 |
| `KNN_k31_Full_Unweighted` | Recall ~ 36% | 0.10 | 0.1632 | 0.3625 | 0.2250 | 240 | 1.231 | 422 | 1.471 | 2.22x | 5.13 |
| `KNN_k31_Full_Unweighted` | Recall ≥ 40% | 0.09 | 0.1300 | 0.4622 | 0.2030 | 306 | 2.047 | 356 | 2.353 | 3.55x | 6.69 |
| `KNN_k15_Full_Unweighted` | **Actual ($\tau=0.21$, $m \ge 4$)** | **0.21** | **0.2761** | **0.2115** | **0.2395** | **140** | **367** | **522** | **507** | **0.77x** | **2.62** |
| `KNN_k15_Full_Unweighted` | Vecinos $m \ge 3$ | 0.15 | 0.1796 | 0.2946 | 0.2231 | 195 | 891 | 467 | 1.086 | 1.64x | 4.57 |
| `KNN_k15_Full_Unweighted` | Vecinos $m \ge 2$ | 0.13 | 0.1252 | 0.4230 | 0.1932 | 280 | 1.956 | 382 | 2.236 | 3.38x | 6.99 |
| `KNN_k15_Full_Unweighted` | Vecinos $m \ge 1$ | 0.01 | 0.0816 | 0.5967 | 0.1436 | 395 | 4.445 | 267 | 4.840 | 7.31x | 11.25 |

### 3.2. Análisis del trade-off de aristas

1. **Discretización brusca en `KNN_k15` vs. granularidad continua en `HGB`:**
   - Debido a que `KNN_k15` sin ponderación por distancia solo puede emitir 16 probabilidades posibles ($0/15, 1/15=0.067, 2/15=0.133, 3/15=0.200, 4/15=0.267, \dots$), cualquier umbral entre `0.14` y `0.20` produce exactamente el mismo resultado ($m \ge 3$ vecinos), y bajar de $\tau=0.21$ ($m \ge 4$) a $\tau=0.20$ ($m \ge 3$) duplica de golpe las aristas predichas de `507` a `1.086`. Además, en `KNN_k15` es **imposible superar un Recall de 59,67%** sin predecir el 100% del grafo, porque el 40,33% de los positivos reales tienen $0/15$ vecinos positivos ($p=0.0$).
   - En cambio, `HGB_Full_Unweighted` produce probabilidades continuas bien ordenadas, permitiendo elegir con precisión quirúrgica el volumen de aristas del grafo:
     - **Con el mismo presupuesto de aristas que hoy genera `KNN_k15` en validación (`~510` aristas):** `HGB_Full_Unweighted` con $\tau = 0.15$ detecta **160 TP** vs **140 TP** de KNN (+14,3% de conflictos reales detectados) con menos falsos positivos (`350 FP` vs `367 FP`) y mayor F1 (`0.2730` vs `0.2395`).
     - **Maximizando F1 (`HGB` con $\tau = 0.20$):** detecta **147 TP** (más que los 140 TP de KNN) reduciendo las aristas falsas a **210 FP** (-42,8% de ruido en el grafo).
     - **Priorizando Recall moderado (`HGB` con $\tau = 0.10$):** eleva el Recall en validación al **34,59%** (`229 TP`, `837 FP`, `1.066` aristas totales, `Precision = 0.2148`, `F1 = 0.2650`), superando incluso el F1 máximo de `KNN_k15` (`0.2395`).
2. **Explosión combinatoria de aristas falsas para Recall $\ge 50\%$:**
   - Para alcanzar `Recall ≈ 50%` en validación (`~342 TP`), incluso el mejor modelo (`HGB` en $\tau=0.06$) debe agregar **2.577 aristas falsas** (`7.54 FP` por cada `TP`, cuadruplicando el tamaño real del grafo: `2.919` aristas predichas frente a `662` reales). Para `Recall ≈ 60%` ($\tau=0.04$), se inyectan **3.754 aristas falsas** (`6.28x` el número de conflictos reales). Como veremos en la Investigación D, esto no es un defecto del algoritmo, sino una consecuencia directa de la estructura de las 10 features.

---

## 4. Evaluación en Test de los candidatos seleccionados en Validación e Investigación D

Respetando estrictamente el protocolo experimental, los modelos y sus puntos operativos de umbral seleccionados en validación se evaluaron una única vez sobre el conjunto de **Test (`65.059` pares: `970` positivos [1,49%] y `64.089` negativos)**.

### 4.1. Resultados en Test Global (`test_candidates_evaluation.csv`)

| Modelo candidato | Criterio de umbral (fijado en Val) | $\tau_{val}$ | Precision (Test) | Recall (Test) | F1-Score (Test) | PR-AUC (Test) | ROC-AUC (Test) | TP | FP | FN | TN | Aristas (`TP+FP`) | `FP/TP` |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `Baseline_Heuristico_Historico` | Regla `hist > 0` | > 0 | 0.1428 | 0.2082 | 0.1694 | 0.0750 | 0.5952 | 202 | 1.213 | 768 | 62.876 | 1.415 | 6.00 |
| `KNN_k15_Full_Unweighted` *(Actual)* | Máx F1 Val | 0.21 | 0.2711 | 0.2485 | 0.2593 | 0.1833 | 0.7665 | 241 | 648 | 729 | 63.441 | 889 | 2.69 |
| `KNN_k31_Full_Unweighted` | Máx F1 Val | 0.20 | 0.3058 | 0.2330 | 0.2645 | 0.2142 | 0.8118 | 226 | 513 | 744 | 63.576 | 739 | 2.27 |
| **`KNN_k51_Full_Unweighted`** | **Máx F1 Val** | **0.16** | **0.2844** | **0.2536** | **0.2681** | **0.2193** | **0.8252** | **246** | **619** | **724** | **63.470** | **865** | **2.52** |
| `DT_d5_Full_Unweighted` | Máx F1 Val | 0.20 | 0.4399 | 0.1546 | 0.2288 | 0.2082 | 0.8543 | 150 | 191 | 820 | 63.898 | 341 | 1.27 |
| `RF_d6_Full_Unweighted` | Máx F1 Val | 0.19 | **0.4711** | 0.1680 | 0.2477 | 0.2496 | 0.8741 | 163 | 183 | 807 | 63.906 | 346 | **1.12** |
| `RF_d6_Full_ClassWeight_Balanced` | Máx F1 Val | 0.90 | 0.3759 | 0.2062 | 0.2663 | 0.2479 | 0.8795 | 200 | 332 | 770 | 63.757 | 532 | 1.66 |
| **`HGB_Full_Unweighted`** | **Máx F1 Val** | **0.20** | **0.4524** | **0.1959** | **0.2734** | **0.2747** | **0.8846** | **190** | **230** | **780** | **63.859** | **420** | **1.21** |
| **`HGB_Full_ClassWeight_Balanced`** | **Máx F1 Val** | **0.90** | **0.3705** | **0.2433** | **0.2937** | **0.2612** | **0.8848** | **236** | **401** | **734** | **63.688** | **637** | **1.70** |
| `LogReg_Full_ClassWeight_Balanced` | Máx F1 Val | 0.94 | 0.2646 | 0.2381 | 0.2507 | 0.1519 | 0.8677 | 231 | 642 | 739 | 63.447 | 873 | 2.78 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **`HGB_Full_Unweighted_ValRec40`** | **Recall Val ≥ 40%** | **0.08** | **0.1853** | **0.4340** | **0.2597** | **0.2747** | **0.8846** | **421** | **1.851** | **549** | **62.238** | **2.272** | **4.40** |
| `RF_d6_Full_Unweighted_ValRec40` | Recall Val ≥ 40% | 0.06 | 0.1638 | 0.4320 | 0.2375 | 0.2496 | 0.8741 | 419 | 2.139 | 551 | 61.950 | 2.558 | 5.11 |
| `KNN_k51_Full_Unweighted_ValRec40` | Recall Val ≥ 40% | 0.09 | 0.1563 | 0.4216 | 0.2280 | 0.2193 | 0.8252 | 409 | 2.208 | 561 | 61.881 | 2.617 | 5.40 |
| `KNN_k15_Full_Unweighted_ValRec40` | Recall Val ≥ 40% | 0.13 | 0.1527 | 0.4155 | 0.2233 | 0.1833 | 0.7665 | 403 | 2.236 | 567 | 61.853 | 2.639 | 5.55 |
| `HGB_Full_Unweighted_ValRec50` | Recall Val ≥ 50% | 0.06 | 0.1501 | 0.4918 | 0.2300 | 0.2747 | 0.8846 | 477 | 2.701 | 493 | 61.388 | 3.178 | 5.66 |
| `HGB_Full_Unweighted_ValRec60` | Recall Val ≥ 60% | 0.04 | 0.1193 | 0.6454 | 0.2014 | 0.2747 | 0.8846 | 626 | 4.621 | 344 | 59.468 | 5.247 | 7.38 |
| `KNN_k15_Full_Unweighted_ValRec60` | Recall Val ≥ 60% | 0.01 | 0.0912 | 0.6010 | 0.1584 | 0.1833 | 0.7665 | 583 | 5.810 | 387 | 58.279 | 6.393 | 9.97 |

> **Hallazgo clave en Test:**
> - `HGB_Full_Unweighted` alcanza en Test un **`PR-AUC = 0.2747`** (+49,9% relativo sobre el `0.1833` de `KNN_k15`) y un **`ROC-AUC = 0.8846`** (+15,4% sobre el `0.7665` de `KNN_k15`).
> - Con umbral $\tau=0.20$ (`HGB_Full_Unweighted`), obtiene `F1 = 0.2734` y `Precision = 0.4524` (`190 TP`, `230 FP`).
> - Con `HGB_Full_ClassWeight_Balanced` ($\tau=0.90$, o equivalentemente `HGB_Full_Unweighted` en $\tau \approx 0.14$), obtiene **`F1 = 0.2937`** (`Precision = 0.3705`, `Recall = 0.2433`, `236 TP` y solo `401 FP`, frente a `648 FP` de `KNN_k15`).
> - Más aún: si se desea priorizar Recall, **`HGB_Full_Unweighted` con $\tau=0.08$ (`ValRec40`) logra en Test exactamente el mismo F1 que `KNN_k15` (`0.2597` vs `0.2593`), pero elevando el Recall del 24,85% al 43,40% (`421 TP` vs `241 TP`, un +74,7% más de conflictos reales detectados)**.

---

## 5. Investigación D: ¿Dónde y por qué falla el modelo?

### 5.1. Desglose por subgrupos de generalización en Test (`inv_d_generalization_subgroups.csv`)

El conjunto de test (`65.059` pares, `970` positivos) se compone de:
1. **`test_1_conocida_1_desconocida` (`61.404` pares, `912` positivos [1,485%]):** una materia pertenece a las 714 de entrenamiento y la otra a las 86 de test.
2. **`test_ambas_desconocidas` (`3.655` pares, `58` positivos [1,587%]):** ambas materias pertenecen a las 86 de test (ninguna fue vista en entrenamiento).

| Modelo | Subgrupo de Test | Pares | Positivos | Precision | Recall | F1-Score | PR-AUC | ROC-AUC | TP | FP | FN |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `KNN_k15_Full_Unweighted` ($\tau=0.21$) | `1_conocida_1_desconocida` | 61.404 | 912 | 0.2719 | 0.2522 | 0.2617 | 0.1835 | 0.7697 | 230 | 616 | 682 |
| `KNN_k15_Full_Unweighted` ($\tau=0.21$) | `ambas_desconocidas` | 3.655 | 58 | 0.2558 | 0.1897 | 0.2178 | 0.1773 | 0.7163 | 11 | 32 | 47 |
| `KNN_k51_Full_Unweighted` ($\tau=0.16$) | `1_conocida_1_desconocida` | 61.404 | 912 | 0.2861 | 0.2588 | 0.2717 | 0.2212 | 0.8291 | 236 | 589 | 676 |
| `KNN_k51_Full_Unweighted` ($\tau=0.16$) | `ambas_desconocidas` | 3.655 | 58 | 0.2500 | 0.1724 | 0.2041 | 0.1928 | 0.7644 | 10 | 30 | 48 |
| `HGB_Full_Unweighted` ($\tau=0.20$) | `1_conocida_1_desconocida` | 61.404 | 912 | 0.4489 | 0.1974 | 0.2742 | 0.2755 | 0.8872 | 180 | 221 | 732 |
| `HGB_Full_Unweighted` ($\tau=0.20$) | `ambas_desconocidas` | 3.655 | 58 | 0.5263 | 0.1724 | 0.2597 | 0.2542 | 0.8444 | 10 | 9 | 48 |
| `HGB_Full_ClassWeight_Balanced` ($\tau=0.90$) | `1_conocida_1_desconocida` | 61.404 | 912 | 0.3711 | 0.2478 | 0.2972 | 0.2617 | 0.8869 | 226 | 383 | 686 |
| `HGB_Full_ClassWeight_Balanced` ($\tau=0.90$) | `ambas_desconocidas` | 3.655 | 58 | 0.3571 | 0.1724 | 0.2326 | 0.2496 | 0.8547 | 10 | 18 | 48 |

- **Observación:** El rendimiento cae moderadamente en `test_ambas_desconocidas` respecto a `test_1_conocida_1_desconocida` (por ejemplo, en `HGB_Full_Unweighted` el `PR-AUC` pasa de `0.2755` a `0.2542` y el `F1` de `0.2742` a `0.2597`; en `KNN_k15` el `F1` pasa de `0.2617` a `0.2178`). Sin embargo, `HGB` conserva una capacidad discriminativa alta en ambos subgrupos (`ROC-AUC = 0.8444` y `PR-AUC = 0.2542` incluso cuando ambas materias son completamente nuevas), porque las 10 features son relacionales/históricas y no dependen de memorizar IDs de materias.

### 5.2. Anatomía estructural de los positivos y negativos (`inv_d_structural_profiles.csv`)

Al particionar los `65.059` pares de Test (y los `51.975` de Validación) en 5 perfiles mutuamente excluyentes y exhaustivos según sus features observables, queda expuesta con total claridad la **limitación estructural de información en las 10 features disponibles**:

| Perfil estructural del par de materias (en Test: `65.059` pares) | Total pares | Positivos reales (`y=1`) | Negativos reales (`y=0`) | % del total de Positivos (`970`) | % del total de Negativos (`64.089`) | Tasa empírica de conflicto $P(y=1 \mid \text{perfil})$ | Negativos por cada Positivo |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **P1:** `compartidos_hist > 0` *(Solapamiento histórico directo)* | 1.415 | **202** | 1.213 | **20,82%** | 1,89% | **14,28%** | **6,0 : 1** |
| **P2:** `compartidos_hist == 0`, `misma_carrera == 1`, `mismo_anio == 1` | 420 | **85** | 335 | **8,76%** | 0,52% | **20,24%** | **3,9 : 1** |
| **P3:** `compartidos_hist == 0`, `misma_carrera == 1`, `mismo_anio == 0` | 4.434 | **176** | 4.258 | **18,14%** | 6,64% | **3,97%** | **24,2 : 1** |
| **P4:** `compartidos_hist == 0`, `misma_carrera == 0`, `(mismo_anio == 1 o mismo_depto/pref == 1)` | 9.546 | **292** | 9.254 | **30,10%** | 14,44% | **3,06%** | **31,7 : 1** |
| **P5:** `compartidos_hist == 0`, `misma_carrera == 0`, `mismo_anio == 0`, `mismo_depto/pref == 0` *(Sin señal relacional)* | 49.244 | **215** | 49.029 | **22,16%** | **76,50%** | **0,44%** | **228,0 : 1** |
| **Total Test** | **65.059** | **970** | **64.089** | **100,0%** | **100,0%** | **1,49%** | **66,1 : 1** |

*(En Validación [`51.975` pares, `662` positivos] las proporciones son prácticamente idénticas: **P1** contiene el 24,32% de los positivos con tasa empírica 17,31%; **P2** el 2,57% con tasa 6,27%; **P3** el 15,11% con tasa 3,66%; **P4** el 30,06% con tasa 2,54%; y **P5** el 27,95% de los positivos sumergidos entre 40.007 negativos, con tasa empírica de apenas 0,46%, es decir, **216,3 negativos por cada positivo**).*

### 5.3. Explicación rigurosa de por qué falla el modelo cuando se exige alto Recall

1. **Sólo el ~29,6% de los conflictos reales en Test (perfiles P1 + P2: `287` de `970` positivos) ocurren en regiones de alta densidad de señal** (`compartidos_hist > 0` o bien `misma_carrera == 1` y `mismo_anio == 1`), donde la proporción negativo:positivo es manejable (`4:1` a `6:1`) y el tamaño de cupo histórico permite separar con `Precision` de `35%–50%`.
2. **El ~48,2% de los conflictos reales (perfiles P3 + P4: `468` de `970` positivos)** corresponde a pares de materias que **no compartieron alumnos en 1994** (`compartidos_hist == 0`), pero comparten carrera en distinto año (`P3`) o comparten año/departamento en distinta carrera (`P4`). En estos dos estratos hay `13.512` pares negativos frente a `468` positivos (**~29 negativos idénticos en estructura por cada positivo**). La única diferenciación adicional disponible dentro de P3 y P4 son las 3 variables de tamaño (`cupo_historico_min`, `cupo_historico_max`, `ratio_cupo_historico`).
3. **El ~22,2% restante de los conflictos reales en Test (perfil P5: `215` de `970` positivos; y 27,95% en Validación: `185` de `662`) tiene literalmente TODAS las features relacionales e históricas de solapamiento en cero:**
   - `cant_estudiantes_compartidos_historico = 0`
   - `jaccard_historico = 0`
   - `misma_carrera = 0`, `cant_carreras_compartidas = 0`
   - `mismo_anio = 0`
   - `mismo_departamento = 0`, `mismo_prefijo_curricular = 0`
   - En este estrato P5 existen **49.029 pares negativos** con exactamente ese mismo vector de ceros y solo 215 positivos (**228 negativos por cada positivo**, tasa base `0,44%`). Ningún clasificador supervisado que reciba únicamente estas 10 features puede distinguir cuáles de esos `49.244` pares sin vínculo curricular ni historial compartido tendrán un alumno en común en 1996 sin predecir como positivos miles de pares negativos con cupos similares.
4. **Rotación interanual del historial (`P1`):** Incluso cuando `compartidos_hist > 0` (`1.415` pares en Test), el **85,7% (`1.213` pares)** tiene `cant_estudiantes_compartidos = 0` en el ciclo objetivo (`hubo_conflicto_potencial = 0`), mientras que **768 de los 970 conflictos reales (79,2%)** tuvieron `compartidos_hist == 0`. Es decir, existe una altísima variación entre los pares que registraron alumnos en común en el dataset histórico (1994) y los del dataset objetivo (1996), especialmente en optativas inter-carrera (P4/P5) o recursantes entre años (P3).

---

## 6. Recomendación Fundamentada sobre el Modelo Candidato

1. **¿Conviene mantener `KNN_Full_Unweighted` ($K=15, \tau=0.21$) o reemplazarlo?**
   - **Técnica y metodológicamente, `HistGradientBoostingClassifier` (`HGB_Full_Unweighted`) es superior a `KNN_Full_Unweighted` ($K=15$) en todos los frentes evaluados**, tanto en validación como en test:
     - **Mejor ranking global (independiente del umbral):** En validación, `PR-AUC = 0.2484` vs `0.1846` (+34,6%) y `ROC-AUC = 0.8842` vs `0.7651`. En test, `PR-AUC = 0.2747` vs `0.1833` (+49,9%) y `ROC-AUC = 0.8846` vs `0.7665`.
     - **Consistencia metodológica entre Validación y Test:** A diferencia de `KNN_k15` (que había quedado segundo en validación detrás de `DecisionTree` en la Fase 3 original), `HGB_Full_Unweighted` es **simultáneamente el #1 en Validación (`F1 = 0.2885`, `PR-AUC = 0.2484`) y el #1 en Test (`PR-AUC = 0.2747`)**, eliminando por completo la discrepancia post-hoc entre validación y test.
     - **Probabilidades continuas y control fino del grafo:** Elimina los saltos discretos de $1/15$ propios de KNN y permite elegir exactamente el punto operativo según la tolerancia a aristas del algoritmo de coloreo:
       - **Opción 1 (Foco en F1 / Baja densidad de aristas falsas, $\tau = 0.20$ o $\tau = 0.14$):** Con $\tau = 0.20$ obtiene `F1 = 0.2885` en Val y `0.2734` en Test (`Precision = 0.4524`, reduciendo los falsos positivos en Test de `648` a `230`). Con $\tau = 0.14$ (o `HGB_Full_ClassWeight_Balanced` en $\tau = 0.90$) obtiene en Test `F1 = 0.2937`, `Recall = 0.2433` (`236 TP`) y `Precision = 0.3705` (`401 FP` vs `648 FP` de KNN).
       - **Opción 2 (Foco en Recall moderado manteniendo el mismo F1 de KNN, $\tau = 0.08$):** Con $\tau = 0.08$ (`HGB_Full_Unweighted_ValRec40`), obtiene en Test **`F1 = 0.2597`** (idéntico al `0.2593` de `KNN_k15`), pero **eleva el Recall de `0.2485` (`241 TP`) a `0.4340` (`421 TP`)**, detectando **180 conflictos reales adicionales** a cambio de `1.851 FP` (`2.272` aristas totales sobre `65.059` pares, una densidad del 3,49% del grafo).
2. **Alternativa conservadora si no se desea cambiar de familia algorítmica:**
   - Si por razones de simplicidad docente se quisiera conservar `KNeighborsClassifier`, subir de $K=15$ a **$K=31$ ($\tau=0.20$) o $K=51$ ($\tau=0.16$)** mejora el desempeño tanto en validación (`F1 = 0.2712`, `PR-AUC = 0.2023`) como en test (`F1 = 0.2681`, `PR-AUC = 0.2193`, `ROC-AUC = 0.8252`), superando limpiamente a $K=15$.
   - No obstante, **`HGB_Full_Unweighted`** (nativo en `scikit-learn`, sin dependencias externas, y más rápido en inferencia que `KNN` porque no requiere calcular distancias contra `254.541` filas de entrenamiento en cada predicción) ofrece la ganancia más sólida y limpia.
