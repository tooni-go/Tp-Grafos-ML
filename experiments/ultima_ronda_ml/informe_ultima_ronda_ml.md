# Informe de Última Ronda de Investigación y Mejora del Modelo ML (`Tp-Grafos-ML`)

**Fecha:** 2026-11-07  
**Directorio experimental:** `experiments/ultima_ronda_ml/`  
**Restricción de integridad:** Ningún archivo oficial de `src/`, `data/` ni `results/` fue modificado. Todos los modelos y umbrales fueron ajustados exclusivamente sobre el split interno por materia (`subtrain`: 637 materias / 254.541 pares; `validation`: 77 materias / 51.975 pares, 662 positivos) con `RANDOM_SEED = 42`, reportando `test` (86 materias / 65.059 pares, 970 positivos) únicamente como verificación final.

---

## 1. Investigación A — Análisis y Selección de Features

### 1.1 Importancia individual y comportamiento univariado (`subtrain` / `validation`)

La tabla completa se encuentra en [`results/inv_a_univariate_features.csv`](file:///w:/home/anton/Tp-Grafos-ML/experiments/ultima_ronda_ml/results/inv_a_univariate_features.csv). Resumen de las 10 features actuales en `subtrain` ($N = 254.541$, tasa base positiva $1,114\%$):

| Feature | Tipo | Tasa de ceros | Media ($y=0$) | Media ($y=1$) | Corr. Pearson ($r$) | Info. Mutua | ROC-AUC (Val) | PR-AUC (Val) | Diagnóstico |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `jaccard_historico` | Continua | 98,14% | 0,0008 | 0,0126 | **+0,1853** | 0,00438 | 0,6147 | **0,1203** | **Señal individual más fuerte** (co-inscripción histórica normalizada). |
| `cant_carreras_compartidas_hist` | Conteo | 88,92% | 0,1455 | 0,8896 | **+0,1757** | 0,00694 | **0,6767** | 0,0780 | **2.ª señal más fuerte**; distingue solapamiento fuerte vs débil de planes. |
| `misma_carrera` | Binaria | 88,92% | 0,1061 | 0,5242 | **+0,1587** | 0,00668 | 0,6743 | 0,0495 | Indicador binario exacto de `cant_carreras_compartidas_hist > 0`. |
| `mismo_prefijo_curricular` | Binaria | 96,17% | 0,0356 | 0,2754 | **+0,1307** | 0,00383 | 0,5873 | 0,0400 | Más específica y limpia que `mismo_departamento`. |
| `cant_estudiantes_compartidos_historico` | Conteo | 98,14% | 0,0371 | 0,7292 | **+0,1248** | 0,00436 | 0,6146 | 0,1025 | Alta cola derecha (máx. 86); mejora al aplicar `log1p`. |
| `mismo_departamento` | Binaria | 95,38% | 0,0435 | 0,2874 | **+0,1207** | 0,00349 | 0,5899 | 0,0368 | Fuertemente colineal con `mismo_prefijo_curricular` ($r = 0,9088$). |
| `mismo_anio` | Binaria | 83,31% | 0,1627 | 0,5363 | **+0,1074** | 0,00368 | 0,6263 | 0,0292 | Muy informativa cuando está disponible, pero limitada por códigos no numéricos. |
| `anio_confirmado` | Binaria | 53,95% | 0,4594 | 0,5504 | +0,0194 | 0,00024 | **0,5008** | 0,0128 | **Casi nula por sí sola**; solo actúa como máscara de validez de `mismo_anio`. |
| `cant_inscriptos_hist_1` | Conteo | 5,97% | **28,12** | **17,29** | **-0,0328** | 0,00210 | **0,4052** | 0,0105 | **Asimétrica e invertida**: depende del orden alfabético del ID y se asocia negativamente con $y=1$. |
| `cant_inscriptos_hist_2` | Conteo | 4,58% | **26,57** | **16,25** | **-0,0329** | 0,00249 | **0,4200** | 0,0107 | **Asimétrica e invertida**: las materias con conflicto en `actual_1994` tienen menos inscriptos en `1989–1993`. |

### 1.2 Correlaciones y redundancias (`inv_a_correlation_matrix.csv`)

Existen **tres pares de features altamente correlacionadas** en [`results/inv_a_correlation_matrix.csv`](file:///w:/home/anton/Tp-Grafos-ML/experiments/ultima_ronda_ml/results/inv_a_correlation_matrix.csv):

1. **`mismo_departamento` vs `mismo_prefijo_curricular` ($r = 0,9088$):** En el parser actual (`parse_nott.py`), `department_code = exam_id[:2]` y ` prefix = description[:3]`. Casi todos los departamentos usan un único prefijo de 3 letras; `mismo_prefijo_curricular` tiene mayor correlación con el target ($0,1307$ vs $0,1207$) y mayor `PR-AUC` univariado ($0,0400$ vs $0,0368$).
2. **`misma_carrera` vs `cant_carreras_compartidas_hist` ($r = 0,8380$):** Por construcción en `build_subject_pairs.py`, `misma_carrera = int(cant_carreras_compartidas_hist > 0)`. En árboles son redundantes (un corte en `0.5` sobre `cant_carreras_compartidas_hist` reproduce exactamente `misma_carrera`), pero en modelos lineales conservar el indicador binario del salto $0 \to 1$ junto con `log1p(cant_carreras_compartidas_hist)` mejora el ajuste.
3. **`cant_estudiantes_compartidos_historico` vs `jaccard_historico` ($r = 0,7562$):** Ambas son cero en el $98,14\%$ de los pares y positivas en el mismo $1,86\%$, pero capturan escalas complementarias (magnitud absoluta vs proporción sobre la unión).

### 1.3 Defecto estructural de `cant_inscriptos_hist_1` y `cant_inscriptos_hist_2`

En `build_subject_pairs.py`, los pares se generan ordenados lexicográficamente (`materia_id_1 < materia_id_2`). Por ende, asignar el tamaño histórico a la columna `_1` o `_2` depende de qué código alfabético venga primero. Además, cuando el par tiene conflicto real ($y=1$), la media histórica de inscriptos cae de **28,1 a 17,3** (correlación negativa $r = -0,033$, `ROC-AUC = 0,405`). Reemplazar `(n1, n2)` por variables simétricas invariantes al orden (`log1p(min(n1, n2))`, `log1p(max(n1, n2))`, y el indicador de materia sin historial `min(n1, n2) == 0`) corrige esta distorsión.

### 1.4 Comparación experimental de representaciones de features (`feature_experiments.csv`)

Resultados en **Validación** (umbrales elegidos en validación; detalle completo en [`results/feature_experiments.csv`](file:///w:/home/anton/Tp-Grafos-ML/experiments/ultima_ronda_ml/results/feature_experiments.csv)):

| Conjunto de Features | N.º Vars | `LogReg (L2, C=1)` F1 / PR-AUC | `KNN (K=31)` F1 / PR-AUC | `DecisionTree (d=6, leaf=50)` F1 / PR-AUC | `LinearSVM Calib` F1 / PR-AUC | `HGB (Ref)` F1 / PR-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`F0_Actuales_10`** (10 oficiales) | 10 | 0,2193 / 0,1451 | **0,2712** / 0,1969 | 0,2662 / 0,2095 | 0,2408 / 0,1616 | 0,2854 / 0,2536 |
| **`F1_Solo_Estructurales_5`** | 5 | 0,1988 / 0,0846 | 0,1961 / 0,0527 | 0,2029 / 0,0878 | 0,2021 / 0,0931 | 0,2029 / 0,0905 |
| **`F2_Solo_Historicas_5`** | 5 | 0,1989 / 0,1239 | 0,2369 / 0,1568 | 0,2553 / 0,1902 | 0,2265 / 0,1406 | 0,2665 / 0,2132 |
| **`F3_Sin_Redundantes_8`** (sin `depto` ni `misma_carrera`) | 8 | 0,2264 / 0,1343 | 0,2596 / 0,1982 | **0,2682 / 0,2099** | 0,2350 / 0,1568 | 0,2799 / 0,2513 |
| **`F4_Simetrizadas_Log_10`** (mismas 10 entradas, `log1p` + `min/max`) | 10 | 0,2230 / 0,1569 | 0,2534 / **0,2004** | 0,2482 / 0,1950 | 0,2393 / 0,1730 | **0,2863 / 0,2546** |
| **`F5_Ingenieria_10Cols_16`** (`F4` + `zero_hist` + interacciones) | 16 | **0,2448 / 0,1792** | 0,2455 / 0,1995 | 0,2439 / 0,1881 | **0,2567 / 0,1721** | 0,2859 / 0,2534 |
| **`D_Diagnostico_Nivel_Y_2Hop_22`** *(diagnóstico Fase 1/2)* | 22 | **0,3541 / 0,3330** | — | **0,3529 / 0,3208** | **0,3407 / 0,3063** | **0,3741 / 0,3621** |

**Conclusiones de la Investigación A:**
1. **Ni las features estructurales solas (`F1`) ni las históricas solas (`F2`) alcanzan:** Combinarlas (`F0`) es indispensable porque la estructura curricular cubre los pares donde la intersección histórica directa es cero.
2. **En Árboles de Decisión y KNN con $K=51$, las 10 features actuales (`F0_Actuales_10`) o su versión simetrizada (`F4_Simetrizadas_Log_10`) ya capturan casi toda la señal disponible en esas 10 columnas:** En `DecisionTree`, quitar las 2 redundantes (`F3`) sube levemente el F1 de validación ($0,2662 \to 0,2682$), pero no cambia sustancialmente el techo ($\text{PR-AUC} \approx 0,21$).
3. **En modelos lineales (`LogisticRegression`, `LinearRegression`, `LinearSVC`), la ingeniería `F5` produce un salto importante:** En `LogisticRegression`, el `PR-AUC` sube de $0,1451 \to 0,1792$ en Validación y de $0,1566 \to 0,2045$ en Test (+30,6%), porque los términos `log1p`, la simetrización `min/max` y los indicadores de historial cero linealizan relaciones que antes solo veían los árboles o KNN.

---

## 2. Investigación B — Evaluación de Modelos Permitidos por la Consigna

Se evaluaron exhaustivamente las **5 familias permitidas por la consigna** (`LinearRegression`, `LogisticRegression`, `KNN`, `DecisionTree`, `SVM`) junto con referencias exploratorias (`RandomForest`, `ExtraTrees`, `HistGradientBoosting`). La tabla completa de las 59 configuraciones (118 filas `validation` + `test`) se encuentra en [`results/model_comparison_val_and_test.csv`](file:///w:/home/anton/Tp-Grafos-ML/experiments/ultima_ronda_ml/results/model_comparison_val_and_test.csv).

### Tabla Comparativa Principal (Selección de Umbral en Validación)

| Modelo | Familia (Permitido) | Features | Umbral ($\tau_{\text{val}}$) | Val Prec | Val Rec | Val F1 | Val PR-AUC | Test Prec | Test Rec | Test F1 | Test PR-AUC | Test ROC-AUC | Test TP / FP / FN | Aristas Test (`FP/TP`) | Fit (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `Baseline_Heuristico_Historico` | Regla (`hist > 0`) | `F0` | — | 0,1731 | 0,2432 | 0,2023 | 0,1025 | 0,1428 | 0,2082 | 0,1694 | 0,0750 | 0,5952 | 202 / 1213 / 768 | 1415 (6,00) | 0,00 |
| **`KNN_k15_Eucl_Unif_Scaled_F0` (Oficial)** | **KNN (Sí)** | **`F0`** | **0,21** | **0,2761** | **0,2115** | **0,2395** | **0,1846** | **0,2711** | **0,2485** | **0,2593** | **0,1833** | **0,7665** | **241 / 648 / 729** | **889 (2,69)** | **0,22** |
| **`KNN_k51_Manh_Unif_Scaled_F0` ★** | **KNN (Sí)** | **`F0`** | **0,16** | **0,3113** | **0,2492** | **0,2768** | **0,2088** | **0,2912** | **0,2567** | **0,2729** | **0,2249** | **0,8289** | **249 / 606 / 721** | **855 (2,43)** | **0,22** |
| **`KNN_k51_Manh_Unif_Scaled_F4_SymLog` ★** | **KNN (Sí)** | **`F4`** | **0,18** | **0,3164** | **0,2160** | **0,2567** | **0,2051** | **0,3153** | **0,2526** | **0,2805** | **0,2273** | **0,8178** | **245 / 532 / 725** | **777 (2,17)** | **0,23** |
| `KNN_k51_Eucl_Dist_Scaled_F0` | KNN (Sí) | `F0` | 0,17 | 0,2864 | **0,2885** | **0,2874** | 0,1802 | 0,2752 | **0,2763** | 0,2757 | 0,1806 | 0,7846 | 268 / 706 / 702 | 974 (2,63) | 0,21 |
| `KNN_k75_Manh_Unif_Scaled_F4_SymLog` | KNN (Sí) | `F4` | 0,17 | 0,3030 | 0,2311 | 0,2622 | 0,2071 | 0,3148 | 0,2515 | 0,2797 | **0,2308** | 0,8268 | 244 / 531 / 726 | 775 (2,18) | 0,24 |
| `DT_d5_leaf1_Gini_Unw_F0` (Fase 3) | Árbol (Sí) | `F0` | 0,20 | 0,3853 | 0,1979 | 0,2615 | 0,1963 | 0,4399 | 0,1546 | 0,2288 | 0,2082 | 0,8543 | 150 / 191 / 820 | 341 (1,27) | 0,10 |
| **`DT_d6_leaf50_Gini_Unw_F0` ★** | **Árbol (Sí)** | **`F0`** | **0,20** | **0,3389** | **0,2145** | **0,2627** | **0,2148** | **0,3842** | **0,2052** | **0,2675** | **0,2270** | **0,8777** | **199 / 319 / 771** | **518 (1,60)** | **0,13** |
| **`DT_d6_leaf50_LogLoss_Unw_F0` ★** | **Árbol (Sí)** | **`F0`** | **0,20** | **0,3226** | **0,2266** | **0,2662** | **0,2095** | **0,3845** | **0,1990** | **0,2622** | **0,2263** | **0,8793** | **193 / 309 / 777** | **502 (1,60)** | **0,11** |
| `DT_d6_leaf50_LogLoss_Unw_F4_SymLog` | Árbol (Sí) | `F4` | 0,15 | 0,3067 | 0,2085 | 0,2482 | 0,1950 | 0,3593 | 0,2186 | 0,2718 | 0,2166 | 0,8653 | 212 / 378 / 758 | 590 (1,78) | 0,11 |
| `DT_d7_leaf75_LogLoss_Unw_F5_Eng` | Árbol (Sí) | `F5` | 0,18 | 0,3161 | 0,2221 | 0,2609 | 0,1961 | 0,3455 | 0,2144 | 0,2646 | 0,2243 | 0,8626 | 208 / 394 / 762 | 602 (1,89) | 0,23 |
| `LogReg_L2_C1_Unw_F0` (Fase 3) | Logística (Sí) | `F0` | 0,19 | 0,3283 | 0,1647 | 0,2193 | 0,1451 | 0,3326 | 0,1649 | 0,2205 | 0,1566 | 0,8632 | 160 / 321 / 810 | 481 (2,01) | 0,66 |
| `LogReg_L2_C1_Unw_F5_Eng` | Logística (Sí) | `F5` | 0,14 | 0,2878 | 0,2130 | 0,2448 | 0,1792 | 0,3022 | 0,2237 | 0,2571 | 0,2043 | 0,8718 | 217 / 501 / 753 | 718 (2,31) | 0,76 |
| `LogReg_L2_C1_Weight5_F5_Eng` | Logística (Sí) | `F5` | 0,42 | 0,2673 | 0,2281 | 0,2461 | 0,1743 | 0,2808 | 0,2423 | 0,2601 | 0,1955 | 0,8727 | 235 / 602 / 735 | 837 (2,56) | 0,89 |
| `LinReg_Ridge_a1_F0` | Lineal (Sí) | `F0` | 0,12 | 0,3119 | 0,1828 | 0,2305 | 0,1532 | 0,2975 | 0,1711 | 0,2173 | 0,1603 | 0,8587 | 166 / 392 / 804 | 558 (2,36) | 0,02 |
| `LinReg_Ridge_a10_F5_Eng` | Lineal (Sí) | `F5` | 0,15 | 0,3561 | 0,1888 | 0,2468 | 0,1664 | 0,3253 | 0,1526 | 0,2077 | 0,1705 | 0,8627 | 148 / 307 / 822 | 455 (2,07) | 0,06 |
| `SVM_Linear_C1_Unw_F0` | SVM Lineal (Sí) | `F0` | 0,09 | 0,3192 | 0,1934 | 0,2408 | 0,1616 | 0,2908 | 0,1763 | 0,2195 | 0,1715 | 0,8698 | 171 / 417 / 799 | 588 (2,44) | 0,89 |
| `SVM_Linear_C1_Unw_F5_Eng` | SVM Lineal (Sí) | `F5` | 0,12 | 0,3761 | 0,1949 | 0,2567 | 0,1721 | 0,3262 | 0,1567 | 0,2117 | 0,1796 | 0,8726 | 152 / 314 / 818 | 466 (2,07) | 1,66 |
| `SVC_RBF_C1_BalUndersamp_ProbTrue_F0` | SVM RBF (Sí) | `F0` | 0,84 | 0,0854 | 0,6284 | 0,1503 | 0,0658 | 0,0970 | 0,6165 | 0,1677 | 0,1187 | 0,8545 | 598 / 5564 / 372 | 6162 (9,30) | 2,65 |
| `SVC_RBF_C1_Sub30k_Unw_ProbTrue_F5_Eng` | SVM RBF (Sí) | `F5` | 0,52 | 0,3077 | 0,2417 | 0,2707 | 0,1579 | 0,3055 | 0,2639 | 0,2832 | 0,1921 | 0,7710 | 256 / 582 / 714 | 838 (2,27) | **52,00** |
| *`Ref_RF_d6_Unw_F0`* | *RF (Ref. No perm.)* | *`F0`* | *0,19* | *0,4121* | *0,2054* | *0,2742* | *0,2405* | *0,4758* | *0,1722* | *0,2528* | *0,2493* | *0,8738* | *167 / 184 / 803* | *351 (1,10)* | *1,95* |
| *`Ref_HGB_d6_Unw_F0`* | *HGB (Ref. No perm.)* | *`F0`* | *0,20* | *0,4096* | *0,2190* | *0,2854* | *0,2536* | *0,4548* | *0,1969* | *0,2748* | *0,2787* | *0,8881* | *191 / 229 / 779* | *420 (1,20)* | *1,12* |
| *`Ref_HGB_d6_Unw_F5_Eng`* | *HGB (Ref. No perm.)* | *`F5`* | *0,13* | *0,2805* | *0,2915* | *0,2859* | *0,2534* | *0,2960* | *0,2969* | *0,2964* | *0,2715* | *0,8890* | *288 / 685 / 682* | *973 (2,38)* | *1,36* |

### Hallazgos por familia de modelos permitidos

1. **KNN (`KNeighborsClassifier`):**
   - El modelo oficial actual usa $K=15$ y distancia Euclídea. Con $K=15$, las probabilidades discretas tienen escalones de $1/15 \approx 0,0667$ y sufren alta varianza local en zonas densas de pares negativos.
   - Al subir a **$K=51$ con distancia Manhattan (`L1`)** sobre las mismas 10 features (`KNN_k51_Manh_Unif_Scaled_F0`, $\tau_{\text{val}}=0,16$), el modelo **supera al `KNN_k15` oficial en absolutamente todas las métricas tanto en Validación como en Test**:
     - **Validación:** `F1` sube de $0,2395 \to \mathbf{0,2768}$ (+15,6%), `PR-AUC` de $0,1846 \to \mathbf{0,2088}$ (+13,1%), `ROC-AUC` de $0,7651 \to \mathbf{0,8154}$, `Precision` de $0,2761 \to \mathbf{0,3113}$ y `Recall` de $0,2115 \to \mathbf{0,2492}$.
     - **Test:** `F1` sube de $0,2593 \to \mathbf{0,2729}$, `PR-AUC` de $0,1833 \to \mathbf{0,2249}$ (+22,7%), `ROC-AUC` de $0,7665 \to \mathbf{0,8289}$, detectando **más conflictos reales (`249` vs `241` TP)** y cometiendo **menos falsos positivos (`606` vs `648` FP)**.
   - Si además se usa la transformación simétrica logarítmica (`KNN_k51_Manh_Unif_Scaled_F4_SymLog`, $\tau_{\text{val}}=0,18$), en Test alcanza **`F1 = 0,2805`**, **`PR-AUC = 0,2273`**, **`Precision = 0,3153`** y **`Recall = 0,2526`** (`245 TP`, `532 FP`).
2. **Árboles de Decisión (`DecisionTreeClassifier`):**
   - El árbol evaluado en la Fase 3 (`max_depth=5, min_samples_leaf=1`) caía a `F1 = 0,2288` en Test porque creaba hojas pequeñas sobreajustadas a combinaciones particulares de `cant_inscriptos_hist_1/2`.
   - Al regularizar las hojas con **`max_depth=6, min_samples_leaf=50, min_samples_split=100`** (`DT_d6_leaf50_Gini_Unw_F0`, $\tau_{\text{val}}=0,20$), el árbol obtiene el **mejor `PR-AUC` en Validación (`0,2148`) y Test (`0,2270`)** y el **mejor `ROC-AUC` (`0,8653` en Val / `0,8777` en Test)** de todos los modelos permitidos sobre `F0`, logrando además una altísima precisión en Test (**`38,42%`**, `199 TP` y solo **`319 FP`**, la mitad de aristas falsas que `KNN_k15`).
3. **Regresión Logística (`LogisticRegression`):**
   - Sobre `F0` queda rezagada (`F1 = 0,2205`, `PR-AUC = 0,1566`), pero con `F5` (`LogReg_L2_C1_Weight5_F5_Eng`) sube a `Val F1 = 0,2461` y `Test F1 = 0,2601` (`Test PR-AUC = 0,1955–0,2045`, `Test ROC-AUC = 0,8727`), empatando al `KNN_k15` actual aunque por debajo de `KNN_k51` y `DT_d6_leaf50`.
4. **Regresión Lineal (`Ridge`):**
   - Aun con `F5` (`LinReg_Ridge_a10_F5_Eng`), su función de pérdida cuadrática sobre etiquetas $\{0, 1\}$ desbalanceadas al $1,4\%$ comprime las predicciones y cae en Test a `F1 = 0,2077`, `PR-AUC = 0,1705`. No es competitiva frente a Logística, KNN ni Árboles.
5. **SVM (`LinearSVC` calibrado y `SVC` con kernel RBF):**
   - **SVM Lineal (`SVM_Linear_C1_Unw_F5_Eng`):** Alcanza `Val F1 = 0,2567` pero cae en Test a `F1 = 0,2117` (`PR-AUC = 0,1796`).
   - **SVM con kernel RBF (`SVC(kernel="rbf")`):**
     - Entrenado sobre `subtrain_balanced` (`5.672` pares), sufre de altísima tasa de falsos positivos (`PR-AUC = 0,0658` en Val y `0,1187` en Test).
     - Entrenado sobre una submuestra estratificada de `30.000` pares sin pesos (`SVC_RBF_C1_Sub30k_Unw_ProbTrue_F5_Eng`), logra un buen `F1` puntual (`0,2707` en Val / `0,2832` en Test), pero tiene **menor `PR-AUC` (`0,1579` en Val / `0,1921` en Test)** y **menor `ROC-AUC` (`0,7387` / `0,7710`)** que `KNN_k51` y `DT_d6_leaf50`.
     - **Costo e inestabilidad de `probability=True`:** En `SVC`, activar `probability=True` ejecuta internamente una validación cruzada de 5 folds con ajuste de Platt, multiplicando el tiempo de entrenamiento por **6,3x** (`52,0 s` vs `8,3 s` en solo 30k filas; inviable en las 254.541 filas completas). Además, la calibración sigmoidea de Platt colapsa en la cola de baja probabilidad: al buscar $\text{Recall} \ge 40\%$ en Validación ($\tau = 0,0696$), en Test dispara **64.624 aristas predichas** (casi el grafo completo). Por ello, `SVC` se descarta como modelo operativo.

---

## 3. Investigación C — Métricas, Umbrales y Costo Estructural en el Grafo

La curva Precision-Recall comparativa está guardada en [`results/pr_curves_ultima_ronda.png`](file:///w:/home/anton/Tp-Grafos-ML/experiments/ultima_ronda_ml/results/pr_curves_ultima_ronda.png) y la tabla completa de puntos operativos en [`results/thresholds_tradeoff.csv`](file:///w:/home/anton/Tp-Grafos-ML/experiments/ultima_ronda_ml/results/thresholds_tradeoff.csv).

### Comparación de Regímenes de Umbral (Elegidos en Validación y Evaluados en Test)

| Modelo | Régimen (fijado en Val) | Umbral $\tau$ | Val Prec / Rec / F1 | Test Prec | Test Recall | Test F1 | Test TP | Test FP | Test FN | Aristas Test (`TP+FP`) | Ratio `FP/TP` (Test) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`KNN_k15_F0` (Actual)** | **Máx. F1 Val** | **0,2100** | 0,276 / 0,212 / 0,240 | **0,2711** | **0,2485** | **0,2593** | **241** | **648** | 729 | **889** | **2,69** |
| `KNN_k15_F0` (Actual) | Recall Val $\ge 30\%$ | 0,1333 | 0,125 / 0,423 / 0,193 | 0,1527 | 0,4155 | 0,2233 | 403 | 2.236 | 567 | 2.639 | 5,55 |
| **`KNN_k51_Manh_F0` ★** | **Máx. F1 Val (Equilibrado)** | **0,1600** | **0,311 / 0,249 / 0,277** | **0,2912** | **0,2567** | **0,2729** | **249** | **606** | 721 | **855** | **2,43** |
| **`KNN_k51_Manh_F0` ★** | **Mayor Recall ($\sim 29\%$)** | **0,1373** | **0,249 / 0,280 / 0,263** | **0,2548** | **0,2856** | **0,2693** | **277** | **810** | 693 | **1.087** | **2,92** |
| `KNN_k51_Manh_F0` | Recall Val $\ge 40\%$ | 0,0980 | 0,144 / 0,406 / 0,213 | 0,1592 | 0,4278 | 0,2321 | 415 | 2.191 | 555 | 2.606 | 5,28 |
| `KNN_k51_Manh_F0` | Recall Val $\ge 50\%$ | 0,0588 | 0,103 / 0,553 / 0,174 | 0,1126 | 0,5711 | 0,1881 | 554 | 4.368 | 416 | 4.922 | 7,88 |
| **`KNN_k51_Manh_F4` ★** | **Máx. F1 Val** | **0,1800** | **0,316 / 0,216 / 0,257** | **0,3153** | **0,2526** | **0,2805** | **245** | **532** | 725 | **777** | **2,17** |
| **`KNN_k51_Manh_F4` ★** | **Mayor Recall ($\sim 30\%$)** | **0,1373** | **0,243 / 0,267 / 0,255** | **0,2674** | **0,3010** | **0,2832** | **292** | **800** | 678 | **1.092** | **2,74** |
| **`DT_d6_leaf50_Gini_F0` ★** | **Máx. F1 Val (Conservador)** | **0,2000** | **0,339 / 0,215 / 0,263** | **0,3842** | **0,2052** | **0,2675** | **199** | **319** | 771 | **518** | **1,60** |
| `DT_d6_leaf50_Gini_F0` | Recall Val $\sim 30\%$ | 0,0808 | 0,213 / 0,289 / 0,245 | 0,1941 | 0,3577 | 0,2516 | 347 | 1.441 | 623 | 1.788 | 4,15 |
| `DT_d6_leaf50_Gini_F0` | Recall Val $\ge 50\%$ | 0,0541 | 0,100 / 0,601 / 0,172 | 0,1260 | 0,5515 | 0,2051 | 535 | 3.712 | 435 | 4.247 | 6,94 |

**Interpretación para el coloreo de grafos:**
- En un grafo no ponderado donde toda arista predicha tiene peso 1, **el límite práctico de Recall sin inflar desproporcionadamente el número cromático está entre $25\%$ y $30\%$**:
  - Con `KNN_k51_Manh_Unif_Scaled_F0` en $\tau = 0,16$ (óptimo F1 en validación), detectamos `249` conflictos reales con `606` falsas aristas (`855` aristas totales, menos que las `889` del modelo actual y con mejor Precision y Recall).
  - Si se desea priorizar ligeramente el Recall (bajar a $\tau = 0,1373$, que equivale a $\ge 7/51$ vecinos positivos), el Recall en Test sube a **`28,56%` (`277 TP`)** con `810 FP` (`1.087` aristas totales, ratio `FP/TP = 2,92`), y con `F4` alcanza **`30,10%` de Recall (`292 TP`)** y **`F1 = 0,2832`** con `800 FP` (`FP/TP = 2,74`).
  - Intentar forzar $\text{Recall} \ge 40\%$ o $\ge 50\%$ exige bajar el umbral por debajo de $0,10$, entrando en la zona donde los pares positivos están diluidos $1:216$ entre negativos y disparando las aristas falsas a **2.200–4.400 FP** (`FP/TP` de $5,3$ a $7,9$).

---

## 4. Investigación D — Limitaciones Reales de los Datos y Origen del Grupo "5 Ceros"

El análisis forense completo sobre `data/interim/` y `data/processed/` está documentado en [`results/inv_a_and_d_data_summary.json`](file:///w:/home/anton/Tp-Grafos-ML/experiments/ultima_ronda_ml/results/inv_a_and_d_data_summary.json).

### 4.1 El grupo "5 ceros" (`compartidos_hist=0, misma_carrera=0, mismo_anio=0, mismo_depto=0, mismo_prefijo=0`)

- **Peso en el dataset:** Representa el **77,33% de todos los pares de Validación (`40.192 / 51.975`)** y el **75,69% de Test (`49.244 / 65.059`)**.
- **Proporción de conflictos reales ocultos allí:** Contiene **185 de los 662 conflictos reales de Validación (27,95%)** y **215 de los 970 conflictos reales de Test (22,16%)**.
- **Indistinguibilidad con las 10 features actuales:** Dentro de este grupo, **7 de las 10 features valen exactamente `0`** para las 40.192 filas (`cant_estudiantes_compartidos_historico`, `jaccard_historico`, `misma_carrera`, `cant_carreras_compartidas_hist`, `mismo_anio`, `mismo_departamento`, `mismo_prefijo_curricular`). Un clasificador solo dispone de `(anio_confirmado, cant_inscriptos_hist_1, cant_inscriptos_hist_2)` con una tasa positiva de apenas **0,46% (`185 / 40.192`, 216 negativos por cada positivo)**. Con esas 3 variables, el `PR-AUC` interno dentro del grupo "5 ceros" es **`0,0060` en Validación y `0,0057` en Test** (prácticamente azar puro).

### 4.2 Tres causas raíz verificadas en los datos crudos/intermedios

1. **Estructura demográfica de un único semestre (`1989–1993` vs `1994`):**
   - El dataset de Nottingham corresponde a los exámenes de **un único semestre lectivo (Semestre 1 de 1994–95)**.
   - Al dividir alumnos por año de ingreso (`cohort_year`), el conjunto `historico_1989_1993` (`4.810` alumnos) está formado por **alumnos de 2.º a 5.º año**, cuyas inscripciones están concentradas en un **91,4% en materias de años superiores (`'2', '3', '4', 'B', 'C'`)**.
   - En cambio, el conjunto `actual_1994` (`3.086` alumnos, donde se mide `hubo_conflicto_potencial`) está formado por **ingresantes de 1.er año de 1994**, cuyas inscripciones están concentradas en un **92,3% en materias de 1.er año (`'1', 'A', '0', 'P'`)**.
   - Como consecuencia, **45 exámenes (incluyendo 34 de nivel inicial `'1'` o `'A'`) tienen `0` inscriptos en `historico_1989_1993`**, provocando que el **17,22% de los conflictos reales de Validación (`114/662`)** ocurran en pares donde al menos una materia tiene `cant_inscriptos_hist == 0` y, por ende, conjunto histórico de carreras vacío (`misma_carrera = 0`).
2. **Pérdida de los niveles curriculares alfabéticos (`'A'`, `'B'`, `'C'`, `'0'`, `'P'`) en `parse_nott.py`:**
   - La función `derive_year_code()` en [`src/parsing/parse_nott.py`](file:///w:/home/anton/Tp-Grafos-ML/src/parsing/parse_nott.py) solo acepta dígitos `'1'..'4'` en el 3.er carácter de `exam_id` (`542` materias) y asigna `year_code = None` (`anio_confirmado = 0, mismo_anio = 0`) a **258 materias (32,25% del catálogo, afectando al 53,95% de todos los pares)**.
   - En la nomenclatura modular de la Universidad de Nottingham, el 3.er carácter `'A'` (`111` materias), `'B'` (`58` materias) y `'C'` (`56` materias) indica el nivel curricular en facultades de Artes y Ciencias Sociales (**Part A = 1.er año, Part B = 2.º año, Part C = 3.er año**), mientras que `'0'` (`18` materias) y `'P'` (`15` materias) corresponden a *Foundation/Preliminary Year*.
   - **Evidencia contundente dentro del grupo "5 ceros":** De los `185` conflictos reales en el grupo "5 ceros" de Validación, **132 conflictos (71,35%) son pares donde AMBAS materias son de 1.er año (`'1'` o `'A'`, p. ej. una materia código `'1'` y otra código `'A'`, o dos código `'A'`)**, mientras que entre los `40.007` negativos del grupo "5 ceros" solo el **4,05% (`1.620 / 40.007`)** cumple esa condición (un enriquecimiento de **17,6x** que hoy queda oculto detrás de `mismo_anio = 0`).
3. **Relaciones históricas de 2 saltos (2-hop) no capturadas:**
   - Incluso cuando dos materias $u$ y $v$ no compartieron alumnos directamente en `historico_1989_1993` (`compartidos_hist = 0`), el **37,8% de los conflictos reales del grupo "5 ceros"** tiene vecinos históricos en común (materias $w$ que compartieron alumnos históricos tanto con $u$ como con $v$).

### 4.3 Prueba cuantitativa del techo de datos (`D_Diagnostico_Nivel_Y_2Hop_22`)

Para verificar empíricamente cuánto del techo proviene del clasificador y cuánto de las features descartadas en Fase 1/2, evaluamos en el script experimental el conjunto `D_Diagnostico_Nivel_Y_2Hop_22` (que agrega a `F5` el nivel curricular equivalente `'1'='A', '2'='B', '3'='C'` extraído del `exam_id`, el indicador de 1.er año en ambas materias, el 1.er carácter de facultad del `exam_id` y el conteo de vecinos históricos a 2 saltos):

- En **`LogisticRegression`**, el `PR-AUC` de Validación salta de **`0,1792` a `0,3330` (+85,8%)**, el `ROC-AUC` de **`0,8525` a `0,9360`**, y el `F1` de **`0,2448` a `0,3541`**.
- En **`DecisionTree (d=6, leaf=50)`**, el `PR-AUC` de Validación salta de **`0,2095` a `0,3208` (+53,1%)**, el `ROC-AUC` de **`0,8680` a `0,9382`**, y el `F1` de **`0,2662` a `0,3529`** (con **`45,0%` de Recall**).

Esto demuestra de manera reproducible que **el techo de $\text{PR-AUC} \approx 0,21\text{–}0,23$ con las 10 columnas actuales no es una limitación de los algoritmos de ML, sino de la información que las 10 features actuales exponen sobre las materias de 1.er año y los códigos `'A'/'B'/'C'`**.

---

## 5. Respuestas Concretas a las 8 Preguntas de la Sección 8

### 1. ¿Las 10 features actuales son adecuadas o conviene modificar el subconjunto / representación?
- **Si no se desea tocar el contrato de entrada de las 10 columnas (`FEATURE_COLUMNS`):** Las 10 columnas actuales son suficientes para árboles regularizados (`DT_d6_leaf50`) y para `KNN` con $K=51$ y distancia Manhattan (`KNN_k51_Manh_Unif_Scaled_F0`), aunque contienen dos redundancias altas (`mismo_departamento` $\leftrightarrow$ `mismo_prefijo_curricular`, $r=0,91$; `misma_carrera` $\leftrightarrow$ `cant_carreras_compartidas_hist`, $r=0,84$) y dos variables asimétricas (`cant_inscriptos_hist_1/2`).
- **Si se aplica un preprocesador interno sobre las mismas 10 columnas (`F4_Simetrizadas_Log_10` / `F5_Ingenieria_10Cols_16`):** Reemplazar `(cant_inscriptos_hist_1, cant_inscriptos_hist_2)` por `log1p(min)` y `log1p(max)` y aplicar `log1p` a los conteos **mejora la precisión y el `F1` de KNN en Test (`0,2729` $\to$ `0,2805`, `PR-AUC = 0,2273`)** y mejora sustancialmente la Regresión Logística (`PR-AUC` de `0,1566` $\to$ `0,2045`), sin cambiar el esquema de entrada del módulo de inferencia.

### 2. ¿Cuál es el mejor modelo dentro de los permitidos por la consigna?
Hay dos ganadores claros dentro del temario permitido, con perfiles complementarios:
1. **`KNN_k51_Manh_Unif_Scaled` ($K=51$, distancia Manhattan `p=1`, pesos uniformes):** Es el mejor modelo permitido en **equilibrio Precision-Recall (`F1`) y detección de conflictos (`Recall`)**. Sobre `F0` alcanza `Val F1 = 0,2768, Val PR-AUC = 0,2088` y `Test F1 = 0,2729, Test PR-AUC = 0,2249, Test ROC-AUC = 0,8289` (`249 TP, 606 FP`), y con la transformación `F4` alcanza `Test F1 = 0,2805, Test PR-AUC = 0,2273` (`245 TP, 532 FP`).
2. **`DecisionTreeClassifier(max_depth=6, min_samples_leaf=50, min_samples_split=100, random_state=42)` (`DT_d6_leaf50`):** Es el mejor modelo permitido en **calidad global de ranking (`Val PR-AUC = 0,2148`, `Test PR-AUC = 0,2270`, `Test ROC-AUC = 0,8777`) y precisión de aristas (`Test Precision = 38,42%`)**, reduciendo los falsos positivos a la mitad (`319 FP` vs `648 FP` del KNN oficial) a cambio de un menor Recall (`20,52%`, `199 TP`).

### 3. ¿Supera realmente al `KNN_Full_Unweighted` actual?
**Sí, de forma consistente tanto en Validación como en Test:**
- Contra `KNN_k15` oficial (`Val F1 = 0,2395, Val PR-AUC = 0,1846` | `Test F1 = 0,2593, Test PR-AUC = 0,1833, Test ROC-AUC = 0,7665`):
  - **`KNN_k51_Manh_Unif_Scaled_F0` ($\tau = 0,16$)** mejora en Validación **`+3,73 puntos de F1` (`0,2768`)** y **`+2,42 puntos de PR-AUC` (`0,2088`)**, y en Test mejora **`+1,36 puntos de F1` (`0,2729`)**, **`+4,16 puntos de PR-AUC` (`0,2249`, +22,7% relativo)** y **`+6,24 puntos de ROC-AUC` (`0,8289`)**, encontrando más conflictos reales (`249` vs `241` TP) con menos aristas falsas (`606` vs `648` FP).
  - Con `F4_Simetrizadas_Log_10` ($\tau = 0,18$), **`KNN_k51_Manh_Unif_Scaled_F4_SymLog`** sube en Test a **`F1 = 0,2805`**, **`PR-AUC = 0,2273`**, **`Precision = 0,3153`** y **`Recall = 0,2526`** (`245 TP`, `532 FP`, ahorrando `116` aristas falsas respecto al modelo actual).

### 4. ¿Cuál es el mejor umbral según validación?
- Para **`KNN_k51_Manh_Unif_Scaled_F0`**:
  - **Umbral óptimo F1 en validación:** $\tau = \mathbf{0,16}$ (equivale a requerir al menos $9$ vecinos positivos de $51$, ya que $9/51 = 0,1765 > 0,16 > 8/51 = 0,1569$). Produce `249 TP` y `606 FP` en Test (`Precision = 29,12%`, `Recall = 25,67%`, `F1 = 0,2729`).
  - **Umbral de mayor sensibilidad moderada:** $\tau = \mathbf{0,1373}$ (al menos $7\text{–}8$ vecinos positivos de $51$). Eleva el Recall en Test a **`28,56%` (`277 TP`)** con `810 FP` (`F1 = 0,2693`).
- Para **`KNN_k51_Manh_Unif_Scaled_F4_SymLog`**:
  - **Umbral óptimo F1 en validación:** $\tau = \mathbf{0,18}$ (`245 TP`, `532 FP`, `F1 = 0,2805` en Test).
  - **Umbral de mayor Recall moderado:** $\tau = \mathbf{0,1373}$ (`292 TP`, `800 FP`, **`Recall = 30,10%`**, **`F1 = 0,2832`** en Test).
- Para **`DT_d6_leaf50_Gini_Unw_F0`**:
  - **Umbral óptimo F1 en validación:** $\tau = \mathbf{0,20}$ (`199 TP`, `319 FP`, `Precision = 38,42%`, `F1 = 0,2675` en Test).

### 5. ¿Cuánto Recall podemos ganar sin disparar falsos positivos?
- Manteniendo un ratio razonable de **`FP/TP` $\le 2,8$** (menos de 3 aristas falsas por conflicto real detectado y $\le 1.100$ aristas totales en Test), **podemos subir el Recall de `24,85%` (`241 TP`) hasta `28,56%–30,10%` (`277–292 TP`)** usando `KNN_k51_Manh` con $\tau = 0,1373$.
- Querer superar el **40% de Recall** con las 10 features actuales obliga a bajar el umbral a $\tau \approx 0,098$, donde la precisión cae al $15,9\%$ y las aristas falsas se triplican a **`2.191 FP`** (`2.606` aristas totales en el grafo).

### 6. ¿Qué parte del límite actual viene del modelo y qué parte viene de los datos?
- **Parte atribuible al modelo / hiperparámetros actuales (~3 a 4,5 puntos de PR-AUC y ~2 puntos de F1):** Pasar de `KNN(K=15, Euclidean)` a `KNN(K=51, Manhattan)` o `DecisionTree(max_depth=6, min_samples_leaf=50)` sube el `PR-AUC` de Test de **`0,1833` a `0,2249–0,2308`** y el `ROC-AUC` de **`0,7665` a `0,8289–0,8793`**.
- **Parte atribuible a las 10 features actuales y al corte de cohortes (~11 a 15 puntos de PR-AUC y ~9 puntos de F1):** Con las 10 features actuales, **ningún modelo (ni siquiera `HistGradientBoosting` o `RandomForest`) supera `PR-AUC = 0,254` en Validación ni `0,279` en Test**, porque el **27,95% de los conflictos de Validación (`185/662`)** cae en el grupo "5 ceros" donde las materias de 1.er año e iniciales con código `'A'` tienen `mismo_anio = 0` y `misma_carrera = 0`. Al incorporar el nivel `'A'/'B'/'C'` del `exam_id` y los vecinos históricos a 2 saltos (`D_Diagnostico_Nivel_Y_2Hop_22`), incluso una simple Regresión Logística o un Árbol de Decisión salta a **`PR-AUC = 0,321–0,333` y `ROC-AUC = 0,936–0,938`**.

### 7. ¿Conviene cambiar el modelo oficial actual o mantenerlo?
- **Conviene actualizar la configuración hiperparamétrica de KNN (o su árbol alternativo), pero manteniendo la misma arquitectura de `Pipeline` y las mismas 10 columnas de entrada:**
  - Actualizar `KNeighborsClassifier(n_neighbors=15)` a **`KNeighborsClassifier(n_neighbors=51, metric="manhattan", weights="uniform")`** con umbral $\tau = 0,16$ (o con un `FunctionTransformer` interno para `F4_Simetrizadas_Log_10` y $\tau = 0,18$) **no rompe ninguna interfaz ni test del proyecto**, sigue siendo 100% un modelo KNN estándar de la consigna, elimina el problema de haber elegido `KNN_k15` cuando era 2.º en validación detrás del árbol en Fase 3, y mejora todas las métricas en validación y test.

### 8. Si recomendamos un cambio, ¿cuál es exactamente la configuración propuesta y por qué?
Ver Sección 6 a continuación.

---

## 6. Recomendación Final (Sección 9)

Siguiendo la restricción de proponer **una única configuración principal recomendada** y **como máximo una alternativa**, todas dentro de los modelos permitidos por la consigna:

### Opción Principal Recomendada: `KNN_k51_Manh_Unif_Scaled` ($K=51$, distancia Manhattan)

- **Definición exacta del Pipeline:**
  - Si se mantiene exactamente la matriz cruda `F0_Actuales_10` sin transformaciones adicionales:
    ```python
    Pipeline([
        ("scaler", StandardScaler()),
        ("model", KNeighborsClassifier(n_neighbors=51, metric="manhattan", weights="uniform")),
    ])
    ```
    con umbral de validación **$\tau = 0,16$** (o $\tau = 0,14$ si se prioriza alcanzar $\sim 28,6\%$ de Recall).
  - Si se incluye dentro del mismo `Pipeline` la simetrización logarítmica de las 10 columnas (`F4_Simetrizadas_Log_10`, que recibe exactamente las mismas 10 columnas de `FEATURE_COLUMNS` y transforma `n1, n2 -> log1p(min), log1p(max)` y `log1p` en conteos):
    - Umbral de validación **$\tau = 0,18$** (para máximo F1: `Test F1 = 0,2805`, `PR-AUC = 0,2273`, `Precision = 31,53%`, `Recall = 25,26%`, `245 TP`, `532 FP`) o **$\tau = 0,1373$** (para mayor Recall: `Test Recall = 30,10%`, `Test F1 = 0,2832`, `292 TP`, `800 FP`).
- **Por qué es la mejor opción principal:**
  1. **Totalmente permitida y central en la consigna (KNN)** y continuista con la arquitectura ya integrada en `src/inference/`.
  2. **Mejora simultáneamente Validación y Test** en `F1`, `PR-AUC`, `ROC-AUC`, `Precision` y `Recall` respecto a `KNN_k15` ($\tau = 0,21$).
  3. **Resuelve la debilidad metodológica de Fase 3:** `KNN_k51_Manh_Unif_Scaled_F0` ($\text{Val F1} = 0,2768$) supera en validación a todos los modelos de Fase 3, por lo que su selección es **100% consistente con el criterio de validación** sin necesidad de justificaciones *post-hoc*.

### Única Alternativa Recomendada (Perfil Conservador / Alta Precisión): `DT_d6_leaf50_Gini_Unw_F0`

- **Definición exacta del modelo:**
  ```python
  DecisionTreeClassifier(
      max_depth=6,
      min_samples_leaf=50,
      min_samples_split=100,
      criterion="gini",
      random_state=42,
  )
  ```
  sobre las 10 features actuales (`F0_Actuales_10`) con umbral de validación **$\tau = 0,20$**.
- **Por qué es la mejor alternativa:**
  1. Logra el **mejor `PR-AUC` (`0,2148` en Val / `0,2270` en Test)** y **mejor `ROC-AUC` (`0,8653` en Val / `0,8777` en Test)** entre todos los modelos permitidos sobre `F0`.
  2. Es **100% interpretable académicamente** (se pueden graficar sus 6 niveles de reglas de decisión) y ofrece un régimen de **alta precisión (`38,42%` en Test, `199 TP` y solo `319 FP`)**, ideal si el algoritmo de coloreo prefiere un grafo más disperso con la mitad de aristas falsas.

---

> [!NOTE] Decisión Final Adoptada
> En la ronda de validación final ([`experiments/cierre_ml/informe_cierre_ml.md`](file:///w:/home/anton/Tp-Grafos-ML/experiments/cierre_ml/informe_cierre_ml.md)) se adoptó formalmente este **Árbol de Decisión regularizado (`DecisionTree_d6_leaf50_Gini`)** como modelo definitivo del proyecto, junto con la incorporación de la corrección de niveles A/B/C (`A=1, B=2, C=3`) en el parser oficial y la regeneración de datasets correspondientes.
