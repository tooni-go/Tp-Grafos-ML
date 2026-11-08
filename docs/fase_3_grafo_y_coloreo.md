# Fase 4 — Grafo de Conflictos y Coloreo alimentados por el modelo de ML

> Reemplaza la versión anterior de este documento (rama `fase-3-grafo-coloreo`), en la que
> las aristas salían de los conflictos reales de 1994. Ahora salen de las predicciones del ML.

**Módulos:** `src/graphs/conflict_graph.py`, `src/graphs/predicted_pairs.py`, `src/graphs/run_schedule.py`
**Tests:** `tests/test_graphs.py` · **Salidas:** `results/graphs/`

## 1. Qué cambió y por qué

| Antes | Ahora |
|---|---|
| Aristas = pares con `cant_estudiantes_afectados > θ` (dato real de 1994) | Aristas = pares con `prob_conflicto >= τ` (predicción del modelo) |
| `validate_coloring` "demostraba" 0 choques | `validate_coloring` es solo un chequeo interno; el cronograma se mide con `evaluate_against_real` |
| Resultado: 19 turnos, 0 choques (no realizable: requiere conocer el futuro) | Resultado realista: turnos + choques reales residuales |

Los conflictos reales se conservan **solo** como (a) referencia (grafo oráculo, 19 turnos) y
(b) métrica de evaluación. Nunca alimentan el grafo del sistema.

## 2. Flujo

```
subject_pairs_poc.csv (319.600 pares, 10 features)
  -> predicted_pairs.get_pair_probabilities()   # prob_conflicto para los 319.600 pares
  -> conflict_graph.build_conflict_graph_from_predictions(τ)   # no dirigido, sin pesos
  -> greedy_coloring (largest_first / Welsh-Powell, o DSATUR)
  -> build_schedule  -> cronograma por turno
  -> evaluate_against_real  -> choques reales, alumnos afectados, TP/FP/FN de aristas
```

El modelo y el umbral son los oficiales de la rama ML, sin modificarlos:
`KNN_Full_Unweighted` (`results/models/knn_full_unweighted.joblib`) y `τ = 0.21`
(leído de `results/metrics/metrics_summary.json`). No se regeneraron datos ni se reentrenó.

## 3. Predicciones honestas para las 800 materias (`predicted_pairs.py`)

El modelo se entrenó con 714 materias; el grafo tiene 800 nodos. Para los pares de entrenamiento
la predicción sería *in-sample* (con KNN, cada punto es su propio vecino) y daría un cronograma
optimista. Por eso hay dos modos:

- **`oof` (por defecto):** validación cruzada de 5 folds **por materia**. Cada par se predice con
  un modelo (hiperparámetros clonados del `.joblib` oficial) que nunca vio ninguna de sus dos
  materias. Es la medición que debe usarse en el informe.
- **`final`:** el modelo oficial ya entrenado (lo que se usaría para materias nuevas). Los pares
  de train son in-sample, por lo que sobreestima la calidad.

Las probabilidades se cachean en `results/predictions/pair_probabilities_oof.csv`.

## 4. Resultados (modo `oof`, `largest_first`)

Referencia oráculo (grafo real, 4.468 aristas): **19 turnos**, 0 choques. Total real: 28.677 alumno-choques.

| τ (≥ k vecinos positivos de 15) | Aristas | Turnos | ¿Cabe en 32? | Alumnos en choque | % | Precisión aristas | Recall aristas |
|---|---|---|---|---|---|---|---|
| 0.033 (k=1) | 31.715 | 53 | No | 215 | 0,75% | 0,080 | 0,569 |
| 0.100 (k=2) | 13.925 | 31 | Sí | 650 | 2,27% | 0,128 | 0,398 |
| 0.167 (k=3) | 7.595 | 25 | Sí | 1.273 | 4,44% | 0,173 | 0,293 |
| **0.21 oficial (k=4)** | **4.342** | **18** | **Sí** | **2.041** | **7,12%** | **0,224** | **0,218** |
| 0.300 (k=5) | 2.529 | 9 | Sí | 3.283 | 11,45% | 0,287 | 0,163 |
| 0.367 (k=6) | 1.338 | 9 | Sí | 5.247 | 18,30% | 0,420 | 0,126 |
| 0.500 (k=8) | 731 | 7 | Sí | 8.029 | 28,00% | 0,536 | 0,088 |

Con el modo `final` y τ = 0.21: 4.287 aristas, 24 turnos, 5,9% de alumnos en choque
(más optimista por el efecto in-sample; no usar como resultado principal).

### Lectura
1. **Tradeoff turnos vs. choques reales.** Bajar τ agrega aristas (más FP), exige más turnos y
   reduce los choques reales; subir τ hace lo contrario. Con τ = 0.21 el cronograma usa 18 turnos
   (incluso uno menos que el oráculo) pero deja ~7% de los alumno-choques sin resolver.
2. **Límite del calendario.** Con el calendario de 32 turnos, el τ más bajo utilizable es ≈ 0.10
   (31 turnos, 2,3% de alumno-choques). Por debajo (k=1) hacen falta 53 turnos y no cabe.
3. **Causa raíz.** La precisión de aristas del ML es baja (~22% en el umbral oficial), y muchos
   conflictos reales son de 1 solo alumno en optativas sin historial (ver
   `experiments/cierre_ml/informe_cierre_ml.md`). El grafo predicho mezcla aristas falsas y omite
   conflictos reales; ese es el costo que reporta `evaluate_against_real`.
4. **Probabilidades discretas.** KNN (k=15) devuelve múltiplos de 1/15, por lo que solo hay 15
   umbrales distintos. El barrido usa el punto medio (k−0,5)/15 para evitar errores de redondeo.

## 5. API de `conflict_graph.py`

| Función | Descripción |
|---|---|
| `load_exams_catalog(path)` | Catálogo de las 800 materias (nombre, depto, duración). |
| `build_conflict_graph_from_predictions(pred_df, τ, catalog)` | Grafo predicho, arista si `prob_conflicto >= τ`. Valida τ ∈ (0,1). |
| `build_real_graph(pairs_df, catalog)` | Grafo oráculo (solo referencia/evaluación). |
| `greedy_coloring(G, strategy)` | `largest_first` (Welsh-Powell, por defecto), `saturation_largest_first` (DSATUR), etc. |
| `build_schedule(coloring, catalog)` | Cronograma `{turno: [materias]}`. |
| `validate_coloring(G, coloring)` | Chequeo interno de que el coloreo respeta el grafo con el que se calculó. |
| `evaluate_against_real(coloring, real_pairs, G)` | Pares y alumnos en choque real, `cabe_en_calendario`, TP/FP/FN, precisión/recall de aristas. |
| `sweep_thresholds(pred_df, real_pairs, thresholds, catalog, strategy)` | Tabla de tradeoff por umbral. |

## 6. Cómo ejecutarlo

```bash
python -m src.graphs.run_schedule                      # τ oficial 0.21, modo oof
python -m src.graphs.run_schedule --threshold 0.10     # más turnos, menos choques
python -m src.graphs.run_schedule --strategy saturation_largest_first
python -m src.graphs.run_schedule --sweep              # results/graphs/threshold_sweep.csv
python -m unittest discover -s tests -v
```

## 7. Integración con las fases siguientes

- **Backend (FastAPI):** `GET /api/graph?threshold=X`, `/api/schedule?threshold=X` y `/api/stats`
  deben cargar `get_pair_probabilities()` una sola vez al iniciar y, por request, llamar a
  `build_conflict_graph_from_predictions`, `greedy_coloring`, `build_schedule` y
  `evaluate_against_real`. Cada respuesta debe incluir turnos, aristas, `cabe_en_calendario`,
  alumnos en choque y el valor de referencia del oráculo.
- **Frontend:** el slider de umbral debe mostrar el tradeoff turnos / % de alumnos en choque.

## 8. Limitaciones a declarar en el informe
- El modelo tiene baja precisión; el cronograma no elimina los choques reales (se miden, no se ocultan).
- La evaluación usa la cohorte 1994 como "realidad"; es un único semestre.
- El umbral oficial (0.21) se calibró para F1 de pares, no para minimizar choques de alumnos.
- El conjunto de test del ML ya fue consultado en rondas previas (ver informe de cierre ML).
