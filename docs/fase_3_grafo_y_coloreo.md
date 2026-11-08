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

El modelo y el umbral son los oficiales de la rama ML:
`DecisionTree_Full_Unweighted` (`results/models/decision_tree_full_unweighted.joblib`) y `τ = 0.20`
(leído de `results/metrics/metrics_summary.json`).

## 3. Predicciones honestas para las 800 materias (`predicted_pairs.py`)

El modelo se entrenó con 714 materias; el grafo tiene 800 nodos. Para los pares de entrenamiento
la predicción sería *in-sample* y daría un cronograma optimista. Por eso hay dos modos:

- **`oof` (por defecto):** validación cruzada de 5 folds **por materia**. Cada par se predice con
  un modelo (hiperparámetros clonados del `.joblib` oficial) que nunca vio ninguna de sus dos
  materias. Es la medición que debe usarse en el informe.
- **`final`:** el modelo oficial ya entrenado (lo que se usaría para materias nuevas). Los pares
  de train son in-sample, por lo que sobreestima la calidad.

Las probabilidades se cachean en `results/predictions/pair_probabilities_oof.csv`.

## 4. Resultados (modo `oof`, `largest_first`)

Referencia oráculo (grafo real, 4.468 aristas): **19 turnos**, 0 choques. Total real: 28.677 alumno-choques.

| τ | Aristas | Turnos | ¿Cabe en 32? | Alumnos en choque | % | Precisión aristas | Recall aristas |
|---|---|---|---|---|---|---|---|
| 0.05 | 23.286 | 139 | No | 307 | 1,07% | 0,106 | 0,554 |
| 0.10 | 3.749 | 15 | Sí | 2.412 | 8,41% | 0,277 | 0,232 |
| 0.15 | 3.019 | 14 | Sí | 2.996 | 10,45% | 0,309 | 0,209 |
| **0.20 oficial** | **2.032** | **11** | **Sí** | **3.723** | **12,98%** | **0,374** | **0,170** |
| 0.25 | 1.659 | 12 | Sí | 4.371 | 15,24% | 0,412 | 0,153 |
| 0.30 | 1.029 | 9 | Sí | 5.466 | 19,06% | 0,505 | 0,116 |
| 0.40 | 669 | 9 | Sí | 6.598 | 23,01% | 0,607 | 0,091 |
| 0.50 | 524 | 9 | Sí | 7.649 | 26,67% | 0,678 | 0,080 |

### Lectura
1. **Tradeoff turnos vs. choques reales.** Bajar τ agrega aristas (más aristas preventivas), exige más turnos y
   reduce los choques reales; subir τ genera menos aristas, compacta el calendario en menos turnos pero deja más choques.
   Con τ = 0.20 el cronograma utiliza **11 turnos** (muy holgado dentro del calendario de 32 turnos) y contiene
   el 87,02% de los alumnos sin conflicto (12,98% en choque).
2. **Mayor precisión del Árbol de Decisión.** Frente a KNN (que tenía una precisión de aristas del 22,4% en τ=0.21),
   el Árbol de Decisión alcanza un **37,35% de precisión** con τ=0.20 (y más del 50% con τ >= 0.30). Esto evita
   la sobrecarga de falsos positivos en el grafo de incompatibilidad.
3. **Límite del calendario.** Con el calendario de 32 turnos, los umbrales utilizables van desde τ ≈ 0.10
   (15 turnos, 8,41% de alumno-choques) hasta τ = 0.50 (9 turnos). Por debajo de 0.05 la cantidad de aristas
   se dispara y exige 139 turnos.
4. **Eficiencia computacional.** A diferencia de KNN (que requería calcular distancias sobre 319.600 pares en 5 folds
   tardando varios minutos), el Árbol de Decisión evalúa reglas de corte ortogonales en milisegundos, permitiendo
   calcular la validación cruzada OOF completa en segundos.

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
python -m src.graphs.run_schedule                      # τ oficial 0.20, modo oof
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
