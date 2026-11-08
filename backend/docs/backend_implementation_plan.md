# Plan de Implementación del Backend — Hoja de Ruta Oficial

## 1. Objetivo y Arquitectura General

El backend de este proyecto actúa como el sistema coordinador para la predicción de conflictos y la asignación automatizada de turnos de exámenes universitarios, basado en el cuerpo de datos de Nottingham 1994 (800 exámenes y 319.600 pares potenciales).

### Principios Arquitectónicos
1. **Desacoplamiento Estricto:** La API web no reimplementa ni modifica los algoritmos de Machine Learning (`src/inference/`, `src/models/`) ni de Grafos y Coloreo (`src/graphs/`). Consume sus interfaces públicas.
2. **Eficiencia en Runtime:** El backend no reentrena modelos ni recalcula validaciones cruzadas (OOF) en peticiones HTTP. Los artefactos pesados se precargan en memoria durante el ciclo de vida del servidor (*lifespan*).
3. **Contratos Tipados:** Toda entrada y salida HTTP se valida mediante esquemas estrictos de Pydantic v2.
4. **Transporte Estándar:** Comunicación exclusiva vía HTTP + JSON, compatible con cualquier frontend SPA (React, Vue, etc.).
5. **Modelo Oficial en Producción:** `DecisionTree_Full_Unweighted` (`results/models/decision_tree_full_unweighted.joblib`) con umbral calibrado en validación de **`0.20`** (leído dinámicamente desde `results/metrics/metrics_summary.json`).

---

## 2. Estructura de Directorios

```
backend/
├── data/
│   ├── interim/exams_clean.csv     # Catálogo oficial de 800 exámenes
│   └── processed/subject_pairs_poc.csv # 319.600 pares de materias con datos históricos
├── docs/                           # Documentación técnica específica del backend
├── results/
│   ├── graphs/                     # Salidas de coloreo y cronogramas de referencia
│   ├── metrics/metrics_summary.json# Fuente única de verdad del umbral (0.20)
│   ├── models/decision_tree_full_unweighted.joblib # Modelo oficial
│   └── predictions/pair_probabilities_oof.csv # Probabilidades OOF cacheadas
├── src/
│   ├── api/
│   │   ├── __init__.py
│   │   ├── main.py                 # Punto de entrada FastAPI, lifespan y CORS
│   │   └── v1/
│   │       ├── __init__.py
│   │       ├── router.py           # Agregador de routers v1
│   │       └── endpoints/
│   │           ├── __init__.py
│   │           ├── health.py       # GET /health
│   │           ├── catalog.py      # GET /exams, GET /exams/{id}, GET /departments
│   │           ├── graph.py        # GET /graph, GET /conflicts/pair
│   │           └── schedule.py     # POST /schedule, POST /schedule/evaluate
│   ├── common/                     # Esquema canónico y PROJECT_ROOT
│   ├── core/                       # Configuración y variables de entorno
│   ├── schemas/                    # Schemas Pydantic v2
│   └── services/                   # Lógica de catálogo, grafo y cronograma
├── tests/                          # Batería de 37 tests automatizados
└── requirements.txt                # Dependencias de Python del backend
```

---

## 3. Contratos de Integración con Módulos Existentes

- **Catálogo de Exámenes:** Consumo directo de `data/interim/exams_clean.csv` mediante la función [`load_exams_catalog(path)`](file:///w:/home/anton/Tp-Grafos-ML/src/graphs/conflict_graph.py#L34-L45).
- **Probabilidades de Conflicto:** Consumo de la matriz OOF precalculada (`results/predictions/pair_probabilities_oof.csv`) cargada en memoria.
- **Construcción del Grafo:** Invocación de [`build_conflict_graph_from_predictions(pred_df, threshold, catalog)`](file:///w:/home/anton/Tp-Grafos-ML/src/graphs/conflict_graph.py#L56-L71). En umbral `0.20`, genera **2.032 aristas** sobre los 800 exámenes.
- **Asignación de Turnos (Coloreo):** Invocación de [`greedy_coloring(G, strategy)`](file:///w:/home/anton/Tp-Grafos-ML/src/graphs/conflict_graph.py#L87-L93) y estructuración con [`build_schedule(coloring, catalog)`](file:///w:/home/anton/Tp-Grafos-ML/src/graphs/conflict_graph.py#L99-L112).
- **Validación del Coloreo:** Invocación de [`validate_coloring(G, coloring)`](file:///w:/home/anton/Tp-Grafos-ML/src/graphs/conflict_graph.py#L114-L123).
- **Evaluación vs. Ground Truth:** Invocación de [`evaluate_against_real(coloring, real_pairs_df, G)`](file:///w:/home/anton/Tp-Grafos-ML/src/graphs/conflict_graph.py#L125-L163).

---

## 4. Endpoints del Sistema

| Método | Ruta | Parámetros (Query / Body) | Respuestas | Errores Posibles | Estado |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `GET` | `/api/v1/health` | Ninguno | `200 OK`: Estado, versión, catálogo cargado | 500 | **Implementado** |
| `GET` | `/api/v1/exams` | `department_code`, `search`, `limit`, `offset` | `200 OK`: Total y lista paginada de exámenes | 422 | **Implementado** |
| `GET` | `/api/v1/exams/{exam_id}` | Path: `exam_id` | `200 OK`: Detalle de la materia | 404 | **Implementado** |
| `GET` | `/api/v1/departments` | Ninguno | `200 OK`: Lista de departamentos y cantidad de exámenes | 500 | **Implementado** |
| `GET` | `/api/v1/graph` | Query: `threshold`, `include_edges` | `200 OK`: Nodos, aristas activas, métricas topológicas | 400, 422 | *Etapa 2* |
| `GET` | `/api/v1/conflicts/pair` | Query: `exam_id_1`, `exam_id_2`, `threshold` | `200 OK`: Probabilidad y predicción del par | 400, 404, 422 | *Etapa 2* |
| `POST` | `/api/v1/schedule` | Body: `threshold`, `strategy` | `200 OK`: Cantidad de turnos, verificación $\le 32$, agenda | 400, 422 | *Etapa 3* |
| `POST` | `/api/v1/schedule/evaluate` | Body: `threshold`, `strategy` | `200 OK`: Métricas vs 1994 (choques, alumnos afectados, TP/FP/FN) | 400, 422 | *Etapa 4* |

---

## 5. Protocolo de Inicio desde Clonación Limpia

1. **Crear y activar entorno virtual:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # o .\venv\Scripts\Activate.ps1 en Windows
   ```
2. **Instalar dependencias:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Asegurar la caché de probabilidades OOF:**
   ```bash
   python -m src.graphs.run_schedule --force
   ```
4. **Iniciar el servidor de desarrollo:**
   ```bash
   uvicorn src.api.main:app --reload --port 8000
   ```
5. **Ejecutar tests:**
   ```bash
   python -m unittest discover -s tests -v
   ```

---

## 6. Etapas de Desarrollo y Criterios de Aceptación

### Etapa 1 — Infraestructura y Catálogo [COMPLETADA]
- [x] Estructura base FastAPI, configuración (`core/config.py`), router v1, endpoint `/health`.
- [x] Lifespan para cargar los 800 exámenes en `app.state.catalog`.
- [x] Schemas Pydantic v2 `ExamOut`, `ExamListResponse`, `DepartmentOut`.
- [x] Endpoints `GET /exams`, `GET /exams/{id}`, `GET /departments`.
- [x] 23 tests automatizados pasando al 100%.

### Etapa 2 — Integración del Grafo [COMPLETADA]
- [x] Precarga de `results/predictions/pair_probabilities_oof.csv` en `app.state.probabilities_df` e indexación de pares para búsqueda instantánea en `app.state.pairs_prob_map`.
- [x] Schemas Pydantic v2 en `src/schemas/graph.py` (`GraphEdgeOut`, `GraphResponse`, `ConflictPairResponse`).
- [x] Servicio `src/services/graph_service.py` (`build_graph_for_threshold`, `get_graph_statistics`, `get_pair_probability_from_map`).
- [x] Endpoints en `src/api/v1/endpoints/graph.py`:
  - `GET /api/v1/graph`: Construye el subgrafo predicho para el umbral solicitado (default: `0.20`), computa densidad, componentes conexas, nodos aislados y lista de aristas.
  - `GET /api/v1/conflicts/pair`: Consulta directa de la probabilidad entre dos materias del catálogo con validación de auto-pares (400) e IDs inexistentes (404).
- [x] Tests automatizados en `tests/test_api_graph.py`.

### Etapa 3 — Generación de Cronogramas [COMPLETADA]
- [x] Conexión con `greedy_coloring`, `build_schedule` y `validate_coloring`.
- [x] Endpoint `POST /api/v1/schedule` validando límite de 32 turnos y generando asignación disjunta (11 turnos a $\theta = 0.20$).
- [x] Endpoint `GET /api/v1/schedule/strategies` exponiendo las 7 heurísticas soportadas.
- [x] Tests automatizados en `tests/test_api_schedule.py` (7 tests, incluyendo unicidad estricta de las 800 materias y cronogramas que exceden 32 turnos).
- [x] Suite completa de 37 tests pasando al 100%.

### Etapa 4 — Evaluación y Cierre [SIGUIENTE INCREMENTO]
- [ ] Conexión con `evaluate_against_real` sobre los pares de 1994.
- [ ] Endpoint `POST /api/v1/schedule/evaluate`.
- [ ] Tests de integración punta a punta y documentación final.
