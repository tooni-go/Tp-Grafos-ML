# Plan de Implementación del Backend — Hoja de Ruta Oficial

## 1. Objetivo y Arquitectura General

El backend de este proyecto actúa como el sistema coordinador para la predicción de conflictos y la asignación automatizada de turnos de exámenes universitarios, basado en el cuerpo de datos de Nottingham 1994 (800 exámenes y 319.600 pares potenciales).

### Principios Arquitectónicos
1. **Desacoplamiento Estricto:** La API web no reimplementa ni modifica los algoritmos de Machine Learning (`src/inference/`, `src/models/`) ni de Grafos y Coloreo (`src/graphs/`). Consume sus interfaces públicas.
2. **Eficiencia en Runtime:** El backend no reentrena modelos ni recalcula validaciones cruzadas (OOF) en peticiones HTTP. Los artefactos computacionalmente pesados se precalculan offline y se consumen en memoria durante el ciclo de vida del servidor (*lifespan*).
3. **Contratos Tipados:** Toda entrada y salida HTTP se valida mediante esquemas estrictos de Pydantic v2.
4. **Transporte Estándar:** Comunicación exclusiva vía HTTP + JSON, compatible con cualquier frontend SPA (React, Vue, etc.).

---

## 2. Estructura de Directorios Prevista

```
src/
├── api/
│   ├── __init__.py
│   ├── main.py                     # Punto de entrada FastAPI, lifespan y CORS
│   └── v1/
│       ├── __init__.py
│       ├── router.py               # Agregador de routers v1
│       └── endpoints/
│           ├── __init__.py
│           ├── health.py           # GET /health
│           ├── catalog.py          # GET /exams, GET /exams/{id}, GET /departments
│           ├── graph.py            # POST /graph, GET /conflicts/pair
│           └── schedule.py         # POST /schedule, POST /schedule/evaluate
├── core/
│   ├── __init__.py
│   └── config.py                   # Rutas canónicas, variables de entorno y defaults
├── schemas/
│   ├── __init__.py
│   ├── catalog.py                  # Schemas de exámenes y departamentos
│   ├── graph.py                    # Schemas de topología y pares
│   └── schedule.py                 # Schemas de cronograma y evaluación
└── services/
    ├── __init__.py
    ├── catalog_service.py          # Consultas optimizadas sobre exams_clean.csv
    └── schedule_service.py         # Orquestación sobre conflict_graph.py
```

---

## 3. Contratos de Integración con Módulos Existentes

- **Catálogo de Exámenes:** Consumo directo de `data/interim/exams_clean.csv` mediante la función [`load_exams_catalog(path)`](file:///w:/home/anton/Tp-Grafos-ML/src/graphs/conflict_graph.py#L34-L45).
- **Probabilidades de Conflicto:** Consumo de la matriz OOF precalculada (`results/predictions/pair_probabilities_oof.csv`) cargada en memoria.
- **Construcción del Grafo:** Invocación de [`build_conflict_graph_from_predictions(pred_df, threshold, catalog)`](file:///w:/home/anton/Tp-Grafos-ML/src/graphs/conflict_graph.py#L56-L71).
- **Asignación de Turnos (Coloreo):** Invocación de [`greedy_coloring(G, strategy)`](file:///w:/home/anton/Tp-Grafos-ML/src/graphs/conflict_graph.py#L87-L93) y estructuración con [`build_schedule(coloring, catalog)`](file:///w:/home/anton/Tp-Grafos-ML/src/graphs/conflict_graph.py#L99-L112).
- **Validación del Coloreo:** Invocación de [`validate_coloring(G, coloring)`](file:///w:/home/anton/Tp-Grafos-ML/src/graphs/conflict_graph.py#L114-L123).
- **Evaluación vs. Ground Truth:** Invocación de [`evaluate_against_real(coloring, real_pairs_df, G)`](file:///w:/home/anton/Tp-Grafos-ML/src/graphs/conflict_graph.py#L125-L163).

---

## 4. Endpoints Previstos

| Método | Ruta | Parámetros (Query / Body) | Respuestas | Errores Posibles |
| :--- | :--- | :--- | :--- | :--- |
| `GET` | `/api/v1/health` | Ninguno | `200 OK`: Estado, versión, catálogo cargado | 500 |
| `GET` | `/api/v1/exams` | `department_code`, `search`, `limit`, `offset` | `200 OK`: Total y lista paginada de exámenes | 422 |
| `GET` | `/api/v1/exams/{exam_id}` | Path: `exam_id` | `200 OK`: Detalle de la materia | 404 |
| `GET` | `/api/v1/departments` | Ninguno | `200 OK`: Lista de departamentos y cantidad de exámenes | 500 |
| `GET` | `/api/v1/conflicts/pair` | Query: `exam_id_1`, `exam_id_2` | `200 OK`: Probabilidad y predicción del par | 400, 404, 422 |
| `POST` | `/api/v1/graph` | Body: `threshold`, `include_isolated_nodes` | `200 OK`: Nodos, aristas activas, métricas topológicas | 400, 422 |
| `POST` | `/api/v1/schedule` | Body: `threshold`, `strategy` | `200 OK`: Cantidad de turnos, verificación $\le 32$, agenda | 400, 422 |
| `POST` | `/api/v1/schedule/evaluate` | Body: `threshold`, `strategy` | `200 OK`: Métricas vs 1994 (choques, alumnos afectados, TP/FP/FN) | 400, 422 |

---

## 5. Datos y Dependencias Requeridas

### Dependencias de Python
- `fastapi>=0.110.0`
- `uvicorn[standard]>=0.28.0`
- `pydantic>=2.6.0`
- `httpx>=0.27.0` (testing)
- Existentes: `networkx>=3.2`, `pandas>=2.2.0`, `scikit-learn==1.9.1`, `joblib>=1.3.0`

### Artefactos de Datos
- `data/interim/exams_clean.csv` (Catálogo de 800 exámenes).
- `data/processed/subject_pairs_poc.csv` (Datos de evaluación histórica).
- `results/predictions/pair_probabilities_oof.csv` (Probabilidades precalculadas).

---

## 6. Protocolo de Inicio desde Clonación Limpia

1. **Crear y activar entorno virtual:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # o .\venv\Scripts\Activate.ps1 en Windows
   ```
2. **Instalar dependencias:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Asegurar la caché de probabilidades:**
   ```bash
   python -c "from src.graphs.predicted_pairs import get_pair_probabilities; get_pair_probabilities(mode='oof')"
   ```
4. **Iniciar el servidor de desarrollo:**
   ```bash
   uvicorn src.api.main:app --reload --port 8000
   ```

---

## 7. Fases de Desarrollo Planificadas

### Etapa 1 — Infraestructura y Catálogo (En Curso)
- **Incremento 1 (Actual):** Estructura base FastAPI, configuración (`core/config.py`), router v1, endpoint `/health`, dependencias en `requirements.txt` y primer test de API.
- **Incremento 2:** Schemas Pydantic del catálogo, carga en `lifespan`, servicio de catálogo y endpoints de consulta (`/exams`, `/exams/{id}`, `/departments`) con tests.

### Etapa 2 — Integración del Grafo
- Carga de probabilidades OOF precalculadas en el `lifespan`.
- Servicio de grafos conectando `build_conflict_graph_from_predictions`.
- Endpoint `POST /api/v1/graph` y `GET /api/v1/conflicts/pair`.
- Tests de integración de grafo.

### Etapa 3 — Generación de Cronogramas
- Conexión con `greedy_coloring`, `build_schedule` y `validate_coloring`.
- Endpoint `POST /api/v1/schedule` validando límite de 32 turnos.
- Tests de validación de coloraciones disjuntas.

### Etapa 4 — Evaluación y Cierre
- Conexión con `evaluate_against_real` sobre los pares de 1994.
- Endpoint `POST /api/v1/schedule/evaluate`.
- Tests de integración punta a punta y documentación de consumo final.
