# Backend — Servicio de Predicción y Asignación de Turnos de Exámenes

Módulo backend desacoplado que coordina la inferencia de Machine Learning, la construcción del grafo de conflictos y el coloreo para la asignación de turnos universitarios (dataset Nottingham 1994).

---

## 1. Estructura del Directorio `backend/`

```
backend/
├── data/
│   ├── interim/
│   │   ├── exams_clean.csv            # Catálogo oficial de 800 exámenes
│   │   ├── enrolments_clean.csv
│   │   └── students_clean.csv
│   └── processed/
│       ├── subject_pairs_poc.csv      # 319.600 pares con ground truth de 1994
│       └── ...
├── docs/
│   ├── backend_implementation_plan.md # Hoja de ruta oficial de etapas
│   ├── backend_stage_1.md             # Documentación de infraestructura y catálogo
│   ├── backend_stage_3.md             # Documentación de cronograma y coloreo
│   └── backend_stage_4.md             # Documentación de evaluación vs cohorte 1994
├── results/
│   ├── graphs/
│   │   ├── schedule_0_20.json         # Cronograma de referencia oficial
│   │   ├── summary.json               # Métricas del grafo y coloreo
│   │   └── threshold_sweep.csv        # Barrido de umbrales
│   ├── metrics/
│   │   └── metrics_summary.json       # Fuente única de verdad del umbral (0.20)
│   ├── models/
│   │   └── decision_tree_full_unweighted.joblib # Modelo oficial entrenado
│   └── predictions/
│       └── pair_probabilities_oof.csv # Caché precalculada OOF de 319.600 pares
├── src/
│   ├── api/                           # FastAPI app, routers y endpoints REST
│   ├── common/                        # Esquema de datos, tipos y PROJECT_ROOT
│   ├── core/                          # Configuración y variables de entorno
│   ├── graphs/                        # Módulo base de NetworkX y coloreo
│   ├── inference/                     # Pipeline desacoplado de ML
│   ├── models/                        # Entrenamiento y comparación
│   ├── parsing/                       # Ingesta de archivos originales
│   ├── schemas/                       # Modelos Pydantic v2
│   ├── services/                      # Lógica de catálogo, grafo y cronograma
│   └── transformation/                # Construcción de features diádicas
├── tests/
│   ├── test_api_catalog.py            # Tests de endpoints de catálogo
│   ├── test_api_graph.py              # Tests de topología del grafo y pares
│   ├── test_api_health.py             # Tests de healthcheck
│   ├── test_api_schedule.py           # Tests de cronogramas y coloreo
│   ├── test_graphs.py                 # Tests base del módulo de grafos
│   └── test_inference.py              # Tests base de inferencia ML
├── requirements.txt                   # Dependencias de Python del backend
└── README.md                          # Este archivo de instrucciones
```

---

## 2. Directorio de Ejecución

> **IMPORTANTE:** Todos los comandos del backend deben ejecutarse posicionándose dentro de la carpeta `backend/`:
> ```bash
> cd backend
> ```

---

## 3. Instalación de Dependencias

1. Crear y activar el entorno virtual de Python:
   ```bash
   python -m venv venv
   source venv/bin/activate       # En Linux / WSL
   # .\venv\Scripts\Activate.ps1  # En Windows PowerShell
   ```

2. Instalar las dependencias:
   ```bash
   pip install -r requirements.txt
   ```

---

## 4. Iniciar el Servidor FastAPI

Desde la carpeta `backend/`:
```bash
uvicorn src.api.main:app --reload --port 8000
```

- **API Base:** [http://localhost:8000](http://localhost:8000)
- **Documentación Swagger UI:** [http://localhost:8000/api/v1/docs](http://localhost:8000/api/v1/docs)
- **Documentación ReDoc:** [http://localhost:8000/api/v1/redoc](http://localhost:8000/api/v1/redoc)

---

## 5. Ejecución de Pruebas Automatizadas

Desde la carpeta `backend/`:
```bash
python -m unittest discover -s tests -v
```
*(Ejecuta los 43 tests automatizados cubriendo API, validaciones, NetworkX e inferencia ML).*

---

## 6. Regenerar Caché de Probabilidades (Si fuera necesario)

Si en una clonación limpia `results/predictions/pair_probabilities_oof.csv` no estuviera presente:
```bash
python -m src.graphs.run_schedule --force
```
