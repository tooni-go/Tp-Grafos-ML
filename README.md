# Tp-Grafos-ML — Planificación de Mesas de Examen Universitarias

Sistema de predicción de conflictos curriculares y asignación automática de turnos de exámenes finales (*Examination Timetabling Problem*) mediante Machine Learning tabular diádico, modelado con grafos y algoritmos de coloreo clásico, evaluado sobre el benchmark histórico de la *University of Nottingham (1994)*.

---

## 1. Problema y Objetivo

- **Problema:** En instituciones universitarias, coordinar el calendario de exámenes finales evitando que un estudiante deba rendir dos asignaturas en el mismo turno requiere conocer las superposiciones de inscripción. En contextos donde estos datos no se conocen de antemano o se desea planificar preventivamente, se necesita estimar qué pares de asignaturas generarán conflicto.
- **Objetivo:** Predecir incompatibilidades entre asignaturas utilizando modelos de Machine Learning entrenados sobre atributos curriculares e históricos, construir un grafo de incompatibilidades no ponderado a partir de dichas predicciones, y resolver la asignación de turnos respetando la capacidad del calendario académico (máximo 32 turnos) mediante algoritmos heurísticos de coloreo de grafos.

---

## 2. Enfoque Metodológico (ML, Grafos y Coloreo)

El flujo de trabajo se compone de tres etapas integradas:

1. **Machine Learning Diádico (Inferencia de Conflictos):**
   - El problema se formula sobre pares no dirigidos de materias $(u, v)$.
   - Se utiliza un modelo de árbol de decisión (**`DecisionTree_Full_Unweighted`**, `max_depth=5`) entrenado sobre 10 variables explicativas curriculares y estructurales.
   - El modelo predice una probabilidad continua $\hat{p} \in [0, 1]$. Se aplica un punto de corte validado **$\theta = 0.20$** (establecido en `metrics_summary.json` como Fuente Única de Verdad). Si $\hat{p} \ge 0.20$, el par se clasifica como incompatible.
2. **Construcción del Grafo de Conflictos (NetworkX):**
   - Cada una de las 800 asignaturas constituye un nodo del grafo.
   - Se incorpora una arista no dirigida $\{u, v\}$ únicamente si el par fue clasificado como conflictivo por el modelo de ML para el umbral dado.
3. **Coloreo de Grafos y Armado de Cronogramas:**
   - La asignación de turnos (slots horarios) se resuelve como un problema clásico de coloreo de vértices, garantizando que dos materias adyacentes reciban colores distintos.
   - Se exploran heurísticas voraces y de grado de saturación (*Largest First*, *DSATUR* / *Saturation Largest First*, *Smallest Last*, etc.).
   - La solución se evalúa posteriormente contra el registro histórico real de 1994 para cuantificar la efectividad práctica y los choques residuales.

---

## 3. Arquitectura General del Repositorio

La estructura del proyecto separa la lógica del servicio y motor de cálculo de los recursos analíticos y documentales:

```
Tp-Grafos-ML/
├── backend/                       # Servicio API REST y motor computacional
│   ├── data/                      # Catálogos limpios y pares procesados
│   ├── docs/                      # Especificaciones y roadmap del backend
│   ├── results/                   # Modelos oficiales, métricas y predicciones
│   ├── src/                       # API FastAPI, inferencia ML, grafos y servicios
│   ├── tests/                     # Suite de pruebas automatizadas (37 tests)
│   ├── requirements.txt           # Dependencias de Python del backend
│   └── README.md                  # Guía completa de instalación y uso del backend
├── docs/                          # Documentación metodológica y diseño de fases
├── experiments/                   # Registros de experimentación y prototipos
├── notebooks/                     # Notebooks de exploración y auditoría
├── Nottingham Dataset/            # Datos fuente originales (Nottingham 1994)
└── README.md                      # Documentación general del proyecto (este archivo)
```

---

## 4. Tecnologías Utilizadas

- **Lenguaje:** Python 3.10+
- **Machine Learning & Datos:** `scikit-learn`, `pandas`, `numpy`, `joblib`
- **Modelado de Grafos & Heurísticas:** `networkx`
- **API & Servicios Web:** `fastapi`, `uvicorn`, `pydantic` v2
- **Testing & Aserciones:** `unittest`

---

## 5. Estado Actual del Desarrollo

- **Investigación y Modelado ML (Completado):** Preparación diádica de datos, validación cruzada estratificada OOF (319.600 pares evaluados), calibración de umbrales óptimos y comparación de algoritmos.
- **Backend API REST (Etapas 1 a 4 Completadas):**
  - **Catálogo de Exámenes:** Paginación, búsqueda y filtros sobre los 800 exámenes y sus departamentos.
  - **Grafo de Conflictos:** Consulta de métricas topológicas, lista de aristas con umbral configurable y consulta simétrica de probabilidades por par.
  - **Generación de Cronogramas:** Asignación de turnos con 7 heurísticas de coloreo, validación formal de adyacencia y control de capacidad del calendario ($\le 32$ turnos).
  - **Evaluación vs. Cohorte 1994:** Auditoría de efectividad contra ground truth histórico, reducción del 87.02% de incidencias potenciales de solapamiento entre pares, métricas de alumnos únicos afectados y matriz de confusión de aristas.
  - **Cobertura de Pruebas:** 43 tests automatizados unitarios y de integración.
- **Próximos Pasos:** Integración con futura interfaz de usuario (frontend SPA).

---

## 6. Ejecución Local

Todo el código ejecutable, la API y la suite de pruebas se operan desde el directorio `backend/`.

Para comenzar, ingresá en la carpeta:

```bash
cd backend
```

Para consultar las instrucciones completas sobre:
- Configuración y activación de entornos virtuales (`venv` en Linux/WSL o Windows),
- Instalación de librerías (`requirements.txt`),
- Puesta en marcha del servidor FastAPI con recarga en vivo (`uvicorn`),
- Ejecución de la suite completa de pruebas unitarias (`unittest`), y
- Comandos CLI del generador de cronogramas,

consultá el manual detallado en [**`backend/README.md`**](backend/README.md).

---

## 7. Módulo de Inferencia de Conflictos (`backend/src/inference/`)

El módulo `src/inference/predict_conflicts.py` encapsula la inferencia productiva:
- Carga el modelo oficial serializado desde `backend/results/models/decision_tree_full_unweighted.joblib`.
- Obtiene el umbral calibrado oficial (`0.20`) de `backend/results/metrics/metrics_summary.json`.
- Calcula probabilidades mediante `model.predict_proba(X)[:, 1]` y aplica el corte `prob_conflicto >= 0.20`.

### Variables Predictoras Requeridas (`FEATURE_COLUMNS`)
1. `mismo_departamento` (`bool` o `0/1`)
2. `mismo_prefijo_curricular` (`bool` o `0/1`)
3. `anio_confirmado` (`bool` o `0/1`)
4. `mismo_anio` (`bool` o `0/1`)
5. `misma_carrera` (`bool` o `0/1`)
6. `cant_carreras_compartidas_hist` (`int`/`float`)
7. `cant_inscriptos_hist_1` (`int`/`float`)
8. `cant_inscriptos_hist_2` (`int`/`float`)
9. `cant_estudiantes_compartidos_historico` (`int`/`float`)
10. `jaccard_historico` (`float`)

### Ejemplo de Integración en Python (desde `backend/`)

```python
import pandas as pd
from src.inference import load_conflict_model, predict_conflict_pairs

# 1. Leer conjunto de pares a evaluar
pares_df = pd.read_csv("data/processed/ml_test_full.csv")

# 2. Cargar modelo oficial (opcional, predict_conflict_pairs lo carga por defecto)
modelo = load_conflict_model()

# 3. Estimar probabilidades y clasificar según el umbral oficial (0.20)
predicciones_df = predict_conflict_pairs(pares_df, model=modelo)

# 4. Extraer los pares conflictivos para alimentar el grafo de NetworkX
aristas = list(
    predicciones_df.loc[
        predicciones_df["pred_conflicto"] == 1,
        ["materia_id_1", "materia_id_2"],
    ].itertuples(index=False, name=None)
)
print(f"Aristas de incompatibilidad identificadas: {len(aristas)}")
```

---

## 8. Documentación Académica y Metodológica

- [`docs/plan_poc_dataset_nottingham.md`](docs/plan_poc_dataset_nottingham.md): Plan general e integración de las fases de datos y ML.
- [`docs/fase_2_5_preparacion_ml.md`](docs/fase_2_5_preparacion_ml.md): Partición estratificada por materias y prevención de *data leakage*.
- [`docs/fase_3_entrenamiento_ml.md`](docs/fase_3_entrenamiento_ml.md): Metodología de entrenamiento, métricas, umbrales y diseño de inferencia.
- [`docs/fase_3_grafo_y_coloreo.md`](docs/fase_3_grafo_y_coloreo.md): Construcción del grafo de conflictos y análisis del barrido de heurísticas de coloreo.
- [`notebooks/Test_modelo_ML.ipynb`](notebooks/Test_modelo_ML.ipynb): Notebook interactivo de auditoría y reproducibilidad.
