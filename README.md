# Tp-Grafos-ML — Planificación de Mesas de Examen Universitarias

Módulo de preparación de datos, entrenamiento e **inferencia de Machine Learning** sobre el dataset *University of Nottingham*, listo para integrarse con el algoritmo de coloreo clásico de grafos no ponderado.

---

## 1. Dependencias e Instalación

Se requiere Python 3.10+ y los paquetes listados en [`requirements.txt`](file:///w:/home/anton/Tp-Grafos-ML/requirements.txt) (`numpy`, `pandas`, `scikit-learn`, `joblib`, `jupyter`, `ipykernel`, `matplotlib`):

```bash
python3 -m venv venv
./venv/bin/pip install -r requirements.txt
```

---

## 2. Cómo consumir el Módulo de Inferencia (`src/inference/`)

El módulo [`src/inference/predict_conflicts.py`](file:///w:/home/anton/Tp-Grafos-ML/src/inference/predict_conflicts.py) expone la función pública `predict_conflict_pairs()`, que encapsula:
- La carga del modelo candidato **`KNN_Full_Unweighted`** desde `results/models/knn_full_unweighted.joblib` (un `Pipeline` que ya incluye su propio `StandardScaler` y `KNeighborsClassifier(n_neighbors=15)`).
- La lectura del umbral validado **`0.21`** desde `results/metrics/metrics_summary.json`.
- El cálculo de probabilidades mediante `model.predict_proba(X)[:, 1]` y la aplicación explícita del corte `prob_conflicto >= 0.21` (nunca utiliza `model.predict(X)`).

### Columnas requeridas en el `DataFrame` de entrada
- **Identificadores del par (`ID_COLUMNS`):** `materia_id_1`, `materia_id_2`.
- **Las 10 features predictoras (`FEATURE_COLUMNS`):**
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

> La función valida la ausencia de nulos e infinitos, verifica tipos numéricos/binarios y **reordena automáticamente las columnas** al orden exacto que exige el modelo antes de predecir.

### Ejemplo de integración para construir el grafo

```python
import pandas as pd
from src.inference import load_conflict_model, predict_conflict_pairs

# 1. Leer pares de materias (por ejemplo, el conjunto de prueba o el universo de pares)
pares_df = pd.read_csv("data/processed/ml_test_full.csv")

# 2. (Opcional) Cargar el modelo una vez en memoria
modelo = load_conflict_model()

# 3. Obtener probabilidades y clasificación binaria (umbral por defecto = 0.21)
predicciones_df = predict_conflict_pairs(pares_df, model=modelo)

# 4. Extraer los pares con conflicto predicho (pred_conflicto == 1) como aristas del grafo
aristas = list(
    predicciones_df.loc[
        predicciones_df["pred_conflicto"] == 1,
        ["materia_id_1", "materia_id_2"],
    ].itertuples(index=False, name=None)
)
print(f"Aristas de incompatibilidad generadas: {len(aristas)}")
```

### Interpretación de la salida
`predict_conflict_pairs()` retorna un `pandas.DataFrame` con:
- `materia_id_1`, `materia_id_2`: identificadores de las dos materias del par $(u, v)$.
- `prob_conflicto`: probabilidad estimada en $[0.0, 1.0]$ (`predict_proba(X)[:, 1]`).
- `pred_conflicto`: entero `1` si `prob_conflicto >= 0.21` (agregar arista de incompatibilidad $\{u, v\}$ al grafo no ponderado) o `0` en caso contrario.

---

## 3. Ejecución de Pruebas Automatizadas

```bash
./venv/bin/python -m unittest discover -s tests -v
```

---

## 4. Documentación del Proyecto

- [`docs/plan_poc_dataset_nottingham.md`](file:///w:/home/anton/Tp-Grafos-ML/docs/plan_poc_dataset_nottingham.md): Plan general e integración de las fases de datos y ML.
- [`docs/fase_2_5_preparacion_ml.md`](file:///w:/home/anton/Tp-Grafos-ML/docs/fase_2_5_preparacion_ml.md): Partición estratificada por materias y prevención de *data leakage*.
- [`docs/fase_3_entrenamiento_ml.md`](file:///w:/home/anton/Tp-Grafos-ML/docs/fase_3_entrenamiento_ml.md): Decisiones metodológicas, métricas, umbrales y especificación del módulo de inferencia.
- [`notebooks/Test_modelo_ML.ipynb`](file:///w:/home/anton/Tp-Grafos-ML/notebooks/Test_modelo_ML.ipynb): Notebook interactivo de inspección y auditoría de reproducibilidad.
