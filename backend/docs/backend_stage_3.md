# Documentación de Implementación — Etapa 3: Generación de Cronogramas y Coloreo

## 1. Resumen de lo Implementado

En esta tercera etapa se implementó el módulo de generación y validación de cronogramas de exámenes mediante algoritmos de coloreo de grafos:

1. **Esquemas Pydantic v2 ([`src/schemas/schedule.py`](file:///w:/home/anton/Tp-Grafos-ML/src/schemas/schedule.py)):**
   - `ScheduleRequest`: Payload con `threshold` (flotante en `(0, 1)`, default `0.20`) y `strategy` (heurística de coloreo, default `"largest_first"`).
   - `ScheduleResponse`: Respuesta completa que incluye:
     - `threshold`: Umbral aplicado.
     - `strategy`: Heurística utilizada.
     - `turns_count`: Cantidad total de turnos de examen asignados (colores).
     - `max_allowed_turns`: Capacidad máxima del calendario de Nottingham (`32`).
     - `fits_in_calendar`: Booleano que valida si `turns_count <= 32`.
     - `is_valid_coloring`: Validación formal de que ningún par adyacente en el grafo predicho comparte turno.
     - `total_exams_scheduled`: Total de materias agendadas (`800`).
     - `turn_distribution`: Conteo de materias asignadas por turno.
     - `schedule`: Diccionario `{ "0": [ExamOut, ...], "1": [ExamOut, ...] }` con la información completa de cada examen asignado.
   - `ScheduleStrategyOut`: Metadatos descriptivos de las heurísticas disponibles (`strategy`, `name`, `description`).
2. **Servicio de Cronogramas ([`src/services/schedule_service.py`](file:///w:/home/anton/Tp-Grafos-ML/src/services/schedule_service.py)):**
   - `generate_schedule`:
     1. Valida la heurística contra `VALID_STRATEGIES`.
     2. Construye el subgrafo predicho NetworkX para el umbral dado.
     3. Ejecuta `greedy_coloring` con la estrategia seleccionada.
     4. Valida formalmente la coloración mediante `validate_coloring`.
     5. Estructura el cronograma asociando los metadatos de los exámenes del catálogo.
   - `get_available_strategies`: Retorna las 7 heurísticas soportadas por NetworkX:
     - `largest_first` (Welsh-Powell, orden por grado descendente).
     - `saturation_largest_first` (DSATUR, grado de saturación de Brélaz).
     - `smallest_last` (Matula-Beck, eliminación iterativa de menor grado).
     - `independent_set` (Conjuntos independientes maximales sucesivos).
     - `connected_sequential_bfs` (Recorrido BFS).
     - `connected_sequential_dfs` (Recorrido DFS).
     - `random_sequential` (Orden aleatorio uniforme).
3. **Endpoints REST ([`src/api/v1/endpoints/schedule.py`](file:///w:/home/anton/Tp-Grafos-ML/src/api/v1/endpoints/schedule.py)):**
   - `POST /api/v1/schedule`: Genera la partición en turnos para los 800 exámenes. Con umbral `0.20` y `largest_first`, produce **11 turnos** (cumpliendo holgadamente el límite de 32 turnos).
   - `GET /api/v1/schedule/strategies`: Retorna la lista de las 7 heurísticas con sus nombres formales y descripciones algorítmicas.
4. **Suite de Pruebas Automatizadas ([`tests/test_api_schedule.py`](file:///w:/home/anton/Tp-Grafos-ML/tests/test_api_schedule.py)):**
   - 5 tests cubriendo: generación oficial por defecto (11 turnos, 800 materias, validación formal `True`), estrategia personalizada DSATUR, rechazo de estrategia inválida (400), validación de umbrales fuera de rango (422) y catálogo de estrategias.

---

## 2. Ejemplos de Uso con `curl`

### 2.1. Generar cronograma oficial por defecto
```bash
curl -s -X POST http://localhost:8000/api/v1/schedule \
  -H "Content-Type: application/json" \
  -d '{}'
```
**Respuesta:**
```json
{
  "threshold": 0.2,
  "strategy": "largest_first",
  "turns_count": 11,
  "max_allowed_turns": 32,
  "fits_in_calendar": true,
  "is_valid_coloring": true,
  "total_exams_scheduled": 800,
  "turn_distribution": {
    "0": 135,
    "1": 112,
    "2": 95,
    ...
  },
  "schedule": {
    "0": [
      {
        "exam_id": "G111GBE1",
        "exam_name": "GENERAL MATHEMATICS 1B",
        "department_code": "MA",
        "duration": "2:00",
        "duration_minutes": 120
      },
      ...
    ]
  }
}
```

### 2.2. Generar cronograma con heurística DSATUR
```bash
curl -s -X POST http://localhost:8000/api/v1/schedule \
  -H "Content-Type: application/json" \
  -d '{"strategy": "saturation_largest_first", "threshold": 0.20}'
```

### 2.3. Consultar heurísticas de coloreo disponibles
```bash
curl -s http://localhost:8000/api/v1/schedule/strategies
```
**Respuesta:**
```json
[
  {
    "strategy": "largest_first",
    "name": "Welsh-Powell (Largest First)",
    "description": "Ordena los vértices por grado descendente. Colorea primero las materias con mayor número de conflictos."
  },
  {
    "strategy": "saturation_largest_first",
    "name": "DSATUR (Saturation Largest First)",
    "description": "Heurística de grado de saturación de Brélaz. Selecciona dinámicamente el vértice con más colores vecinos distintos."
  },
  ...
]
```

---

## 3. Pruebas Automatizadas

```bash
python -m unittest discover -s tests -v
```
**35 tests ejecutados en 3.26 segundos — 100% exitosos.**
