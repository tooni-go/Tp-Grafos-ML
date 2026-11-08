# Documentación de Implementación — Etapa 4: Evaluación de Cronogramas vs. Ground Truth 1994

## 1. Resumen de lo Implementado

En esta cuarta y última etapa del backend se implementó el módulo de auditoría y evaluación empírica de cronogramas contra los conflictos reales observados en el cuerpo histórico de la *University of Nottingham (1994)*:

1. **Esquema Pydantic v2 ([`src/schemas/schedule.py`](../src/schemas/schedule.py)):**
   - `ScheduleEvaluationResponse`: Modelo tipado y validado que expone métricas operativas, estadísticas y de impacto estudiantil:
     - **Configuración y Turnos:** `threshold` (flotante), `strategy` (texto), `turns_count` (int), `max_allowed_turns` (32), `fits_in_calendar` (bool), `is_valid_coloring` (bool).
     - **Incidencias de Conflicto Estudiante-Par (Pares de Exámenes en Conflicto):**
       - `real_conflict_pairs`: Total de pares no dirigidos de materias con alumnos compartidos en 1994 (`4.468` pares).
       - `clashing_pairs`: Pares reales que quedaron asignados al mismo turno (`830` pares en configuración oficial).
       - `total_affected_students`: Suma total de incidencias potenciales de conflicto estudiante-par en toda la cohorte (`28.677` eventos de inscripción simultánea $\sum_{s} \binom{k_s}{2}$).
       - `clashing_students`: Total acumulado de incidencias de conflicto estudiante-par que permanecen en el mismo turno (`3.723` violaciones horarias). Un estudiante puede sumar múltiples incidencias si tiene $\ge 3$ materias en el mismo turno.
       - `clash_student_percentage`: Proporción de incidencias de conflicto estudiante-par que permanecen en el cronograma respecto al peor caso teórico (`12.98%`). Esto equivale a decir que el cronograma logró evitar el **87.02%** de las incidencias potenciales de colisión.
     - **Métricas de Personas Físicas Únicas (Headcount):**
       - `unique_students_affected`: Cantidad de personas únicas de la cohorte que sufren al menos un choque de horarios en el cronograma (`1.945` alumnos de los 3.086 totales).
       - `unique_students_percentage`: Porcentaje de personas físicas afectadas sobre el total de estudiantes evaluados (`63.03%`).
     - **Matriz de Confusión y Calidad de Aristas del Grafo:**
       - `predicted_edges`: Total de aristas del grafo predicho (`2.032`).
       - `true_positives` (`TP`): Conflictos reales correctamente predichos y evitados (`759`).
       - `false_positives` (`FP`): Conflictos predichos donde en la realidad no hubo inscriptos compartidos (`1.273`).
       - `false_negatives` (`FN`): Conflictos reales no capturados por el modelo en este umbral (`3.709`).
       - `edge_precision`: Precisión sobre las aristas predichas (`0.3735`).
       - `edge_recall`: Cobertura de las aristas reales (`0.1699`).
     - **Métricas Crudas:** `raw_metrics` diccionario con la salida directa de `evaluate_against_real`.

2. **Servicio de Evaluación ([`src/services/schedule_service.py`](../src/services/schedule_service.py)):**
   - `evaluate_schedule`:
     1. Valida la heurística contra `VALID_STRATEGIES`.
     2. Construye el grafo de incompatibilidades NetworkX para el umbral solicitado.
     3. Ejecuta el algoritmo voraz de coloreo (`greedy_coloring`).
     4. Valida internamente la consistencia formal (`validate_coloring`).
     5. Cruza el cronograma contra el ground truth de 1994 mediante `evaluate_against_real`, calculando dinámicamente las incidencias y el impacto en estudiantes únicos.

3. **Ciclo de Vida y Eficiencia ([`src/api/main.py`](../src/api/main.py)):**
   - En el evento `lifespan`, el servidor precarga en memoria `app.state.real_pairs_df` y `app.state.student_exams_1994`, permitiendo que el endpoint `/schedule/evaluate` responda en ~10-15 milisegundos sin lecturas a disco durante las peticiones.

4. **Endpoint REST ([`src/api/v1/endpoints/schedule.py`](../src/api/v1/endpoints/schedule.py)):**
   - `POST /api/v1/schedule/evaluate`: Acepta `threshold` y `strategy` (ambos opcionales, con valores por defecto `0.20` y `"largest_first"`).

---

## 2. Ejemplos de Uso con `curl`

### 2.1. Evaluar cronograma con configuración oficial (0.20, Largest First)

```bash
curl -s -X POST http://localhost:8000/api/v1/schedule/evaluate \
  -H "Content-Type: application/json" \
  -d '{}'
```

**Respuesta HTTP 200:**
```json
{
  "threshold": 0.2,
  "strategy": "largest_first",
  "turns_count": 11,
  "max_allowed_turns": 32,
  "fits_in_calendar": true,
  "is_valid_coloring": true,
  "real_conflict_pairs": 4468,
  "clashing_pairs": 830,
  "total_affected_students": 28677,
  "clashing_students": 3723,
  "clash_student_percentage": 12.98,
  "unique_students_affected": 1945,
  "unique_students_percentage": 63.03,
  "predicted_edges": 2032,
  "true_positives": 759,
  "false_positives": 1273,
  "false_negatives": 3709,
  "edge_precision": 0.3735,
  "edge_recall": 0.1699,
  "raw_metrics": {
    "turnos": 11,
    "cabe_en_calendario": true,
    "pares_reales": 4468,
    "pares_reales_en_choque": 830,
    "alumnos_choque_total": 28677,
    "alumnos_en_choque": 3723,
    "pct_alumnos_en_choque": 12.98,
    "alumnos_unicos_total": 3086,
    "alumnos_unicos_en_choque": 1945,
    "pct_alumnos_unicos_en_choque": 63.03,
    "aristas_predichas": 2032,
    "TP": 759,
    "FP": 1273,
    "FN": 3709,
    "precision_aristas": 0.3735,
    "recall_aristas": 0.1699
  }
}
```

---

## 3. Interpretación Metodológica y Trade-off

| Dimensión | Métrica | Configuración Oficial ($\theta = 0.20$) | Grafo Oráculo (Ground Truth 1994) | Significado Metodológico |
|:---|:---|:---:|:---:|:---|
| **Turnos** | Turnos Asignados | **11 turnos** | 19 turnos | La compacidad del calendario predicho optimiza fuertemente el tiempo total de exámenes ($\le 32$). |
| **Incidencias Estudiante-Par** | Incidencias en Choque | **3.723** de 28.677 | 0 de 28.677 | Penalización de solapamiento horario (student-clash instances). El cronograma elimina el **87.02%** de las incidencias potenciales. |
| **Incidencias Estudiante-Par** | Tasa Residual de Incidencias | **12.98%** | 0.00% | Proporción de solapamientos ocurridos respecto al peor escenario teórico (todos en un solo turno). |
| **Personas Únicas (Headcount)** | Alumnos Únicos con $\ge 1$ Choque | **1.945** de 3.086 | 0 de 3.086 | Estudiantes reales que sufren al menos una superposición en su agenda de examen personal. |
| **Personas Únicas (Headcount)** | % Alumnos Únicos Afectados | **63.03%** | 0.00% | Proporción de la población estudiantil física afectada. |
| **Calidad del Grafo** | TP / FP / FN | 759 / 1.273 / 3.709 | 4.468 / 0 / 0 | Los 830 choques provienen exclusivamente de los 3.709 FN ($830 \subset FN$); los 759 TP jamás chocan. |
