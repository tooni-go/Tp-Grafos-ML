# Fase 3 — Grafo de Conflictos y Asignación de Turnos por Coloreo

**Proyecto:** Planificación de Mesas de Examen Universitarias  
**Módulo Implementado:** [`src/graphs/conflict_graph.py`](file:///w:/home/anton/Tp-Grafos-ML/src/graphs/conflict_graph.py)  
**Catálogo Base:** `data/interim/exams_clean.csv` (800 exámenes)  
**Dataset de Relaciones:** `data/processed/subject_pairs_poc.csv` (319.600 pares de materias)  

---

## 1. Contexto y Objetivos del Módulo

El objetivo de esta fase es implementar el núcleo algorítmico de teoría de grafos solicitado en el enunciado de la cátedra para la planificación de exámenes:

1. **Modelar las materias y sus choques como un grafo de conflictos no dirigido $G = (V, E)$:**
   - **Vértices ($V$):** Cada nodo representa una materia o examen a rendir en el período académico (800 materias en el catálogo).
   - **Aristas ($E$):** Existe una arista no dirigida entre dos materias $u$ y $v$ si y sólo si existe un conflicto significativo entre ambas (estudiantes en común o conflicto predicho).
2. **Definición de aristas sin ponderación de costo:**
   - A diferencia de problemas de camino mínimo (Dijkstra) o árboles de expansión (Prim/Kruskal), en el problema clásico de coloreo de grafos las aristas son restricciones de incompatibilidad pura (binarias). No llevan costos asociados; indican si dos exámenes pueden o no compartir el mismo turno horario.
3. **Control dinámico mediante umbral de conflicto (*threshold*):**
   - El sistema permite ajustar un parámetro de sensibilidad $\theta \ge 0$. Una arista entre el par $(u, v)$ se activa únicamente si la métrica de conflicto supera dicho umbral ($\text{conflicto} > \theta$). Esto permite relajar conflictos menores (por ejemplo, pares con 1 o 2 alumnos compartidos) para reducir la cantidad total de turnos requeridos.
4. **Asignación de turnos mediante Coloreo Goloso (*Greedy Coloring*):**
   - Asignar a cada materia $v \in V$ un color (entero no negativo representativo del turno de examen) de manera tal que para toda arista $(u, v) \in E$, $\text{color}(u) \neq \text{color}(v)$.
5. **Generación del cronograma y validación formal:**
   - Agrupar los exámenes por turno asignado para alimentar los endpoints del backend y las vistas de calendario del frontend.
   - Auditar de forma exhaustiva que no existan aristas adyacentes con colores idénticos (0 choques de horario).

---

## 2. Decisiones de Diseño y Algoritmos

### 2.1. Representación del Grafo
Se utiliza la librería `networkx` mediante un grafo simple no dirigido `nx.Graph()`:
- Los 800 exámenes del catálogo se precargan como nodos del grafo con sus metadatos (`exam_id`, `exam_name`, `department_code`, `duration_minutes`).
- Las aristas se incorporan únicamente cuando el valor de conflicto supera el umbral configurado, manteniéndose sin atributos de peso para respetar el modelo clásico de coloración.

### 2.2. Algoritmo de Coloreo Goloso: Estrategia `largest_first` (Welsh-Powell)
En problemas de coloreo de grafos generales, hallar el número cromático exacto $\chi(G)$ es un problema NP-hard. Por este motivo, se emplea una heurística golosa constructiva:

- **Estrategia por defecto:** `largest_first` (también conocida como heurística de Welsh-Powell).
  1. Se calcula el grado de cada vértice $d(v)$ (cantidad de vecinos en conflicto).
  2. Se ordenan los vértices de mayor a menor según su grado: $d(v_1) \ge d(v_2) \ge \dots \ge d(v_n)$.
  3. Se itera en ese orden estricto asignando a cada vértice el menor entero disponible $\{0, 1, 2, \dots\}$ que no haya sido utilizado previamente por ninguno de sus vecinos adyacentes ya coloreados.
- **Justificación:** Colorear primero los nodos con más restricciones (materias masivas o con múltiples choques) evita acorralar al algoritmo hacia el final de la ejecución, minimizando la cantidad total de colores (turnos) necesarios sin retroceso (*backtracking*).

---

## 3. Funciones Implementadas en `src/graphs/conflict_graph.py`

| Función | Parámetros Principales | Salida | Descripción Técnica |
| :--- | :--- | :--- | :--- |
| `load_exams_catalog` | `exams_csv_path: Path` | `Dict[str, Dict]` | Lee `exams_clean.csv` e indexa los 800 exámenes con su nombre, departamento y duración. |
| `build_conflict_graph` | `pairs: List[Dict]`, `threshold: float`, `columna_conflicto: str`, `exams_catalog: Dict` | `nx.Graph` | Construye el grafo no dirigido agregando nodos del catálogo y aristas donde `valor_conflicto > threshold`. |
| `greedy_coloring` | `G: nx.Graph`, `strategy: str` | `Dict[str, int]` | Aplica el algoritmo goloso sobre el grafo y devuelve un mapa `{exam_id: turno}`. |
| `build_schedule` | `coloring: Dict[str, int]`, `exams_catalog: Dict` | `Dict[int, List[Dict]]` | Transforma el diccionario de coloreo en un cronograma ordenado por número de turno con información detallada de cada materia. |
| `validate_coloring` | `G: nx.Graph`, `coloring: Dict[str, int]` | `Tuple[bool, List]` | Verifica arista por arista que ningún par en conflicto comparta turno. Devuelve `True` si el cronograma es 100% libre de colisiones. |

---

## 4. Resultados Experimentales con el Dataset de Nottingham

Se evaluó el comportamiento del módulo sobre los datos reales de la cohorte 1994 (4.468 pares con conflicto positivo de estudiantes afectados), variando el umbral de conflicto $\theta$:

| Umbral ($\theta$) | Nodos ($|V|$) | Aristas Activas ($|E|$) | Turnos Requeridos | ¿Coloreo Válido? | Conflictos Residuales |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **$\theta \ge 0$** (Conflicto $\ge 1$ alumno) | 800 | 4.468 | **19 turnos** | ✅ True | 0 |
| **$\theta \ge 2$** (Conflicto $> 2$ alumnos) | 800 | 1.462 | **11 turnos** | ✅ True | 0 |
| **$\theta \ge 5$** (Conflicto $> 5$ alumnos) | 800 | 812 | **10 turnos** | ✅ True | 0 |
| **$\theta \ge 10$** (Conflicto $> 10$ alumnos) | 800 | 506 | **10 turnos** | ✅ True | 0 |

### Observaciones Relevantes para la Defensa:
1. **Factibilidad del cronograma:** Incluso considerando absolutamente todos los choques de horario posibles ($\theta = 0$, 4.468 restricciones), el algoritmo resuelve la planificación completa de las 800 asignaturas en **19 turnos**. Dado que el calendario de Nottingham disponía de **32 turnos disponibles** en 2 semanas, el resultado demuestra holgura y viabilidad operativa real.
2. **Impacto de la relajación:** Al aumentar el umbral a $\theta = 2$ (tolerando que hasta 2 estudiantes rindan en el mismo turno), la cantidad de turnos requeridos baja drásticamente de 19 a **11 turnos**, ilustrando de forma directa el balance entre restricciones y capacidad académica.

---

## 5. Integración con las Fases Siguientes

- **Entrada desde Machine Learning (Fase 4):** Cuando se entrenen los modelos de regresión o clasificación, la lista `pairs` suministrará las predicciones generadas ($\hat{y}$) en el campo `columna_conflicto`, sustituyendo los valores reales por los estimados.
- **Salida hacia Backend API (Fase 5):** `build_conflict_graph` y `build_schedule` se llamarán dinámicamente desde los endpoints de FastAPI (`/api/graph` y `/api/schedule`) en respuesta a las peticiones del usuario desde el frontend.
