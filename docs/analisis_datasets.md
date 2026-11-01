# Análisis Comparativo de Datasets para Planificación de Mesas de Examen

**Proyecto:** Planificación de Mesas de Examen Universitarias (Teoría de Grafos y Machine Learning)  
**Fecha de Análisis:** 01/11/2026  
**Documento de Referencia:** [Enunciado Antonio-Mamani.md](file:///w:/home/anton/Tp-Grafos-ML/Enunciado%20Antonio-Mamani.md)

---

## 1. Contexto y Objetivos del Análisis

El trabajo práctico integrador exige modelar la asignación de turnos de examen mediante un grafo de conflictos no dirigido, donde:
* **Nodos:** Materias que se rinden en el período académico.
* **Aristas:** Conflicto entre pares de materias (estudiantes que necesitan rendir ambas).
* **Pesos:** Grado o complejidad del conflicto (cantidad de estudiantes afectados o probabilidad de conflicto predicha por un modelo de Machine Learning).
* **Algoritmo de Grafos:** Coloreo de grafos (estrategia greedy) para asignar turnos (colores) minimizando la cantidad total de turnos sin solapamiento de exámenes en conflicto.

El enunciado requiere explícitamente contar con (o derivar) un dataset con las siguientes columnas mínimas:
1. `materia_id_1`, `materia_id_2`: Identificadores de las materias.
2. `cant_estudiantes_compartidos_historico`: Cantidad histórica de estudiantes compartidos.
3. `mismo_anio`: Booleano (si ambas pertenecen al mismo año de cursada).
4. `misma_carrera`: Booleano (si pertenecen al mismo plan o carrera).
5. `cant_estudiantes_afectados`: Variable objetivo numérica (regresión).
6. `hubo_conflicto`: Variable objetivo booleana alternativa (clasificación).

Se analizaron en profundidad los datasets reales descargados en el repositorio:
- **University of Nottingham Exam Timetabling Dataset** (`Nott/`)
- **Toronto Benchmark Exam Timetabling Dataset — Carter et al., 1996** (`Toronto/`)
- **UniTime / Purdue University Exam Timetabling Dataset — MISTA 2013** (`pu-exam-mista13/`)

---

## 2. Análisis de University of Nottingham Dataset (`Nott/`)

### 2.1. Archivos Disponibles y Formatos
El directorio `Nott/` contiene **5 archivos de texto plano ASCII de formato fijo y delimitado**, correspondientes al **Semestre 1 del ciclo académico 1994–95**, con un peso total inferior a **1 MB**:

| Archivo | Formato / Longitudes de Campo | Tamaño | Registros | Descripción |
| :--- | :--- | :---: | :---: | :--- |
| [README](file:///w:/home/anton/Tp-Grafos-ML/Nott/README) | Texto plano | 950 B | 26 líneas | Documentación oficial de la estructura del dataset y longitudes de campo. |
| [exams](file:///w:/home/anton/Tp-Grafos-ML/Nott/exams) | Fijo: Código (8), Título (40), Duración (4), Depto (2) | 46.4 KB | 800 líneas | Padrón completo de los 800 exámenes provistos por 46 departamentos. |
| [students](file:///w:/home/anton/Tp-Grafos-ML/Nott/students) | Fijo: Código de estudiante (10), Código de carrera (4) | 126.3 KB | 7,896 líneas | Padrón de 7,896 alumnos con su código de carrera/plan (`course code`). |
| [enrolements](file:///w:/home/anton/Tp-Grafos-ML/Nott/enrolements) | Fijo: Código de estudiante (10), Código de examen (8) | 679.9 KB | 33,997 líneas | Relación de inscripciones individuales alumno–examen. |
| [data](file:///w:/home/anton/Tp-Grafos-ML/Nott/data) | Texto plano semi-estructurado | 3.77 KB | 182 líneas | Definición de fechas, turnos (32 turnos posibles), aulas y restricciones. |

### 2.2. Estructura Interna y Relaciones (Estudiantes, Materias y Períodos)
El dataset conserva la nomenclatura modular británica real de la Universidad de Nottingham, permitiendo extraer atributos académicos determinantes:
1. **Archivo [exams](file:///w:/home/anton/Tp-Grafos-ML/Nott/exams):**
   - `exam code` (8 caracteres, ej. `B31101E1`, `B32321E1`, `C13562E1`):
     - **Caracteres 1–2 (`B3`, `C1`, `F1`, `M3`, `V7`, etc.):** Código de la escuela/carrera base a la que pertenece la materia.
     - **Carácter 3 (`1`, `2`, `3`, `4` o `A`, `B`, `C`, `P`):** **Nivel / Año de la materia en el plan de estudios** (`1`/`A` = 1.er año, `2`/`B` = 2.do año, `3`/`C` = 3.er año, `4`/`P` = 4.to año o posgrado). Se comprobó empíricamente cruzando las inscripciones con el año de ingreso del estudiante codificado en su matrícula:
       - Estudiantes ingresantes en 1994 (prefijo `94`, ej. `D94`, `B94`, `A94`) rinden masivamente materias con 3.er carácter `1` o `A`.
       - Estudiantes ingresantes en 1993 (prefijo `93`, ej. `D93`, `B93`, `A93`) rinden masivamente materias con 3.er carácter `2` o `B`.
       - Estudiantes ingresantes en 1992 (prefijo `92`, ej. `D92`, `B92`, `A92`) rinden masivamente materias con 3.er carácter `3` o `C`.
   - `description` (40 caracteres): Nombre real de la asignatura (ej. `MOLECULAR PHARMACOLOGY`, `ANIMAL BEHAVIOUR`, `PARASITE IMMUNOLOGY`).
   - `duration` (4 caracteres): Duración del examen (`1:00`, `1:30`, `2:00`, `3:00`).
   - `department code` (2 caracteres): Código del departamento académico (46 departamentos, ej. `PI`, `LE`, `TH`, `HI`, `CM`).
2. **Archivo [students](file:///w:/home/anton/Tp-Grafos-ML/Nott/students):**
   - `student code` (10 caracteres, ej. `A890186790` a `D94...`): incluye el año de cohorte (`89` a `94`).
   - `course code` (4 caracteres, ej. `M300` con 389 alumnos, `F100` con 327, `B300` con 310; total de **223 carreras/planes distintos**).
3. **Archivo [enrolements](file:///w:/home/anton/Tp-Grafos-ML/Nott/enrolements):**
   - 33,997 inscripciones individuales exactas (promedio de 4.3 exámenes por estudiante y 42.5 estudiantes por examen).
4. **Archivo [data](file:///w:/home/anton/Tp-Grafos-ML/Nott/data):**
   - Período de exámenes: Lunes 23 de enero al Sábado 4 de febrero de 1995 (2 semanas).
   - Turnos por día (`TIMES`): Lunes a Viernes 3 turnos (`9:00`, `13:30`, `16:30`) y Sábados 1 turno (`9:00`), conformando **32 turnos (períodos)** en total.
   - Aulas (`ROOMS`, `ROOM ASSIGNMENTS`) y restricciones (`COINCIDENCES`, `EARLINESS PRIORITY`, `MISC`).

### 2.3. Evaluación Punto por Punto de Nottingham
* **¿Inscripciones individuales o matriz de conflictos?** Dispone de **inscripciones individuales completas** en `enrolements`.
* **¿Podemos calcular cuántos estudiantes comparten cada par de materias?** **Sí, de forma exacta.** De los $\binom{800}{2} = 319,600$ pares posibles de materias:
  - **10,113 pares** comparten $\ge 1$ estudiante.
  - **2,392 pares** comparten $\ge 5$ estudiantes.
  - **1,313 pares** comparten $\ge 10$ estudiantes (máximo: 167 estudiantes compartidos en un par).
* **¿Existen datos de distintos semestres o períodos históricos?** Contiene **1 único semestre real** (Semestre 1, 1994–95). Sin embargo, al incluir la carrera (`course code`) y cohorte de ingreso de cada alumno (`89` a `94`), permite particionar cohortes o simular el histórico basándose en la estructura curricular real (práctica avalada en el enunciado bajo *"Dataset real adaptado"*).
* **¿Podemos reconstruir conflictos efectivos a partir de los cronogramas?**
  - **Conflictos reales de demanda (`cant_estudiantes_afectados`):** **Sí**, de forma exacta a partir de `enrolements`.
  - **Choques sobre un cronograma ya resuelto:** No incluye un cronograma final asignado (solo las restricciones previas en `data`), ya que la asignación de turnos es la tarea que resuelve nuestro algoritmo de coloreo.

---

## 3. Análisis de Toronto Benchmark Dataset (`Toronto/`)

### 3.1. ¿Qué es cada archivo dentro de la carpeta `Toronto/`?
Aunque este conjunto de datos se conoce mundialmente en la literatura científica como el **"Toronto Benchmark Dataset"** (publicado por *Michael W. Carter, Gilbert Laporte y Sau Yan Lee, 1996, Journal of the Operational Research Society*), en realidad es una **colección de 13 instancias independientes pertenecientes a 12 instituciones educativas distintas** (universidades y colegios secundarios de Canadá, Reino Unido, EE. UU. y Arabia Saudita).

Cada instancia consta exactamente de **2 archivos de texto plano** con el mismo prefijo (`<institución>-<semestre>-<año>`):
1. **Archivo `.crs` (Courses / Exams):** Lista los exámenes de esa institución. Cada línea tiene únicamente **2 números enteros separados por espacio**:
   - `ID_examen` (secuencial de `0001` a `N`) y `cantidad_inscriptos` (ej. `0001 123`).
2. **Archivo `.stu` (Students):** Lista las inscripciones por estudiante. **Cada línea representa a un estudiante anónimo** y contiene los `ID_examen` a los que está inscripto separados por espacios (ej. `0082 0239 0249`).

A continuación se detalla **qué es cada uno de los 26 archivos (13 pares `.crs` / `.stu`)** presentes en tu carpeta `Toronto/`:

| Prefijo de Archivos (`.crs` y `.stu`) | Institución Educativa Real | Período | Exámenes (líneas en `.crs`) | Estudiantes (líneas en `.stu`) | Inscripciones Totales | ¿Es de la Univ. de Toronto? |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| [uta-s-92.crs](file:///w:/home/anton/Tp-Grafos-ML/Toronto/uta-s-92.crs) / [.stu](file:///w:/home/anton/Tp-Grafos-ML/Toronto/uta-s-92.stu) | **University of Toronto** (Faculty of Arts & Science) | Spring 1992 | 622 | 21,266 | 58,979 | **SÍ (Artes y Ciencias)** |
| [ute-s-92.crs](file:///w:/home/anton/Tp-Grafos-ML/Toronto/ute-s-92.crs) / [.stu](file:///w:/home/anton/Tp-Grafos-ML/Toronto/ute-s-92.stu) | **University of Toronto** (Faculty of Engineering) | Spring 1992 | 184 | 2,749 | 11,793 | **SÍ (Ingeniería)** |
| [car-s-91.crs](file:///w:/home/anton/Tp-Grafos-ML/Toronto/car-s-91.crs) / [.stu](file:///w:/home/anton/Tp-Grafos-ML/Toronto/car-s-91.stu) | **Carleton University** (Ottawa, Canadá) | Spring 1991 | 682 | 16,925 | 56,877 | No |
| [car-f-92.crs](file:///w:/home/anton/Tp-Grafos-ML/Toronto/car-f-92.crs) / [.stu](file:///w:/home/anton/Tp-Grafos-ML/Toronto/car-f-92.stu) | **Carleton University** (Ottawa, Canadá) | Fall 1992 | 543 | 18,419 | 55,522 | No |
| [rye-s-93.crs](file:///w:/home/anton/Tp-Grafos-ML/Toronto/rye-s-93.crs) / [.stu](file:///w:/home/anton/Tp-Grafos-ML/Toronto/rye-s-93.stu) | **Ryerson Polytechnical Institute** (Toronto, Canadá) | Spring 1993 | 486 | 11,483 | 45,051 | No (otra univ. en Toronto) |
| [tre-s-92.crs](file:///w:/home/anton/Tp-Grafos-ML/Toronto/tre-s-92.crs) / [.stu](file:///w:/home/anton/Tp-Grafos-ML/Toronto/tre-s-92.stu) | **Trent University** (Peterborough, Canadá) | Spring 1992 | 261 | 4,360 | 14,901 | No |
| [hec-s-92.crs](file:///w:/home/anton/Tp-Grafos-ML/Toronto/hec-s-92.crs) / [.stu](file:///w:/home/anton/Tp-Grafos-ML/Toronto/hec-s-92.stu) | **HEC Montréal** (Escuela de Negocios, Canadá) | Spring 1992 | 81 | 2,823 | 10,632 | No |
| [lse-f-91.crs](file:///w:/home/anton/Tp-Grafos-ML/Toronto/lse-f-91.crs) / [.stu](file:///w:/home/anton/Tp-Grafos-ML/Toronto/lse-f-91.stu) | **London School of Economics (LSE)** (Londres, UK) | Fall 1991 | 381 | 2,726 | 10,918 | No |
| [kfu-s-93.crs](file:///w:/home/anton/Tp-Grafos-ML/Toronto/kfu-s-93.crs) / [.stu](file:///w:/home/anton/Tp-Grafos-ML/Toronto/kfu-s-93.stu) | **King Fahd University (KFUPM)** (Arabia Saudita) | Spring 1993 | 461 | 5,349 | 25,113 | No |
| [pur-s-93.crs](file:///w:/home/anton/Tp-Grafos-ML/Toronto/pur-s-93.crs) / [.stu](file:///w:/home/anton/Tp-Grafos-ML/Toronto/pur-s-93.stu) | **Purdue University** (Indiana, EE. UU.) | Spring 1993 | 2,419 | 30,029 | 120,681 | No |
| [ear-f-83.crs](file:///w:/home/anton/Tp-Grafos-ML/Toronto/ear-f-83.crs) / [.stu](file:///w:/home/anton/Tp-Grafos-ML/Toronto/ear-f-83.stu) | **Earl Haig Collegiate Institute** (Secundario, Toronto) | Fall 1983 | 190 | 1,125 | 8,109 | No (escuela secundaria) |
| [yor-f-83.crs](file:///w:/home/anton/Tp-Grafos-ML/Toronto/yor-f-83.crs) / [.stu](file:///w:/home/anton/Tp-Grafos-ML/Toronto/yor-f-83.stu) | **York Mills Collegiate Institute** (Secundario, Toronto) | Fall 1983 | 181 | 941 | 6,034 | No (escuela secundaria) |
| [sta-f-83.crs](file:///w:/home/anton/Tp-Grafos-ML/Toronto/sta-f-83.crs) / [.stu](file:///w:/home/anton/Tp-Grafos-ML/Toronto/sta-f-83.stu) | **St. Andrew's Junior High School** (Secundario, Toronto) | Fall 1983 | 139 | 611 | 5,751 | No (escuela secundaria) |

### 3.2. Estructura Interna y Relaciones (Estudiantes, Materias y Períodos)
El formato de Carter/Toronto es **extremadamente minimalista y completamente anonimizado**:
* **Archivo `.crs`:** Solo contiene `ID_examen` (numerado siempre como `0001`, `0002`, ..., `N`) y la cantidad total de alumnos que rinden ese examen. **No tiene nombre de materia, ni carrera, ni departamento, ni año de cursada, ni duración del examen.**
* **Archivo `.stu`:** Cada renglón es un estudiante anónimo (ni siquiera tiene `ID_estudiante` ni código de carrera) seguido de los números `0001..N` de los exámenes que rinde.
* **Períodos / Turnos:** Los archivos `.crs` y `.stu` **no incluyen definición de períodos ni de aulas** (en la literatura de investigación operativa, la cantidad de turnos para cada instancia de Carter se fija por fuera como una constante en el paper, por ejemplo 35 turnos para `uta-s-92` o 10 turnos para `ute-s-92`).

### 3.3. Evaluación Punto por Punto de Toronto (`Toronto/`)
* **¿Inscripciones individuales o matriz de conflictos?**
  - **Sí tiene inscripciones individuales** en los archivos `.stu` (cada línea representa los exámenes que rinde un alumno individual), a partir de las cuales se construye la matriz de conflictos.
* **¿Podemos calcular cuántos estudiantes comparten cada par de materias?**
  - **Sí, de forma exacta dentro de cada instancia** contando en cuántas líneas de `.stu` aparecen juntos dos códigos de examen (ej. `0111` y `0489`).
* **¿Existen datos de distintos semestres o períodos históricos?**
  - **No son combinables entre sí:** Los 13 pares de archivos corresponden a **12 instituciones distintas** (mezclando universidades de distintos países y hasta 3 escuelas secundarias: `ear`, `yor`, `sta`).
  - La única institución con dos semestres en la carpeta es **Carleton University** (`car-s-91` con 682 exámenes y `car-f-92` con 543 exámenes), pero al igual que en el dataset de Purdue XML, **los exámenes fueron re-enumerados secuencialmente de `0001` a `N` en cada archivo**, por lo que `0001` en `car-s-91` (152 alumnos) no es la misma materia que `0001` en `car-f-92` (280 alumnos). No existe forma de vincular qué materia es cuál entre ambos semestres.
* **¿Podemos reconstruir conflictos efectivos a partir de los cronogramas?**
  - **Conflictos de demanda (estudiantes que rinden ambas materias):** **Sí**, se calculan exactamente desde el archivo `.stu`.
  - **Conflictos sobre un cronograma asignado:** **No**, los archivos `.crs` y `.stu` no contienen fechas, horarios ni asignaciones de turnos.
* **¿Qué columnas de nuestro dataset final podemos obtener y cuáles NO están disponibles en Toronto?**
  - `materia_id_1`, `materia_id_2`: **Disponible parcialmente** (solo como código numérico anónimo `0001`, `0002`, sin nombre de materia ni departamento).
  - `cant_estudiantes_compartidos_historico`: **No disponible** (no hay correspondencia histórica entre archivos).
  - `mismo_anio`: **No disponible** (el archivo `.crs` solo tiene el número `0001` y la cantidad de inscriptos; no hay ningún dato de año de cursada).
  - `misma_carrera`: **No disponible** (ni `.crs` ni `.stu` contienen carrera, facultad o departamento).
  - `cant_estudiantes_afectados` / `hubo_conflicto`: **Calculable** contando co-ocurrencias en `.stu`.
* **¿Permite entrenar un modelo de regresión o clasificación con las features pedidas?**
  - **No de manera genuina sin inventar los atributos:** Como en cada archivo `.crs` el único dato de una materia es su cantidad de alumnos inscriptos (`size`), para un par de materias $(i, j)$ solo sabríamos cuántos alumnos tiene la materia $i$ y cuántos tiene la materia $j$. **No existen features explicativas reales (`mismo_anio`, `misma_carrera`, departamento, carrera del alumno)** para que el modelo de Machine Learning aprenda qué pares de materias entran en conflicto.

---

## 4. Resumen del Tercer Dataset Evaluado Previamente: UniTime / Purdue (`pu-exam-mista13/`)

Para mantener la documentación completa en un único archivo, se conserva el resumen de `pu-exam-mista13/`:
* **Archivos:** 9 archivos XML (~153 MB en total) de Purdue University entre Fall 2008 (`pu-exam-fal08.xml`) y Fall 2012 (`pu-exam-fal12.xml`).
* **Contenido:** ~1,800 a ~2,223 exámenes y ~30,000 a ~35,000 estudiantes por semestre, con 29 períodos y ~500 aulas.
* **Limitación crítica (idéntica a Toronto):** Los `exam id` están anonimizados y numerados de `1` a `N` independientemente en cada archivo XML. No incluyen nombre de materia, ni año (`mismo_anio`), ni carrera (`misma_carrera`), impidiendo tanto el cruce histórico entre semestres como la construcción de las variables académicas pedidas en el enunciado.

---

## 5. Comparación General entre los 3 Datasets Descargados

| Criterio / Requisito del Proyecto | **University of Nottingham (`Nott/`)** | **Toronto Benchmark (`Toronto/`)** | **UniTime / Purdue (`pu-exam-mista13/`)** |
| :--- | :--- | :--- | :--- |
| **Cantidad y tipo de archivos** | 4 archivos de datos ASCII + `README` (< 1 MB total). Una única universidad coherente. | 26 archivos (13 pares `.crs`/`.stu`, ~2.3 MB) de **12 instituciones distintas** (univ. y secundarios). | 9 archivos XML (~153 MB) de 9 semestres de Purdue University. |
| **Inscripciones individuales** | **Sí** (`enrolements`: 33,997 inscripciones de 7,896 alumnos). | **Sí** (archivos `.stu`: una línea por alumno con los IDs de examen). | **Sí** (nodos `<student>` con hijos `<exam>`). |
| **Nombres reales de materias (para UI y Grafo)** | **Sí, completos** (`exams`: ej. `MOLECULAR PHARMACOLOGY`, `OPERA STUDIES`). | **No** (solo números `0001`, `0002`, ...). | **No** (solo números `1`, `2`, ...). |
| **Carrera del estudiante (`misma_carrera`)** | **Sí, disponible y calculable** (`students` incluye `course code` de 223 carreras + 46 deptos en `exams`). | **No disponible** (sin datos de carrera ni departamento). | **No disponible** (sin datos de carrera ni departamento). |
| **Año de las materias (`mismo_anio`)** | **Sí, calculable directamente** (3.er carácter del `exam code` indica nivel 1, 2, 3, 4; validado con cohorte del alumno). | **No disponible** (sin datos de año ni nivel). | **No disponible** (sin datos de año ni nivel). |
| **Información de turnos / períodos de calendario** | **Sí** (`data`: 32 turnos reales con fecha y hora, de Lun 23 Ene a Sáb 4 Feb 1995). | **No incluida en los archivos** (solo hay `.crs` y `.stu`). | **Sí** (29 nodos `<period>` con fecha y hora en cada XML). |
| **Información histórica multi-semestre** | 1 semestre real, pero con **6 cohortes de ingreso (`A89`–`D94`) y carreras** que permiten adaptar/simular el histórico. | 12 instituciones separadas; la única con 2 semestres (`car-s-91` y `car-f-92`) tiene IDs re-enumerados sin vínculo. | 9 semestres, pero con IDs `1..N` re-enumerados en cada XML sin vínculo entre sí. |
| **Viabilidad para ML (Regresión y Clasificación)** | **Alta:** posee features reales y explicativas (`mismo_anio`, `misma_carrera`, depto, tamaño, duración). | **Muy baja:** solo se conoce la cantidad de inscriptos de cada examen; faltan todas las variables curriculares. | **Muy baja:** faltan todas las variables curriculares (`mismo_anio`, `misma_carrera`). |
| **Facilidad de procesamiento** | **Muy alta** (archivos de texto fijo, < 1 MB, lectura instantánea). | **Muy alta** (archivos de texto simples, pero faltan metadatos). | **Media/Baja** (9 archivos XML pesados de ~846,000 líneas cada uno). |

---

## 6. Conclusión y Recomendación Final

1. **Sobre la carpeta `Toronto/`:**
   - Contiene las **13 instancias del benchmark de Carter (1996)**. Si alguien quisiera usar propiamente la Universidad de Toronto de esa carpeta, los únicos archivos correspondientes son **`uta-s-92.crs` / `uta-s-92.stu`** (Facultad de Artes y Ciencias, 622 exámenes) y **`ute-s-92.crs` / `ute-s-92.stu`** (Facultad de Ingeniería, 184 exámenes); los otros 22 archivos son de otras universidades y colegios secundarios.
   - Sin embargo, **ninguno de los archivos de `Toronto/` contiene nombres de materias, ni carreras, ni departamentos, ni año de cursada, ni grilla de turnos**, ya que fueron publicados únicamente con números secuenciales (`0001`, `0002`, ...) para probar algoritmos puros de coloreo de grafos sin Machine Learning.

2. **Por qué `University of Nottingham (Nott/)` sigue siendo el dataset claramente superior e ideal para este Trabajo Práctico:**
   - **Es el único de los tres datasets (`Nott/`, `Toronto/`, `pu-exam-mista13/`) que no está anonimizado académicamente:** conserva los códigos curriculares reales (`B31101E1`), los **nombres reales de las 800 materias**, los **46 departamentos**, la **duración** de cada examen, los **32 turnos del calendario** y la **carrera (`course code`) y cohorte de ingreso de cada uno de los 7,896 estudiantes**.
   - Gracias a ello, en **Nottingham** podemos construir de forma **100% real y defendible en el coloquio oral** las variables `mismo_anio`, `misma_carrera`, `cant_estudiantes_afectados` y `hubo_conflicto`, y adaptar limpiamente `cant_estudiantes_compartidos_historico` a partir de las cohortes/carreras reales del padrón.
