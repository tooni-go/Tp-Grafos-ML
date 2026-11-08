"""Grafo de conflictos y coloreo para asignar turnos de examen (Fase 4: integración con ML).

Cambio respecto de la versión previa: las aristas ya NO salen de los conflictos reales de
1994 (`cant_estudiantes_afectados`), sino de la probabilidad predicha por el modelo
(`prob_conflicto`). Los conflictos reales se usan solo para EVALUAR el cronograma
(`evaluate_against_real`) y como referencia (grafo oráculo).

El grafo sigue siendo no dirigido y sin pesos: la arista {u, v} existe si
`prob_conflicto >= threshold` (misma regla `>=` que `src/inference`).
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import networkx as nx
import pandas as pd

TURNOS_CALENDARIO = 32  # turnos disponibles en el calendario de Nottingham (2 semanas)

VALID_STRATEGIES = {
    "largest_first",  # Welsh-Powell
    "smallest_last",
    "saturation_largest_first",  # DSATUR
    "independent_set",
    "random_sequential",
    "connected_sequential_bfs",
    "connected_sequential_dfs",
}


def load_exams_catalog(exams_csv_path: Path) -> Dict[str, Dict[str, Any]]:
    catalog: Dict[str, Dict[str, Any]] = {}
    with Path(exams_csv_path).open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            catalog[row["exam_id"]] = {
                "exam_id": row["exam_id"],
                "exam_name": row["exam_name"],
                "department_code": row.get("department_code", ""),
                "duration": row.get("duration", ""),
                "duration_minutes": int(row["duration_minutes"]) if row.get("duration_minutes") else 0,
            }
    return catalog


def _init_graph(exams_catalog: Optional[Dict[str, Dict[str, Any]]]) -> nx.Graph:
    G = nx.Graph()
    if exams_catalog:
        for exam_id, attrs in exams_catalog.items():
            G.add_node(exam_id, **attrs)
    return G


def build_conflict_graph_from_predictions(
    pred_df: pd.DataFrame,
    threshold: float,
    exams_catalog: Optional[Dict[str, Dict[str, Any]]] = None,
    prob_col: str = "prob_conflicto",
) -> nx.Graph:
    """Grafo predicho: arista {u, v} si `prob_conflicto >= threshold`. Sin pesos."""
    if not (0.0 < float(threshold) < 1.0):
        raise ValueError(f"threshold debe estar en (0, 1); se recibió {threshold!r}")
    for col in ("materia_id_1", "materia_id_2", prob_col):
        if col not in pred_df.columns:
            raise ValueError(f"Falta la columna requerida '{col}'.")
    G = _init_graph(exams_catalog)
    edges = pred_df.loc[pred_df[prob_col] >= threshold, ["materia_id_1", "materia_id_2"]]
    G.add_edges_from(edges.itertuples(index=False, name=None))
    return G


def build_real_graph(
    pairs_df: pd.DataFrame,
    exams_catalog: Optional[Dict[str, Dict[str, Any]]] = None,
    min_students: int = 1,
) -> nx.Graph:
    """Grafo ORÁCULO con conflictos reales (cohorte 1994). Solo para referencia/evaluación."""
    G = _init_graph(exams_catalog)
    e = pairs_df.loc[pairs_df["cant_estudiantes_afectados"] >= min_students,
                     ["materia_id_1", "materia_id_2"]]
    G.add_edges_from(e.itertuples(index=False, name=None))
    return G


def greedy_coloring(G: nx.Graph, strategy: str = "largest_first") -> Dict[str, int]:
    if len(G) == 0:
        return {}
    if strategy not in VALID_STRATEGIES:
        strategy = "largest_first"
    return nx.coloring.greedy_color(G, strategy=strategy)


def count_turns(coloring: Dict[str, int]) -> int:
    return (max(coloring.values()) + 1) if coloring else 0


def build_schedule(
    coloring: Dict[str, int],
    exams_catalog: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[int, List[Dict[str, Any]]]:
    schedule: Dict[int, List[Dict[str, Any]]] = {}
    for exam_id, turno in coloring.items():
        info = {"exam_id": exam_id}
        if exams_catalog and exam_id in exams_catalog:
            info.update(exams_catalog[exam_id])
        else:
            info["exam_name"] = exam_id
        schedule.setdefault(turno, []).append(info)
    return dict(sorted(schedule.items()))


def validate_coloring(G: nx.Graph, coloring: Dict[str, int]) -> Tuple[bool, List[Tuple[str, str, int]]]:
    """Chequeo INTERNO: el coloreo respeta las aristas del grafo con el que se calculó.

    Con un grafo predicho esto es una verificación del algoritmo, no del cronograma:
    para medir el cronograma contra la realidad usar `evaluate_against_real`.
    """
    bad = [(u, v, coloring[u]) for u, v in G.edges()
           if u in coloring and v in coloring and coloring[u] == coloring[v]]
    return len(bad) == 0, bad


def evaluate_against_real(
    coloring: Dict[str, int],
    real_pairs_df: pd.DataFrame,
    predicted_graph: Optional[nx.Graph] = None,
) -> Dict[str, Any]:
    """Mide un cronograma contra los conflictos reales de la cohorte 1994.

    Un par real está en choque si sus dos materias quedaron en el mismo turno.
    Devuelve pares reales en choque, alumnos afectados y % cubierto.
    Si se pasa `predicted_graph`, agrega precisión/recall de aristas (TP/FP/FN).
    """
    real = real_pairs_df.loc[real_pairs_df["cant_estudiantes_afectados"] > 0,
                             ["materia_id_1", "materia_id_2", "cant_estudiantes_afectados"]]
    same = [coloring.get(a) is not None and coloring.get(a) == coloring.get(b)
            for a, b in zip(real["materia_id_1"], real["materia_id_2"])]
    clash = real[same]
    total_students = int(real["cant_estudiantes_afectados"].sum())
    clash_students = int(clash["cant_estudiantes_afectados"].sum())
    out: Dict[str, Any] = {
        "turnos": count_turns(coloring),
        "cabe_en_calendario": count_turns(coloring) <= TURNOS_CALENDARIO,
        "pares_reales": int(len(real)),
        "pares_reales_en_choque": int(len(clash)),
        "alumnos_choque_total": total_students,
        "alumnos_en_choque": clash_students,
        "pct_alumnos_en_choque": round(100.0 * clash_students / total_students, 2) if total_students else 0.0,
    }
    if predicted_graph is not None:
        real_edges = {frozenset(e) for e in zip(real["materia_id_1"], real["materia_id_2"])}
        pred_edges = {frozenset(e) for e in predicted_graph.edges()}
        tp = len(real_edges & pred_edges)
        fp = len(pred_edges - real_edges)
        fn = len(real_edges - pred_edges)
        out.update({
            "aristas_predichas": len(pred_edges), "TP": tp, "FP": fp, "FN": fn,
            "precision_aristas": round(tp / (tp + fp), 4) if tp + fp else 0.0,
            "recall_aristas": round(tp / (tp + fn), 4) if tp + fn else 0.0,
        })
    return out


def sweep_thresholds(
    pred_df: pd.DataFrame,
    real_pairs_df: pd.DataFrame,
    thresholds: Iterable[float],
    exams_catalog: Optional[Dict[str, Dict[str, Any]]] = None,
    strategy: str = "largest_first",
) -> pd.DataFrame:
    """Para cada umbral: arma grafo, colorea y evalúa contra lo real."""
    rows = []
    for th in thresholds:
        G = build_conflict_graph_from_predictions(pred_df, th, exams_catalog)
        coloring = greedy_coloring(G, strategy)
        ok, _ = validate_coloring(G, coloring)
        row = {"threshold": th, "estrategia": strategy, "coloreo_valido_en_grafo_predicho": ok,
               "nodos": G.number_of_nodes()}
        row.update(evaluate_against_real(coloring, real_pairs_df, G))
        rows.append(row)
    return pd.DataFrame(rows)
