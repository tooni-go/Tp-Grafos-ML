from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import networkx as nx


def load_exams_catalog(exams_csv_path: Path) -> Dict[str, Dict[str, Any]]:
    catalog = {}
    with exams_csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            catalog[row["exam_id"]] = {
                "exam_id": row["exam_id"],
                "exam_name": row["exam_name"],
                "department_code": row.get("department_code", ""),
                "duration": row.get("duration", ""),
                "duration_minutes": int(row["duration_minutes"]) if row.get("duration_minutes") else 0,
            }
    return catalog


def build_conflict_graph(
    pairs: List[Dict[str, Any]],
    threshold: float = 0.0,
    columna_conflicto: str = "conflicto",
    exams_catalog: Optional[Dict[str, Dict[str, Any]]] = None,
) -> nx.Graph:
    G = nx.Graph()

    if exams_catalog:
        for exam_id, attrs in exams_catalog.items():
            G.add_node(exam_id, **attrs)

    for p in pairs:
        u = p["materia_id_1"]
        v = p["materia_id_2"]
        valor_conflicto = float(p.get(columna_conflicto, 0.0))

        if valor_conflicto > threshold:
            if not G.has_node(u):
                G.add_node(u, exam_name=p.get("materia_nombre_1", u))
            if not G.has_node(v):
                G.add_node(v, exam_name=p.get("materia_nombre_2", v))
            G.add_edge(u, v)

    return G


def greedy_coloring(
    G: nx.Graph,
    strategy: str = "largest_first"
) -> Dict[str, int]:
    if len(G) == 0:
        return {}

    valid_strategies = {
        "largest_first",
        "random_sequential",
        "smallest_last",
        "independent_set",
        "connected_sequential_bfs",
        "connected_sequential_dfs",
        "saturation_largest_first",
    }
    if strategy not in valid_strategies:
        strategy = "largest_first"

    coloring = nx.coloring.greedy_color(G, strategy=strategy)
    return coloring


def build_schedule(
    coloring: Dict[str, int],
    exams_catalog: Optional[Dict[str, Dict[str, Any]]] = None
) -> Dict[int, List[Dict[str, Any]]]:
    schedule: Dict[int, List[Dict[str, Any]]] = {}

    for exam_id, turno in coloring.items():
        if turno not in schedule:
            schedule[turno] = []

        exam_info = {"exam_id": exam_id}
        if exams_catalog and exam_id in exams_catalog:
            exam_info.update(exams_catalog[exam_id])
        else:
            exam_info["exam_name"] = exam_id

        schedule[turno].append(exam_info)

    return dict(sorted(schedule.items()))


def validate_coloring(G: nx.Graph, coloring: Dict[str, int]) -> Tuple[bool, List[Tuple[str, str, int]]]:
    conflicts = []
    for u, v in G.edges():
        c_u = coloring.get(u)
        c_v = coloring.get(v)
        if c_u is not None and c_v is not None and c_u == c_v:
            conflicts.append((u, v, c_u))
    return len(conflicts) == 0, conflicts
