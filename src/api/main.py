"""API REST con FastAPI para el Frontend de Planificación de Mesas de Examen.

Expone los endpoints acordados en el contrato del frontend:
- GET /api/schedule?threshold=X
- GET /api/stats?threshold=X
- GET /api/graph?threshold=X&subject=Y
- GET /api/sweep
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware

from src.common.schema import PROJECT_ROOT
from src.graphs.conflict_graph import (
    build_conflict_graph_from_predictions,
    build_schedule,
    evaluate_against_real,
    greedy_coloring,
    load_exams_catalog,
)
from src.graphs.predicted_pairs import get_pair_probabilities, load_all_pairs

app = FastAPI(
    title="Planificación de Mesas de Examen API",
    description="Backend API que integra Teoría de Grafos y Machine Learning",
    version="1.0.0",
)

# Habilitar CORS para permitir llamadas desde el frontend de desarrollo (localhost:5173)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Carga en memoria única al iniciar
EXAMS_CSV = PROJECT_ROOT / "data" / "interim" / "exams_clean.csv"
PAIRS_CSV = PROJECT_ROOT / "data" / "processed" / "subject_pairs_poc.csv"
SWEEP_CSV = PROJECT_ROOT / "results" / "graphs" / "threshold_sweep.csv"

CATALOG = load_exams_catalog(EXAMS_CSV)
REAL_PAIRS = load_all_pairs(PAIRS_CSV)
PRED_PAIRS = get_pair_probabilities(mode="oof")

# Caché en memoria por umbral
_CACHE_SCHEDULE: Dict[float, Any] = {}
_CACHE_STATS: Dict[float, Any] = {}
_CACHE_GRAPH: Dict[float, Any] = {}


def _compute_for_threshold(th: float):
    if th in _CACHE_SCHEDULE:
        return _CACHE_SCHEDULE[th], _CACHE_STATS[th], _CACHE_GRAPH[th]

    G = build_conflict_graph_from_predictions(PRED_PAIRS, th, CATALOG)
    coloring = greedy_coloring(G, "largest_first")
    ev = evaluate_against_real(coloring, REAL_PAIRS, G)
    sch = build_schedule(coloring, CATALOG)

    schedule_data = {
        str(k): [
            {
                "code": item["exam_id"],
                "name": item["exam_name"],
                "department": item["department_code"],
                "duration": item["duration_minutes"],
            }
            for item in items
        ]
        for k, items in sch.items()
    }

    stats_data = {
        "threshold": th,
        "turnos_usados": ev["turnos"],
        "max_turnos": 32,
        "alumnos_protegidos_pct": round(100.0 - ev["pct_alumnos_en_choque"], 2),
        "precision_arbol": round(ev["precision_aristas"] * 100.0, 2),
        "precision_knn": 22.41,
        "turnos_oraculo": 19,
    }

    nodes = [
        {"id": exam_id, "label": info.get("exam_name", exam_id)}
        for exam_id, info in CATALOG.items()
    ]
    edges = [
        {
            "from": u,
            "to": v,
            "prob": round(float(d.get("prob", th)), 4),
        }
        for u, v, d in G.edges(data=True)
    ]
    graph_data = {"nodes": nodes, "edges": edges}

    _CACHE_SCHEDULE[th] = schedule_data
    _CACHE_STATS[th] = stats_data
    _CACHE_GRAPH[th] = graph_data

    return schedule_data, stats_data, graph_data


@app.get("/api/schedule")
def get_schedule(threshold: float = Query(0.20, ge=0.01, le=0.99)):
    """Devuelve la asignación de materias por turno para el umbral dado."""
    sch, _, _ = _compute_for_threshold(round(threshold, 2))
    return sch


@app.get("/api/stats")
def get_stats(threshold: float = Query(0.20, ge=0.01, le=0.99)):
    """Devuelve las métricas operativas del cronograma y evaluación contra lo real."""
    _, stats, _ = _compute_for_threshold(round(threshold, 2))
    return stats


@app.get("/api/graph")
def get_graph(
    threshold: float = Query(0.20, ge=0.01, le=0.99),
    subject: Optional[str] = Query(None),
):
    """Devuelve nodos y aristas del grafo predicho (completo o filtrado por materia)."""
    _, _, full_graph = _compute_for_threshold(round(threshold, 2))
    if not subject:
        return full_graph

    sub_edges = [
        e
        for e in full_graph["edges"]
        if e["from"] == subject or e["to"] == subject
    ]
    sub_ids = {subject} | {e["from"] for e in sub_edges} | {e["to"] for e in sub_edges}
    sub_nodes = [n for n in full_graph["nodes"] if n["id"] in sub_ids]
    return {"nodes": sub_nodes, "edges": sub_edges}


@app.get("/api/sweep")
def get_sweep():
    """Devuelve la curva de tradeoff calculada en el barrido de umbrales."""
    if SWEEP_CSV.exists():
        df = pd.read_csv(SWEEP_CSV)
        return [
            {
                "threshold": float(r["threshold"]),
                "turnos": int(r["turnos"]),
                "alumnos_protegidos_pct": round(
                    100.0 - float(r["pct_alumnos_en_choque"]), 2
                ),
            }
            for _, r in df.iterrows()
        ]
    return []
