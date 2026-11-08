"""Servicio para construcción, métricas y consulta de grafos de conflicto."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import pandas as pd

from src.graphs.conflict_graph import build_conflict_graph_from_predictions


def build_graph_for_threshold(
    pred_df: pd.DataFrame,
    threshold: float,
    catalog: Optional[Dict[str, Dict[str, Any]]] = None,
) -> nx.Graph:
    """Construye el grafo NetworkX no dirigido a partir de probabilidades y umbral."""
    return build_conflict_graph_from_predictions(
        pred_df=pred_df,
        threshold=threshold,
        exams_catalog=catalog,
    )


def get_graph_statistics(
    G: nx.Graph,
    pred_df: pd.DataFrame,
    threshold: float,
    include_edges: bool = True,
    prob_col: str = "prob_conflicto",
) -> Dict[str, Any]:
    """Calcula las métricas topológicas del grafo y extrae sus aristas activas."""
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    density = round(float(nx.density(G)), 6) if num_nodes > 1 else 0.0
    isolated_nodes_count = len(list(nx.isolates(G)))
    connected_components_count = nx.number_connected_components(G) if num_nodes > 0 else 0

    edges: List[Dict[str, Any]] = []
    if include_edges:
        mask = pred_df[prob_col] >= threshold
        filtered = pred_df.loc[mask, ["materia_id_1", "materia_id_2", prob_col]]
        edges = [
            {
                "source": row[0],
                "target": row[1],
                "probability": round(float(row[2]), 4),
            }
            for row in filtered.itertuples(index=False, name=None)
        ]

    return {
        "threshold": threshold,
        "num_nodes": num_nodes,
        "num_edges": num_edges,
        "density": density,
        "isolated_nodes_count": isolated_nodes_count,
        "connected_components_count": connected_components_count,
        "edges": edges,
    }


def get_pair_probability_from_map(
    pairs_prob_map: Dict[Tuple[str, str], float],
    id1: str,
    id2: str,
) -> Optional[float]:
    """Obtiene la probabilidad de conflicto de un par en el mapa en memoria $O(1)$."""
    canonical_pair = (min(id1, id2), max(id1, id2))
    return pairs_prob_map.get(canonical_pair)
