"""Servicio para la generación de cronogramas mediante coloreo de grafos."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd

from src.graphs.conflict_graph import (
    TURNOS_CALENDARIO,
    VALID_STRATEGIES,
    build_conflict_graph_from_predictions,
    build_schedule,
    count_turns,
    greedy_coloring,
    validate_coloring,
)

STRATEGY_METADATA: List[Dict[str, str]] = [
    {
        "strategy": "largest_first",
        "name": "Welsh-Powell (Largest First)",
        "description": "Ordena los vértices por grado descendente. Colorea primero las materias con mayor número de conflictos.",
    },
    {
        "strategy": "saturation_largest_first",
        "name": "DSATUR (Saturation Largest First)",
        "description": "Heurística de grado de saturación de Brélaz. Selecciona dinámicamente el vértice con más colores vecinos distintos.",
    },
    {
        "strategy": "smallest_last",
        "name": "Smallest Last (Matula-Beck)",
        "description": "Elimina iterativamente vértices de menor grado para colorear en orden inverso.",
    },
    {
        "strategy": "independent_set",
        "name": "Independent Set",
        "description": "Encuentra conjuntos independientes maximales sucesivos para asignar a cada turno.",
    },
    {
        "strategy": "connected_sequential_bfs",
        "name": "Connected Sequential (BFS)",
        "description": "Ordena los vértices mediante un recorrido en anchura (BFS) del grafo.",
    },
    {
        "strategy": "connected_sequential_dfs",
        "name": "Connected Sequential (DFS)",
        "description": "Ordena los vértices mediante un recorrido en profundidad (DFS) del grafo.",
    },
    {
        "strategy": "random_sequential",
        "name": "Random Sequential",
        "description": "Colorea los vértices en un ordenamiento aleatorio uniforme.",
    },
]


def get_available_strategies() -> List[Dict[str, str]]:
    """Retorna la lista descriptiva de todas las estrategias de coloreo soportadas."""
    return STRATEGY_METADATA


def generate_schedule(
    pred_df: pd.DataFrame,
    catalog: Dict[str, Dict[str, Any]],
    threshold: float,
    strategy: str = "largest_first",
) -> Dict[str, Any]:
    """Genera la partición en turnos de examen sobre el grafo predicho.

    Pasos:
    1. Construye el grafo no dirigido para el umbral dado.
    2. Ejecuta el algoritmo greedy de coloreo con la heurística indicada.
    3. Valida formalmente que ninguna arista predicha comparta color.
    4. Estructura el cronograma asociando los metadatos de catálogo a cada turno.
    """
    clean_strategy = strategy.strip().lower()
    if clean_strategy not in VALID_STRATEGIES:
        valid_options = ", ".join(sorted(VALID_STRATEGIES))
        raise ValueError(
            f"Estrategia de coloreo '{strategy}' no válida. Opciones disponibles: {valid_options}"
        )

    # 1. Construir grafo predicho
    G = build_conflict_graph_from_predictions(
        pred_df=pred_df,
        threshold=threshold,
        exams_catalog=catalog,
    )

    # 2. Ejecutar coloreo
    coloring = greedy_coloring(G, strategy=clean_strategy)

    # 3. Validación formal interna
    is_valid, _ = validate_coloring(G, coloring)

    # 4. Cálculo de turnos y verificación de calendario
    turns = count_turns(coloring)
    fits = bool(turns <= TURNOS_CALENDARIO)

    # 5. Agrupación y estructuración de turnos
    raw_schedule = build_schedule(coloring, exams_catalog=catalog)
    formatted_schedule: Dict[str, List[Dict[str, Any]]] = {
        str(turn): items for turn, items in raw_schedule.items()
    }
    turn_distribution = {
        str(turn): len(items) for turn, items in raw_schedule.items()
    }
    total_scheduled = sum(len(items) for items in raw_schedule.values())

    return {
        "threshold": threshold,
        "strategy": clean_strategy,
        "turns_count": turns,
        "max_allowed_turns": TURNOS_CALENDARIO,
        "fits_in_calendar": fits,
        "is_valid_coloring": is_valid,
        "total_exams_scheduled": total_scheduled,
        "turn_distribution": turn_distribution,
        "schedule": formatted_schedule,
    }
