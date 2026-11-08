"""Endpoints para análisis del grafo de conflictos y consulta de incompatibilidades entre pares."""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException, Query, Request

from src.schemas.graph import ConflictPairResponse, GraphEdgeOut, GraphResponse
from src.services.graph_service import (
    build_graph_for_threshold,
    get_graph_statistics,
    get_pair_probability_from_map,
)

router = APIRouter()


@router.get("/graph", response_model=GraphResponse, summary="Obtener topología del grafo predicho")
def get_graph(
    request: Request,
    threshold: Optional[float] = Query(
        None,
        gt=0.0,
        lt=1.0,
        description="Umbral de probabilidad de conflicto (0, 1). Por defecto: umbral calibrado oficial (0.20)",
        examples=[0.20],
    ),
    include_edges: bool = Query(
        True,
        description="Indica si se debe incluir la lista detallada de aristas activas en la respuesta",
    ),
) -> GraphResponse:
    """Construye y analiza el grafo de conflictos predicho para el umbral solicitado."""
    pred_df = getattr(request.app.state, "probabilities_df", None)
    if pred_df is None:
        raise HTTPException(
            status_code=503,
            detail="La matriz de probabilidades OOF no está cargada en memoria. Verifique la inicialización del servidor.",
        )

    resolved_threshold = (
        float(threshold)
        if threshold is not None
        else float(getattr(request.app.state, "default_threshold", 0.20))
    )

    catalog = getattr(request.app.state, "catalog", {})
    G = build_graph_for_threshold(
        pred_df=pred_df,
        threshold=resolved_threshold,
        catalog=catalog,
    )

    stats = get_graph_statistics(
        G=G,
        pred_df=pred_df,
        threshold=resolved_threshold,
        include_edges=include_edges,
    )

    return GraphResponse(
        threshold=stats["threshold"],
        num_nodes=stats["num_nodes"],
        num_edges=stats["num_edges"],
        density=stats["density"],
        isolated_nodes_count=stats["isolated_nodes_count"],
        connected_components_count=stats["connected_components_count"],
        edges=[GraphEdgeOut(**e) for e in stats["edges"]],
    )


@router.get("/conflicts/pair", response_model=ConflictPairResponse, summary="Consultar conflicto entre dos materias")
def get_conflict_pair(
    request: Request,
    exam_id_1: str = Query(..., description="ID de la primera materia", examples=["ACC1"]),
    exam_id_2: str = Query(..., description="ID de la segunda materia", examples=["ACC2"]),
    threshold: Optional[float] = Query(
        None,
        gt=0.0,
        lt=1.0,
        description="Umbral de corte para determinar si es conflicto (0, 1). Por defecto: oficial (0.20)",
        examples=[0.20],
    ),
) -> ConflictPairResponse:
    """Verifica si dos materias del catálogo presentan conflicto predicho y su probabilidad."""
    id1 = exam_id_1.strip()
    id2 = exam_id_2.strip()

    if id1 == id2:
        raise HTTPException(
            status_code=400,
            detail="No se pueden comparar una materia consigo misma (auto-par no permitido).",
        )

    catalog = getattr(request.app.state, "catalog", {})
    if id1 not in catalog:
        raise HTTPException(
            status_code=404,
            detail=f"Materia '{id1}' no encontrada en el catálogo.",
        )
    if id2 not in catalog:
        raise HTTPException(
            status_code=404,
            detail=f"Materia '{id2}' no encontrada en el catálogo.",
        )

    pairs_prob_map = getattr(request.app.state, "pairs_prob_map", {})
    prob = get_pair_probability_from_map(pairs_prob_map, id1, id2)
    if prob is None:
        raise HTTPException(
            status_code=404,
            detail=f"No se encontró la predicción para el par ({id1}, {id2}) en la matriz de probabilidades.",
        )

    resolved_threshold = (
        float(threshold)
        if threshold is not None
        else float(getattr(request.app.state, "default_threshold", 0.20))
    )

    return ConflictPairResponse(
        exam_id_1=id1,
        exam_id_2=id2,
        conflict_probability=round(float(prob), 4),
        has_conflict=bool(prob >= resolved_threshold),
        applied_threshold=resolved_threshold,
    )
