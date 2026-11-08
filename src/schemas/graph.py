"""Esquemas Pydantic para el grafo de conflictos y consulta de pares."""

from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field


class GraphEdgeOut(BaseModel):
    """Representación de una arista de conflicto entre dos materias."""

    source: str = Field(..., description="ID de la primera materia", examples=["ACC1"])
    target: str = Field(..., description="ID de la segunda materia", examples=["MAT101"])
    probability: float = Field(..., description="Probabilidad de conflicto predicha por el modelo", examples=[0.75])


class GraphResponse(BaseModel):
    """Métricas topológicas y aristas activas del grafo predicho según el umbral."""

    threshold: float = Field(..., description="Umbral de decisión aplicado", examples=[0.20])
    num_nodes: int = Field(..., description="Cantidad total de nodos en el grafo", examples=[800])
    num_edges: int = Field(..., description="Cantidad de aristas activas (prob_conflicto >= threshold)", examples=[2032])
    density: float = Field(..., description="Densidad del grafo (relación de aristas existentes vs posibles)", examples=[0.006358])
    isolated_nodes_count: int = Field(..., description="Cantidad de materias sin ningún conflicto predicho", examples=[115])
    connected_components_count: int = Field(..., description="Cantidad de componentes conexas en el grafo", examples=[122])
    edges: List[GraphEdgeOut] = Field(default_factory=list, description="Lista de aristas activas en el grafo")


class ConflictPairResponse(BaseModel):
    """Respuesta a la consulta de probabilidad y conflicto entre dos materias."""

    materia_id_1: str = Field(..., description="ID de la primera materia", examples=["ACC1"])
    materia_id_2: str = Field(..., description="ID de la segunda materia", examples=["ACC2"])
    prob_conflicto: float = Field(..., description="Probabilidad predicha de conflicto", examples=[0.8542])
    es_conflicto: bool = Field(..., description="Indica si prob_conflicto >= threshold_aplicado", examples=[True])
    threshold_aplicado: float = Field(..., description="Umbral de corte aplicado para la clasificación", examples=[0.20])
