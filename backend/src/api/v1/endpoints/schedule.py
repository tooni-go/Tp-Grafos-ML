"""Endpoints para generación de cronogramas y consulta de estrategias de coloreo."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import APIRouter, Body, HTTPException, Query, Request

from src.common.schema import PROJECT_ROOT
from src.core.config import settings
from src.schemas.catalog import ExamOut
from src.schemas.schedule import (
    ScheduleEvaluationResponse,
    ScheduleRequest,
    ScheduleResponse,
    ScheduleStrategyOut,
)
from src.services.schedule_service import (
    evaluate_schedule,
    generate_schedule,
    get_available_strategies,
)

router = APIRouter()


@router.get(
    "/schedule",
    response_model=ScheduleResponse,
    summary="Generar cronograma de exámenes (GET)",
)
def get_schedule(
    request: Request,
    threshold: Optional[float] = Query(
        None,
        gt=0.0,
        lt=1.0,
        description="Umbral de probabilidad de conflicto en (0, 1). Por defecto: calibrado oficial (0.20)",
        examples=[0.20],
    ),
    strategy: str = Query(
        "largest_first",
        description="Heurística de coloreo greedy (ej. 'largest_first', 'saturation_largest_first')",
        examples=["largest_first"],
    ),
) -> ScheduleResponse:
    """Genera una asignación de exámenes a turnos sin conflictos mediante coloreo de grafos (método GET)."""
    payload = ScheduleRequest(threshold=threshold, strategy=strategy)
    return post_schedule(request=request, payload=payload)


@router.post("/schedule", response_model=ScheduleResponse, summary="Generar cronograma de exámenes")
def post_schedule(
    request: Request,
    payload: ScheduleRequest = Body(default_factory=ScheduleRequest),
) -> ScheduleResponse:
    """Genera una asignación de exámenes a turnos sin conflictos mediante coloreo de grafos."""
    pred_df = getattr(request.app.state, "probabilities_df", None)
    if pred_df is None:
        raise HTTPException(
            status_code=503,
            detail="La matriz de probabilidades OOF no está cargada en memoria. Verifique la inicialización del servidor.",
        )

    resolved_threshold = (
        float(payload.threshold)
        if payload.threshold is not None
        else float(getattr(request.app.state, "default_threshold", 0.20))
    )

    catalog = getattr(request.app.state, "catalog", {})

    try:
        result = generate_schedule(
            pred_df=pred_df,
            catalog=catalog,
            threshold=resolved_threshold,
            strategy=payload.strategy,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    # Mapear exámenes a ExamOut en cada turno
    formatted_schedule = {
        turn: [ExamOut(**exam) for exam in exams]
        for turn, exams in result["schedule"].items()
    }

    return ScheduleResponse(
        threshold=result["threshold"],
        strategy=result["strategy"],
        turns_count=result["turns_count"],
        max_allowed_turns=result["max_allowed_turns"],
        fits_in_calendar=result["fits_in_calendar"],
        is_valid_coloring=result["is_valid_coloring"],
        total_exams_scheduled=result["total_exams_scheduled"],
        turn_distribution=result["turn_distribution"],
        schedule=formatted_schedule,
    )


@router.get(
    "/schedule/strategies",
    response_model=List[ScheduleStrategyOut],
    summary="Listar heurísticas de coloreo disponibles",
)
def get_strategies() -> List[ScheduleStrategyOut]:
    """Retorna la lista de todas las heurísticas y algoritmos de coloreo disponibles."""
    strategies = get_available_strategies()
    return [ScheduleStrategyOut(**item) for item in strategies]


@router.get(
    "/schedule/sweep",
    summary="Obtener barrido de umbrales precomputado",
)
def get_schedule_sweep() -> List[Dict[str, Any]]:
    """Retorna los puntos del barrido de umbrales precalculado para análisis de compensación."""
    sweep_path = PROJECT_ROOT / "results" / "graphs" / "threshold_sweep.csv"
    if not sweep_path.exists():
        raise HTTPException(
            status_code=404,
            detail="El archivo threshold_sweep.csv no existe.",
        )
    df = pd.read_csv(sweep_path)
    points: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        pct_choque = float(row.get("pct_alumnos_en_choque", 0.0))
        points.append({
            "threshold": float(row["threshold"]),
            "turnos": int(row["turnos"]),
            "alumnos_protegidos_pct": round(100.0 - pct_choque, 2),
            "precision_aristas": float(row.get("precision_aristas", 0.0)),
            "aristas": int(row.get("aristas_predichas", 0)),
        })
    return points


@router.get(
    "/schedule/evaluate",
    response_model=ScheduleEvaluationResponse,
    summary="Evaluar cronograma contra conflictos reales de 1994 (GET)",
)
def get_schedule_evaluate(
    request: Request,
    threshold: Optional[float] = Query(
        None,
        gt=0.0,
        lt=1.0,
        description="Umbral de probabilidad de conflicto en (0, 1). Por defecto: calibrado oficial (0.20)",
        examples=[0.20],
    ),
    strategy: str = Query(
        "largest_first",
        description="Heurística de coloreo greedy (ej. 'largest_first', 'saturation_largest_first')",
        examples=["largest_first"],
    ),
) -> ScheduleEvaluationResponse:
    """Evalúa la eficacia de un cronograma generado por ML comparándolo con los choques reales observados (método GET)."""
    payload = ScheduleRequest(threshold=threshold, strategy=strategy)
    return post_schedule_evaluate(request=request, payload=payload)


@router.post(
    "/schedule/evaluate",
    response_model=ScheduleEvaluationResponse,
    summary="Evaluar cronograma contra conflictos reales de 1994",
)
def post_schedule_evaluate(
    request: Request,
    payload: ScheduleRequest = Body(default_factory=ScheduleRequest),
) -> ScheduleEvaluationResponse:
    """Evalúa la eficacia de un cronograma generado por ML comparándolo con los choques reales observados."""
    pred_df = getattr(request.app.state, "probabilities_df", None)
    if pred_df is None:
        raise HTTPException(
            status_code=503,
            detail="La matriz de probabilidades OOF no está cargada en memoria. Verifique la inicialización del servidor.",
        )

    real_df = getattr(request.app.state, "real_pairs_df", None)
    if real_df is None:
        raise HTTPException(
            status_code=503,
            detail="El dataset de pares reales históricos (1994) no está cargado en memoria. Verifique la inicialización del servidor.",
        )

    resolved_threshold = (
        float(payload.threshold)
        if payload.threshold is not None
        else float(getattr(request.app.state, "default_threshold", 0.20))
    )

    catalog = getattr(request.app.state, "catalog", {})
    student_exams = getattr(request.app.state, "student_exams_1994", None)

    try:
        result = evaluate_schedule(
            pred_df=pred_df,
            real_pairs_df=real_df,
            catalog=catalog,
            threshold=resolved_threshold,
            strategy=payload.strategy,
            student_exams=student_exams,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return ScheduleEvaluationResponse(**result)


@router.get(
    "/stats",
    response_model=ScheduleEvaluationResponse,
    summary="Alias para evaluar cronograma y métricas (GET)",
    include_in_schema=False,
)
def get_stats(
    request: Request,
    threshold: Optional[float] = Query(None, gt=0.0, lt=1.0),
    strategy: str = Query("largest_first"),
) -> ScheduleEvaluationResponse:
    """Alias compatible con /api/stats para frontend."""
    return get_schedule_evaluate(request=request, threshold=threshold, strategy=strategy)


@router.get(
    "/sweep",
    summary="Alias para barrido de umbrales",
    include_in_schema=False,
)
def get_sweep() -> List[Dict[str, Any]]:
    """Alias compatible con /api/sweep para frontend."""
    return get_schedule_sweep()
