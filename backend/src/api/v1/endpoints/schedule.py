"""Endpoints para generación de cronogramas y consulta de estrategias de coloreo."""

from __future__ import annotations

from typing import List

from fastapi import APIRouter, Body, HTTPException, Request

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
