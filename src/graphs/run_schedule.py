"""Pipeline completo: ML -> grafo predicho -> coloreo -> evaluación contra lo real.

Uso (desde la raíz del repo):
    python -m src.graphs.run_schedule                 # umbral oficial (0.20), modo oof
    python -m src.graphs.run_schedule --threshold 0.15 --strategy saturation_largest_first
    python -m src.graphs.run_schedule --mode final --sweep

Salidas en results/graphs/: schedule_<th>.json, threshold_sweep.csv, summary.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.common.schema import PROJECT_ROOT
from src.graphs.conflict_graph import (
    build_conflict_graph_from_predictions, build_real_graph, build_schedule,
    evaluate_against_real, greedy_coloring, load_exams_catalog, sweep_thresholds,
    validate_coloring,
)
from src.graphs.predicted_pairs import default_threshold, get_pair_probabilities, load_all_pairs

OUT_DIR = PROJECT_ROOT / "results" / "graphs"
EXAMS_CSV = PROJECT_ROOT / "data" / "interim" / "exams_clean.csv"
# Barrido de umbrales para evaluar el tradeoff entre turnos y choques reales.
# El umbral calibrado en validación para Decision Tree es 0.20.
DEFAULT_SWEEP = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--threshold", type=float, default=None, help="Umbral de prob_conflicto (default: calibrado, 0.20)")
    ap.add_argument("--strategy", default="largest_first")
    ap.add_argument("--mode", choices=["oof", "final"], default="oof")
    ap.add_argument("--sweep", action="store_true", help="Barrido de umbrales -> threshold_sweep.csv")
    ap.add_argument("--force", action="store_true", help="Recalcular probabilidades (ignora caché)")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    catalog = load_exams_catalog(EXAMS_CSV)
    real = load_all_pairs()
    pred = get_pair_probabilities(mode=args.mode, force=args.force)
    th = args.threshold if args.threshold is not None else default_threshold()

    # Referencia: grafo oráculo (conflictos reales de 1994)
    G_real = build_real_graph(real, catalog)
    c_real = greedy_coloring(G_real, args.strategy)
    ref = evaluate_against_real(c_real, real, G_real)

    G = build_conflict_graph_from_predictions(pred, th, catalog)
    coloring = greedy_coloring(G, args.strategy)
    ok, _ = validate_coloring(G, coloring)
    ev = evaluate_against_real(coloring, real, G)
    schedule = build_schedule(coloring, catalog)

    tag = f"{th:.2f}".replace(".", "_")
    (OUT_DIR / f"schedule_{tag}.json").write_text(
        json.dumps({str(k): v for k, v in schedule.items()}, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {"modo_predicciones": args.mode, "threshold": th, "estrategia": args.strategy,
               "coloreo_valido_en_grafo_predicho": ok, "evaluacion_vs_real": ev, "referencia_oraculo": ref}
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=int), encoding="utf-8")

    print(f"Umbral {th} | modo {args.mode} | estrategia {args.strategy}")
    print(f"  Grafo predicho : {ev['aristas_predichas']} aristas (TP={ev['TP']}, FP={ev['FP']}, FN={ev['FN']})")
    print(f"  Turnos         : {ev['turnos']} (calendario: 32) | oráculo: {ref['turnos']}")
    print(f"  Choques reales : {ev['pares_reales_en_choque']}/{ev['pares_reales']} pares, "
          f"{ev['alumnos_en_choque']}/{ev['alumnos_choque_total']} alumnos ({ev['pct_alumnos_en_choque']}%)")

    if args.sweep:
        df = sweep_thresholds(pred, real, DEFAULT_SWEEP, catalog, args.strategy)
        df.to_csv(OUT_DIR / "threshold_sweep.csv", index=False)
        print("\n", df[["threshold", "aristas_predichas", "turnos", "alumnos_en_choque",
                        "pct_alumnos_en_choque", "precision_aristas", "recall_aristas"]].to_string(index=False))


if __name__ == "__main__":
    main()
