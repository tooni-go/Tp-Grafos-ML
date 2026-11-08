"""Pruebas del módulo de grafos adaptado al ML (`src/graphs/`). Ejecutar:
    python -m unittest discover -s tests -v
"""

from __future__ import annotations

import unittest

import pandas as pd

from src.graphs.conflict_graph import (
    build_conflict_graph_from_predictions, build_real_graph, build_schedule,
    count_turns, count_unique_students_in_clash, evaluate_against_real,
    greedy_coloring, sweep_thresholds, validate_coloring,
)


def _pred():
    return pd.DataFrame({
        "materia_id_1": ["A", "A", "B", "C"],
        "materia_id_2": ["B", "C", "C", "D"],
        "prob_conflicto": [0.9, 0.5, 0.2, 0.05],
    })


def _real():
    return pd.DataFrame({
        "materia_id_1": ["A", "B", "C", "A"],
        "materia_id_2": ["B", "C", "D", "D"],
        "cant_estudiantes_afectados": [4, 2, 1, 0],
    })


class TestGraphs(unittest.TestCase):
    def test_threshold_es_inclusivo_y_con_pesos(self):
        G = build_conflict_graph_from_predictions(_pred(), 0.5)
        self.assertEqual({frozenset(e) for e in G.edges()}, {frozenset("AB"), frozenset("AC")})
        self.assertTrue(all("weight" in d for _, _, d in G.edges(data=True)))
        self.assertEqual(G["A"]["B"]["weight"], 0.9)
        self.assertEqual(G["A"]["C"]["weight"], 0.5)

    def test_coloreo_estrategias(self):
        G = build_conflict_graph_from_predictions(_pred(), 0.2)
        for strategy in ("largest_first", "weighted_largest_first", "random_sequential", "smallest_last"):
            c = greedy_coloring(G, strategy=strategy)
            self.assertTrue(validate_coloring(G, c)[0])

    def test_threshold_invalido(self):
        for bad in (0.0, 1.0, -1, 2):
            with self.assertRaises(ValueError):
                build_conflict_graph_from_predictions(_pred(), bad)

    def test_coloreo_valido_en_grafo_predicho(self):
        G = build_conflict_graph_from_predictions(_pred(), 0.2)
        c = greedy_coloring(G)
        self.assertTrue(validate_coloring(G, c)[0])

    def test_evaluacion_contra_real(self):
        # Todo en el mismo turno -> chocan los 3 pares reales con alumnos (4+2+1=7)
        ev = evaluate_against_real({k: 0 for k in "ABCD"}, _real())
        self.assertEqual(ev["pares_reales"], 3)
        self.assertEqual(ev["pares_reales_en_choque"], 3)
        self.assertEqual(ev["alumnos_en_choque"], 7)
        # Coloreo propio del grafo real -> 0 choques
        G = build_real_graph(_real())
        ev0 = evaluate_against_real(greedy_coloring(G), _real(), G)
        self.assertEqual(ev0["alumnos_en_choque"], 0)
        self.assertEqual((ev0["FP"], ev0["FN"]), (0, 0))

    def test_grafo_predicho_incompleto_deja_choques(self):
        G = build_conflict_graph_from_predictions(_pred(), 0.5)  # no tiene B-C ni C-D
        c = greedy_coloring(G)
        ev = evaluate_against_real(c, _real(), G)
        self.assertGreater(ev["FN"], 0)
        self.assertGreater(ev["alumnos_en_choque"], 0)

    def test_sweep_y_schedule(self):
        df = sweep_thresholds(_pred(), _real(), [0.1, 0.5, 0.9])
        self.assertEqual(len(df), 3)
        self.assertTrue(df["coloreo_valido_en_grafo_predicho"].all())
        sch = build_schedule({"A": 0, "B": 1, "C": 1})
        self.assertEqual(sorted(sch), [0, 1])
        self.assertEqual(count_turns({"A": 0, "B": 3}), 4)

    def test_unique_students_synthetic_case(self):
        """Verifica conteo de personas únicas afectadas en un caso sintético controlado.

        Escenario:
        - Estudiante A: inscripto en materias '1', '2', '3' (3 materias).
        - Estudiante B: inscripto en materias '1', '2' (2 materias).
        - Estudiante C: inscripto en materias '3', '4' (2 materias).
        - Cronograma: materias '1', '2' y '3' asignadas al turno 0; materia '4' al turno 1.

        Resultado esperado:
        - A tiene 3 pares en choque, pero cuenta como 1 sola persona.
        - B tiene 1 par en choque, pero cuenta como 1 sola persona.
        - C no tiene choques ('3' en turno 0 y '4' en turno 1).
        - Total de personas únicas afectadas: 2 de 3 (66.67%).
        """
        coloring = {"1": 0, "2": 0, "3": 0, "4": 1}
        student_exams = {
            "Estudiante_A": ["1", "2", "3"],
            "Estudiante_B": ["1", "2"],
            "Estudiante_C": ["3", "4"],
        }
        total, affected, pct = count_unique_students_in_clash(coloring, student_exams)
        self.assertEqual(total, 3)
        self.assertEqual(affected, 2)
        self.assertEqual(pct, 66.67)


if __name__ == "__main__":
    unittest.main()
