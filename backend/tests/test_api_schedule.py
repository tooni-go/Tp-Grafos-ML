"""Pruebas unitarias y de integración para los endpoints de generación de cronogramas."""

from __future__ import annotations

import unittest

from fastapi.testclient import TestClient

from src.api.main import app
from src.core.config import settings


class TestApiSchedule(unittest.TestCase):
    """Pruebas del módulo de cronogramas: coloreo de grafos, turnos y validación."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.client_cm = TestClient(app)
        cls.client = cls.client_cm.__enter__()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.client_cm.__exit__(None, None, None)

    def test_generate_schedule_default(self) -> None:
        """Verifica generación de cronograma con configuración oficial (0.20, largest_first)."""
        response = self.client.post(f"{settings.API_V1_STR}/schedule", json={})
        self.assertEqual(response.status_code, 200)

        data = response.json()
        self.assertEqual(data.get("threshold"), 0.20)
        self.assertEqual(data.get("strategy"), "largest_first")
        self.assertEqual(data.get("turns_count"), 11)
        self.assertEqual(data.get("max_allowed_turns"), 32)
        self.assertTrue(data.get("fits_in_calendar"))
        self.assertTrue(data.get("is_valid_coloring"))
        self.assertEqual(data.get("total_exams_scheduled"), 800)

        # Verificar consistencia de la distribución de turnos
        dist = data.get("turn_distribution", {})
        self.assertEqual(len(dist), 11)
        self.assertEqual(sum(dist.values()), 800)

        # Verificar estructura de los turnos en el cronograma
        schedule = data.get("schedule", {})
        self.assertEqual(len(schedule), 11)
        for turn_idx in range(11):
            turn_key = str(turn_idx)
            self.assertIn(turn_key, schedule)
            self.assertGreater(len(schedule[turn_key]), 0)

            # Verificar campos del primer examen del turno
            first_exam = schedule[turn_key][0]
            self.assertIn("exam_id", first_exam)
            self.assertIn("exam_name", first_exam)
            self.assertIn("department_code", first_exam)
            self.assertIn("duration_minutes", first_exam)

    def test_schedule_ids_uniqueness_and_catalog_parity(self) -> None:
        """Verifica que el cronograma contenga exactamente las 800 materias del catálogo sin duplicados."""
        response = self.client.post(f"{settings.API_V1_STR}/schedule", json={})
        self.assertEqual(response.status_code, 200)

        data = response.json()
        schedule = data.get("schedule", {})

        all_scheduled_ids = [
            exam["exam_id"]
            for exams in schedule.values()
            for exam in exams
        ]

        # 1. Total exacto de materias agendadas
        self.assertEqual(len(all_scheduled_ids), 800)

        # 2. Sin IDs duplicados entre turnos
        unique_ids = set(all_scheduled_ids)
        self.assertEqual(len(unique_ids), 800)

        # 3. Exactamente el mismo conjunto de IDs del catálogo oficial
        catalog_response = self.client.get(f"{settings.API_V1_STR}/exams?limit=1000")
        self.assertEqual(catalog_response.status_code, 200)
        catalog_ids = {exam["exam_id"] for exam in catalog_response.json()["items"]}
        self.assertEqual(unique_ids, catalog_ids)

    def test_schedule_exceeds_calendar_capacity(self) -> None:
        """Verifica comportamiento cuando el cronograma supera el límite de 32 turnos (fits_in_calendar=false)."""
        # A umbral muy exigente (0.0333), la alta cantidad de aristas demanda > 32 turnos
        payload = {"threshold": 0.0333, "strategy": "largest_first"}
        response = self.client.post(f"{settings.API_V1_STR}/schedule", json=payload)
        self.assertEqual(response.status_code, 200)

        data = response.json()
        turns_count = data.get("turns_count")

        # 1. Supera el calendario de 32 turnos
        self.assertGreater(turns_count, 32)
        self.assertEqual(data.get("max_allowed_turns"), 32)
        self.assertFalse(data.get("fits_in_calendar"))

        # 2. El coloreo sigue siendo formalmente válido sobre el grafo predicho
        self.assertTrue(data.get("is_valid_coloring"))

        # 3. No se trunca silenciosamente para forzar 32 turnos
        schedule = data.get("schedule", {})
        self.assertEqual(len(schedule), turns_count)
        self.assertEqual(data.get("total_exams_scheduled"), 800)

    def test_generate_schedule_custom_strategy(self) -> None:
        """Verifica coloreo con la heurística DSATUR (saturation_largest_first)."""
        payload = {"strategy": "saturation_largest_first", "threshold": 0.20}
        response = self.client.post(f"{settings.API_V1_STR}/schedule", json=payload)
        self.assertEqual(response.status_code, 200)

        data = response.json()
        self.assertEqual(data.get("strategy"), "saturation_largest_first")
        self.assertTrue(data.get("is_valid_coloring"))
        self.assertTrue(data.get("fits_in_calendar"))
        self.assertEqual(data.get("total_exams_scheduled"), 800)

    def test_generate_schedule_invalid_strategy(self) -> None:
        """Verifica rechazo con código 400 ante una estrategia inexistente."""
        payload = {"strategy": "heuristica_invalida"}
        response = self.client.post(f"{settings.API_V1_STR}/schedule", json=payload)
        self.assertEqual(response.status_code, 400)
        self.assertIn("no válida", response.json().get("detail", "").lower())

    def test_generate_schedule_invalid_threshold(self) -> None:
        """Verifica rechazo con código 422 para umbrales fuera del rango (0, 1)."""
        for bad_th in (0.0, 1.0, -0.2, 1.5):
            payload = {"threshold": bad_th}
            response = self.client.post(f"{settings.API_V1_STR}/schedule", json=payload)
            self.assertEqual(response.status_code, 422)

    def test_get_available_strategies(self) -> None:
        """Verifica que /schedule/strategies liste las 8 heurísticas soportadas."""
        response = self.client.get(f"{settings.API_V1_STR}/schedule/strategies")
        self.assertEqual(response.status_code, 200)

        strategies = response.json()
        self.assertIsInstance(strategies, list)
        self.assertEqual(len(strategies), 8)

        strategy_keys = {item["strategy"] for item in strategies}
        self.assertIn("largest_first", strategy_keys)
        self.assertIn("weighted_largest_first", strategy_keys)
        self.assertIn("saturation_largest_first", strategy_keys)
        self.assertIn("smallest_last", strategy_keys)

    def test_evaluate_schedule_weighted_strategy(self) -> None:
        """Verifica evaluación con heurística Welsh-Powell ponderada (weighted_largest_first)."""
        payload = {"strategy": "weighted_largest_first", "threshold": 0.20}
        response = self.client.post(f"{settings.API_V1_STR}/schedule/evaluate", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data.get("strategy"), "weighted_largest_first")
        self.assertEqual(data.get("turns_count"), 11)
        self.assertTrue(data.get("fits_in_calendar"))
        self.assertTrue(data.get("is_valid_coloring"))

    def test_evaluate_schedule_default(self) -> None:
        """Verifica evaluación del cronograma oficial (0.20, largest_first) contra la realidad de 1994."""
        response = self.client.post(f"{settings.API_V1_STR}/schedule/evaluate", json={})
        self.assertEqual(response.status_code, 200)

        data = response.json()
        # 1. Configuración y turnos
        self.assertEqual(data.get("threshold"), 0.20)
        self.assertEqual(data.get("strategy"), "largest_first")
        self.assertEqual(data.get("turns_count"), 11)
        self.assertEqual(data.get("max_allowed_turns"), 32)
        self.assertTrue(data.get("fits_in_calendar"))
        self.assertTrue(data.get("is_valid_coloring"))

        # 2. Métricas de choques reales de 1994 (incidencias estudiante-par)
        self.assertEqual(data.get("real_conflict_pairs"), 4468)
        self.assertEqual(data.get("clashing_pairs"), 830)
        self.assertEqual(data.get("total_affected_students"), 28677)
        self.assertEqual(data.get("clashing_students"), 3723)
        self.assertEqual(data.get("clash_student_percentage"), 12.98)

        # 3. Métricas de personas únicas afectadas (headcount cohorte 1994)
        self.assertEqual(data.get("unique_students_affected"), 1945)
        self.assertEqual(data.get("unique_students_percentage"), 63.03)

        # 4. Matriz de confusión y calidad de aristas del grafo
        self.assertEqual(data.get("predicted_edges"), 2032)
        self.assertEqual(data.get("true_positives"), 759)
        self.assertEqual(data.get("false_positives"), 1273)
        self.assertEqual(data.get("false_negatives"), 3709)
        self.assertEqual(data.get("edge_precision"), 0.3735)
        self.assertEqual(data.get("edge_recall"), 0.1699)

        # 5. Métricas crudas
        raw = data.get("raw_metrics")
        self.assertIsInstance(raw, dict)
        self.assertEqual(raw.get("TP"), 759)
        self.assertEqual(raw.get("alumnos_en_choque"), 3723)
        self.assertEqual(raw.get("alumnos_unicos_total"), 3086)
        self.assertEqual(raw.get("alumnos_unicos_en_choque"), 1945)
        self.assertEqual(raw.get("pct_alumnos_unicos_en_choque"), 63.03)

    def test_evaluate_schedule_custom_strategy(self) -> None:
        """Verifica evaluación con heurística DSATUR (saturation_largest_first)."""
        payload = {"strategy": "saturation_largest_first", "threshold": 0.20}
        response = self.client.post(f"{settings.API_V1_STR}/schedule/evaluate", json=payload)
        self.assertEqual(response.status_code, 200)

        data = response.json()
        self.assertEqual(data.get("strategy"), "saturation_largest_first")
        self.assertTrue(data.get("is_valid_coloring"))
        self.assertTrue(data.get("fits_in_calendar"))
        self.assertEqual(data.get("real_conflict_pairs"), 4468)
        self.assertGreater(data.get("clashing_students"), 0)
        self.assertGreater(data.get("clash_student_percentage"), 0.0)

    def test_evaluate_schedule_consistency_with_schedule_endpoint(self) -> None:
        """Verifica coherencia estricta entre /schedule y /schedule/evaluate."""
        payload = {"threshold": 0.20, "strategy": "largest_first"}
        resp_schedule = self.client.post(f"{settings.API_V1_STR}/schedule", json=payload)
        resp_eval = self.client.post(f"{settings.API_V1_STR}/schedule/evaluate", json=payload)

        self.assertEqual(resp_schedule.status_code, 200)
        self.assertEqual(resp_eval.status_code, 200)

        data_sched = resp_schedule.json()
        data_eval = resp_eval.json()

        self.assertEqual(data_sched.get("turns_count"), data_eval.get("turns_count"))
        self.assertEqual(data_sched.get("fits_in_calendar"), data_eval.get("fits_in_calendar"))
        self.assertEqual(data_sched.get("is_valid_coloring"), data_eval.get("is_valid_coloring"))

    def test_evaluate_schedule_invalid_strategy(self) -> None:
        """Verifica rechazo con código 400 ante estrategia inválida en evaluación."""
        payload = {"strategy": "estrategia_inexistente"}
        response = self.client.post(f"{settings.API_V1_STR}/schedule/evaluate", json=payload)
        self.assertEqual(response.status_code, 400)
        self.assertIn("no válida", response.json().get("detail", "").lower())

    def test_evaluate_schedule_invalid_threshold(self) -> None:
        """Verifica validación 422 para umbrales fuera de (0, 1) en evaluación."""
        for bad_th in (0.0, 1.0, -0.5, 2.0):
            payload = {"threshold": bad_th}
            response = self.client.post(f"{settings.API_V1_STR}/schedule/evaluate", json=payload)
            self.assertEqual(response.status_code, 422)

    def test_evaluate_schedule_exceeds_calendar_capacity(self) -> None:
        """Verifica que a umbral bajo el evaluador detecte fits_in_calendar=false sin rechazar la solicitud."""
        payload = {"threshold": 0.0333, "strategy": "largest_first"}
        response = self.client.post(f"{settings.API_V1_STR}/schedule/evaluate", json=payload)
        self.assertEqual(response.status_code, 200)

        data = response.json()
        self.assertGreater(data.get("turns_count"), 32)
        self.assertFalse(data.get("fits_in_calendar"))
        self.assertTrue(data.get("is_valid_coloring"))
        self.assertGreater(data.get("predicted_edges"), 2032)

    def test_evaluate_schedule_confusion_matrix_invariants(self) -> None:
        """Verifica los invariantes algebraicos de la matriz de confusión de aristas."""
        response = self.client.post(f"{settings.API_V1_STR}/schedule/evaluate", json={})
        self.assertEqual(response.status_code, 200)

        data = response.json()
        tp = data["true_positives"]
        fp = data["false_positives"]
        fn = data["false_negatives"]
        real_pairs = data["real_conflict_pairs"]
        pred_edges = data["predicted_edges"]
        precision = data["edge_precision"]
        recall = data["edge_recall"]

        # Invariante 1: TP + FN == Pares con conflicto real
        self.assertEqual(tp + fn, real_pairs)

        # Invariante 2: TP + FP == Total de aristas predichas en el grafo
        self.assertEqual(tp + fp, pred_edges)

        # Invariante 3: Precisión de aristas == TP / (TP + FP)
        expected_precision = round(tp / (tp + fp), 4) if (tp + fp) else 0.0
        self.assertEqual(precision, expected_precision)

        # Invariante 4: Recall de aristas == TP / (TP + FN)
        expected_recall = round(tp / (tp + fn), 4) if (tp + fn) else 0.0
        self.assertEqual(recall, expected_recall)

    def test_evaluate_schedule_invariants_and_clash_sources(self) -> None:
        """Verifica que ningún TP choque en el cronograma y que todo choque provenga de Falsos Negativos."""
        # Obtener evaluación y cronograma con la misma configuración
        payload = {"threshold": 0.20, "strategy": "largest_first"}
        resp_eval = self.client.post(f"{settings.API_V1_STR}/schedule/evaluate", json=payload)
        resp_sched = self.client.post(f"{settings.API_V1_STR}/schedule", json=payload)

        self.assertEqual(resp_eval.status_code, 200)
        self.assertEqual(resp_sched.status_code, 200)

        eval_data = resp_eval.json()
        sched_data = resp_sched.json()

        # Invariante 1: El coloreo es válido sobre el grafo predicho (ninguna arista predicha comparte turno)
        self.assertTrue(eval_data["is_valid_coloring"])
        self.assertTrue(sched_data["is_valid_coloring"])

        # Invariante 2: Los pares reales que chocan son como máximo los falsos negativos (clashing_pairs <= FN)
        clashing_pairs = eval_data["clashing_pairs"]
        fn = eval_data["false_negatives"]
        self.assertLessEqual(clashing_pairs, fn)
        self.assertEqual(clashing_pairs, 830)
        self.assertEqual(fn, 3709)

        # Invariante 3: Mapear materias a turnos y verificar aristas predichas
        exam_to_turn = {}
        for turn_str, exams in sched_data["schedule"].items():
            for ex in exams:
                exam_to_turn[ex["exam_id"]] = int(turn_str)

        # Consultar aristas predichas desde /graph
        graph_resp = self.client.get(f"{settings.API_V1_STR}/graph?threshold=0.20&include_edges=true")
        self.assertEqual(graph_resp.status_code, 200)
        pred_edges = graph_resp.json()["edges"]

        # Ninguna arista predicha puede tener ambos extremos en el mismo turno
        for edge in pred_edges:
            t_src = exam_to_turn.get(edge["source"])
            t_tgt = exam_to_turn.get(edge["target"])
            self.assertIsNotNone(t_src)
            self.assertIsNotNone(t_tgt)
            self.assertNotEqual(t_src, t_tgt)

    def test_evaluate_schedule_percentage_formulas_and_headcount_invariants(self) -> None:
        """Verifica coherencia aritmética de porcentajes de incidencias y personas únicas."""
        response = self.client.post(f"{settings.API_V1_STR}/schedule/evaluate", json={})
        self.assertEqual(response.status_code, 200)

        data = response.json()
        total_incidences = data["total_affected_students"]
        clashing_incidences = data["clashing_students"]
        pct_incidences = data["clash_student_percentage"]

        unique_affected = data["unique_students_affected"]
        unique_pct = data["unique_students_percentage"]

        # 1. Fórmula de porcentaje de incidencias estudiante-par
        expected_pct_incidences = round(100.0 * clashing_incidences / total_incidences, 2)
        self.assertEqual(pct_incidences, expected_pct_incidences)

        # 2. Invariante de headcount: personas únicas afectadas <= total de la cohorte (3.086)
        self.assertLessEqual(unique_affected, 3086)
        self.assertEqual(unique_affected, 1945)

        # 3. Fórmula de porcentaje de personas únicas sobre los 3.086 alumnos
        expected_pct_unique = round(100.0 * unique_affected / 3086, 2)
        self.assertEqual(unique_pct, expected_pct_unique)
        self.assertEqual(unique_pct, 63.03)

        # 4. Invariante: cada persona única afectada aporta al menos 1 choque individual (unique_affected <= clashing_incidences)
        self.assertLessEqual(unique_affected, clashing_incidences)


if __name__ == "__main__":
    unittest.main()
