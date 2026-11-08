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
        """Verifica que /schedule/strategies liste las 7 heurísticas soportadas."""
        response = self.client.get(f"{settings.API_V1_STR}/schedule/strategies")
        self.assertEqual(response.status_code, 200)

        strategies = response.json()
        self.assertIsInstance(strategies, list)
        self.assertEqual(len(strategies), 7)

        strategy_keys = {item["strategy"] for item in strategies}
        self.assertIn("largest_first", strategy_keys)
        self.assertIn("saturation_largest_first", strategy_keys)
        self.assertIn("smallest_last", strategy_keys)


if __name__ == "__main__":
    unittest.main()
