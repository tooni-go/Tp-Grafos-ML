"""Pruebas unitarias y de integración para los endpoints del catálogo de exámenes."""

from __future__ import annotations

import unittest

from fastapi.testclient import TestClient

from src.api.main import app
from src.core.config import settings


class TestApiCatalog(unittest.TestCase):
    """Pruebas del módulo de catálogo: paginación, filtros, detalle y departamentos."""

    @classmethod
    def setUpClass(cls) -> None:
        # Usar context manager para asegurar la ejecución del lifespan en TestClient
        cls.client_cm = TestClient(app)
        cls.client = cls.client_cm.__enter__()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.client_cm.__exit__(None, None, None)

    def test_list_exams_default_pagination(self) -> None:
        """Verifica paginación por defecto: 800 exámenes en total, 100 devueltos."""
        response = self.client.get(f"{settings.API_V1_STR}/exams")
        self.assertEqual(response.status_code, 200)

        data = response.json()
        self.assertEqual(data.get("total"), 800)
        self.assertEqual(data.get("limit"), 100)
        self.assertEqual(data.get("offset"), 0)
        self.assertEqual(len(data.get("items", [])), 100)

        # Verificar estructura del primer elemento
        first = data["items"][0]
        for field in ("exam_id", "exam_name", "department_code", "duration", "duration_minutes"):
            self.assertIn(field, first)

    def test_list_exams_custom_pagination(self) -> None:
        """Verifica paginación con offset y limit personalizados."""
        response = self.client.get(f"{settings.API_V1_STR}/exams?limit=25&offset=50")
        self.assertEqual(response.status_code, 200)

        data = response.json()
        self.assertEqual(data.get("total"), 800)
        self.assertEqual(data.get("limit"), 25)
        self.assertEqual(data.get("offset"), 50)
        self.assertEqual(len(data.get("items", [])), 25)

    def test_filter_by_department(self) -> None:
        """Verifica que el filtro por departamento restrinja adecuadamente la búsqueda."""
        dept = "AC"
        response = self.client.get(f"{settings.API_V1_STR}/exams?department_code={dept}")
        self.assertEqual(response.status_code, 200)

        data = response.json()
        self.assertGreater(data.get("total"), 0)
        self.assertTrue(all(item["department_code"] == dept for item in data["items"]))

    def test_search_by_text(self) -> None:
        """Verifica búsqueda textual en código o nombre de examen."""
        response = self.client.get(f"{settings.API_V1_STR}/exams?search=OPERA")
        self.assertEqual(response.status_code, 200)

        data = response.json()
        self.assertGreater(data.get("total"), 0)
        self.assertTrue(
            any("OPERA" in item["exam_name"].upper() for item in data["items"])
        )

    def test_get_exam_by_id_success(self) -> None:
        """Verifica consulta exitosa de una materia existente."""
        exam_id = "AA2016E1"
        response = self.client.get(f"{settings.API_V1_STR}/exams/{exam_id}")
        self.assertEqual(response.status_code, 200)

        data = response.json()
        self.assertEqual(data.get("exam_id"), exam_id)
        self.assertEqual(data.get("duration_minutes"), 90)
        self.assertEqual(data.get("department_code"), "GM")

    def test_get_exam_by_id_not_found(self) -> None:
        """Verifica respuesta 404 al solicitar un ID inexistente."""
        response = self.client.get(f"{settings.API_V1_STR}/exams/ID_INEXISTENTE_XYZ")
        self.assertEqual(response.status_code, 404)

        data = response.json()
        self.assertIn("detail", data)

    def test_list_departments_consistency(self) -> None:
        """Verifica que la lista de departamentos sea consistente y la suma de exámenes totalice 800."""
        response = self.client.get(f"{settings.API_V1_STR}/departments")
        self.assertEqual(response.status_code, 200)

        departments = response.json()
        self.assertIsInstance(departments, list)
        self.assertGreater(len(departments), 0)

        total_exams = sum(dept["exams_count"] for dept in departments)
        self.assertEqual(total_exams, 800)


if __name__ == "__main__":
    unittest.main()
