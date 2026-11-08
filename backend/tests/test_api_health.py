"""Pruebas unitarias para el endpoint de healthcheck y raíz de la API."""

from __future__ import annotations

import unittest

from fastapi.testclient import TestClient

from src.api.main import app
from src.core.config import settings


class TestApiHealth(unittest.TestCase):
    """Pruebas de disponibilidad y respuesta del endpoint /api/v1/health."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.client_cm = TestClient(app)
        cls.client = cls.client_cm.__enter__()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.client_cm.__exit__(None, None, None)

    def test_healthcheck_returns_healthy_status_and_catalog_count(self) -> None:
        """Verifica que /api/v1/health retorne 200, versión y el catálogo cargado."""
        response = self.client.get(f"{settings.API_V1_STR}/health")
        self.assertEqual(response.status_code, 200)

        data = response.json()
        self.assertEqual(data.get("status"), "healthy")
        self.assertEqual(data.get("version"), settings.VERSION)
        self.assertEqual(data.get("project_name"), settings.PROJECT_NAME)
        self.assertEqual(data.get("catalog_exams_count"), 800)

    def test_root_endpoint_returns_welcome_and_docs_link(self) -> None:
        """Verifica que la raíz / responda con 200 y el link a la documentación."""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)

        data = response.json()
        self.assertIn("docs", data)
        self.assertEqual(data["docs"], f"{settings.API_V1_STR}/docs")


if __name__ == "__main__":
    unittest.main()
