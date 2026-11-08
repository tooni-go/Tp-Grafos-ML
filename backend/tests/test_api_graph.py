"""Pruebas unitarias y de integración para los endpoints de grafo y consulta de pares."""

from __future__ import annotations

import unittest

from fastapi.testclient import TestClient

from src.api.main import app
from src.core.config import settings


class TestApiGraph(unittest.TestCase):
    """Pruebas del módulo de grafo: métricas topológicas, aristas y consulta de pares."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.client_cm = TestClient(app)
        cls.client = cls.client_cm.__enter__()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.client_cm.__exit__(None, None, None)

    def test_get_graph_default_threshold(self) -> None:
        """Verifica que el grafo con umbral por defecto (0.20) devuelva 800 nodos y 2032 aristas."""
        response = self.client.get(f"{settings.API_V1_STR}/graph")
        self.assertEqual(response.status_code, 200)

        data = response.json()
        self.assertEqual(data.get("threshold"), 0.20)
        self.assertEqual(data.get("num_nodes"), 800)
        self.assertEqual(data.get("num_edges"), 2032)
        self.assertGreater(data.get("density"), 0.0)
        self.assertGreater(data.get("isolated_nodes_count"), 0)
        self.assertGreater(data.get("connected_components_count"), 0)
        self.assertEqual(len(data.get("edges", [])), 2032)

        # Verificar estructura de la primera arista
        first_edge = data["edges"][0]
        self.assertIn("source", first_edge)
        self.assertIn("target", first_edge)
        self.assertIn("probability", first_edge)
        self.assertGreaterEqual(first_edge["probability"], 0.20)

    def test_get_graph_without_edges_payload(self) -> None:
        """Verifica que include_edges=false omita la lista de aristas para respuestas livianas."""
        response = self.client.get(f"{settings.API_V1_STR}/graph?include_edges=false")
        self.assertEqual(response.status_code, 200)

        data = response.json()
        self.assertEqual(data.get("num_edges"), 2032)
        self.assertEqual(len(data.get("edges", [])), 0)

    def test_get_graph_invalid_threshold(self) -> None:
        """Verifica validación de umbrales fuera de (0, 1) con respuesta 422."""
        for bad_th in (0.0, 1.0, -0.5, 1.5):
            response = self.client.get(f"{settings.API_V1_STR}/graph?threshold={bad_th}")
            self.assertEqual(response.status_code, 422)

    def test_get_conflict_pair_success(self) -> None:
        """Verifica consulta exitosa de probabilidad entre dos materias del catálogo."""
        id1 = "AA2016E1"
        id2 = "AAA011E1"
        response = self.client.get(
            f"{settings.API_V1_STR}/conflicts/pair?exam_id_1={id1}&exam_id_2={id2}"
        )
        self.assertEqual(response.status_code, 200)

        data = response.json()
        self.assertEqual(data.get("materia_id_1"), id1)
        self.assertEqual(data.get("materia_id_2"), id2)
        self.assertIsInstance(data.get("prob_conflicto"), float)
        self.assertIsInstance(data.get("es_conflicto"), bool)
        self.assertEqual(data.get("threshold_aplicado"), 0.20)
        self.assertEqual(
            data.get("es_conflicto"), data["prob_conflicto"] >= data["threshold_aplicado"]
        )

    def test_get_conflict_pair_order_invariance(self) -> None:
        """Verifica que el orden de los IDs no altere la probabilidad del par (grafo no dirigido)."""
        id1 = "AA2016E1"
        id2 = "AA3008E1"
        res_direct = self.client.get(
            f"{settings.API_V1_STR}/conflicts/pair?exam_id_1={id1}&exam_id_2={id2}"
        )
        res_inverse = self.client.get(
            f"{settings.API_V1_STR}/conflicts/pair?exam_id_1={id2}&exam_id_2={id1}"
        )
        self.assertEqual(res_direct.status_code, 200)
        self.assertEqual(res_inverse.status_code, 200)
        self.assertEqual(
            res_direct.json()["prob_conflicto"], res_inverse.json()["prob_conflicto"]
        )

    def test_get_conflict_pair_self_pair_rejected(self) -> None:
        """Verifica rechazo con código 400 de auto-pares (misma materia)."""
        same_id = "ACC1"
        response = self.client.get(
            f"{settings.API_V1_STR}/conflicts/pair?exam_id_1={same_id}&exam_id_2={same_id}"
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("auto-par", response.json().get("detail", "").lower())

    def test_get_conflict_pair_not_found(self) -> None:
        """Verifica respuesta 404 si alguna de las materias no existe en el catálogo."""
        response = self.client.get(
            f"{settings.API_V1_STR}/conflicts/pair?exam_id_1=ACC1&exam_id_2=INEXISTENTE_XYZ"
        )
        self.assertEqual(response.status_code, 404)


if __name__ == "__main__":
    unittest.main()
