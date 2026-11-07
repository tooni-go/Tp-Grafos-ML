"""Pruebas automatizadas para el módulo de inferencia (`src/inference/predict_conflicts.py`).

Verifica:
1. Carga y validación del modelo `.joblib` y lectura del umbral `0.21` desde `metrics_summary.json`.
2. Cálculo de probabilidades (`predict_proba(X)[:, 1]`) y discretización en múltiplos de 1/15.
3. Aplicación explícita del umbral `0.21` (y no `model.predict(X)` con corte 0.50).
4. Validación exhaustiva de entradas incorrectas (columnas faltantes, nulos, infinitos,
   tipos inválidos, strings booleanos, conteos negativos, Jaccard fuera de [0, 1],
   auto-pares, pares duplicados y umbrales fuera de rango).
5. Consistencia de dimensiones, reordenamiento automático de columnas y reproducción exacta
   de las predicciones oficiales de test.
6. Independencia del directorio de trabajo (`os.chdir`) y desacoplamiento total respecto
   al módulo de entrenamiento (`src.models.compare_models`).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.common.schema import (
    FEATURE_COLUMNS as SCHEMA_FEATURE_COLUMNS,
    ID_COLUMNS as SCHEMA_ID_COLUMNS,
    PROJECT_ROOT,
    RESULTS_PREDICTIONS_DIR,
)
from src.inference import (
    DEFAULT_METRICS_SUMMARY_PATH,
    DEFAULT_MODEL_NAME,
    DEFAULT_MODEL_PATH,
    FEATURE_COLUMNS,
    ID_COLUMNS,
    PREDICTION_COLUMN,
    PROBABILITY_COLUMN,
    load_calibrated_threshold,
    load_conflict_model,
    predict_conflict_pairs,
    validate_and_prepare_features,
)

TEST_FULL_PATH = PROJECT_ROOT / "data" / "processed" / "ml_test_full.csv"


class TestConflictInference(unittest.TestCase):
    """Pruebas unitarias y de integración del contrato de inferencia ML."""

    @classmethod
    def setUpClass(cls) -> None:
        """Carga el modelo entrenado una sola vez para agilizar la batería de pruebas."""
        cls.model = load_conflict_model(DEFAULT_MODEL_PATH)
        cls.sample_df = pd.DataFrame(
            [
                {
                    "materia_id_1": "AA2016E1",
                    "materia_id_2": "AAA034E1",
                    "mismo_departamento": False,
                    "mismo_prefijo_curricular": True,
                    "anio_confirmado": False,
                    "mismo_anio": False,
                    "misma_carrera": True,
                    "cant_carreras_compartidas_hist": 2,
                    "cant_inscriptos_hist_1": 19,
                    "cant_inscriptos_hist_2": 15,
                    "cant_estudiantes_compartidos_historico": 0,
                    "jaccard_historico": 0.0,
                },
                {
                    "materia_id_1": "C11101E1",
                    "materia_id_2": "C11105E1",
                    "mismo_departamento": True,
                    "mismo_prefijo_curricular": True,
                    "anio_confirmado": True,
                    "mismo_anio": True,
                    "misma_carrera": True,
                    "cant_carreras_compartidas_hist": 8,
                    "cant_inscriptos_hist_1": 120,
                    "cant_inscriptos_hist_2": 115,
                    "cant_estudiantes_compartidos_historico": 45,
                    "jaccard_historico": 0.236842,
                },
            ]
        )

    def test_01_load_model_and_threshold_ssot(self) -> None:
        """Verifica que el modelo .joblib y el umbral 0.21 se carguen correctamente."""
        self.assertIsInstance(self.model, Pipeline)
        self.assertIn("scaler", self.model.named_steps)
        self.assertIn("clf", self.model.named_steps)
        self.assertEqual(list(self.model.feature_names_in_), FEATURE_COLUMNS)
        self.assertEqual(len(FEATURE_COLUMNS), 10)
        self.assertIs(FEATURE_COLUMNS, SCHEMA_FEATURE_COLUMNS)
        self.assertIs(ID_COLUMNS, SCHEMA_ID_COLUMNS)

        th = load_calibrated_threshold(DEFAULT_MODEL_NAME, DEFAULT_METRICS_SUMMARY_PATH)
        self.assertAlmostEqual(th, 0.21, places=6)

    def test_02_missing_or_corrupt_files_handling(self) -> None:
        """Verifica el manejo de errores ante rutas inexistentes o JSON malformado."""
        with self.assertRaises(FileNotFoundError):
            load_conflict_model(Path("ruta/inexistente/modelo.joblib"))

        with self.assertRaises(FileNotFoundError):
            load_calibrated_threshold(
                metrics_summary_path=Path("ruta/inexistente/summary.json")
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            bad_json_path = Path(tmpdir) / "bad.json"
            bad_json_path.write_text("{corrupt_json", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_calibrated_threshold(metrics_summary_path=bad_json_path)

            missing_key_path = Path(tmpdir) / "no_key.json"
            missing_key_path.write_text(json.dumps({"other": 1}), encoding="utf-8")
            with self.assertRaises(KeyError):
                load_calibrated_threshold(metrics_summary_path=missing_key_path)

            missing_model_path = Path(tmpdir) / "no_model.json"
            missing_model_path.write_text(
                json.dumps({"calibrated_thresholds_from_validation": {"Other": 0.5}}),
                encoding="utf-8",
            )
            with self.assertRaises(KeyError):
                load_calibrated_threshold(
                    model_name="KNN_Full_Unweighted",
                    metrics_summary_path=missing_model_path,
                )

    def test_03_probabilities_and_threshold_application(self) -> None:
        """Verifica que las probabilidades sean múltiplos de 1/15 y se aplique el umbral 0.21."""
        result = predict_conflict_pairs(self.sample_df, model=self.model)

        self.assertEqual(len(result), len(self.sample_df))
        self.assertEqual(
            list(result.columns),
            ID_COLUMNS + [PROBABILITY_COLUMN, PREDICTION_COLUMN],
        )

        probs = result[PROBABILITY_COLUMN].to_numpy()
        preds = result[PREDICTION_COLUMN].to_numpy()

        # Probabilidades dentro de [0, 1] y múltiplos exactos de 1/15 (K=15 uniforme)
        self.assertTrue(np.all((probs >= 0.0) & (probs <= 1.0)))
        scaled_15 = np.round(probs * 15, 6)
        self.assertTrue(np.all(np.isclose(scaled_15, np.round(scaled_15))))

        # El primer par tiene baja probabilidad (< 0.21 -> 0) y el segundo alta (>= 0.21 -> 1)
        self.assertEqual(preds[0], 0)
        self.assertEqual(preds[1], 1)
        self.assertTrue(
            np.array_equal(preds, (probs >= 0.21).astype(int))
        )

    def test_04_column_reordering_and_input_immutability(self) -> None:
        """Verifica que el orden desordenado de columnas se corrija sin mutar el DataFrame original."""
        reversed_cols = list(reversed(self.sample_df.columns))
        scrambled_df = self.sample_df[reversed_cols].copy()
        scrambled_df["columna_extra_ignorada"] = 999
        original_cols_snapshot = list(scrambled_df.columns)

        res_canonical = predict_conflict_pairs(self.sample_df, model=self.model)
        res_scrambled = predict_conflict_pairs(scrambled_df, model=self.model)

        self.assertEqual(list(scrambled_df.columns), original_cols_snapshot)
        pd.testing.assert_frame_equal(res_canonical, res_scrambled)

        # Opción include_features=True y require_ids=False
        only_features_df = self.sample_df[FEATURE_COLUMNS].copy()
        res_no_ids = predict_conflict_pairs(
            only_features_df, model=self.model, require_ids=False, include_features=True
        )
        self.assertEqual(
            list(res_no_ids.columns),
            FEATURE_COLUMNS + [PROBABILITY_COLUMN, PREDICTION_COLUMN],
        )

    def test_05_invalid_inputs_validation(self) -> None:
        """Verifica que todas las entradas inválidas lancen las excepciones esperadas."""
        # No es DataFrame
        with self.assertRaises(TypeError):
            predict_conflict_pairs([{"a": 1}], model=self.model)  # type: ignore[arg-type]

        # DataFrame vacío
        with self.assertRaises(ValueError):
            predict_conflict_pairs(self.sample_df.iloc[0:0], model=self.model)

        # Falta una feature requerida
        missing_feat_df = self.sample_df.drop(columns=["jaccard_historico"])
        with self.assertRaises(ValueError):
            predict_conflict_pairs(missing_feat_df, model=self.model)

        # Falta columna ID cuando require_ids=True
        missing_id_df = self.sample_df.drop(columns=["materia_id_1"])
        with self.assertRaises(ValueError):
            predict_conflict_pairs(missing_id_df, model=self.model, require_ids=True)

        # Valor nulo (NaN) en feature
        null_df = self.sample_df.copy()
        null_df.loc[0, "cant_inscriptos_hist_1"] = np.nan
        with self.assertRaises(ValueError):
            predict_conflict_pairs(null_df, model=self.model)

        # Valor infinito (np.inf) en feature
        inf_df = self.sample_df.copy()
        inf_df["jaccard_historico"] = [np.inf, 0.1]
        with self.assertRaises(ValueError):
            predict_conflict_pairs(inf_df, model=self.model)

        # Valor inválido en columna booleana
        bad_bool_df = self.sample_df.copy()
        bad_bool_df["mismo_anio"] = ["invalido", "True"]
        with self.assertRaises(ValueError):
            predict_conflict_pairs(bad_bool_df, model=self.model)

        # String "False" debe parsearse como 0 (no como 1)
        str_bool_df = self.sample_df.copy()
        str_bool_df["mismo_departamento"] = ["False", "True"]
        X_clean, _ = validate_and_prepare_features(str_bool_df)
        self.assertEqual(X_clean["mismo_departamento"].tolist(), [0, 1])

        # Valor no numérico en columna numérica
        bad_num_df = self.sample_df.copy()
        bad_num_df["cant_inscriptos_hist_1"] = ["abc", 10]
        with self.assertRaises(ValueError):
            predict_conflict_pairs(bad_num_df, model=self.model)

        # Valor booleano en columna numérica de conteo
        bool_in_num_df = self.sample_df.copy()
        bool_in_num_df["cant_inscriptos_hist_1"] = [True, False]
        with self.assertRaises(ValueError):
            predict_conflict_pairs(bool_in_num_df, model=self.model)

        # Conteo histórico negativo (< 0)
        neg_count_df = self.sample_df.copy()
        neg_count_df.loc[0, "cant_estudiantes_compartidos_historico"] = -1
        with self.assertRaises(ValueError):
            predict_conflict_pairs(neg_count_df, model=self.model)

        # jaccard_historico fuera de [0, 1]
        for bad_jaccard in (-0.01, 1.05):
            bad_jac_df = self.sample_df.copy()
            bad_jac_df.loc[0, "jaccard_historico"] = bad_jaccard
            with self.assertRaises(ValueError):
                predict_conflict_pairs(bad_jac_df, model=self.model)

        # Cadena vacía en columna identificadora
        empty_id_df = self.sample_df.copy()
        empty_id_df.loc[0, "materia_id_1"] = "   "
        with self.assertRaises(ValueError):
            predict_conflict_pairs(empty_id_df, model=self.model)

        # Auto-par (misma materia en ambos extremos)
        self_pair_df = self.sample_df.copy()
        self_pair_df.loc[0, "materia_id_2"] = self_pair_df.loc[0, "materia_id_1"]
        with self.assertRaises(ValueError):
            predict_conflict_pairs(self_pair_df, model=self.model)

        # Par duplicado directo (u, v) repetido
        dup_direct_df = pd.concat([self.sample_df, self.sample_df.iloc[[0]]], ignore_index=True)
        with self.assertRaises(ValueError):
            predict_conflict_pairs(dup_direct_df, model=self.model)

        # Par duplicado simétrico (u, v) y (v, u)
        sym_row = self.sample_df.iloc[[0]].copy()
        sym_row["materia_id_1"] = self.sample_df.loc[0, "materia_id_2"]
        sym_row["materia_id_2"] = self.sample_df.loc[0, "materia_id_1"]
        dup_sym_df = pd.concat([self.sample_df, sym_row], ignore_index=True)
        with self.assertRaises(ValueError):
            predict_conflict_pairs(dup_sym_df, model=self.model)

        # Umbral inválido
        for invalid_th in (0.0, 1.0, -0.5, 1.5, float("nan"), float("inf"), True):
            with self.assertRaises(ValueError):
                predict_conflict_pairs(
                    self.sample_df, model=self.model, threshold=invalid_th  # type: ignore[arg-type]
                )

    def test_06_reproducibility_against_official_test_predictions(self) -> None:
        """Verifica que la función reproduzca exactamente las predicciones oficiales de test."""
        df_test = pd.read_csv(TEST_FULL_PATH)
        df_official = pd.read_csv(
            RESULTS_PREDICTIONS_DIR / "test_predictions_all_models.csv"
        )

        out = predict_conflict_pairs(df_test, model=self.model)

        # Coincidencia 100% en las 65.059 filas (IDs y clasificación binaria con umbral 0.21)
        self.assertEqual(len(out), 65059)
        self.assertTrue((out["materia_id_1"] == df_official["materia_id_1"]).all())
        self.assertTrue((out["materia_id_2"] == df_official["materia_id_2"]).all())
        self.assertTrue(
            np.array_equal(
                out[PREDICTION_COLUMN].to_numpy(),
                df_official["pred_KNN_Full_Unweighted_thVal"].to_numpy(),
            )
        )
        # Coincidencia de probabilidades en las 65.059 filas (el CSV oficial redondea a 6 decimales)
        max_diff = np.max(
            np.abs(
                out[PROBABILITY_COLUMN].to_numpy()
                - df_official["prob_KNN_Full_Unweighted"].to_numpy()
            )
        )
        self.assertLessEqual(max_diff, 5e-7)

    def test_07_working_directory_independence(self) -> None:
        """Verifica que las rutas por defecto funcionen desde cualquier directorio de trabajo."""
        original_cwd = Path.cwd()
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                os.chdir(tmpdir)
                out = predict_conflict_pairs(self.sample_df)
                self.assertEqual(len(out), 2)
                self.assertEqual(out[PREDICTION_COLUMN].tolist(), [0, 1])
        finally:
            os.chdir(original_cwd)

    def test_08_decoupling_and_single_source_of_truth(self) -> None:
        """Verifica que importar src.inference NO cargue src.models.compare_models."""
        check_code = (
            "import sys; "
            "import src.inference; "
            "assert 'src.models.compare_models' not in sys.modules, "
            "'src.models.compare_models fue importado innecesariamente'"
        )
        proc = subprocess.run(
            [sys.executable, "-c", check_code],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, msg=proc.stderr)


if __name__ == "__main__":
    unittest.main()
