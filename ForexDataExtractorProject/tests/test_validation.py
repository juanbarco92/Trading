"""
test_validation.py - Pruebas unitarias para TICK-09 (Purged & Embargoed Cross-Validation).
"""

import unittest
import numpy as np
import pandas as pd
from src.models.validation import (
    PurgedKFold,
    PurgedWalkForwardCV,
    compute_t1_series,
    verify_no_leakage,
)


class TestPurgedCrossValidation(unittest.TestCase):
    def setUp(self):
        # Crear serie temporal sintética de 100 horas con 2 tickers
        dates = pd.date_range("2024-01-01 09:30:00", periods=50, freq="1h")
        df_aapl = pd.DataFrame({
            "timestamp": dates,
            "ticker": "AAPL",
            "trade_exit_bar": [3.0] * 50,
            "feature": np.random.randn(50),
            "target_quality_trade": np.random.randint(0, 2, 50),
        })
        df_meli = pd.DataFrame({
            "timestamp": dates,
            "ticker": "MELI",
            "trade_exit_bar": [5.0] * 50,
            "feature": np.random.randn(50),
            "target_quality_trade": np.random.randint(0, 2, 50),
        })
        self.df = pd.concat([df_aapl, df_meli], ignore_index=True)
        self.dates = dates

    def test_compute_t1_series(self):
        """Verifica el cálculo exacto de t1 según trade_exit_bar y ticker."""
        t1 = compute_t1_series(self.df)
        self.assertEqual(len(t1), len(self.df))

        # Fila 0 de AAPL entra en dates[0] y dura 3 barras -> sale en dates[3]
        self.assertEqual(t1.iloc[0], self.dates[3])
        # Fila 50 (Fila 0 de MELI) entra en dates[0] y dura 5 barras -> sale en dates[5]
        self.assertEqual(t1.iloc[50], self.dates[5])

    def test_purged_kfold_no_temporal_leakage(self):
        """Criterio de Aceptación DoD: Cero solapamiento temporal en cada fold de PurgedKFold."""
        cv = PurgedKFold(n_splits=5, pct_embargo=0.02)
        t0, t1 = cv._resolve_time_series(self.df)

        fold_count = 0
        for train_idx, test_idx in cv.split(self.df):
            fold_count += 1
            # Verificar auditoría matemática
            audit = verify_no_leakage(train_idx, test_idx, t0, t1)
            self.assertTrue(
                audit["is_leakage_free"],
                f"Fuga temporal detectada en fold {fold_count}: {audit}",
            )
            self.assertEqual(audit["temporal_overlap_count"], 0)
            self.assertEqual(audit["index_intersection"], 0)

        self.assertEqual(fold_count, 5)

    def test_purging_explicitly_drops_boundary_trade(self):
        """Verifica que un trade cuyo horizonte cruza la frontera del test es purgado."""
        # Fechas simples continuas
        dates = pd.date_range("2024-01-01", periods=20, freq="1h")
        df = pd.DataFrame({
            "timestamp": dates,
            "trade_exit_bar": [1.0] * 20,
        })
        # Forzar que la observación en índice 3 tenga duración de 10 barras (sale en barra 13)
        df.loc[3, "trade_exit_bar"] = 10.0

        # Particionar en 2 folds: Fold 0 (0-9), Fold 1 (10-19)
        cv = PurgedKFold(n_splits=2, pct_embargo=0.0)
        splits = list(cv.split(df))

        # En el split donde Fold 1 (índices 10 a 19) es el test:
        # La observación 3 de train termina en la fecha 13 (dentro del test).
        # Debe haber sido PURGADA del train.
        train_idx_f1, test_idx_f1 = splits[1]
        self.assertNotIn(3, train_idx_f1, "La observación 3 debía ser purgada del train.")

    def test_embargo_drops_post_test_samples(self):
        """Criterio de Aceptación DoD: Verifica que el embargo descarta muestras tras el test."""
        dates = pd.date_range("2024-01-01", periods=30, freq="1h")
        df = pd.DataFrame({
            "timestamp": dates,
            "trade_exit_bar": [1.0] * 30,
        })

        # Test en Fold 0 (primer tercio). Train en tercios posteriores.
        cv_no_embargo = PurgedKFold(n_splits=3, pct_embargo=0.0)
        cv_with_embargo = PurgedKFold(n_splits=3, bars_embargo=5)

        train_no_emb, _ = next(cv_no_embargo.split(df))
        train_with_emb, _ = next(cv_with_embargo.split(df))

        # El entrenamiento con embargo debe tener menos muestras que sin embargo
        self.assertLess(len(train_with_emb), len(train_no_emb))

    def test_purged_walk_forward_cv(self):
        """Verifica que WalkForward expande el train cronológicamente y purga la frontera."""
        cv = PurgedWalkForwardCV(n_splits=4)
        t0, t1 = cv._resolve_time_series(self.df)

        fold_count = 0
        prev_train_max_date = None
        for train_idx, test_idx in cv.split(self.df):
            fold_count += 1
            audit = verify_no_leakage(train_idx, test_idx, t0, t1)
            self.assertTrue(audit["is_leakage_free"])

            test_min_date = t0.iloc[test_idx].min()
            train_max_date = t0.iloc[train_idx].max()

            # En Walk-Forward, ningún dato de train puede ser posterior al inicio de test
            self.assertLess(train_max_date, test_min_date)

            # Además, las salidas de los trades de train deben ser anteriores a test_min_date
            train_max_exit = t1.iloc[train_idx].max()
            self.assertLess(train_max_exit, test_min_date)

        self.assertEqual(fold_count, 3)

    def test_multi_asset_synchronized_splitting(self):
        """Verifica que la partición es simultánea por fecha para todos los activos."""
        cv = PurgedKFold(n_splits=3)
        for train_idx, test_idx in cv.split(self.df):
            test_sub = self.df.iloc[test_idx]
            # Ambos tickers deben estar presentes en el conjunto de test
            self.assertIn("AAPL", test_sub["ticker"].values)
            self.assertIn("MELI", test_sub["ticker"].values)
            # Las fechas de AAPL en test deben ser idénticas a las fechas de MELI en test
            aapl_dates = set(test_sub[test_sub["ticker"] == "AAPL"]["timestamp"])
            meli_dates = set(test_sub[test_sub["ticker"] == "MELI"]["timestamp"])
            self.assertEqual(aapl_dates, meli_dates)


if __name__ == "__main__":
    unittest.main()
