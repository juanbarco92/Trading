"""
test_train_swing_model.py - Pruebas unitarias para TICK-10 (Entrenamiento y Calibración LightGBM).
"""

import unittest
import numpy as np
import pandas as pd
from src.models.train_swing_model import (
    get_base_lgbm_classifier,
    evaluate_out_of_sample_cv,
    train_final_calibrated_model,
    DEFAULT_FEATURES,
)


class TestTrainSwingModel(unittest.TestCase):
    def setUp(self):
        # Crear dataset sintético pequeño con 60 observaciones y 2 tickers
        np.random.seed(42)
        n = 60
        dates = pd.date_range("2024-01-01", periods=n // 2, freq="1h")
        timestamps = list(dates) + list(dates)

        # 20 características
        feature_data = {feat: np.random.randn(n) for feat in DEFAULT_FEATURES}
        self.X = pd.DataFrame(feature_data)

        # Target con señal sintética (si feature 'vwap_dist_pct' > 0 -> más probable target=1)
        prob = 1 / (1 + np.exp(-self.X["vwap_dist_pct"].values))
        self.y = pd.Series((np.random.rand(n) < prob).astype(int))

        self.t0 = pd.Series(timestamps)
        # Exit en 2 horas
        self.t1 = self.t0 + pd.Timedelta(hours=2)

        self.meta = pd.DataFrame({
            "ticker": ["AAPL"] * (n // 2) + ["MELI"] * (n // 2),
            "timestamp": timestamps,
            "trade_net_return": np.where(self.y == 1, 3.2, -1.9),
            "trade_gross_return": np.where(self.y == 1, 3.3, -1.8),
            "trade_exit_type": np.where(self.y == 1, "TP", "SL"),
            "trade_exit_bar": [2.0] * n,
            "close": [150.0] * n,
        })

    def test_get_base_lgbm_classifier(self):
        """Verifica la configuración de hiperparámetros del clasificador."""
        clf = get_base_lgbm_classifier(random_state=42)
        self.assertEqual(clf.objective, "binary")
        self.assertEqual(clf.max_depth, 5)
        self.assertLessEqual(clf.learning_rate, 0.05)

    def test_evaluate_out_of_sample_cv_synthetic(self):
        """Verifica la ejecución de CV con cálculo de probabilidades Out-of-Sample."""
        results_df, metrics = evaluate_out_of_sample_cv(
            self.X, self.y, self.t0, self.t1, self.meta, n_splits=2, pct_embargo=0.0, min_child_samples=5
        )

        self.assertIn("prob_calibrated", results_df.columns)
        self.assertTrue((results_df["prob_calibrated"] >= 0.0).all())
        self.assertTrue((results_df["prob_calibrated"] <= 1.0).all())

        self.assertIn("global_roc_auc", metrics)
        self.assertIn("threshold_analysis", metrics)
        self.assertGreaterEqual(metrics["global_roc_auc"], 0.0)

    def test_train_final_calibrated_model(self):
        """Verifica el entrenamiento y extracción de importancia de variables."""
        cal_model, feat_imp = train_final_calibrated_model(
            self.X, self.y, self.t0, self.t1, pct_embargo=0.0, min_child_samples=5, save_path="models/test_model.joblib"
        )

        # Probar predicción con modelo calibrado
        preds = cal_model.predict_proba(self.X[:5])
        self.assertEqual(preds.shape, (5, 2))
        np.testing.assert_allclose(preds.sum(axis=1), 1.0, atol=1e-5)

        # Verificar tabla de importancia de características
        self.assertEqual(len(feat_imp), len(DEFAULT_FEATURES))
        self.assertIn("gain_pct", feat_imp.columns)
        self.assertAlmostEqual(feat_imp["gain_pct"].sum(), 100.0, delta=0.1)

        # Limpiar archivo de prueba
        test_file = pd.Series([None])
        import os
        if os.path.exists("models/test_model.joblib"):
            os.remove("models/test_model.joblib")


if __name__ == "__main__":
    unittest.main()
