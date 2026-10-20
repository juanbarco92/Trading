import unittest
import pandas as pd
import numpy as np
from pathlib import Path

from src.features.feature_pipeline import (
    run_full_feature_pipeline,
    analyze_collinearity,
    clean_and_validate_features,
    CORE_FEATURE_NAMES,
)


class TestFeaturePipeline(unittest.TestCase):
    def test_core_features_list(self):
        self.assertEqual(len(CORE_FEATURE_NAMES), 20)
        self.assertIn("rvol", CORE_FEATURE_NAMES)
        self.assertIn("alpha_1d", CORE_FEATURE_NAMES)
        self.assertIn("atr_pct", CORE_FEATURE_NAMES)
        self.assertIn("dist_to_res20_atr", CORE_FEATURE_NAMES)

    def test_collinearity_detection(self):
        # Create synthetic data with 2 highly correlated features
        n = 100
        x1 = np.linspace(0, 10, n)
        x2 = x1 * 0.99 + np.random.normal(0, 0.01, n)  # Highly correlated with x1 (r ~ 0.99)
        x3 = np.random.normal(0, 1, n)  # Independent

        df = pd.DataFrame({"feat1": x1, "feat2": x2, "feat3": x3})
        _, high_corr = analyze_collinearity(df, ["feat1", "feat2", "feat3"], max_correlation=0.75)

        self.assertEqual(len(high_corr), 1)
        self.assertEqual(high_corr[0][0], "feat1")
        self.assertEqual(high_corr[0][1], "feat2")

    def test_clean_and_validate(self):
        # DataFrame with NaNs and Infs
        df = pd.DataFrame({
            "ticker": ["A", "A", "A", "B", "B"],
            "feat1": [1.0, np.nan, 3.0, np.inf, 2.0],
            "feat2": [True, False, True, False, True],
        })
        cleaned = clean_and_validate_features(df, ["feat1", "feat2"])

        self.assertEqual(cleaned["feat1"].isna().sum(), 0)
        self.assertFalse(np.isinf(cleaned["feat1"]).any())
        self.assertTrue(np.issubdtype(cleaned["feat2"].dtype, np.integer))

    def test_generated_features_parquet(self):
        parquet_path = Path("data/processed/features_universe_1hour.parquet")
        self.assertTrue(parquet_path.exists(), "El archivo de características no fue generado.")

        df = pd.read_parquet(parquet_path)
        self.assertGreater(len(df), 80000)

        # Verificar que contiene las 20 características
        for feat in CORE_FEATURE_NAMES:
            self.assertIn(feat, df.columns)
            self.assertEqual(df[feat].isna().sum(), 0, f"Columna {feat} contiene NaNs!")
            self.assertFalse(np.isinf(df[feat]).any(), f"Columna {feat} contiene Infs!")

        # Verificar colinealidad nula sobre el dataset real
        _, high_pairs = analyze_collinearity(df, CORE_FEATURE_NAMES, max_correlation=0.75)
        self.assertEqual(len(high_pairs), 0, f"Existen pares colineales en el dataset final: {high_pairs}")


if __name__ == "__main__":
    unittest.main()
