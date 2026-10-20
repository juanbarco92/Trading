import unittest
import pandas as pd
import numpy as np

from src.features.volatility_features import (
    compute_normalized_atr,
    compute_bollinger_features,
    compute_support_resistance_distances,
    compute_volatility_features,
)
from src.data_extraction.dataset_builder import load_universe_data


class TestVolatilityFeatures(unittest.TestCase):
    def setUp(self):
        # Create synthetic 60-bar trending hourly data
        timestamps = pd.date_range("2026-10-01", periods=60, freq="1h", tz="UTC")
        n = len(timestamps)

        self.sample_df = pd.DataFrame({
            "timestamp": timestamps,
            "ticker": ["TEST"] * n,
            "open": [100.0 + i for i in range(n)],
            "high": [102.0 + i for i in range(n)],
            "low": [99.0 + i for i in range(n)],
            "close": [101.0 + i for i in range(n)],
        })

    def test_compute_normalized_atr(self):
        df_atr = compute_normalized_atr(self.sample_df)
        self.assertIn("atr14", df_atr.columns)
        self.assertIn("atr_pct", df_atr.columns)
        self.assertIn("vol_regime", df_atr.columns)
        self.assertIn("is_vol_compression", df_atr.columns)
        self.assertIn("is_vol_expansion", df_atr.columns)

        # ATR% should be positive
        self.assertTrue((df_atr["atr_pct"] > 0).all())
        # No NaNs in generated features
        self.assertEqual(df_atr["atr_pct"].isna().sum(), 0)

    def test_compute_bollinger_features(self):
        df_bb = compute_bollinger_features(self.sample_df)
        self.assertIn("bb_percent_b", df_bb.columns)
        self.assertIn("bb_bandwidth", df_bb.columns)
        self.assertIn("is_bb_squeeze", df_bb.columns)

        # %B should be finite and within clipped boundaries
        self.assertTrue((df_bb["bb_percent_b"] >= -0.5).all())
        self.assertTrue((df_bb["bb_percent_b"] <= 1.5).all())
        self.assertTrue((df_bb["bb_bandwidth"] >= 0).all())

    def test_support_resistance_distances(self):
        df_sr = compute_support_resistance_distances(self.sample_df)
        self.assertIn("dist_to_res20_atr", df_sr.columns)
        self.assertIn("dist_to_sup20_atr", df_sr.columns)
        self.assertIn("near_resistance_breakout", df_sr.columns)
        self.assertIn("near_support_bounce", df_sr.columns)

        # In an uptrend, price is closer to resistance than support
        last_row = df_sr.iloc[-1]
        self.assertTrue(np.isfinite(last_row["dist_to_res20_atr"]))
        self.assertTrue(np.isfinite(last_row["dist_to_sup20_atr"]))

    def test_universe_integration(self):
        # Run on real universe slice (MELI and SPY)
        universe = load_universe_data(candlesize="1hour")
        sample_slice = universe[universe["ticker"].isin(["MELI", "SPY"])].copy()

        result = compute_volatility_features(sample_slice)

        expected_features = [
            "atr_pct", "vol_regime", "is_vol_compression", "is_vol_expansion",
            "bb_percent_b", "bb_bandwidth", "is_bb_squeeze",
            "dist_to_res20_atr", "dist_to_sup20_atr", "near_resistance_breakout", "near_support_bounce"
        ]
        for feat in expected_features:
            self.assertIn(feat, result.columns)
            self.assertEqual(result[feat].isna().sum(), 0, f"Feature {feat} contains NaN values!")
            self.assertFalse(np.isinf(result[feat]).any(), f"Feature {feat} contains Inf values!")


if __name__ == "__main__":
    unittest.main()
