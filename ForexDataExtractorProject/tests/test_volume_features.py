import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.features.volume_features import (
    compute_vwap,
    compute_rvol,
    compute_volume_momentum,
    compute_absorption_and_climax,
    compute_volume_features,
)
from src.data_extraction.dataset_builder import load_universe_data


class TestVolumeFeatures(unittest.TestCase):
    def setUp(self):
        # Create synthetic 2-day hourly data (7 bars per day, standard US market)
        base_time = pd.Timestamp("2026-10-01 13:30:00", tz="UTC")  # 9:30 AM EST
        timestamps = []
        for d in range(2):
            for h in range(7):
                timestamps.append(base_time + pd.Timedelta(days=d, hours=h))

        n = len(timestamps)
        self.sample_df = pd.DataFrame({
            "timestamp": timestamps,
            "ticker": ["TEST"] * n,
            "open": [100.0 + i for i in range(n)],
            "high": [102.0 + i for i in range(n)],
            "low": [99.0 + i for i in range(n)],
            "close": [101.0 + i for i in range(n)],
            "volume": [1000.0 * (1 + (i % 7)) for i in range(n)],
        })

    def test_compute_vwap(self):
        df_vwap = compute_vwap(self.sample_df)
        self.assertIn("vwap", df_vwap.columns)
        self.assertIn("vwap_dist_pct", df_vwap.columns)
        self.assertIn("is_above_vwap", df_vwap.columns)

        # Check that VWAP is between Low and High
        for _, row in df_vwap.iterrows():
            self.assertGreaterEqual(row["vwap"], row["low"] - 1.0)
            self.assertLessEqual(row["vwap"], row["high"] + 1.0)

        # Verify daily reset: Bar 7 (start of day 2) should have typical price == VWAP
        day2_bar0 = df_vwap.iloc[7]
        expected_tp = (day2_bar0["high"] + day2_bar0["low"] + day2_bar0["close"]) / 3.0
        self.assertAlmostEqual(day2_bar0["vwap"], expected_tp, places=4)

    def test_compute_rvol(self):
        df_rvol = compute_rvol(self.sample_df, window=5)
        self.assertIn("rvol", df_rvol.columns)
        self.assertIn("rvol_surge", df_rvol.columns)
        self.assertIn("rvol_extreme", df_rvol.columns)
        self.assertFalse(df_rvol["rvol"].isna().any())
        self.assertTrue((df_rvol["rvol"] >= 0).all())

    def test_compute_volume_momentum(self):
        df_mom = compute_volume_momentum(self.sample_df, window=5)
        self.assertIn("volume_zscore", df_mom.columns)
        self.assertIn("volume_ratio_ema", df_mom.columns)
        self.assertFalse(df_mom["volume_zscore"].isna().any())

    def test_compute_absorption_and_climax(self):
        # Create a hammer candle with high volume
        custom_df = self.sample_df.copy()
        custom_df.loc[5, "rvol"] = 3.0
        custom_df.loc[5, "open"] = 105.0
        custom_df.loc[5, "close"] = 106.0
        custom_df.loc[5, "high"] = 106.5
        custom_df.loc[5, "low"] = 98.0  # Big lower wick rejection

        df_abs = compute_absorption_and_climax(custom_df)
        self.assertIn("is_bullish_absorption", df_abs.columns)
        self.assertIn("is_bearish_exhaustion", df_abs.columns)
        # Bar 5 should have lower wick ratio > 0.40
        self.assertEqual(df_abs.loc[5, "is_bullish_absorption"], 1)

    def test_universe_integration(self):
        # Test on real universe dataset (small slice of SPY and MELI)
        universe = load_universe_data(candlesize="1hour")
        sample_universe = universe[universe["ticker"].isin(["SPY", "MELI"])].head(200).copy()

        result = compute_volume_features(sample_universe)

        expected_features = [
            "vwap", "vwap_dist_pct", "is_above_vwap",
            "rvol", "rvol_surge", "rvol_extreme",
            "volume_zscore", "volume_ratio_ema",
            "is_bullish_absorption", "is_bearish_exhaustion", "volume_per_range"
        ]
        for feat in expected_features:
            self.assertIn(feat, result.columns)
            self.assertEqual(result[feat].isna().sum(), 0, f"Feature {feat} contains NaN values!")


if __name__ == "__main__":
    unittest.main()
