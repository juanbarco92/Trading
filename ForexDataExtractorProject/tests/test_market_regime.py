import unittest
import pandas as pd
import numpy as np

from src.features.market_regime import (
    compute_benchmark_regime,
    compute_relative_strength,
    compute_market_regime_features,
)
from src.data_extraction.dataset_builder import load_universe_data


class TestMarketRegime(unittest.TestCase):
    def setUp(self):
        # Create synthetic data with SPY and a stock (e.g. MELI) over 50 bars
        timestamps = pd.date_range("2026-10-01", periods=50, freq="1h", tz="UTC")
        n = len(timestamps)

        # SPY in steady uptrend
        spy_df = pd.DataFrame({
            "timestamp": timestamps,
            "ticker": ["SPY"] * n,
            "close": [500.0 + (i * 0.5) for i in range(n)],
        })

        # MELI outperforming SPY (stronger uptrend)
        meli_df = pd.DataFrame({
            "timestamp": timestamps,
            "ticker": ["MELI"] * n,
            "close": [1500.0 + (i * 3.0) for i in range(n)],
        })

        self.sample_df = pd.concat([spy_df, meli_df], ignore_index=True)

    def test_compute_benchmark_regime(self):
        spy_slice = self.sample_df[self.sample_df["ticker"] == "SPY"].copy()
        regime = compute_benchmark_regime(spy_slice)

        self.assertIn("spy_ema_50", regime.columns)
        self.assertIn("is_market_bullish", regime.columns)
        self.assertIn("spy_dist_ema50_pct", regime.columns)
        self.assertEqual(len(regime), 50)

    def test_relative_strength(self):
        result = compute_market_regime_features(self.sample_df, benchmark_symbol="SPY")

        expected_cols = [
            "spy_close", "is_market_bullish", "is_market_bearish",
            "stock_ret_1d", "alpha_1d", "alpha_5d",
            "rs_ratio", "rs_momentum_5d", "is_outperforming_spy", "rs_divergence_bullish"
        ]
        for col in expected_cols:
            self.assertIn(col, result.columns)

        # MELI should outperform SPY because its growth rate is higher
        meli_res = result[result["ticker"] == "MELI"].iloc[-5:]
        self.assertTrue((meli_res["alpha_1d"] >= 0).all())

        # SPY's own alpha should be 0.0
        spy_res = result[result["ticker"] == "SPY"].iloc[-5:]
        self.assertTrue((spy_res["alpha_1d"] == 0.0).all())

    def test_universe_integration(self):
        # Run on a slice of real universe data (SPY, QQQ, MELI, NVDA)
        universe = load_universe_data(candlesize="1hour")
        sample_slice = universe[universe["ticker"].isin(["SPY", "QQQ", "MELI", "NVDA"])].copy()

        result = compute_market_regime_features(sample_slice, benchmark_symbol="SPY")

        self.assertEqual(len(result), len(sample_slice))
        # Ensure no NaN values in generated features
        check_cols = ["alpha_1d", "alpha_5d", "is_market_bullish", "is_outperforming_spy"]
        for c in check_cols:
            self.assertEqual(result[c].isna().sum(), 0, f"Column {c} contains NaN values!")


if __name__ == "__main__":
    unittest.main()
