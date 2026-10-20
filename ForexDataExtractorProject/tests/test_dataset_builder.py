import unittest
import pandas as pd
from datetime import datetime
from src.data_extraction.dataset_builder import (
    parse_datetime_series,
    validate_price_integrity,
    load_universe_data,
)


class TestDatasetBuilder(unittest.TestCase):
    def test_parse_datetime_series(self):
        # Sample dates with different formatting
        sample_dates = pd.Series([
            "20261001  08:30:00",
            "20261001 09:30:00",
            "20261001 10:30:00 US/Eastern",
        ])
        parsed = parse_datetime_series(sample_dates)

        self.assertEqual(len(parsed), 3)
        self.assertIsNotNone(parsed.dt.tz)
        # Should be UTC
        self.assertEqual(str(parsed.dt.tz), "UTC")
        self.assertEqual(parsed.isna().sum(), 0)

    def test_validate_price_integrity(self):
        # Valid DataFrame
        df_valid = pd.DataFrame({
            "open": [100.0],
            "high": [105.0],
            "low": [99.0],
            "close": [102.0],
            "volume": [1000.0],
        })
        issues = validate_price_integrity(df_valid, "TEST")
        self.assertEqual(sum(issues.values()), 0)

        # Invalid DataFrame with High < Low
        df_invalid = pd.DataFrame({
            "open": [100.0],
            "high": [95.0],  # Invalid: high < open/low
            "low": [98.0],
            "close": [96.0],
            "volume": [-10.0],  # Invalid: negative volume
        })
        issues_invalid = validate_price_integrity(df_invalid, "TEST_BAD")
        self.assertGreater(sum(issues_invalid.values()), 0)
        self.assertEqual(issues_invalid["high_less_than_low"], 1)
        self.assertEqual(issues_invalid["negative_volume"], 1)

    def test_load_universe_data(self):
        df_universe = load_universe_data(candlesize="1hour")
        self.assertGreater(len(df_universe), 50000)
        self.assertIn("ticker", df_universe.columns)
        self.assertIn("timestamp", df_universe.columns)
        self.assertIn("close", df_universe.columns)
        self.assertIn("volume", df_universe.columns)

        unique_tickers = df_universe["ticker"].unique()
        self.assertIn("SPY", unique_tickers)
        self.assertIn("MELI", unique_tickers)
        self.assertIn("NVDA", unique_tickers)


if __name__ == "__main__":
    unittest.main()
