import unittest
import numpy as np
import pandas as pd
from src.utils.exploration_utils import generate_triple_barrier_targets, engineer_curated_features

class TestCuratedPipeline(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        n = 150
        dates = pd.date_range('2025-01-01', periods=n, freq='5min', tz='UTC')
        close_prices = 1.1000 + np.cumsum(np.random.randn(n) * 0.0003)
        self.df = pd.DataFrame({
            'datetime': dates,
            'open': close_prices - 0.0001,
            'high': close_prices + 0.0005,
            'low': close_prices - 0.0005,
            'close': close_prices
        })

    def test_engineer_curated_features(self):
        feat_df = engineer_curated_features(self.df)
        self.assertIn('ATR_14', feat_df.columns)
        self.assertIn('rsi_14', feat_df.columns)
        self.assertIn('bb_percent_b', feat_df.columns)
        self.assertIn('macd_hist_norm', feat_df.columns)
        self.assertIn('dist_to_ema20_atr', feat_df.columns)
        self.assertIn('is_london_session', feat_df.columns)
        self.assertEqual(len(feat_df), len(self.df))

    def test_generate_triple_barrier_targets(self):
        feat_df = engineer_curated_features(self.df)
        labeled_df = generate_triple_barrier_targets(
            feat_df,
            atr_col='ATR_14',
            k_tp=2.5,
            k_sl=1.0,
            future_periods=10,
            spread_pips=1.0
        )
        self.assertIn('target_buy', labeled_df.columns)
        self.assertIn('target_sell', labeled_df.columns)
        self.assertIn('multiclass_target', labeled_df.columns)
        self.assertIn('realized_pips_buy', labeled_df.columns)
        self.assertIn('realized_pips_sell', labeled_df.columns)
        
        # Verify tail rows are undefined
        self.assertTrue((labeled_df['multiclass_target'].iloc[-10:] == 'undefined').all())

if __name__ == '__main__':
    unittest.main()
