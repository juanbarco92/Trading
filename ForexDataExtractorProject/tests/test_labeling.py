"""
test_labeling.py - Pruebas unitarias para TICK-07 (Triple Barrier) y TICK-08 (Fricción Sintética)
"""

import unittest
import numpy as np
import pandas as pd
from src.models.labeling import (
    generate_triple_barrier_labels,
    compute_atr_series,
    _label_single_series,
)


class TestTripleBarrierLabeling(unittest.TestCase):
    def setUp(self):
        # Generar un DataFrame base sintético con 40 barras
        self.n_bars = 40
        self.horizon = 10
        self.dates = pd.date_range("2024-01-01", periods=self.n_bars, freq="1h")
        self.base_price = 100.0

    def test_take_profit_hit(self):
        """Verifica que un trade que sube directamente alcanza TP (+1 / Target 1)."""
        df = pd.DataFrame({
            "open": [100.0] * self.n_bars,
            "high": [100.5] * self.n_bars,
            "low": [99.5] * self.n_bars,
            "close": [100.0] * self.n_bars,
            "atr14": [2.0] * self.n_bars,
        })
        # Forzar que en la barra 3 (paso 3 tras la entrada en 0) el High toque el TP (100 + 3*2 = 106)
        df.loc[3, "high"] = 110.0
        df.loc[3, "close"] = 108.0

        res = generate_triple_barrier_labels(
            df, k_tp=3.0, k_sl=1.5, horizon_bars=self.horizon, friction_pct=0.0005
        )

        # En la barra 0, el trade debió salir en la barra 3 por TP
        self.assertEqual(res.loc[0, "target_triple_barrier"], 1.0)
        self.assertEqual(res.loc[0, "target_quality_trade"], 1.0)
        self.assertEqual(res.loc[0, "trade_exit_type"], "TP")
        self.assertEqual(res.loc[0, "trade_exit_bar"], 3.0)
        self.assertGreater(res.loc[0, "trade_net_return"], 0.0)

    def test_stop_loss_hit(self):
        """Verifica que un trade que cae directamente alcanza SL (-1 / Target 0)."""
        df = pd.DataFrame({
            "open": [100.0] * self.n_bars,
            "high": [100.5] * self.n_bars,
            "low": [99.5] * self.n_bars,
            "close": [100.0] * self.n_bars,
            "atr14": [2.0] * self.n_bars,
        })
        # Forzar caída en barra 2 por debajo de SL (100 - 1.5*2 = 97)
        df.loc[2, "low"] = 95.0
        df.loc[2, "close"] = 96.0

        res = generate_triple_barrier_labels(
            df, k_tp=3.0, k_sl=1.5, horizon_bars=self.horizon, friction_pct=0.0005
        )

        self.assertEqual(res.loc[0, "target_triple_barrier"], -1.0)
        self.assertEqual(res.loc[0, "target_quality_trade"], 0.0)
        self.assertEqual(res.loc[0, "trade_exit_type"], "SL")
        self.assertEqual(res.loc[0, "trade_exit_bar"], 2.0)
        self.assertLess(res.loc[0, "trade_net_return"], 0.0)

    def test_vertical_time_barrier_expiration(self):
        """Verifica que si el precio lateraliza, vence el tiempo (0 / Target 0)."""
        df = pd.DataFrame({
            "open": [100.0] * self.n_bars,
            "high": [100.5] * self.n_bars,
            "low": [99.5] * self.n_bars,
            "close": [100.0] * self.n_bars,
            "atr14": [2.0] * self.n_bars,
        })

        res = generate_triple_barrier_labels(
            df, k_tp=3.0, k_sl=1.5, horizon_bars=self.horizon, friction_pct=0.0005
        )

        self.assertEqual(res.loc[0, "target_triple_barrier"], 0.0)
        self.assertEqual(res.loc[0, "target_quality_trade"], 0.0)
        self.assertEqual(res.loc[0, "trade_exit_type"], "TIME")
        self.assertEqual(res.loc[0, "trade_exit_bar"], self.horizon)

    def test_conservative_intrabar_conflict(self):
        """Verifica que ante toque simultáneo de TP y SL en la misma vela, asume SL primero."""
        df = pd.DataFrame({
            "open": [100.0] * self.n_bars,
            "high": [100.5] * self.n_bars,
            "low": [99.5] * self.n_bars,
            "close": [100.0] * self.n_bars,
            "atr14": [2.0] * self.n_bars,
        })
        # Barra 1 toca tanto TP (>= 106) como SL (<= 97)
        df.loc[1, "high"] = 115.0
        df.loc[1, "low"] = 85.0

        # Con regla conservadora
        res_cons = generate_triple_barrier_labels(
            df, k_tp=3.0, k_sl=1.5, horizon_bars=self.horizon, conservative_intrabar=True
        )
        self.assertEqual(res_cons.loc[0, "target_triple_barrier"], -1.0)
        self.assertEqual(res_cons.loc[0, "trade_exit_type"], "SL")

        # Sin regla conservadora
        res_aggr = generate_triple_barrier_labels(
            df, k_tp=3.0, k_sl=1.5, horizon_bars=self.horizon, conservative_intrabar=False
        )
        self.assertEqual(res_aggr.loc[0, "target_triple_barrier"], 1.0)
        self.assertEqual(res_aggr.loc[0, "trade_exit_type"], "TP")

    def test_synthetic_friction_deduction(self):
        """TICK-08: Verifica que el retorno neto es estrictamente menor al retorno bruto."""
        df = pd.DataFrame({
            "open": [100.0] * self.n_bars,
            "high": [100.5] * self.n_bars,
            "low": [99.5] * self.n_bars,
            "close": [100.0] * self.n_bars,
            "atr14": [2.0] * self.n_bars,
        })
        df.loc[2, "high"] = 110.0

        res = generate_triple_barrier_labels(
            df, k_tp=3.0, k_sl=1.5, horizon_bars=self.horizon, friction_pct=0.001
        )

        gross = res.loc[0, "trade_gross_return"]
        net = res.loc[0, "trade_net_return"]
        # El retorno neto debe ser menor al bruto aproximadamente por el spread de ida y vuelta (~0.2%)
        self.assertLess(net, gross)
        self.assertAlmostEqual(gross - net, 0.2, delta=0.05)

    def test_gap_down_slippage_on_stop(self):
        """Verifica penalización por gap bajista en la apertura de la vela de salida."""
        df = pd.DataFrame({
            "open": [100.0] * self.n_bars,
            "high": [100.5] * self.n_bars,
            "low": [99.5] * self.n_bars,
            "close": [100.0] * self.n_bars,
            "atr14": [2.0] * self.n_bars,
        })
        # SL normal está en 97. En la barra 2 abre en 94 (gap down severo)
        df.loc[2, "open"] = 94.0
        df.loc[2, "high"] = 94.5
        df.loc[2, "low"] = 92.0
        df.loc[2, "close"] = 93.0

        res = generate_triple_barrier_labels(
            df, k_tp=3.0, k_sl=1.5, horizon_bars=self.horizon, friction_pct=0.0
        )
        # El precio de salida de SL debió penalizarse al precio de apertura (94.0) en lugar del 97.0
        self.assertAlmostEqual(res.loc[0, "trade_exit_price"], 94.0, places=2)

    def test_multi_ticker_isolation(self):
        """Verifica que el procesamiento por ticker no tiene fuga de información entre activos."""
        df1 = pd.DataFrame({
            "ticker": ["SPY"] * 30,
            "open": [100.0] * 30,
            "high": [100.5] * 30,
            "low": [99.5] * 30,
            "close": [100.0] * 30,
            "atr14": [2.0] * 30,
        })
        df1.loc[3, "high"] = 110.0  # SPY toca TP

        df2 = pd.DataFrame({
            "ticker": ["MELI"] * 30,
            "open": [100.0] * 30,
            "high": [100.5] * 30,
            "low": [99.5] * 30,
            "close": [100.0] * 30,
            "atr14": [2.0] * 30,
        })
        df2.loc[3, "low"] = 90.0  # MELI toca SL

        combined = pd.concat([df1, df2], ignore_index=True)
        res = generate_triple_barrier_labels(combined, horizon_bars=10)

        # SPY barra 0 debió ser TP
        spy_row0 = res[res["ticker"] == "SPY"].iloc[0]
        self.assertEqual(spy_row0["target_triple_barrier"], 1.0)
        self.assertEqual(spy_row0["trade_exit_type"], "TP")

        # MELI barra 0 debió ser SL
        meli_row0 = res[res["ticker"] == "MELI"].iloc[0]
        self.assertEqual(meli_row0["target_triple_barrier"], -1.0)
        self.assertEqual(meli_row0["trade_exit_type"], "SL")

        # Últimas horizon_bars de cada activo deben ser NaN
        spy_tail = res[res["ticker"] == "SPY"].iloc[-10:]
        meli_tail = res[res["ticker"] == "MELI"].iloc[-10:]
        self.assertTrue(spy_tail["target_triple_barrier"].isna().all())
        self.assertTrue(meli_tail["target_triple_barrier"].isna().all())

    def test_missing_columns_raises_error(self):
        """Verifica que falten columnas obligatorias arroje ValueError."""
        df = pd.DataFrame({"close": [100.0, 101.0]})
        with self.assertRaises(ValueError):
            generate_triple_barrier_labels(df)


if __name__ == "__main__":
    unittest.main()
