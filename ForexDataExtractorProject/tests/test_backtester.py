"""
test_backtester.py - Pruebas unitarias para TICK-11 (Motor de Backtesting y Métricas Institucionales).
"""

import unittest
import numpy as np
import pandas as pd
from src.models.backtester import (
    run_portfolio_backtest,
    compute_backtest_metrics,
    compute_spy_benchmark,
    Trade,
)


class TestBacktester(unittest.TestCase):
    def setUp(self):
        # Crear datos sintéticos para 2 activos (SPY y AAPL) a lo largo de 30 velas horarias
        dates = pd.date_range("2024-01-01 09:30:00", periods=30, freq="1h")

        spy_rows = []
        aapl_rows = []
        for i, dt in enumerate(dates):
            # SPY sube de forma constante de 400 a 430
            spy_rows.append({
                "timestamp": dt,
                "ticker": "SPY",
                "open": 400.0 + i,
                "high": 401.0 + i,
                "low": 399.0 + i,
                "close": 400.0 + i,
                "atr14": 4.0,
                "signal": False,
                "trade_exit_bar": 5,
                "trade_exit_price": 405.0 + i,
                "trade_exit_type": "TIME",
                "trade_net_return": 1.25,
            })
            # AAPL oscila
            aapl_rows.append({
                "timestamp": dt,
                "ticker": "AAPL",
                "open": 150.0,
                "high": 152.0,
                "low": 148.0,
                "close": 150.0,
                "atr14": 2.0,
                "signal": (i == 2),  # Señal en la barra 2
                "trade_exit_bar": 4,   # Sale 4 barras después (en barra 6)
                "trade_exit_price": 156.0,  # TP en 156
                "trade_exit_type": "TP",
                "trade_net_return": 3.9,
            })

        self.df = pd.DataFrame(spy_rows + aapl_rows).sort_values("timestamp").reset_index(drop=True)

    def test_fixed_dollar_risk_position_sizing(self):
        """Verifica que el dimensionamiento de posición en $1R produce la ganancia/pérdida esperada."""
        equity_df, closed_trades, metrics = run_portfolio_backtest(
            self.df,
            initial_capital=100_000.0,
            risk_per_trade_usd=1_000.0,
            max_open_positions=5,
            max_pos_capital_pct=0.60,
        )

        self.assertEqual(len(closed_trades), 1)
        t = closed_trades[0]
        self.assertEqual(t.ticker, "AAPL")
        self.assertEqual(t.exit_type, "TP")

        # Con ATR = 2.0, stop distance = 1.5 * 2.0 = 3.0.
        # Shares = 1000 / 3.0 = 333 acciones.
        # En TP gana ~6 USD por acción -> ~2000 USD (+2R)
        self.assertGreater(t.pnl_usd, 1500.0)
        self.assertAlmostEqual(t.r_multiple, 2.0, delta=0.2)

    def test_max_open_positions_concurrency_limit(self):
        """Verifica que no se abren más de max_open_positions simultáneas."""
        # Generar 10 tickers con señal en la misma barra
        dates = pd.date_range("2024-01-01 09:30:00", periods=10, freq="1h")
        rows = []
        for ticker_idx in range(10):
            t_name = f"TICKER_{ticker_idx}"
            for i, dt in enumerate(dates):
                rows.append({
                    "timestamp": dt,
                    "ticker": t_name,
                    "close": 100.0,
                    "atr14": 2.0,
                    "signal": (i == 0),  # Todos dan señal en la barra 0
                    "trade_exit_bar": 5,
                    "trade_exit_price": 105.0,
                    "trade_exit_type": "TP",
                    "trade_net_return": 4.9,
                })

        test_df = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)

        equity_df, closed_trades, metrics = run_portfolio_backtest(
            test_df,
            max_open_positions=3,  # Límite estricto de 3
        )

        # Máximo de posiciones abiertas no debe superar 3
        self.assertLessEqual(equity_df["open_positions_count"].max(), 3)
        self.assertEqual(len(closed_trades), 3)

    def test_no_duplicate_positions_on_same_ticker(self):
        """Verifica que no se duplica posición en un activo ya abierto."""
        dates = pd.date_range("2024-01-01 09:30:00", periods=10, freq="1h")
        rows = []
        for i, dt in enumerate(dates):
            rows.append({
                "timestamp": dt,
                "ticker": "AAPL",
                "close": 100.0,
                "atr14": 2.0,
                "signal": True,  # Da señal en todas las barras
                "trade_exit_bar": 6,  # Dura 6 barras
                "trade_exit_price": 106.0,
                "trade_exit_type": "TP",
                "trade_net_return": 5.0,
            })

        test_df = pd.DataFrame(rows)
        equity_df, closed_trades, metrics = run_portfolio_backtest(test_df)

        # En 10 barras con trade de 6 barras, solo debió entrar 1 vez (o 2 si da tiempo a reentrar)
        self.assertLessEqual(len(closed_trades), 2)
        self.assertLessEqual(equity_df["open_positions_count"].max(), 1)

    def test_metrics_calculation_integrity(self):
        """Verifica integridad de métricas calculadas."""
        _, _, metrics = run_portfolio_backtest(self.df)
        self.assertIn("sharpe_ratio", metrics)
        self.assertIn("max_drawdown_pct", metrics)
        self.assertIn("win_rate_pct", metrics)
        self.assertIn("profit_factor", metrics)
        self.assertIn("benchmark_spy", metrics)


if __name__ == "__main__":
    unittest.main()
