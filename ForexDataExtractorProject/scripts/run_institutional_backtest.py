"""
run_institutional_backtest.py - Ejecución y Comparativa de Estrategias Swing Trading (2022-2026).
"""

import sys
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from src.models.backtester import run_portfolio_backtest


def main():
    print("=" * 85)
    print("BACKTESTING VECTORIZADO DE CARTERA SWING TRADING (TICK-11: 2022 - 2026)")
    print("=" * 85)

    data_path = Path("data/processed/labeled_universe_1hour.parquet")
    oos_path = Path("data/processed/oos_predictions.parquet")

    if not data_path.exists():
        print(f"Error: {data_path} no existe.")
        return

    df = pd.read_parquet(data_path).dropna(subset=["target_quality_trade"]).reset_index(drop=True)

    # Fusionar probabilidades Out-of-Sample si existen
    if oos_path.exists():
        oos_df = pd.read_parquet(oos_path)[["timestamp", "ticker", "prob_calibrated"]]
        df = df.merge(oos_df, on=["timestamp", "ticker"], how="left")
    else:
        df["prob_calibrated"] = 0.33

    print(f"Dataset cargado: {len(df):,} velas horarias en 10 activos (Dec 2021 a Oct 2026).")
    print(f"Capital Inicial: $100,000 USD | Riesgo Fijo por Operación: $1,000 USD (1R) | Máx Posiciones: 5\n")

    # Definir Estrategias Cuantitativas a Comparar
    strategies = {
        "1. Breakout Institucional + RVOL > 1.2": (
            (df["dist_to_res20_atr"] < 0.5) &
            (df["is_market_bullish"] == 1) &
            (df["rvol"] > 1.2)
        ),
        "2. Soporte + Compresión + Alpha > 0": (
            (df["dist_to_sup20_atr"] < 1.0) &
            (df["is_vol_compression"] == 1) &
            (df["alpha_5d"] > 0)
        ),
        "3. Meta-Model ML (Top 5% Convicción)": (
            df["prob_calibrated"] >= df["prob_calibrated"].quantile(0.95)
        ),
        "4. Híbrida: Breakout + Filtro ML": (
            (df["dist_to_res20_atr"] < 0.6) &
            (df["is_market_bullish"] == 1) &
            (df["rvol"] > 1.1) &
            (df["prob_calibrated"] >= df["prob_calibrated"].median())
        ),
    }

    summary_rows = []
    equity_curves = {}
    all_metrics = {}

    for strat_name, signal_series in strategies.items():
        df_strat = df.copy()
        df_strat["signal"] = signal_series

        equity_df, trades, metrics = run_portfolio_backtest(
            df_strat,
            initial_capital=100_000.0,
            risk_per_trade_usd=1_000.0,
            max_open_positions=5,
            max_pos_capital_pct=0.25,
            friction_pct=0.0005,
        )

        equity_curves[strat_name] = equity_df.set_index("timestamp")["total_equity"]
        all_metrics[strat_name] = metrics

        spy = metrics.get("benchmark_spy", {})
        summary_rows.append({
            "Estrategia": strat_name,
            "Retorno Total": f"{metrics['retorno_total_pct']:+.2f}%",
            "Ganancia USD": f"${metrics['ganancia_neta_usd']:+,.0f}",
            "CAGR %": f"{metrics['cagr_pct']:.2f}%",
            "Sharpe": metrics["sharpe_ratio"],
            "Sortino": metrics["sortino_ratio"],
            "Max DD %": f"{metrics['max_drawdown_pct']:.2f}%",
            "Trades": metrics["total_trades"],
            "Win Rate": f"{metrics['win_rate_pct']:.2f}%",
            "Profit Factor": metrics["profit_factor"],
            "Exp (R)": f"{metrics['expectativa_r']:+.2f}R",
        })

    summary_df = pd.DataFrame(summary_rows)
    print("--- TABLA COMPARATIVA DE ESTRATEGIAS (2022 - 2026) ---")
    print(summary_df.to_string(index=False))

    # Benchmark SPY
    ref_spy = list(all_metrics.values())[0].get("benchmark_spy", {})
    print("\n--- BENCHMARK PASIVO: SPY (BUY & HOLD) EN EL MISMO PERÍODO ---")
    print(f"  * Retorno Total SPY:   {ref_spy.get('spy_retorno_total_pct', 0.0):+.2f}%")
    print(f"  * CAGR SPY:            {ref_spy.get('spy_cagr_pct', 0.0):.2f}%")
    print(f"  * Max Drawdown SPY:    {ref_spy.get('spy_max_drawdown_pct', 0.0):.2f}%")
    print(f"  * Sharpe Ratio SPY:    {ref_spy.get('spy_sharpe_ratio', 0.0):.2f}")

    # Guardar reporte JSON y curvas de capital
    report_file = Path("data/processed/backtest_report.json")
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=4)
    print(f"\nReporte completo guardado en {report_file}")

    curves_df = pd.DataFrame(equity_curves)
    curves_path = Path("data/processed/backtest_equity_curves.parquet")
    curves_df.to_parquet(curves_path)
    print(f"Curvas de capital guardadas en {curves_path}")
    print("=" * 85)


if __name__ == "__main__":
    main()
