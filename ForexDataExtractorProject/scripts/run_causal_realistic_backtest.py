"""
run_causal_realistic_backtest.py - TICK-17 & TICK-18: Backtest Causal con Ejecución en Open[t+1] y Error Analysis Forense.

Integra:
1. Modelos purificados y calibrados sin fuga de datos (TICK-15 & TICK-16).
2. Motor de ejecución realista en Open[t+1] con resolución pesimista intra-barra (TICK-17).
3. Diagnóstico forense de errores y falsos positivos (TICK-18).
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
import pandas as pd
from src.models.realistic_backtester import run_realistic_portfolio_backtest
from scripts.exhaustive_error_analysis import run_error_analysis


def main():
    features_path = PROJECT_ROOT / "data" / "processed" / "features_universe_full_structural.parquet"
    models_dir = PROJECT_ROOT / "models"
    buy_file = models_dir / "pipeline_universe_buy_calibrated.joblib"
    sell_file = models_dir / "pipeline_universe_sell_calibrated.joblib"

    print("1. Cargando dataset estructural causal y modelos purificados...")
    df = pd.read_parquet(features_path)
    model_buy = joblib.load(buy_file)
    model_sell = joblib.load(sell_file)

    feature_cols = list(model_buy.feature_names_in_)
    print(f"Features predictoras: {len(feature_cols)}")
    X = df[feature_cols].fillna(0.0)

    print("2. Generando probabilidades out-of-sample...")
    raw_buy = model_buy.predict_proba(X)[:, 1]
    raw_sell = model_sell.predict_proba(X)[:, 1]

    # Winner Takes All
    df["prob_buy"] = np.where(raw_buy > raw_sell, raw_buy, 0.0)
    df["prob_sell"] = np.where(raw_sell > raw_buy, raw_sell, 0.0)

    # 3. Ejecutar Backtest con Ejecución Obligatoria en Open[t+1]
    print("\n3. Ejecutando Backtest Institucional (Ejecución Estricta Open[t+1] + Fricción 10 bps)...")
    trades_df, annual_df = run_realistic_portfolio_backtest(
        df,
        threshold_buy=0.45,
        threshold_sell=0.45,
        initial_capital=25000.0,
        max_position_size=8000.0,
        risk_per_trade_pct=0.015,
        max_open_positions=6,
        friction_pct=0.0010,
        filter_morning_traps=True,
        max_horizon_bars=35,
    )

    print("\n" + "="*80)
    print("MÉTRICAS ANUALES REALISTAS (EJECUCIÓN EN OPEN[t+1]):")
    print("="*80)
    print(annual_df.to_string(index=False))

    total_cagr = 0.0
    if len(annual_df) > 0 and len(trades_df) > 0:
        final_cap = trades_df.iloc[-1]["capital_after"]
        total_cagr = ((final_cap / 25000.0) ** (1.0 / len(annual_df)) - 1.0) * 100.0
        print(f"\n  * Capital Inicial:    $25,000.00 USD")
        print(f"  * Capital Final:      ${final_cap:,.2f} USD")
        print(f"  * CAGR Global (% EA): {total_cagr:+.2f}% EA")
        print(f"  * Benchmark CDT:      13.00% EA")
    print("="*80)

    # 4. Error Analysis Forense
    print("\n4. Ejecutando Diagnóstico Forense de Errores...")
    run_error_analysis(trades_df, df)


if __name__ == "__main__":
    main()
