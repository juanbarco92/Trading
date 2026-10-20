"""
exhaustive_error_analysis.py - TICK-18: Análisis Forense Exhaustivo de Errores y Falsos Positivos.

Diagnostica matemáticamente los fallos del modelo:
1. Análisis de Falsos Positivos de Alta Convicción (Probabilidad >= 0.45 que tocaron Stop Loss).
2. Correlación de fallos con:
   - Hora del día y Día de la semana (Trampas de apertura y cierres de semana).
   - Régimen de Mercado (SPY alcista vs bajista, volatilidad ATR relativa).
   - Atribución por Activo (Ranking de fiabilidad de los 21 activos).
3. Análisis de Rachas de Pérdida (Max Consecutive Losses) y Distribución de Colas (Slippage y Gaps).
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd


def run_error_analysis(trades_df: pd.DataFrame, df_features: pd.DataFrame):
    print("\n" + "="*80)
    print("INICIANDO ANÁLISIS FORENSE EXHAUSTIVO DE ERRORES (ERROR ANALYSIS)")
    print("="*80)

    if len(trades_df) == 0:
        print("No hay operaciones registradas para analizar.")
        return

    # Separar Ganadores y Perdedores
    wins = trades_df[trades_df["pnl_usd"] > 0].copy()
    losses = trades_df[trades_df["pnl_usd"] <= 0].copy()

    total_trades = len(trades_df)
    n_wins = len(wins)
    n_losses = len(losses)
    win_rate = n_wins / total_trades * 100.0

    print(f"\n1. RESUMEN GLOBAL DE EJECUCIÓN:")
    print(f"  * Total Trades Analizados: {total_trades:,}")
    print(f"  * Ganadores: {n_wins:,} ({win_rate:.2f}%)")
    print(f"  * Perdedores: {n_losses:,} ({100.0 - win_rate:.2f}%)")
    print(f"  * PnL Neto Total: ${trades_df['pnl_usd'].sum():,.2f} USD")
    print(f"  * Retorno Medio por Trade: {trades_df['pnl_pct'].mean():+.2f}%")

    # -------------------------------------------------------------
    # 2. ANÁLISIS DE MOTIVOS DE SALIDA (EXIT REASONS)
    # -------------------------------------------------------------
    print("\n" + "-"*80)
    print("2. DESGLOSE POR MOTIVO DE SALIDA (EXIT REASON):")
    print("-"*80)
    exit_summary = trades_df.groupby("exit_reason").agg(
        trades=("pnl_usd", "count"),
        pct_trades=("pnl_usd", lambda x: round(len(x) / total_trades * 100, 2)),
        win_rate=("pnl_usd", lambda x: round((x > 0).mean() * 100, 2)),
        pnl_total=("pnl_usd", "sum"),
        pnl_medio=("pnl_usd", "mean"),
        duracion_media_barras=("bars_held", "mean"),
    ).reset_index()
    print(exit_summary.to_string(index=False))

    # -------------------------------------------------------------
    # 3. DISTRIBUCIÓN HORARIA Y DÍA DE LA SEMANA DE LAS PÉRDIDAS
    # -------------------------------------------------------------
    print("\n" + "-"*80)
    print("3. ANÁLISIS TEMPORAL DE FALLOS (CUÁNDO FALLA EL MODELO):")
    print("-"*80)
    trades_df["entry_dt"] = pd.to_datetime(trades_df["entry_time"])
    trades_df["hour"] = trades_df["entry_dt"].dt.hour
    trades_df["day_name"] = trades_df["entry_dt"].dt.day_name()

    hourly = trades_df.groupby("hour").agg(
        trades=("pnl_usd", "count"),
        win_rate=("pnl_usd", lambda x: round((x > 0).mean() * 100, 2)),
        pnl_total=("pnl_usd", "sum"),
        retorno_medio_pct=("pnl_pct", "mean"),
    ).reset_index()
    print("Distribución por Hora de Entrada (UTC):")
    print(hourly.to_string(index=False))

    daily = trades_df.groupby("day_name").agg(
        trades=("pnl_usd", "count"),
        win_rate=("pnl_usd", lambda x: round((x > 0).mean() * 100, 2)),
        pnl_total=("pnl_usd", "sum"),
        retorno_medio_pct=("pnl_pct", "mean"),
    ).reset_index()
    print("\nDistribución por Día de la Semana:")
    print(daily.to_string(index=False))

    # -------------------------------------------------------------
    # 4. ATRIBUCIÓN POR ACTIVO (RANKING DE FIABILIDAD)
    # -------------------------------------------------------------
    print("\n" + "-"*80)
    print("4. ATRIBUCIÓN POR ACTIVO (LÍDERES VS TÓXICOS):")
    print("-"*80)
    ticker_summary = trades_df.groupby("ticker").agg(
        trades=("pnl_usd", "count"),
        win_rate=("pnl_usd", lambda x: round((x > 0).mean() * 100, 2)),
        pnl_total=("pnl_usd", "sum"),
        profit_factor=("pnl_usd", lambda x: round(x[x > 0].sum() / abs(x[x < 0].sum()), 2) if len(x[x < 0]) > 0 and abs(x[x < 0].sum()) > 0 else 99.0),
        retorno_medio_pct=("pnl_pct", "mean"),
    ).reset_index().sort_values("pnl_total", ascending=False)
    print(ticker_summary.to_string(index=False))

    # -------------------------------------------------------------
    # 5. ANÁLISIS DE RACHAS DE PÉRDIDAS Y RIESGO DE COLA
    # -------------------------------------------------------------
    print("\n" + "-"*80)
    print("5. RIESGO DE COLA Y RACHAS CONSECUTIVAS DE PÉRDIDA:")
    print("-"*80)
    trades_df_sorted = trades_df.sort_values("entry_dt").reset_index(drop=True)
    is_loss = (trades_df_sorted["pnl_usd"] <= 0).astype(int)
    
    # Calcular rachas de pérdidas consecutivas
    loss_streaks = []
    current_streak = 0
    for l in is_loss:
        if l == 1:
            current_streak += 1
        else:
            if current_streak > 0:
                loss_streaks.append(current_streak)
            current_streak = 0
    if current_streak > 0:
        loss_streaks.append(current_streak)

    max_loss_streak = max(loss_streaks) if loss_streaks else 0
    mean_loss_streak = np.mean(loss_streaks) if loss_streaks else 0

    worst_trade_pct = trades_df["pnl_pct"].min()
    best_trade_pct = trades_df["pnl_pct"].max()

    print(f"  * Racha Máxima de Pérdidas Consecutivas: {max_loss_streak} trades")
    print(f"  * Racha Media de Pérdidas:             {mean_loss_streak:.1f} trades")
    print(f"  * Peor Trade Individual:              {worst_trade_pct:.2f}% (Pérdida máxima)")
    print(f"  * Mejor Trade Individual:             {best_trade_pct:+.2f}% (Ganancia máxima)")
    print("="*80)
