"""
Análisis Cuantitativo de Errores - Motor 2 (Shorts & Cobertura Activa)
Audita a fondo los trades ganadores vs perdedores del pipeline Walk-Forward
para descubrir la causa raíz de los fallos y plantear nuevas features e hipótesis.
"""

import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Importar funciones del pipeline Walk-Forward
from scripts.run_motor_2_walkforward_pipeline import (
    PARQUET_IBKR, EARNINGS_PARQUET, extract_all_motor_2_events,
    run_walk_forward_ml, simulate_dynamic_trading_2b, simulate_portfolio_slots, FEATURES
)

def run_error_analysis():
    print("=" * 90)
    print("INICIANDO ANALISIS CUANTITATIVO DE ERRORES: MOTOR 2 (SHORTS)")
    print("=" * 90)
    
    df_raw = pd.read_parquet(PARQUET_IBKR)
    earn_df = pd.read_parquet(EARNINGS_PARQUET)
    
    # 1. Extracción y simulación
    df_events = extract_all_motor_2_events(df_raw, earn_df)
    oos_df, _ = run_walk_forward_ml(df_events)
    trades_df = simulate_dynamic_trading_2b(oos_df, df_raw)
    
    prob_cutoff = oos_df["prob_2a"].median()
    qualified = trades_df[trades_df["prob_2a"] >= prob_cutoff].copy().reset_index(drop=True)
    exec_df, p_stats = simulate_portfolio_slots(qualified, initial_capital=25000.0, max_slots=5)
    
    # Cruzar features originales con los trades ejecutados
    merged = pd.merge(exec_df, df_events, on="event_id", suffixes=("", "_ev"))
    
    winners = merged[merged["net_ret"] > 0].copy()
    losers = merged[merged["net_ret"] <= 0].copy()
    
    print(f"\nTotal trades ejecutados analizados: {len(merged)}")
    print(f"Ganadores: {len(winners)} ({len(winners)/len(merged)*100:.1f}%) | Perdedores: {len(losers)} ({len(losers)/len(merged)*100:.1f}%)")
    
    # -------------------------------------------------------------------------
    # 1. COMPARATIVA DE FEATURES: GANADORES VS PERDEDORES
    # -------------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("1. DISCRIMINACION DE FEATURES (GANADORES VS PERDEDORES):")
    print("=" * 90)
    rows = []
    for f in FEATURES:
        if f not in merged.columns:
            continue
        w_val = winners[f].dropna()
        l_val = losers[f].dropna()
        
        # Test de hipótesis t de Student / Mann-Whitney
        t_stat, p_val = stats.mannwhitneyu(w_val, l_val)
        rows.append({
            "Feature": f,
            "Media_Ganadores": round(w_val.mean(), 4),
            "Media_Perdedores": round(l_val.mean(), 4),
            "Diferencia": round(w_val.mean() - l_val.mean(), 4),
            "p-value": round(p_val, 4),
            "Relevante": "***" if p_val < 0.01 else ("**" if p_val < 0.05 else ("*" if p_val < 0.10 else ""))
        })
    feat_df = pd.DataFrame(rows).sort_values("p-value").reset_index(drop=True)
    print(feat_df.to_string(index=False))
    
    # -------------------------------------------------------------------------
    # 2. ANÁLISIS DE MOTIVOS DE SALIDA
    # -------------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("2. ANÁLISIS POR MOTIVO DE SALIDA (EXIT REASON):")
    print("=" * 90)
    exit_summary = merged.groupby("exit_reason").agg(
        Trades=("event_id", "count"),
        WinRate=("net_ret", lambda x: f"{(x > 0).mean()*100:.1f}%"),
        AvgRetPct=("net_ret_pct", lambda x: f"{x.mean():+.2f}%"),
        PnL_Total=("pnl_usd", lambda x: f"${x.sum():+,.0f}")
    )
    print(exit_summary.to_string())
    
    # -------------------------------------------------------------------------
    # 3. INTERACCIÓN MACRO (¿QUÉ ESTABA HACIENDO EL SPY CUANDO EL SHORT FALLÓ?)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("3. INTERACCIÓN CON EL MERCADO GENERAL (SPY):")
    print("=" * 90)
    print(f"Retorno medio 3d de SPY en Ganadores:  {winners['spy_ret_3d'].mean()*100:+.2f}%")
    print(f"Retorno medio 3d de SPY en Perdedores: {losers['spy_ret_3d'].mean()*100:+.2f}%")
    print(f"Retorno medio 5d de SPY en Ganadores:  {winners['spy_ret_5d'].mean()*100:+.2f}%")
    print(f"Retorno medio 5d de SPY en Perdedores: {losers['spy_ret_5d'].mean()*100:+.2f}%")
    print(f"Distancia a SMA200 SPY en Ganadores:   {winners['dist_spy_sma200'].mean()*100:+.2f}%")
    print(f"Distancia a SMA200 SPY en Perdedores:  {losers['dist_spy_sma200'].mean()*100:+.2f}%")
    
    # -------------------------------------------------------------------------
    # 4. TRAMPA DE SOBREVENDIDO (¿VENDIMOS CUANDO YA ERA MUY TARDE?)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("4. DIAGNOSTICO DE SOBRE-EXTENSION (TRAMPA DE VENDER TARDE):")
    print("=" * 90)
    print(f"Retorno acumulado 3d previo en Ganadores:  {winners['ret_3d'].mean()*100:+.2f}%")
    print(f"Retorno acumulado 3d previo en Perdedores: {losers['ret_3d'].mean()*100:+.2f}%")
    print(f"Z-Score 3d previo en Ganadores:            {winners['z_3d'].mean():+.2f}")
    print(f"Z-Score 3d previo en Perdedores:           {losers['z_3d'].mean():+.2f}")
    print(f"RSI-14 previo en Ganadores:                {winners['rsi_14'].mean():.1f}")
    print(f"RSI-14 previo en Perdedores:               {losers['rsi_14'].mean():.1f}")
    
    # -------------------------------------------------------------------------
    # 5. ATRIBUCIÓN POR ACTIVO (MEJORES VS PEORES TICKERS PARA SHORT)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("5. TOP 10 ACTIVOS MÁS RENTABLES EN SHORT:")
    print("=" * 90)
    by_tk = merged.groupby("ticker").agg(
        Trades=("event_id", "count"),
        WinRate=("net_ret", lambda x: (x > 0).mean()*100),
        PnL_Total=("pnl_usd", "sum")
    ).sort_values("PnL_Total", ascending=False)
    
    top10_win = by_tk.head(10).copy()
    top10_win["WinRate"] = top10_win["WinRate"].map(lambda x: f"{x:.1f}%")
    top10_win["PnL_Total"] = top10_win["PnL_Total"].map(lambda x: f"${x:+,.0f}")
    print(top10_win.to_string())
    
    print("\n" + "=" * 90)
    print("TOP 10 PEORES ACTIVOS (TRAMPAS CRÓNICAS DE SHORT):")
    print("=" * 90)
    top10_lose = by_tk.tail(10).copy()
    top10_lose["WinRate"] = top10_lose["WinRate"].map(lambda x: f"{x:.1f}%")
    top10_lose["PnL_Total"] = top10_lose["PnL_Total"].map(lambda x: f"${x:+,.0f}")
    print(top10_lose.to_string())
    print("=" * 90)

if __name__ == "__main__":
    run_error_analysis()
