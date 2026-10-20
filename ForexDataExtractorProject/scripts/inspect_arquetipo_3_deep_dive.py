"""
Deep Dive exclusivo en Arquetipo 3 (Rollover / Exhaustion Top)
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_motor_2_walkforward_pipeline import (
    PARQUET_IBKR, EARNINGS_PARQUET, extract_all_motor_2_events,
    run_walk_forward_ml, simulate_dynamic_trading_2b, simulate_portfolio_slots
)

def analyze_arquetipo_3():
    df_raw = pd.read_parquet(PARQUET_IBKR)
    earn_df = pd.read_parquet(EARNINGS_PARQUET)
    
    df_events = extract_all_motor_2_events(df_raw, earn_df)
    oos_df, _ = run_walk_forward_ml(df_events)
    trades_df = simulate_dynamic_trading_2b(oos_df, df_raw)
    
    prob_cutoff = oos_df["prob_2a"].median()
    qual = trades_df[trades_df["prob_2a"] >= prob_cutoff].copy().reset_index(drop=True)
    exec_df, _ = simulate_portfolio_slots(qual, initial_capital=25000.0, max_slots=5)
    
    # Filtrar solo Arquetipo 3
    a3_events = df_events[df_events["archetype_name"] == "Rollover_Top"].copy()
    a3_oos = oos_df[oos_df["archetype_name"] == "Rollover_Top"].copy()
    a3_trades = exec_df[exec_df["archetype_name"] == "Rollover_Top"].copy()
    
    print("=" * 80)
    print("DEEP DIVE EXCLUSIVO: ARQUETIPO 3 (ROLLOVER / EXHAUSTION TOP)")
    print("=" * 80)
    print(f"1. UNIVERSO Y OCURRENCIA:")
    print(f"   - Total eventos detectados en 7 años: {len(a3_events)}")
    print(f"   - Eventos evaluados en periodo OOS (2022-2026): {len(a3_oos)}")
    print(f"   - Trades efectivamente ejecutados por el portafolio: {len(a3_trades)}")
    
    print("\n2. DESEMPEÑO POR AÑO (OOS PURO):")
    for yr in sorted(a3_trades["year"].unique()):
        sub_yr = a3_trades[a3_trades["year"] == yr]
        wr = (sub_yr["net_ret"] > 0).mean() * 100
        pnl = sub_yr["pnl_usd"].sum()
        ret_mean = sub_yr["net_ret_pct"].mean()
        print(f"   Año {yr}: {len(sub_yr):2d} trades | WinRate: {wr:4.1f}% | Ret Medio: {ret_mean:+.2f}% | PnL: ${pnl:+,.0f} USD")
        
    print("\n3. DESGLOSE POR MOTIVO DE SALIDA EN ARQUETIPO 3:")
    exit_grp = a3_trades.groupby("exit_reason").agg(
        Trades=("event_id", "count"),
        WinRate=("net_ret", lambda x: f"{(x > 0).mean()*100:.1f}%"),
        AvgRet=("net_ret_pct", lambda x: f"{x.mean():+.2f}%"),
        PnL=("pnl_usd", lambda x: f"${x.sum():+,.0f}")
    )
    print(exit_grp.to_string())
    
    print("\n4. VELOCIDAD Y DURACIÓN DEL TRADE:")
    print(f"   - Duración media: {a3_trades['duration_bars'].mean():.1f} barras horarias ({a3_trades['duration_bars'].mean()/7:.1f} días)")
    print(f"   - Trades cerrados en <= 3 barras: {(a3_trades['duration_bars'] <= 3).mean()*100:.1f}%")
    
    print("\n5. TOP TICKERS MÁS RENTABLES EN ARQUETIPO 3:")
    by_tk = a3_trades.groupby("ticker").agg(
        Trades=("event_id", "count"),
        WinRate=("net_ret", lambda x: f"{(x > 0).mean()*100:.1f}%"),
        PnL=("pnl_usd", "sum")
    ).sort_values("PnL", ascending=False)
    print(by_tk.head(8).to_string())
    
    print("\n6. TICKERS QUE MÁS PERDIERON EN ARQUETIPO 3:")
    print(by_tk.tail(5).to_string())
    
    # 7. Comparación directa con Arquetipo 2 (Breakdown)
    a2_trades = exec_df[exec_df["archetype_name"] == "Breakdown_Short"].copy()
    print("\n" + "=" * 80)
    print("7. COMPARACIÓN ARQUETIPO 3 (ROLLOVER) VS ARQUETIPO 2 (BREAKDOWN):")
    print("=" * 80)
    print(f"Métrica                     | Arquetipo 2 (Breakdown) | Arquetipo 3 (Rollover)")
    print(f"Trades Ejecutados           | {len(a2_trades):<23} | {len(a3_trades)}")
    print(f"Win Rate                    | {(a2_trades['net_ret']>0).mean()*100:.1f}%                   | {(a3_trades['net_ret']>0).mean()*100:.1f}%")
    w2 = a2_trades[a2_trades['net_ret']>0]['pnl_usd'].sum()
    l2 = abs(a2_trades[a2_trades['net_ret']<=0]['pnl_usd'].sum())
    w3 = a3_trades[a3_trades['net_ret']>0]['pnl_usd'].sum()
    l3 = abs(a3_trades[a3_trades['net_ret']<=0]['pnl_usd'].sum())
    print(f"Profit Factor               | {w2/l2:.2f}                    | {w3/l3:.2f}")
    print(f"Retorno Medio por Trade     | {a2_trades['net_ret_pct'].mean():+.2f}%                  | {a3_trades['net_ret_pct'].mean():+.2f}%")
    print(f"Duración Media (Barras)     | {a2_trades['duration_bars'].mean():.1f}                    | {a3_trades['duration_bars'].mean():.1f}")
    print("=" * 80)

if __name__ == "__main__":
    analyze_arquetipo_3()
