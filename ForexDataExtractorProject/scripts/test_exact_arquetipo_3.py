"""
Test exacto de Arquetipo 3 usando las funciones oficiales del pipeline.
"""

import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_motor_2_walkforward_pipeline import (
    PARQUET_IBKR, EARNINGS_PARQUET, simulate_dynamic_trading_2b, simulate_portfolio_slots
)

from scripts.test_arquetipo_3_and_runners import extract_motor_2_events_v3, run_walk_forward_ml_v3

def test_exact():
    df_raw = pd.read_parquet(PARQUET_IBKR)
    earn_df = pd.read_parquet(EARNINGS_PARQUET)
    
    df_events = extract_motor_2_events_v3(df_raw, earn_df)
    oos_df = run_walk_forward_ml_v3(df_events)
    
    trades_df = simulate_dynamic_trading_2b(oos_df, df_raw)
    prob_cutoff = oos_df["prob_2a"].median()
    
    qual = trades_df[trades_df["prob_2a"] >= prob_cutoff].copy().reset_index(drop=True)
    exec_df, p_stats = simulate_portfolio_slots(qual, initial_capital=25000.0, max_slots=5)
    
    print("\n" + "="*80)
    print("RESULTADOS EXACTOS CON ARQUETIPO 3 (CON simulate_dynamic_trading_2b OFICIAL):")
    print("="*80)
    for k, v in p_stats.items():
        print(f"  {k}: {v}")
        
    print("\nDesglose por Arquetipo:")
    for arch, grp in exec_df.groupby("archetype_name"):
        wr = (grp['net_ret'] > 0).mean() * 100
        pnl = grp['pnl_usd'].sum()
        print(f"  {arch:<20} -> Trades: {len(grp):<4} | WinRate: {wr:.1f}% | PnL: ${pnl:+,.0f} USD")
        
    print("\nAtribución Anual:")
    for yr, grp in exec_df.groupby("year"):
        wr = (grp['net_ret'] > 0).mean() * 100
        pnl = grp['pnl_usd'].sum()
        print(f"  Año {yr} -> Trades: {len(grp):<4} | WinRate: {wr:.1f}% | PnL: ${pnl:+,.0f} USD")
    print("="*80)

if __name__ == "__main__":
    test_exact()
