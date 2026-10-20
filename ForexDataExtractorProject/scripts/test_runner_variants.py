"""
Test runner integrado respetando el trailing ceiling exacto de simulate_dynamic_trading_2b
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_motor_2_walkforward_pipeline import (
    PARQUET_IBKR, EARNINGS_PARQUET, simulate_portfolio_slots
)
from scripts.test_arquetipo_3_and_runners import extract_motor_2_events_v3, run_walk_forward_ml_v3

def simulate_dynamic_trading_with_clean_runner(oos_df: pd.DataFrame, df_raw: pd.DataFrame, runner_ratio=0.50):
    raw_by_tk = {tk: grp.sort_values("timestamp").reset_index(drop=True) for tk, grp in df_raw.groupby("ticker")}
    trades = []
    
    for _, row in oos_df.iterrows():
        tk = row["ticker"]
        stk_df = raw_by_tk.get(tk)
        if stk_df is None:
            continue
            
        matches = np.where(stk_df["timestamp"].to_numpy() == row["entry_ts"])[0]
        if len(matches) == 0:
            continue
        idx_entry = matches[0]
        
        entry_p = row["entry_p"]
        atr = row["atr_d"]
        
        stop_p = max(row["stop_loss_p"], entry_p * (1.0 + 0.7 * atr))
        target_p = entry_p * (1.0 - 1.2 * atr)
        
        fut_bars = stk_df.iloc[idx_entry : idx_entry + 14]
        if len(fut_bars) == 0:
            continue
            
        dyn_stop = stop_p
        be_locked = False
        min_seen = entry_p
        duration_bars = len(fut_bars)
        
        target_hit_bar = -1
        exit_p_part1 = None
        exit_p_runner = None
        exit_reason = "time_stop_2d"
        
        for step, (_, bar) in enumerate(fut_bars.iterrows()):
            bh = bar["high"]
            bl = bar["low"]
            bc = bar["close"]
            bo = bar["open"]
            rng = max(bh - bl, 1e-6)
            lower_wick = (min(bo, bc) - bl) / rng
            
            min_seen = min(min_seen, bl)
            gain_now = (entry_p - bl) / entry_p
            
            # Si aún no hemos tocado target:
            if target_hit_bar == -1:
                # 1. Stop loss
                if bh >= dyn_stop:
                    exit_p_part1 = dyn_stop
                    exit_p_runner = dyn_stop
                    exit_reason = "stop_loss" if not be_locked else "breakeven_stop"
                    duration_bars = step + 1
                    break
                    
                # 2. Target hit
                if bl <= target_p:
                    target_hit_bar = step
                    exit_p_part1 = target_p
                    if runner_ratio == 0.0:
                        exit_p_runner = target_p
                        exit_reason = "target_hit"
                        duration_bars = step + 1
                        break
                    else:
                        # Para el runner: el stop se mueve a Breakeven garantizado
                        dyn_stop = min(dyn_stop, entry_p * 0.999)
                        be_locked = True
                        
                # 3. Quick BE
                if gain_now >= (0.5 * atr) and not be_locked:
                    dyn_stop = entry_p * 0.999
                    be_locked = True
                    
                # 4. Trailing Ceiling
                if gain_now >= (0.8 * atr):
                    trailing_ceiling = min_seen + (0.35 * atr * entry_p)
                    dyn_stop = min(dyn_stop, trailing_ceiling)
                    
                # 5. Absorción
                if gain_now >= (0.60 * atr) and lower_wick >= 0.40 and bc > bo:
                    exit_p_part1 = bc
                    exit_p_runner = bc
                    exit_reason = "absorption_hammer"
                    duration_bars = step + 1
                    break
            else:
                # El runner sigue corriendo:
                # Actualizamos trailing ceiling con mayor holgura (0.5x ATR del mínimo)
                trailing_ceiling = min_seen + (0.50 * atr * entry_p)
                dyn_stop = min(dyn_stop, trailing_ceiling)
                
                # Si toca dyn_stop:
                if bh >= dyn_stop:
                    exit_p_runner = dyn_stop
                    exit_reason = "target_hit+runner_trail"
                    duration_bars = step + 1
                    break
                    
                # Absorción profunda en el runner (caída > 2x ATR con martillo)
                if gain_now >= (2.0 * atr) and lower_wick >= 0.40 and bc > bo:
                    exit_p_runner = bc
                    exit_reason = "target_hit+runner_absorption"
                    duration_bars = step + 1
                    break
                    
        # Salida por tiempo
        if exit_p_part1 is None:
            last_c = fut_bars.iloc[-1]["close"]
            exit_p_part1 = last_c
            exit_p_runner = last_c
            exit_reason = "time_stop_2d"
        elif exit_p_runner is None:
            last_c = fut_bars.iloc[-1]["close"]
            exit_p_runner = last_c
            exit_reason = "target_hit+runner_time_14b"
            
        ret_part1 = (entry_p - exit_p_part1) / entry_p
        ret_runner = (entry_p - exit_p_runner) / entry_p
        gross_ret = (1.0 - runner_ratio) * ret_part1 + runner_ratio * ret_runner
        net_ret = gross_ret - 0.0010
        
        trades.append({
            "event_id": row["event_id"],
            "ticker": tk,
            "sector": row["sector"],
            "archetype_name": row["archetype_name"],
            "year": row["year"],
            "macro_regime": row["macro_regime"],
            "window_name": row["window_name"],
            "entry_ts": row["entry_ts"],
            "exit_ts": fut_bars.iloc[min(len(fut_bars)-1, duration_bars-1)]["timestamp"],
            "duration_bars": duration_bars,
            "prob_2a": row["prob_2a"],
            "target_2a": row["target_2a"],
            "entry_p": entry_p,
            "gross_ret": gross_ret,
            "net_ret": net_ret,
            "net_ret_pct": net_ret * 100.0,
            "exit_reason": exit_reason
        })
        
    return pd.DataFrame(trades).sort_values("entry_ts").reset_index(drop=True)

def test_runner_variants():
    df_raw = pd.read_parquet(PARQUET_IBKR)
    earn_df = pd.read_parquet(EARNINGS_PARQUET)
    df_events = extract_motor_2_events_v3(df_raw, earn_df)
    oos_df = run_walk_forward_ml_v3(df_events)
    prob_cutoff = oos_df["prob_2a"].median()
    
    for r_ratio in [0.0, 0.30, 0.50]:
        t_df = simulate_dynamic_trading_with_clean_runner(oos_df, df_raw, runner_ratio=r_ratio)
        qual = t_df[t_df["prob_2a"] >= prob_cutoff].copy().reset_index(drop=True)
        exec_df, p_stats = simulate_portfolio_slots(qual, initial_capital=25000.0, max_slots=5)
        print(f"\nRunner Ratio {r_ratio*100:.0f}%:")
        print(f"  Trades: {p_stats['trades']} | WinRate: {p_stats['wr']:.1f}% | PF: {p_stats['pf']:.2f}")
        print(f"  PnL Total: ${p_stats['total_pnl']:+,.0f} USD ({p_stats['total_ret_pct']:+.1f}%) | Max DD: {p_stats['max_dd']:.2f}%")

if __name__ == "__main__":
    test_runner_variants()
