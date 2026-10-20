"""
[TICK-M2-A2-01]: Auditoría Forense de Falsos Quiebres vs Calidad Estructural de S/R (stock_sr.py)
Analiza la relación entre el número de toques confirmados (sr_support_touches), la frescura (sr_bars_since_sup),
la posición de canal (sr_channel_position) y la tasa de supervivencia de los quiebres en 107 activos de IBKR.
"""

import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_model_2a_breakdown_with_structural_sr import (
    extract_breakdown_with_structural_sr, FEATURES_SR_BREAKDOWN, PARQUET_IBKR
)
from scripts.run_motor_2_walkforward_pipeline import (
    simulate_dynamic_trading_2b, simulate_portfolio_slots
)
import lightgbm as lgb

def run_sr_quality_audit():
    print("=" * 90)
    print("EJECUTANDO [TICK-M2-A2-01]: AUDITORÍA DE CALIDAD ESTRUCTURAL S/R EN BREAKDOWNS")
    print("=" * 90)
    
    df_raw = pd.read_parquet(PARQUET_IBKR)
    df_events = extract_breakdown_with_structural_sr(df_raw)
    
    # 1. Walk-Forward ML
    windows = [
        ("Ventana 1 (2022)", list(range(2020, 2022)), [2022]),
        ("Ventana 2 (2023)", list(range(2020, 2023)), [2023]),
        ("Ventana 3 (2024)", list(range(2020, 2024)), [2024]),
        ("Ventana 4 (2025-26)", list(range(2020, 2025)), [2025, 2026])
    ]
    
    oos_preds = []
    for win_name, train_years, test_years in windows:
        train_df = df_events[df_events["year"].isin(train_years)].copy()
        test_df = df_events[df_events["year"].isin(test_years)].copy()
        
        clf = lgb.LGBMClassifier(
            n_estimators=80, learning_rate=0.03, max_depth=3, num_leaves=7,
            min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
            random_state=42, verbose=-1
        )
        clf.fit(train_df[FEATURES_SR_BREAKDOWN], train_df["target_2a"])
        test_df["prob_2a"] = clf.predict_proba(test_df[FEATURES_SR_BREAKDOWN])[:, 1]
        test_df["window_name"] = win_name
        oos_preds.append(test_df)
        
    oos_df = pd.concat(oos_preds, ignore_index=True)
    trades_df = simulate_dynamic_trading_2b(oos_df, df_raw)
    
    prob_cutoff = oos_df["prob_2a"].median()
    qual = trades_df[trades_df["prob_2a"] >= prob_cutoff].copy().reset_index(drop=True)
    exec_df, p_stats = simulate_portfolio_slots(qual, initial_capital=25000.0, max_slots=5)
    
    # Cruzar trades ejecutados con features S/R originales
    merged = pd.merge(exec_df, df_events, on="event_id", suffixes=("", "_ev"))
    
    print("\n" + "=" * 90)
    print("1. DISCRIMINACIÓN ESTADÍSTICA: VARIABLES S/R (GANADORES VS PERDEDORES)")
    print("=" * 90)
    winners = merged[merged["net_ret"] > 0]
    losers = merged[merged["net_ret"] <= 0]
    
    sr_vars = [
        "sr_support_touches", "sr_resistance_touches", "sr_channel_position",
        "sr_bars_since_sup", "sr_bars_since_res", "sr_dist_support_atr", "sr_dist_resistance_atr",
        "penetration_pct", "lower_wick_ratio", "rvol_break_power"
    ]
    
    stat_rows = []
    for v in sr_vars:
        w_val = winners[v].dropna()
        l_val = losers[v].dropna()
        t_stat, p_val = stats.mannwhitneyu(w_val, l_val)
        sig = "***" if p_val < 0.01 else ("**" if p_val < 0.05 else ("*" if p_val < 0.10 else ""))
        stat_rows.append({
            "Variable S/R": v,
            "Media Ganadores": round(w_val.mean(), 4),
            "Media Perdedores": round(l_val.mean(), 4),
            "Diferencia": round(w_val.mean() - l_val.mean(), 4),
            "p-value": round(p_val, 4),
            "Significancia": sig
        })
    stat_df = pd.DataFrame(stat_rows).sort_values("p-value").reset_index(drop=True)
    print(stat_df.to_string(index=False))
    
    # -------------------------------------------------------------------------
    # 2. DESGLOSE POR NÚMERO DE TOQUES DE SOPORTE (sr_support_touches)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("2. DESEMPEÑO POR NÚMERO DE TOQUES CONFIRMADOS DEL SOPORTE (sr_support_touches):")
    print("=" * 90)
    merged["touches_bin"] = pd.cut(
        merged["sr_support_touches"],
        bins=[-0.1, 0.5, 1.5, 2.5, 99.0],
        labels=["0 Toques (Nivel No Valorado)", "1 Toque (Soporte Débil)", "2 Toques (Confirmado)", "3+ Toques (Fuerte / Institucional)"]
    )
    
    touch_summary = merged.groupby("touches_bin", observed=False).agg(
        Trades=("event_id", "count"),
        WinRate=("net_ret", lambda x: f"{(x > 0).mean()*100:.1f}%"),
        ProfitFactor=("pnl_usd", lambda x: f"{x[x>0].sum() / abs(x[x<=0].sum()) if abs(x[x<=0].sum()) > 0 else 99.0:.2f}"),
        AvgRet=("net_ret_pct", lambda x: f"{x.mean():+.2f}%"),
        PnL_Total=("pnl_usd", lambda x: f"${x.sum():+,.0f}")
    )
    print(touch_summary.to_string())
    
    # -------------------------------------------------------------------------
    # 3. DESGLOSE POR FRESCURA DEL SOPORTE (sr_bars_since_sup)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("3. DESEMPEÑO POR FRESCURA / ANTIGÜEDAD DEL SOPORTE (sr_bars_since_sup):")
    print("=" * 90)
    merged["freshness_bin"] = pd.cut(
        merged["sr_bars_since_sup"],
        bins=[-1, 20, 50, 100, 9999],
        labels=["Muy Fresco (<= 20 barras)", "Medio (21 a 50 barras)", "Maduro (51 a 100 barras)", "Añejo (> 100 barras)"]
    )
    
    fresh_summary = merged.groupby("freshness_bin", observed=False).agg(
        Trades=("event_id", "count"),
        WinRate=("net_ret", lambda x: f"{(x > 0).mean()*100:.1f}%"),
        ProfitFactor=("pnl_usd", lambda x: f"{x[x>0].sum() / abs(x[x<=0].sum()) if abs(x[x<=0].sum()) > 0 else 99.0:.2f}"),
        AvgRet=("net_ret_pct", lambda x: f"{x.mean():+.2f}%"),
        PnL_Total=("pnl_usd", lambda x: f"${x.sum():+,.0f}")
    )
    print(fresh_summary.to_string())
    
    # -------------------------------------------------------------------------
    # 4. SIMULACIÓN DE FILTRADO: ¿QUÉ PASA SI EXIGIMOS SOPORTES CON >= 1 TOQUE?
    # -------------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("4. SIMULACIÓN DE IMPACTO: EXIGIR sr_support_touches >= 1:")
    print("=" * 90)
    
    for min_t in [0, 1, 2]:
        filtered_qual = qual[qual["event_id"].isin(merged[merged["sr_support_touches"] >= min_t]["event_id"])].copy().reset_index(drop=True)
        f_exec, f_stats = simulate_portfolio_slots(filtered_qual, initial_capital=25000.0, max_slots=5)
        print(f"Condición: sr_support_touches >= {min_t}")
        print(f"  Trades: {f_stats['trades']} | Win Rate: {f_stats['wr']:.1f}% | PF: {f_stats['pf']:.2f}")
        print(f"  PnL Total: ${f_stats['total_pnl']:+,.0f} USD ({f_stats['total_ret_pct']:+.1f}%) | Max DD: {f_stats['max_dd']:.2f}%\n")
    print("=" * 90)

if __name__ == "__main__":
    run_sr_quality_audit()
