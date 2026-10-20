"""
Auditoría Forense y Diagnóstico de Vulnerabilidad Macro en Rebotes Violentos (Autopsia 2023)
Ticket [TICK-M2-A2-02]
Aisla la causa raíz del desempeño negativo en 2023 (-$1,862 USD en 80 trades)
durante el rally en 'V' de la renta variable.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_model_2a_breakdown_with_structural_sr import (
    extract_breakdown_with_structural_sr, PARQUET_IBKR
)

def audit_macro_failures_2023():
    print("=" * 90)
    print("AUTOPSIA MACRO 2023: ¿POR QUÉ FALLARON LOS SHORTS DE BREAKDOWN EN EL RALLY EN 'V'?")
    print("=" * 90)
    
    cache_file = PROJECT_ROOT / "data_cache" / "events_breakdown_107_tickers_sr.parquet"
    if cache_file.exists():
        print(f"Cargando eventos desde caché: {cache_file}")
        df_events = pd.read_parquet(cache_file)
    else:
        df_raw = pd.read_parquet(PARQUET_IBKR)
        df_events = extract_breakdown_with_structural_sr(df_raw)
        df_events.to_parquet(cache_file, index=False)
        print(f"Eventos guardados en caché: {cache_file}")
    
    # Filtrar exclusivamente los eventos ocurridos en 2023
    ev_2023 = df_events[df_events["year"] == 2023].copy().reset_index(drop=True)
    print(f"\nTotal eventos de Breakdown en 2023: {len(ev_2023)}")
    
    winners_2023 = ev_2023[ev_2023["target_2a"] == 1]
    losers_2023 = ev_2023[ev_2023["target_2a"] == 0]
    
    print(f"Quiebres Genuinos (Target 1): {len(winners_2023)} ({len(winners_2023)/len(ev_2023)*100:.1f}%)")
    print(f"Trampas Bajistas (Target 0): {len(losers_2023)} ({len(losers_2023)/len(ev_2023)*100:.1f}%)")
    
    # -------------------------------------------------------------------------
    # 1. COMPARATIVA BIVARIADA MACRO & TENDENCIA EN 2023
    # -------------------------------------------------------------------------
    macro_features = [
        "spy_ret_3d", "spy_ret_5d", "dist_spy_sma200", "rel_strength_3d",
        "upside_tail_risk_20d", "dist_sma_20", "dist_sma_40", "rsi_14",
        "sr_dist_support_pct", "sr_support_touches", "sr_bars_since_sup",
        "penetration_pct", "lower_wick_ratio", "rvol_break_power"
    ]
    
    rows = []
    for f in macro_features:
        if f not in ev_2023.columns:
            continue
        w_val = winners_2023[f].dropna()
        l_val = losers_2023[f].dropna()
        if len(w_val) == 0 or len(l_val) == 0:
            continue
        t_stat, p_val = stats.mannwhitneyu(w_val, l_val)
        sig = "***" if p_val < 0.01 else ("**" if p_val < 0.05 else ("*" if p_val < 0.10 else ""))
        rows.append({
            "Feature": f,
            "Media Genuinos (1)": round(w_val.mean(), 4),
            "Media Trampas (0)": round(l_val.mean(), 4),
            "Diferencia": round(w_val.mean() - l_val.mean(), 4),
            "p-value": round(p_val, 4),
            "Sig": sig
        })
    res_df = pd.DataFrame(rows).sort_values("p-value").reset_index(drop=True)
    print("\n" + "=" * 90)
    print("PRUEBA STATS MANN-WHITNEY U: QUÉ SEPARÓ A GANADORES DE PERDEDORES EN 2023")
    print("=" * 90)
    print(res_df.to_string(index=False))
    
    # -------------------------------------------------------------------------
    # 2. ANÁLISIS DE REGÍMENES DE MERCADO GENERAL (SPY)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("DESGLOSE DE WIN RATE SEGÚN EL RÉGIMEN DE SPY EN 2023:")
    print("=" * 90)
    
    # Régimen 1: SPY por encima o debajo de SMA 200
    ev_2023["spy_bull_regime"] = ev_2023["dist_spy_sma200"] > 0.0
    spy_bull_summary = ev_2023.groupby("spy_bull_regime").agg(
        Total=("target_2a", "count"),
        Genuinos=("target_2a", "sum"),
        WinRate=("target_2a", lambda x: f"{x.mean()*100:.1f}%")
    ).rename(index={True: "SPY sobre SMA200 (Bull Market)", False: "SPY bajo SMA200 (Bear/Transition)"})
    print(spy_bull_summary.to_string())
    
    # Régimen 2: SPY Momentum de 3 días
    print("\nDESGLOSE SEGÚN EL IMPULSO ALCISTA DE SPY (spy_ret_3d):")
    ev_2023["spy_momentum_bin"] = pd.cut(
        ev_2023["spy_ret_3d"] * 100,
        bins=[-99.0, -1.0, 0.0, 1.5, 99.0],
        labels=["SPY Cayendo (< -1%)", "SPY Ligeramente Bajista (-1% a 0%)", "SPY Moderado (0% a +1.5%)", "SPY Rally Euforia (> +1.5%)"]
    )
    spy_mom_summary = ev_2023.groupby("spy_momentum_bin", observed=False).agg(
        Total=("target_2a", "count"),
        Genuinos=("target_2a", "sum"),
        WinRate=("target_2a", lambda x: f"{x.mean()*100:.1f}%")
    )
    print(spy_mom_summary.to_string())
    
    # Régimen 3: Fuerza Relativa frente a SPY (rel_strength_3d)
    print("\nDESGLOSE SEGÚN LA DEBILIDAD RELATIVA FRENTE A SPY (rel_strength_3d):")
    ev_2023["rel_str_bin"] = pd.cut(
        ev_2023["rel_strength_3d"] * 100,
        bins=[-99.0, -4.0, -1.0, 1.0, 99.0],
        labels=["Ultra Débil vs SPY (< -4%)", "Moderadamente Débil (-4% a -1%)", "En línea con SPY (-1% a +1%)", "Más Fuerte que SPY (> +1%)"]
    )
    rel_summary = ev_2023.groupby("rel_str_bin", observed=False).agg(
        Total=("target_2a", "count"),
        Genuinos=("target_2a", "sum"),
        WinRate=("target_2a", lambda x: f"{x.mean()*100:.1f}%")
    )
    print(rel_summary.to_string())
    print("=" * 90)

if __name__ == "__main__":
    audit_macro_failures_2023()
