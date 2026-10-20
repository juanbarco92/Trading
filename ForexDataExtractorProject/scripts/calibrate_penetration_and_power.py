"""
Calibración Forense de Penetración Óptima y Poder de Quiebre en Arquetipo 2
Ticket [TICK-M2-A2-03]
Evalúa el poder predictivo continuo de penetración normalizada por ATR,
absorción de mecha inferior y volumen relativo para eliminar falsas rupturas sin sobreajuste.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_model_2a_breakdown_with_structural_sr import (
    extract_breakdown_with_structural_sr, PARQUET_IBKR
)

def calibrate_penetration_and_microstructure():
    print("=" * 90)
    print("CALIBRACIÓN FORENSE: PENETRACIÓN / ATR, PODER DE QUIEBRE Y ABSORCIÓN")
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
    
    # 1. Penetración normalizada por volatilidad intrínseca (ATR diario)
    df_events["penetration_atr_ratio"] = df_events["penetration_pct"] / (df_events["atr_d"] + 1e-4)
    
    # -------------------------------------------------------------------------
    # BIVARIADO: PENETRATION / ATR vs WIN RATE
    # -------------------------------------------------------------------------
    print("\n1. DESGLOSE DE WIN RATE POR PENETRACIÓN NORMALIZADA (penetration / atr_d):")
    df_events["pen_atr_bin"] = pd.cut(
        df_events["penetration_atr_ratio"],
        bins=[-99.0, 0.20, 0.50, 0.80, 1.20, 99.0],
        labels=["Superficial (< 0.2x ATR)", "Moderada (0.2x a 0.5x ATR)", "Sólida (0.5x a 0.8x ATR)", "Profunda (0.8x a 1.2x ATR)", "Extrema (> 1.2x ATR)"]
    )
    biv_pen = df_events.groupby("pen_atr_bin", observed=False).agg(
        Total=("target_2a", "count"),
        Genuinos=("target_2a", "sum"),
        WinRate=("target_2a", lambda x: f"{x.mean()*100:.1f}%")
    )
    print(biv_pen.to_string())
    
    # -------------------------------------------------------------------------
    # BIVARIADO: ABSORCIÓN INFERIOR (lower_wick_ratio) vs WIN RATE
    # -------------------------------------------------------------------------
    print("\n2. DESGLOSE DE WIN RATE POR ABSORCIÓN COMPRADORA (lower_wick_ratio):")
    df_events["wick_bin"] = pd.cut(
        df_events["lower_wick_ratio"],
        bins=[-0.01, 0.10, 0.25, 0.40, 1.01],
        labels=["Cero Absorción (mecha <= 10%)", "Baja Absorción (10% a 25%)", "Absorción Moderada (25% a 40%)", "Fuerte Absorción (> 40%)"]
    )
    biv_wick = df_events.groupby("wick_bin", observed=False).agg(
        Total=("target_2a", "count"),
        Genuinos=("target_2a", "sum"),
        WinRate=("target_2a", lambda x: f"{x.mean()*100:.1f}%")
    )
    print(biv_wick.to_string())
    
    # -------------------------------------------------------------------------
    # BIVARIADO: PODER DE QUIEBRE (rvol_break_power) vs WIN RATE
    # -------------------------------------------------------------------------
    print("\n3. DESGLOSE DE WIN RATE POR PODER DE QUIEBRE (rvol_break_power):")
    df_events["power_bin"] = pd.qcut(
        df_events["rvol_break_power"],
        q=4,
        labels=["Q1 (Bajo Poder)", "Q2 (Poder Medio-Bajo)", "Q3 (Poder Medio-Alto)", "Q4 (Poder Extremo)"]
    )
    biv_pow = df_events.groupby("power_bin", observed=False).agg(
        Total=("target_2a", "count"),
        Genuinos=("target_2a", "sum"),
        WinRate=("target_2a", lambda x: f"{x.mean()*100:.1f}%")
    )
    print(biv_pow.to_string())
    
    # -------------------------------------------------------------------------
    # ROC-AUC DE CADA VARIABLE INDIVIDUAL EN PREDECIR ÉXITO DEL QUIEBRE
    # -------------------------------------------------------------------------
    print("\n4. CAPACIDAD PREDICTIVA UNIVARIADA (ROC-AUC):")
    cand_feats = [
        "penetration_atr_ratio", "penetration_pct", "lower_wick_ratio",
        "rvol_break_power", "marubozu_bear_strength", "sr_support_touches",
        "sr_bars_since_sup", "dist_sma_20", "dist_sma_40", "rsi_14"
    ]
    auc_rows = []
    y_true = df_events["target_2a"].to_numpy()
    for f in cand_feats:
        vals = df_events[f].to_numpy()
        # Manejar nan
        mask = ~np.isnan(vals)
        auc = roc_auc_score(y_true[mask], vals[mask])
        if auc < 0.5:
            # Si tiene correlación inversa
            auc_eff = 1.0 - auc
            direction = "Inversa (-)"
        else:
            auc_eff = auc
            direction = "Directa (+)"
        auc_rows.append({
            "Variable": f,
            "AUC Efectivo": round(auc_eff, 4),
            "Dirección": direction,
            "AUC Raw": round(auc, 4)
        })
    auc_df = pd.DataFrame(auc_rows).sort_values("AUC Efectivo", ascending=False).reset_index(drop=True)
    print(auc_df.to_string(index=False))
    print("=" * 90)

if __name__ == "__main__":
    calibrate_penetration_and_microstructure()
