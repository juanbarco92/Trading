"""
Auditoría Cuantitativa de Recall y Falsos Negativos ("El Oro Dejado Sobre la Mesa")
Modelo 2A (Ingreso) - Arquetipo 2 (Breakdown Estructural con stock_sr.py)
Evalúa decil por decil qué colapsos masivos estamos descartando y por qué.
"""

import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, brier_score_loss

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_model_2a_breakdown_with_structural_sr import (
    extract_breakdown_with_structural_sr, PARQUET_IBKR
)

# Conjunto parsimonioso de features (Cero Overfitting)
FEATURES_CLEAN = [
    # 1. Soporte y Resistencia Causal (stock_sr.py)
    "sr_dist_support_atr", "sr_channel_position", "sr_bars_since_sup",
    "is_untested_air_pocket", "is_double_bottom_risk", "is_exhausted_support",
    
    # 2. Microestructura Normalizada y Cierre
    "penetration_atr", "lower_wick_ratio", "upper_wick_ratio",
    "body_ratio", "close_loc_1h", "rvol_break_power",
    
    # 3. Momentum y Volatilidad
    "dist_sma_20", "ret_3d", "z_3d", "rsi_14", "atr_d",
    
    # 4. Contexto Macro
    "upside_tail_risk_20d", "spy_ret_3d", "dist_spy_sma200"
]

def add_parsimonious_features(df_events: pd.DataFrame) -> pd.DataFrame:
    df = df_events.copy()
    
    # 1. Penetración normalizada por ATR (elimina distorsión de precios nominales)
    df["penetration_atr"] = (df["penetration_pct"] / 100.0) / (df["atr_d"] + 1e-6)
    
    # 2. Variables estructurales derivadas de stock_sr.py (parsimoniosas, no sobreajustadas)
    touches = df["sr_support_touches"]
    df["is_untested_air_pocket"] = (touches == 0).astype(float)
    df["is_exhausted_support"] = (touches >= 3).astype(float)
    df["is_double_bottom_risk"] = ((touches == 2) & (df["penetration_pct"] < 2.5)).astype(float)
    
    return df

def run_recall_audit():
    print("=" * 90)
    print("AUDITORÍA DE RECALL Y FALSOS NEGATIVOS (EL ORO DEJADO EN LA MESA) - MODELO 2A")
    print("=" * 90)
    
    df_raw = pd.read_parquet(PARQUET_IBKR)
    df_events = extract_breakdown_with_structural_sr(df_raw)
    df_events = add_parsimonious_features(df_events)
    
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
            n_estimators=70, learning_rate=0.03, max_depth=3, num_leaves=7,
            min_child_samples=25, subsample=0.8, colsample_bytree=0.8,
            random_state=42, verbose=-1
        )
        clf.fit(train_df[FEATURES_CLEAN], train_df["target_2a"])
        test_df["prob_2a"] = clf.predict_proba(test_df[FEATURES_CLEAN])[:, 1]
        test_df["window_name"] = win_name
        oos_preds.append(test_df)
        
    oos_df = pd.concat(oos_preds, ignore_index=True)
    
    # -------------------------------------------------------------------------
    # 1. ANÁLISIS DE DECILES (PRECISIÓN VS RECALL POR PERCENTILES)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("1. CALIBRACIÓN POR DECILES DE PROBABILIDAD (DEL PEOR AL MEJOR PERCENTIL):")
    print("=" * 90)
    oos_df["decil"] = pd.qcut(oos_df["prob_2a"], q=10, labels=[f"D{i}" for i in range(1, 11)])
    
    # Calcular caída máxima real posterior (14 barras) para cada evento
    raw_by_tk = {tk: grp.sort_values("timestamp").reset_index(drop=True) for tk, grp in df_raw.groupby("ticker")}
    max_drops = []
    for _, row in oos_df.iterrows():
        tk = row["ticker"]
        stk_df = raw_by_tk.get(tk)
        if stk_df is None:
            max_drops.append(0.0)
            continue
        ts_e = row["entry_ts"]
        matches = np.where(stk_df["timestamp"].to_numpy() == ts_e)[0]
        if len(matches) == 0:
            max_drops.append(0.0)
            continue
        i = matches[0]
        fut_l = stk_df.iloc[i : min(len(stk_df), i + 14)]["low"].values
        drop = (np.min(fut_l) - row["entry_p"]) / row["entry_p"] if len(fut_l) > 0 else 0.0
        max_drops.append(abs(drop) * 100.0)
        
    oos_df["max_drop_pct"] = max_drops
    
    decil_table = oos_df.groupby("decil", observed=False).agg(
        Total_Eventos=("event_id", "count"),
        Genuinos_Exitosos=("target_2a", "sum"),
        Tasa_Acierto=("target_2a", lambda x: f"{x.mean()*100:.1f}%"),
        Caida_Media_Pct=("max_drop_pct", lambda x: f"{x.mean():.2f}%"),
        Super_Crashes_8pct=("max_drop_pct", lambda x: (x >= 8.0).sum())
    )
    print(decil_table.to_string())
    
    # -------------------------------------------------------------------------
    # 2. EL ORO EN EL FONDO DE LA TABLA (LOS FALSOS NEGATIVOS)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("2. AUDITORÍA DEL ORO DESPERDICIADO (DECILES 1 A 5 - LA MITAD DESCARTADA):")
    print("=" * 90)
    discarded = oos_df[oos_df["prob_2a"] < oos_df["prob_2a"].median()].copy()
    approved = oos_df[oos_df["prob_2a"] >= oos_df["prob_2a"].median()].copy()
    
    gold_discarded = discarded[discarded["target_2a"] == 1]
    super_gold_discarded = discarded[discarded["max_drop_pct"] >= 8.0]
    
    print(f"Total eventos descartados por Modelo 2A: {len(discarded)}")
    print(f"  - De los cuales fueron QUIEBRES GENUINOS (Falsos Negativos): {len(gold_discarded)} ({len(gold_discarded)/len(discarded)*100:.1f}%)")
    print(f"  - De los cuales fueron SUPER CRASHES (Caída >= 8% en 2 días): {len(super_gold_discarded)} ({len(super_gold_discarded)/len(discarded)*100:.1f}%)")
    print(f"  - Caída máxima potencial media de los eventos descartados: {discarded['max_drop_pct'].mean():.2f}%")
    
    print("\nTOP 10 SUPER CRASHES DESCARTADOS POR EL MODELO (EL ORO QUE DEJAMOS IR):")
    top_missed = super_gold_discarded.sort_values("max_drop_pct", ascending=False).head(10)
    for _, r in top_missed.iterrows():
        print(f"  {r['ticker']:5s} ({str(r['timestamp'])[:16]}): Prob 2A = {r['prob_2a']:.3f} (Decil {r['decil']}) | CAYÓ UN -{r['max_drop_pct']:.1f}% | RVOL: {r['rvol_break_power']:.1f}")
        
    # -------------------------------------------------------------------------
    # 3. CAUSA RAÍZ: ¿POR QUÉ EL MODELO DESCARTÓ ESE ORO?
    # -------------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("3. ANÁLISIS FORENSE: ¿POR QUÉ EL MODELO LE DIO BAJA PROBABILIDAD A ESOS GANADORES?")
    print("   Comparativa: Falsos Negativos (Oro Descartado) vs True Positives (Oro Aprobado)")
    print("=" * 90)
    
    comp_rows = []
    tp = approved[approved["target_2a"] == 1]
    fn = gold_discarded
    
    for col in FEATURES_CLEAN:
        v_tp = tp[col].dropna()
        v_fn = fn[col].dropna()
        t_stat, p_val = stats.mannwhitneyu(v_tp, v_fn)
        sig = "***" if p_val < 0.01 else ("**" if p_val < 0.05 else ("*" if p_val < 0.10 else ""))
        comp_rows.append({
            "Variable": col,
            "Media Oro Aprobado (TP)": round(v_tp.mean(), 4),
            "Media Oro Descartado (FN)": round(v_fn.mean(), 4),
            "Diferencia": round(v_tp.mean() - v_fn.mean(), 4),
            "p-value": round(p_val, 4),
            "Culpable": sig
        })
    comp_df = pd.DataFrame(comp_rows).sort_values("p-value").reset_index(drop=True)
    print(comp_df.to_string(index=False))
    print("=" * 90)

if __name__ == "__main__":
    run_recall_audit()
