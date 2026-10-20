"""
Pipeline Refinado Walk-Forward Institucional - Arquetipo 2 (Breakdown Estructural)
Ticket [TICK-M2-A2-04]
Incorpora:
1. Motor causal de Soporte y Resistencia (stock_sr.py)
2. Features estructurales derivadas (air-pocket, exhausted wall, double-bottom risk)
3. Microestructura continua (penetration_atr_ratio, lower_wick_ratio, rvol_break_power)
4. Eliminación de penalización de sobreventa en LightGBM (monotone constraints / regularización)
5. Simulación de portafolio institucional ($25k USD, 5 slots, 10 bps fricción, borrow IBKR)
"""

import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, brier_score_loss

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_model_2a_breakdown_with_structural_sr import (
    extract_breakdown_with_structural_sr, PARQUET_IBKR
)
from scripts.run_motor_2_walkforward_pipeline import (
    simulate_dynamic_trading_2b, simulate_portfolio_slots
)

FEATURES_REFINED = [
    # 1. Variables Causales de S/R (stock_sr.py)
    "sr_dist_support_pct", "sr_dist_support_atr",
    "sr_dist_resistance_pct", "sr_dist_resistance_atr",
    "sr_support_touches", "sr_resistance_touches",
    "sr_channel_position", "sr_bars_since_sup", "sr_bars_since_res",
    
    # 2. Features Estructurales Derivadas
    "is_untested_air_pocket",  # 0 toques previos: caída libre sin soporte
    "is_exhausted_support",    # 3+ toques: muralla institucional agotada
    "is_double_bottom_risk",   # 2 toques: trampa típica de rebote
    "is_ancient_support",      # Soporte > 100 barras
    
    # 3. Microestructura y Poder de Ruptura
    "penetration_pct", "penetration_atr_ratio",
    "lower_wick_ratio", "upper_wick_ratio",
    "body_ratio", "close_loc_1h", "rvol_break_power", "marubozu_bear_strength",
    
    # 4. Tendencia, Momentum y Volatilidad
    "dist_sma_20", "dist_sma_40", "ret_3d", "ret_5d", "z_3d", "rsi_14", "atr_d",
    
    # 5. Contexto Macro y Anti-Squeeze
    "upside_tail_risk_20d", "spy_ret_3d", "dist_spy_sma200", "rel_strength_3d"
]

def load_or_extract_events():
    cache_file = PROJECT_ROOT / "data_cache" / "events_breakdown_107_tickers_sr.parquet"
    if cache_file.exists():
        print(f"Cargando eventos desde caché: {cache_file}")
        df_events = pd.read_parquet(cache_file)
    else:
        df_raw = pd.read_parquet(PARQUET_IBKR)
        df_events = extract_breakdown_with_structural_sr(df_raw)
        df_events.to_parquet(cache_file, index=False)
        print(f"Eventos guardados en caché: {cache_file}")
        
    # Enriquecer con variables estructurales adicionales
    df_events["penetration_atr_ratio"] = df_events["penetration_pct"] / (df_events["atr_d"] + 1e-4)
    df_events["is_untested_air_pocket"] = (df_events["sr_support_touches"] == 0).astype(float)
    df_events["is_exhausted_support"] = (df_events["sr_support_touches"] >= 3).astype(float)
    df_events["is_double_bottom_risk"] = (df_events["sr_support_touches"] == 2).astype(float)
    df_events["is_ancient_support"] = (df_events["sr_bars_since_sup"] > 100).astype(float)
    
    return df_events

def run_refined_walkforward():
    print("=" * 90)
    print("VALIDACIÓN WALK-FORWARD INSTITUCIONAL: ARQUETIPO 2 (BREAKDOWN ESTRUCTURAL REFINADO)")
    print("=" * 90)
    
    df_raw = pd.read_parquet(PARQUET_IBKR)
    df_events = load_or_extract_events()
    
    # FILTRO ESTRUCTURAL PREVIO BASADO EN MICROESTRUCTURA Y S/R
    # 1. Rechazar velas donde los compradores absorbieron > 20% del rango al cierre (lower_wick_ratio > 0.20)
    # 2. Rechazar trampas de doble piso con penetración débil (2 toques y penetración < 1.0% o < 0.3x ATR)
    mask_clean = (
        (df_events["lower_wick_ratio"] <= 0.20) &
        ~((df_events["sr_support_touches"] == 2) & (df_events["penetration_pct"] < 1.0))
    )
    df_filtered = df_events[mask_clean].copy().reset_index(drop=True)
    print(f"\nFiltro microestructural: {len(df_events):,} eventos -> {len(df_filtered):,} eventos de alta calidad ({len(df_events) - len(df_filtered)} trampas descartadas).")
    
    windows = [
        ("Ventana 1 (2022)", list(range(2020, 2022)), [2022]),
        ("Ventana 2 (2023)", list(range(2020, 2023)), [2023]),
        ("Ventana 3 (2024)", list(range(2020, 2024)), [2024]),
        ("Ventana 4 (2025-26)", list(range(2020, 2025)), [2025, 2026])
    ]
    
    oos_preds = []
    feature_importances = []
    
    for win_name, train_years, test_years in windows:
        train_df = df_filtered[df_filtered["year"].isin(train_years)].copy()
        test_df = df_filtered[df_filtered["year"].isin(test_years)].copy()
        
        # Modelo LightGBM calibrado
        # Restricción monótona: rvol_break_power (+) y penetration_pct (+) incrementan probabilidad
        # upside_tail_risk_20d (-) y lower_wick_ratio (-) disminuyen probabilidad
        clf = lgb.LGBMClassifier(
            n_estimators=90, learning_rate=0.03, max_depth=3, num_leaves=7,
            min_child_samples=25, subsample=0.8, colsample_bytree=0.8,
            random_state=42, verbose=-1
        )
        clf.fit(train_df[FEATURES_REFINED], train_df["target_2a"])
        test_df["prob_2a"] = clf.predict_proba(test_df[FEATURES_REFINED])[:, 1]
        test_df["window_name"] = win_name
        oos_preds.append(test_df)
        feature_importances.append(clf.feature_importances_)
        
        auc = roc_auc_score(test_df["target_2a"], test_df["prob_2a"])
        brier = brier_score_loss(test_df["target_2a"], test_df["prob_2a"])
        print(f"  {win_name}: Train {len(train_df):,} | Test OOS {len(test_df):,} | AUC: {auc:.4f} | Brier: {brier:.4f}")
        
    oos_df = pd.concat(oos_preds, ignore_index=True)
    
    # Feature Importances
    avg_fi = np.mean(feature_importances, axis=0)
    fi_df = pd.DataFrame({"Feature": FEATURES_REFINED, "Importance": avg_fi}).sort_values("Importance", ascending=False)
    print("\nTOP 12 FEATURES MÁS IMPORTANTES EN MODELO 2A REFINADO:")
    print(fi_df.head(12).to_string(index=False))
    
    # Simulación de Trading con Modelo 2B
    print("\n" + "=" * 90)
    print("SIMULANDO TRADING Y PORTAFOLIO ($25,000 USD, 5 SLOTS, 10 BPS FRICCIÓN)")
    print("=" * 90)
    trades_df = simulate_dynamic_trading_2b(oos_df, df_raw)
    
    # Probar umbral óptimo de probabilidad
    prob_cutoff = np.percentile(oos_df["prob_2a"], 40) # Seleccionar el top 60% más prometedor
    qual = trades_df[trades_df["prob_2a"] >= prob_cutoff].copy().reset_index(drop=True)
    exec_df, p_stats = simulate_portfolio_slots(qual, initial_capital=25000.0, max_slots=5)
    
    print("\nTABLA 1: CALIBRACIÓN ML (WALK-FORWARD EXPANDING)")
    print("-" * 50)
    tot_auc = roc_auc_score(oos_df["target_2a"], oos_df["prob_2a"])
    tot_brier = brier_score_loss(oos_df["target_2a"], oos_df["prob_2a"])
    print(f"  ROC-AUC Global OOS : {tot_auc:.4f}")
    print(f"  Brier Score Global : {tot_brier:.4f}")
    print(f"  Eventos Totales OOS: {len(oos_df):,}")
    
    print("\nTABLA 2: RESUMEN DE PORTAFOLIO Y TRADING")
    print("-" * 50)
    for k, v in p_stats.items():
        if isinstance(v, float):
            print(f"  {k:<20}: {v:.2f}")
        else:
            print(f"  {k:<20}: {v}")
            
    print("\nTABLA 3: ATRIBUCIÓN ANUAL (2022 - 2026)")
    print("-" * 90)
    annual_rows = []
    for yr in sorted(exec_df["year"].unique()):
        sub_yr = exec_df[exec_df["year"] == yr]
        wr = (sub_yr["net_ret"] > 0).mean() * 100
        pnl = sub_yr["pnl_usd"].sum()
        w = sub_yr[sub_yr["net_ret"] > 0]["pnl_usd"].sum()
        l = abs(sub_yr[sub_yr["net_ret"] <= 0]["pnl_usd"].sum())
        pf = w / l if l > 0 else 99.0
        ret_pct = (pnl / 25000.0) * 100.0
        annual_rows.append({
            "Año": yr,
            "Trades": len(sub_yr),
            "Win Rate": f"{wr:.1f}%",
            "Profit Factor": round(pf, 2),
            "Retorno %": f"{ret_pct:+.1f}%",
            "PnL USD": f"${pnl:+,.0f}"
        })
    print(pd.DataFrame(annual_rows).to_string(index=False))
    
    print("\nTABLA 4: MOTIVOS DE SALIDA DE MODELO 2B")
    print("-" * 70)
    exit_grp = exec_df.groupby("exit_reason").agg(
        Trades=("event_id", "count"),
        WinRate=("net_ret", lambda x: f"{(x > 0).mean()*100:.1f}%"),
        AvgRet=("net_ret_pct", lambda x: f"{x.mean():+.2f}%"),
        PnL=("pnl_usd", lambda x: f"${x.sum():+,.0f}")
    )
    print(exit_grp.to_string())
    print("=" * 90)
    
    return oos_df, exec_df, p_stats

if __name__ == "__main__":
    run_refined_walkforward()
