"""
Entrenamiento y Validación Walk-Forward de Modelo 2A (Ingreso) - Arquetipo 2 (Breakdown Estructural)
INTEGRANDO EL MOTOR CAUSAL DE SOPORTE Y RESISTENCIA (stock_sr.py) DEL PROYECTO
Evaluación en el Universo Completo de 107 Acciones de IBKR (2020 - 2026).
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

from src.utils.stock_sr import compute_stock_support_resistance_features
from scripts.run_motor_2_walkforward_pipeline import (
    PARQUET_IBKR, simulate_dynamic_trading_2b, simulate_portfolio_slots
)

FEATURES_SR_BREAKDOWN = [
    # 1. Variables Estructurales del Usuario (stock_sr.py)
    "sr_dist_support_pct", "sr_dist_support_atr",
    "sr_dist_resistance_pct", "sr_dist_resistance_atr",
    "sr_support_touches", "sr_resistance_touches",
    "sr_channel_position", "sr_bars_since_sup", "sr_bars_since_res",
    
    # 2. Microestructura y Poder de Ruptura Forense
    "penetration_pct", "lower_wick_ratio", "upper_wick_ratio",
    "body_ratio", "close_loc_1h", "rvol_break_power", "marubozu_bear_strength",
    
    # 3. Tendencia, Momentum y Volatilidad
    "dist_sma_20", "dist_sma_40", "ret_3d", "ret_5d", "z_3d", "rsi_14", "atr_d",
    
    # 4. Contexto Macro y Anti-Squeeze
    "upside_tail_risk_20d", "spy_ret_3d", "dist_spy_sma200", "rel_strength_3d"
]

def extract_breakdown_with_structural_sr(df_raw: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "=" * 90)
    print("EXTRAYENDO BREAKDOWNS CON MOTOR ESTRUCTURAL S/R (stock_sr.py) EN 107 ACTIVOS")
    print("=" * 90)
    t0 = time.time()
    
    # SPY benchmark
    df_spy = df_raw[df_raw["ticker"] == "SPY"].sort_values("timestamp").drop_duplicates("timestamp").set_index("timestamp")
    spy_c = df_spy["close"]
    spy_sma200_d = spy_c.rolling(1400, min_periods=200).mean()
    spy_ret_3d = spy_c.pct_change(21).fillna(0.0)
    spy_ret_5d = spy_c.pct_change(35).fillna(0.0)
    
    df_stocks = df_raw[~df_raw["ticker"].isin(["SPY", "QQQ"])].copy()
    tickers = sorted(df_stocks["ticker"].unique())
    print(f"Calculando S/R causal barra a barra para {len(tickers)} activos...")
    
    all_events = []
    event_id = 0
    
    for idx_tk, ticker in enumerate(tickers):
        tdf = df_stocks[df_stocks["ticker"] == ticker].sort_values("timestamp").reset_index(drop=True)
        if len(tdf) < 250:
            continue
            
        # 1. EJECUTAR EL MOTOR DE SOPORTE Y RESISTENCIA CAUSAL DEL USUARIO
        tdf_sr = compute_stock_support_resistance_features(tdf, distance=5, atr_window=14, lookback_window=250)
        
        c = tdf_sr["close"].to_numpy()
        o = tdf_sr["open"].to_numpy()
        h = tdf_sr["high"].to_numpy()
        l = tdf_sr["low"].to_numpy()
        v = tdf_sr["volume"].to_numpy()
        ts = tdf_sr["timestamp"].to_numpy()
        sec = tdf_sr["sector"].iloc[0] if "sector" in tdf_sr.columns else "General"
        
        # Variables de S/R del usuario
        sr_dist_sup_pct = tdf_sr["sr_dist_support_pct"].to_numpy()
        sr_dist_sup_atr = tdf_sr["sr_dist_support_atr"].to_numpy()
        sr_dist_res_pct = tdf_sr["sr_dist_resistance_pct"].to_numpy()
        sr_dist_res_atr = tdf_sr["sr_dist_resistance_atr"].to_numpy()
        sr_sup_touches = tdf_sr["sr_support_touches"].to_numpy()
        sr_res_touches = tdf_sr["sr_resistance_touches"].to_numpy()
        sr_chan_pos = tdf_sr["sr_channel_position"].to_numpy()
        sr_breakdown = tdf_sr["sr_breakdown_sup"].to_numpy()
        sr_bars_sup = tdf_sr["sr_bars_since_sup"].to_numpy()
        sr_bars_res = tdf_sr["sr_bars_since_res"].to_numpy()
        
        # Medias y ATR diario
        sma_20 = pd.Series(c).rolling(20, min_periods=10).mean().to_numpy()
        sma_40 = pd.Series(c).rolling(40, min_periods=20).mean().to_numpy()
        
        tr = np.maximum(h[1:] - l[1:], np.maximum(np.abs(h[1:] - c[:-1]), np.abs(l[1:] - c[:-1])))
        tr = np.insert(tr, 0, h[0] - l[0])
        atr_d = pd.Series(tr).rolling(98, min_periods=20).mean().to_numpy() / (c + 1e-6)
        
        ret_1d = np.zeros(len(tdf_sr))
        ret_1d[7:] = (c[7:] - c[:-7]) / (c[:-7] + 1e-6)
        roll_max_up = pd.Series(ret_1d).rolling(140, min_periods=20).max().fillna(0.05).to_numpy()
        
        ret_3d = np.zeros(len(tdf_sr))
        ret_5d = np.zeros(len(tdf_sr))
        ret_3d[21:] = (c[21:] - c[:-21]) / (c[:-21] + 1e-6)
        ret_5d[35:] = (c[35:] - c[:-35]) / (c[:-35] + 1e-6)
        roll_std_3d = pd.Series(ret_3d).rolling(250, min_periods=50).std().fillna(0.04).to_numpy()
        z_3d = ret_3d / (roll_std_3d + 1e-6)
        
        rng = np.maximum(h - l, 1e-6)
        close_loc = (c - l) / rng
        upper_wick = (h - np.maximum(o, c)) / rng
        lower_wick = (np.minimum(o, c) - l) / rng
        body_ratio = np.abs(c - o) / rng
        rvol_1h = v / (pd.Series(v).rolling(35, min_periods=10).mean().to_numpy() + 1e-6)
        
        delta = pd.Series(c).diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        rs = gain.rolling(14, min_periods=14).mean() / (loss.rolling(14, min_periods=14).mean() + 1e-9)
        rsi_14 = (100.0 - (100.0 / (1.0 + rs))).fillna(50.0).to_numpy()
        
        for i in range(120, len(tdf_sr) - 20):
            if atr_d[i] < 0.020:
                continue
                
            cur_ts = pd.Timestamp(ts[i])
            
            # REGLA ESTRUCTURAL DE RUPTURA:
            # Opción 1: Ruptura detectada por el motor de valles causales del usuario (sr_breakdown_sup == 1.0)
            # Opción 2: Quiebre con penetración confirmada de soporte de 40 barras
            min_prev_40 = np.min(l[max(0, i-40): i])
            is_break_40 = (c[i] < min_prev_40)
            is_break_user_sr = (sr_breakdown[i] == 1.0)
            
            if not (is_break_40 or is_break_user_sr):
                continue
                
            # Filtros de microestructura validados en el Error Analysis:
            # 1. Volumen institucional
            if rvol_1h[i] < 1.4:
                continue
            # 2. Cierre en la parte baja de la vela (control de vendedores)
            if close_loc[i] > 0.35:
                continue
            # 3. Filtro forense: mecha inferior no debe mostrar absorción masiva (menor a 25%)
            if lower_wick[i] > 0.25:
                continue
                
            # Profundidad de penetración real bajo el nivel roto
            ref_support = min_prev_40 if is_break_40 else c[i] * (1.0 + sr_dist_sup_pct[i]/100.0)
            penetration_pct = max(0.0, (ref_support - c[i]) / ref_support) * 100.0
            
            entry_p = o[i + 1]
            if entry_p <= 0:
                continue
                
            stop_loss_p = max(h[i], entry_p * (1.0 + 0.7 * atr_d[i]))
            target_drop_p = entry_p * (1.0 - 1.0 * atr_d[i])
            
            fut_h = h[i + 1 : i + 1 + 14]
            fut_l = l[i + 1 : i + 1 + 14]
            
            target_hit = 0
            for step in range(len(fut_h)):
                if fut_l[step] <= target_drop_p:
                    target_hit = 1
                    break
                if fut_h[step] >= stop_loss_p:
                    target_hit = 0
                    break
                    
            spy_val = spy_c.get(cur_ts, np.nan)
            spy_sma_val = spy_sma200_d.get(cur_ts, np.nan)
            dist_spy_sma200 = (spy_val - spy_sma_val) / (spy_sma_val + 1e-6) if not np.isnan(spy_sma_val) else 0.0
            spy_r3 = float(spy_ret_3d.get(cur_ts, 0.0))
            spy_r5 = float(spy_ret_5d.get(cur_ts, 0.0))
            rel_strength_3d = ret_3d[i] - spy_r3
            
            event_id += 1
            all_events.append({
                "event_id": event_id,
                "ticker": ticker,
                "sector": sec,
                "timestamp": cur_ts,
                "entry_ts": pd.Timestamp(ts[i + 1]),
                "year": cur_ts.year,
                "archetype_code": 2,
                "archetype_name": "Breakdown_Short",
                "entry_p": entry_p,
                "target_drop_p": target_drop_p,
                "stop_loss_p": stop_loss_p,
                "target_2a": target_hit,
                
                # VARIABLES DEL USUARIO (stock_sr.py)
                "sr_dist_support_pct": sr_dist_sup_pct[i],
                "sr_dist_support_atr": sr_dist_sup_atr[i],
                "sr_dist_resistance_pct": sr_dist_res_pct[i],
                "sr_dist_resistance_atr": sr_dist_res_atr[i],
                "sr_support_touches": sr_sup_touches[i],
                "sr_resistance_touches": sr_res_touches[i],
                "sr_channel_position": sr_chan_pos[i],
                "sr_bars_since_sup": sr_bars_sup[i],
                "sr_bars_since_res": sr_bars_res[i],
                
                # FORENSE Y MICROESTRUCTURA
                "penetration_pct": penetration_pct,
                "lower_wick_ratio": lower_wick[i],
                "upper_wick_ratio": upper_wick[i],
                "body_ratio": body_ratio[i],
                "close_loc_1h": close_loc[i],
                "rvol_break_power": rvol_1h[i] * (1.0 - close_loc[i]),
                "marubozu_bear_strength": body_ratio[i] * (1.0 - upper_wick[i]) if c[i] < o[i] else 0.0,
                
                # TENDENCIA Y MOMENTUM
                "dist_sma_20": (c[i] - sma_20[i]) / (sma_20[i] + 1e-6),
                "dist_sma_40": (c[i] - sma_40[i]) / (sma_40[i] + 1e-6),
                "ret_3d": ret_3d[i],
                "ret_5d": ret_5d[i],
                "z_3d": z_3d[i],
                "rsi_14": rsi_14[i],
                "atr_d": atr_d[i],
                
                # MACRO
                "upside_tail_risk_20d": roll_max_up[i],
                "spy_ret_3d": spy_r3,
                "dist_spy_sma200": dist_spy_sma200,
                "rel_strength_3d": rel_strength_3d,
                "macro_regime": "Bear_Crash" if spy_r3 <= -0.02 else ("Bull_Trend" if spy_r3 >= 0.015 else "Neutral_Chop")
            })
            
    df_events = pd.DataFrame(all_events)
    print(f"[OK] {len(df_events):,} eventos extraídos con S/R estructural en {time.time() - t0:.1f}s.")
    return df_events

def run_walkforward_with_structural_sr():
    df_raw = pd.read_parquet(PARQUET_IBKR)
    df_events = extract_breakdown_with_structural_sr(df_raw)
    
    windows = [
        ("Ventana 1 (2022)", list(range(2020, 2022)), [2022]),
        ("Ventana 2 (2023)", list(range(2020, 2023)), [2023]),
        ("Ventana 3 (2024)", list(range(2020, 2024)), [2024]),
        ("Ventana 4 (2025-26)", list(range(2020, 2025)), [2025, 2026])
    ]
    
    print("\n" + "=" * 90)
    print("ENTRENANDO MODELO 2A CON VARIABLES S/R DEL USUARIO (WALK-FORWARD EXPANDING)")
    print("=" * 90)
    
    oos_preds = []
    feature_importances = []
    
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
        feature_importances.append(clf.feature_importances_)
        
        auc = roc_auc_score(test_df["target_2a"], test_df["prob_2a"])
        brier = brier_score_loss(test_df["target_2a"], test_df["prob_2a"])
        print(f"  {win_name}: Train {len(train_df):,} | Test OOS {len(test_df):,} | AUC: {auc:.4f} | Brier: {brier:.4f}")
        
    oos_df = pd.concat(oos_preds, ignore_index=True)
    
    # Importancia de features con S/R del usuario
    avg_fi = np.mean(feature_importances, axis=0)
    fi_df = pd.DataFrame({"Feature": FEATURES_SR_BREAKDOWN, "Importance": avg_fi}).sort_values("Importance", ascending=False)
    print("\nTOP 10 FEATURES MÁS IMPORTANTES EN MODELO 2A (CON S/R):")
    print(fi_df.head(10).to_string(index=False))
    
    # Simulación dinámica con Modelo 2B
    print("\n" + "=" * 90)
    print("SIMULANDO PORTAFOLIO ($25,000 USD, 5 SLOTS DE $5,000 USD, 10 BPS FRICCIÓN)")
    print("=" * 90)
    trades_df = simulate_dynamic_trading_2b(oos_df, df_raw)
    
    prob_cutoff = oos_df["prob_2a"].median()
    qual = trades_df[trades_df["prob_2a"] >= prob_cutoff].copy().reset_index(drop=True)
    exec_df, p_stats = simulate_portfolio_slots(qual, initial_capital=25000.0, max_slots=5)
    
    print("\nRESULTADOS GLOBALES WALK-FORWARD (BREAKDOWN CON S/R ESTRUCTURAL):")
    print("=" * 90)
    for k, v in p_stats.items():
        if isinstance(v, float):
            print(f"  {k:<15}: {v:.2f}")
        else:
            print(f"  {k:<15}: {v}")
            
    print("\nATRIBUCIÓN ANUAL:")
    for yr in sorted(exec_df["year"].unique()):
        sub_yr = exec_df[exec_df["year"] == yr]
        wr = (sub_yr["net_ret"] > 0).mean() * 100
        pnl = sub_yr["pnl_usd"].sum()
        w = sub_yr[sub_yr["net_ret"] > 0]["pnl_usd"].sum()
        l = abs(sub_yr[sub_yr["net_ret"] <= 0]["pnl_usd"].sum())
        pf = w / l if l > 0 else 99.0
        ret_pct = (pnl / 25000.0) * 100.0
        print(f"  Año {yr}: {len(sub_yr):3d} trades | WinRate: {wr:4.1f}% | PF: {pf:4.2f} | Ret: {ret_pct:+5.1f}% | PnL: ${pnl:+,.0f} USD")
        
    print("\nMOTIVOS DE SALIDA:")
    exit_grp = exec_df.groupby("exit_reason").agg(
        Trades=("event_id", "count"),
        WinRate=("net_ret", lambda x: f"{(x > 0).mean()*100:.1f}%"),
        AvgRet=("net_ret_pct", lambda x: f"{x.mean():+.2f}%"),
        PnL=("pnl_usd", lambda x: f"${x.sum():+,.0f}")
    )
    print(exit_grp.to_string())
    print("=" * 90)

if __name__ == "__main__":
    run_walkforward_with_structural_sr()
