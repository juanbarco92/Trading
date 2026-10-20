"""
Pipeline Científico de Walk-Forward Puro (Entrenar en el Pasado -> Predecir a Ciegas en el Futuro):
- Año de Prueba 2022 (Bear Market Inflacionario / Tasas):
  * Entrenar modelos LightGBM (1A + Guardián + 1B) EXCLUSIVAMENTE con datos de 2020 y 2021 (pre-2022).
  * Predecir a ciegas en todo 2022 y medir Win Rate, Drawdown y Retorno.
- Año de Prueba 2024:
  * Entrenar con 2020 a 2023. Predecir a ciegas en 2024.
- Año de Prueba 2025-2026:
  * Entrenar con 2020 a 2024. Predecir a ciegas en 2025 y 2026.
"""

import sys
import time
from pathlib import Path
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PARQUET_IBKR = PROJECT_ROOT / "data_ibkr" / "sp500_ibkr_multisector_7years_1hour.parquet"

def extract_features_for_ibkr(df_stocks: pd.DataFrame, df_spy: pd.DataFrame, vix_map: dict) -> pd.DataFrame:
    """Extrae las features técnicas de 1A y Guardián sobre datos horarios limpios de IBKR"""
    events = []
    
    # Pre-calcular SPY retornos
    spy_c = df_spy.sort_values("timestamp").drop_duplicates("timestamp").set_index("timestamp")["close"]
    spy_ret_3d_map = spy_c.pct_change(21).to_dict()
    spy_ret_2h_map = spy_c.pct_change(2).to_dict()
    
    for ticker, tdf in df_stocks.groupby("ticker"):
        tdf = tdf.sort_values("timestamp").reset_index(drop=True)
        if len(tdf) < 200:
            continue
            
        c = tdf["close"].to_numpy()
        o = tdf["open"].to_numpy()
        h = tdf["high"].to_numpy()
        l = tdf["low"].to_numpy()
        v = tdf["volume"].to_numpy()
        ts = tdf["timestamp"].to_numpy()
        sec = tdf["sector"].iloc[0]
        
        # Medias móviles
        sma_40 = pd.Series(c).rolling(40, min_periods=20).mean().to_numpy()
        sma_100 = pd.Series(c).rolling(100, min_periods=50).mean().to_numpy()
        slope_100 = (sma_100 - np.roll(sma_100, 10)) / (np.roll(sma_100, 10) + 1e-6)
        
        # ATR 14d (98 barras)
        tr = np.maximum(h[1:] - l[1:], np.maximum(np.abs(h[1:] - c[:-1]), np.abs(l[1:] - c[:-1])))
        tr = np.insert(tr, 0, h[0] - l[0])
        atr_d = pd.Series(tr).rolling(98, min_periods=20).mean().to_numpy() / (c + 1e-6)
        
        # Retornos
        ret_3d = np.zeros(len(tdf))
        ret_5d = np.zeros(len(tdf))
        ret_3d[21:] = (c[21:] - c[:-21]) / c[:-21]
        ret_5d[35:] = (c[35:] - c[:-35]) / c[:-35]
        
        roll_std_3d = pd.Series(ret_3d).rolling(250, min_periods=50).std().fillna(0.04).to_numpy()
        roll_std_5d = pd.Series(ret_5d).rolling(250, min_periods=50).std().fillna(0.05).to_numpy()
        z_3d = ret_3d / (roll_std_3d + 1e-6)
        z_5d = ret_5d / (roll_std_5d + 1e-6)
        
        # Microestructura
        rng = np.maximum(h - l, 1e-6)
        close_loc = (c - l) / rng
        lower_wick = (np.minimum(o, c) - l) / rng
        rvol_1h = v / (pd.Series(v).rolling(35, min_periods=10).mean().to_numpy() + 1e-6)
        
        for i in range(100, len(tdf) - 105):
            is_shock = (z_3d[i] <= -2.0 or z_5d[i] <= -2.0)
            if not is_shock:
                continue
            if atr_d[i] < 0.020:
                continue
            is_abs = (close_loc[i] >= 0.50 or lower_wick[i] >= 0.20)
            if not is_abs:
                continue
                
            entry_p = o[i + 1]
            shock_drop = abs(min(ret_3d[i], ret_5d[i]))
            if shock_drop < 0.03:
                continue
                
            entry_ts = pd.Timestamp(ts[i + 1])
            entry_day = entry_ts.tz_localize(None).floor("D")
            vix_val = vix_map.get(entry_day, 20.0)
            
            # Features de microestructura y rebote
            c_now = c[i]
            ret_2h = (c_now - min(l[i-1], l[i])) / (min(l[i-1], l[i]) + 1e-6)
            bounce_thrust = (ret_2h / atr_d[i]) * rvol_1h[i]
            
            win3 = c[max(0, i-2): i+1]
            floor_def = float(np.mean(close_loc[max(0, i-2): i+1]))
            
            # Futuro para Target 1A (recupera +45% del shock)
            fut_h = h[i + 1 : i + 1 + 105]
            fut_l = l[i + 1 : i + 1 + 105]
            fut_c = c[i + 1 : i + 1 + 105]
            fut_o = o[i + 1 : i + 1 + 105]
            fut_ts = ts[i + 1 : i + 1 + 105]
            
            target_meli = entry_p * (1.0 + 0.45 * shock_drop)
            sl_p = entry_p * (1.0 - 1.5 * atr_d[i])
            
            reached_meli = 0
            exit_p = fut_c[-1]
            exit_ts = fut_ts[-1]
            floor_p = None
            peak_p = entry_p
            
            for step in range(len(fut_c)):
                if floor_p is not None and step > 0:
                    if fut_c[step] <= floor_p:
                        exit_p = fut_o[step + 1] if step + 1 < len(fut_c) else fut_c[step]
                        exit_ts = fut_ts[step + 1] if step + 1 < len(fut_c) else fut_ts[step]
                        break
                if (step + 1) % 7 == 0 and fut_c[step] <= sl_p:
                    exit_p = fut_o[step + 1] if step + 1 < len(fut_c) else fut_c[step]
                    exit_ts = fut_ts[step + 1] if step + 1 < len(fut_c) else fut_ts[step]
                    break
                if fut_h[step] > peak_p:
                    peak_p = fut_h[step]
                if fut_h[step] >= target_meli:
                    reached_meli = 1
                    floor_p = max(floor_p or 0.0, entry_p * (1.0 + 0.35 * shock_drop), peak_p * (1.0 - 0.70 * atr_d[i]))
                    
            net_ret = ((exit_p * 0.9995 - entry_p) / entry_p) * 100.0
            
            events.append({
                "ticker": ticker,
                "sector": sec,
                "entry_ts": entry_ts,
                "exit_ts": pd.Timestamp(exit_ts),
                "year": entry_ts.year,
                "entry_p": entry_p,
                "exit_p": exit_p,
                "net_ret_pct": net_ret,
                "target_meli": reached_meli,
                # Features
                "z_3d": z_3d[i],
                "z_5d": z_5d[i],
                "atr_d": atr_d[i],
                "close_loc_1h": close_loc[i],
                "lower_wick_ratio": lower_wick[i],
                "rvol_1h": rvol_1h[i],
                "close_diff_SMA_100": (c[i] - sma_100[i]) / (sma_100[i] + 1e-6),
                "slope_SMA_100": slope_100[i],
                "bounce_thrust_2h": bounce_thrust,
                "floor_defense_3h": floor_def,
                "vix_level": vix_val,
                "shock_drop": shock_drop
            })
            
    return pd.DataFrame(events)

def simulate_portfolio(events_df: pd.DataFrame, initial_capital: float = 25000.0, max_slots: int = 5):
    df_s = events_df.sort_values("entry_ts").reset_index(drop=True)
    if df_s.empty:
        return {"trades": 0, "wr": 0.0, "ret_pct": 0.0, "pf": 0.0, "max_dd": 0.0}
    cash = initial_capital
    open_pos = []
    closed = []
    eq_curve = []
    for _, row in df_s.iterrows():
        cur_ts = row["entry_ts"]
        still = []
        for p in open_pos:
            if p["exit_ts"] <= cur_ts:
                cash += p["returned"]
                closed.append(p)
            else:
                still.append(p)
        open_pos = still
        cur_eq = cash + sum(p["invested"] for p in open_pos)
        eq_curve.append({"ts": cur_ts, "eq": cur_eq})
        if len(open_pos) >= max_slots:
            continue
        if any(p["ticker"] == row["ticker"] for p in open_pos):
            continue
        alloc = min(cash / (max_slots - len(open_pos)), cur_eq / max_slots)
        if alloc < 500:
            continue
        cash -= alloc
        ret_f = row["net_ret_pct"] / 100.0
        open_pos.append({
            "ticker": row["ticker"],
            "entry_ts": cur_ts,
            "exit_ts": row["exit_ts"],
            "invested": alloc,
            "returned": alloc * (1.0 + ret_f),
            "net_ret_pct": row["net_ret_pct"]
        })
    for p in open_pos:
        cash += p["returned"]
        closed.append(p)
    tdf = pd.DataFrame(closed)
    wins = tdf[tdf["net_ret_pct"] > 0]["net_ret_pct"].sum()
    loss = abs(tdf[tdf["net_ret_pct"] <= 0]["net_ret_pct"].sum())
    pf = wins / loss if loss > 0 else 9.99
    eq_df = pd.DataFrame(eq_curve)
    max_dd = 0.0
    if not eq_df.empty:
        eq_df["peak"] = eq_df["eq"].cummax()
        eq_df["dd"] = (eq_df["eq"] - eq_df["peak"]) / eq_df["peak"] * 100.0
        max_dd = float(eq_df["dd"].min())
    return {
        "trades": len(tdf),
        "wr": float((tdf["net_ret_pct"] > 0).mean() * 100.0) if len(tdf) > 0 else 0.0,
        "pf": pf,
        "ret_pct": float((cash - initial_capital) / initial_capital * 100.0),
        "final_capital": cash,
        "max_dd": max_dd
    }

def main():
    print("=" * 90)
    print("WALK-FORWARD CIENTÍFICO PURO: ENTRENAR EN EL PASADO -> PREDECIR EN EL FUTURO (IBKR 2020-2026)")
    print("=" * 90)
    
    df_all = pd.read_parquet(PARQUET_IBKR)
    df_spy = df_all[df_all["ticker"] == "SPY"].copy()
    df_stocks = df_all[~df_all["ticker"].isin(["SPY", "QQQ"])].copy()
    
    # Cargar VIX
    vix_df = pd.read_parquet(PROJECT_ROOT / "data_ibkr" / "vix_daily_2020_2026.parquet")
    vix_df.columns = [c[0] if isinstance(c, tuple) else str(c).replace("('", "").replace("', '')", "").replace("', '^VIX')", "").replace("'", "") for c in vix_df.columns]
    date_col = [c for c in vix_df.columns if "Date" in c][0]
    close_col = [c for c in vix_df.columns if "Close" in c][0]
    vix_df["date_key"] = pd.to_datetime(vix_df[date_col]).dt.tz_localize(None).dt.floor("D")
    vix_map = vix_df.set_index("date_key")[close_col].to_dict()
    
    print("Extrayendo features para todas las barras horarias de IBKR...")
    df_feat = extract_features_for_ibkr(df_stocks, df_spy, vix_map)
    print(f"Total eventos con features extraídas: {len(df_feat):,}")
    
    # Aplicar Escudo Macro VIX
    df_feat = df_feat[df_feat["vix_level"] < 32.0].copy().reset_index(drop=True)
    
    features = [
        "z_3d", "z_5d", "atr_d", "close_loc_1h", "lower_wick_ratio", "rvol_1h",
        "close_diff_SMA_100", "slope_SMA_100", "bounce_thrust_2h", "floor_defense_3h", "shock_drop"
    ]
    
    # Pruebas Walk-Forward
    test_windows = [
        # (Año Test, Nombre Régimen, Años de Entrenamiento previos)
        (2022, "Bear Market Inflación/Tasas", [2020, 2021]),
        (2023, "Inicio Rally IA / Consolidación", [2020, 2021, 2022]),
        (2024, "Expansión Bull Market", [2020, 2021, 2022, 2023]),
        (2025, "Madurez de Ciclo", [2020, 2021, 2022, 2023, 2024]),
        (2026, "Año Corriente", [2020, 2021, 2022, 2023, 2024, 2025]),
    ]
    
    wf_results = []
    
    for test_year, reg_name, train_years in test_windows:
        tr_df = df_feat[df_feat["year"].isin(train_years)].copy()
        te_df = df_feat[df_feat["year"] == test_year].copy()
        
        if len(tr_df) < 50 or len(te_df) < 10:
            continue
            
        # Entrenar LightGBM EXCLUSIVAMENTE con el pasado
        clf = lgb.LGBMClassifier(
            n_estimators=100, learning_rate=0.03, max_depth=3, num_leaves=8,
            min_child_samples=25, subsample=0.8, colsample_bytree=0.7,
            random_state=42, verbose=-1
        )
        clf.fit(tr_df[features].fillna(0.0), tr_df["target_meli"].values)
        
        # Predecir a ciegas en el año de prueba
        te_df["prob_1a"] = clf.predict_proba(te_df[features].fillna(0.0))[:, 1]
        auc = roc_auc_score(te_df["target_meli"], te_df["prob_1a"]) if te_df["target_meli"].nunique() > 1 else 0.50
        
        # Simular portafolio filtrando con el modelo de ML (Corte >= 0.55 / 0.58)
        # Comparar: Sin ML (todas las señales) vs Con ML (solo prob_1a >= 0.55)
        res_raw = simulate_portfolio(te_df, 25000.0, 5)
        
        ml_approved = te_df[te_df["prob_1a"] >= 0.55].copy()
        res_ml = simulate_portfolio(ml_approved, 25000.0, 5)
        
        wf_results.append({
            "Año_Test": test_year,
            "Régimen": reg_name,
            "Entrenamiento": f"{min(train_years)}-{max(train_years)} ({len(tr_df)} casos)",
            "AUC_OOS": round(auc, 4),
            "Trades_SinML": res_raw["trades"],
            "WR_SinML": f"{res_raw['wr']:.1f}%",
            "Retorno_SinML": f"{'+' if res_raw['ret_pct']>=0 else ''}{res_raw['ret_pct']:.1f}%",
            "Trades_ConML": res_ml["trades"],
            "WR_ConML": f"{res_ml['wr']:.1f}%",
            "Retorno_ConML": f"{'+' if res_ml['ret_pct']>=0 else ''}{res_ml['ret_pct']:.1f}%",
            "MaxDD_ConML": f"{res_ml['max_dd']:.2f}%",
            "PF_ConML": round(res_ml["pf"], 2)
        })
        
    print("\n" + "=" * 125)
    print("RESULTADOS DEL WALK-FORWARD CIENTÍFICO (MODELO ENTRENADO SOLO EN EL PASADO -> PROBADO A CIEGAS EN EL FUTURO):")
    print("=" * 125)
    print(pd.DataFrame(wf_results).to_string(index=False))
    print("=" * 125)

if __name__ == "__main__":
    main()
