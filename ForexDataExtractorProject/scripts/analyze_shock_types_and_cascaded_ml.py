"""
analyze_shock_types_and_cascaded_ml.py -
1. Clasifica y analiza empíricamente los tipos de caída en las 314 acciones de alto ATR (7,201 golpes):
   - Caída Macro / Mercado (SPY cae fuerte)
   - Caída Sectorial (El sector cae fuerte aunque el SPY aguante)
   - Caída por Resultados Negativos (Post-Earnings <= 5d con EPS Surprise < 0)
   - Caída por Sobre-reacción en Buenos Resultados (Post-Earnings <= 5d con EPS Surprise >= 0)
   - Caída Idiosincrática Aislada (Mercado y Sector estables, lejos de balance)
2. Entrena la Arquitectura en Cascada (Stacked Dual ML):
   - Modelo 1A (Entrada) enriquecido con el ADN del Origen de la Caída.
   - Modelo 1B (Salida en el Hito del 50% de recuperación) que recibe:
     a) El resultado del Modelo 1A (`prob_1a`, generado Out-of-Fold en Train para evitar overfitting).
     b) El ADN completo de la caída de la que viene (tipo, magnitud, fundamentales, exceso vs sector/SPY).
     c) La física en vivo del rebote (velocidad, acompañamiento del sector/SPY, rechazo en velas y techo S/R o SMA_100).
"""

import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from scripts.evaluate_multisector_motor1 import compute_core_user_features_per_ticker, deduplicate_events


def classify_shock_origin(row) -> str:
    if row["days_since_earnings"] <= 5.0:
        if row["eps_surprise_pct"] < 0:
            return "1_PostEarnings_Negativo (Castigo Balance)"
        else:
            return "2_PostEarnings_Positivo (Sobre-reaccion)"
    if row["spy_ret_3d"] <= -0.022 or row["spy_ret_5d"] <= -0.030:
        return "3_Caida_Macro_Mercado (SPY Cae Fuerte)"
    if row["sector_ret_3d"] <= -0.022 or row["sector_ret_5d"] <= -0.030:
        return "4_Caida_Sectorial (Sector Cae Fuerte)"
    return "5_Caida_Idiosincratica_Aislada (Solo la Accion)"


def main():
    t0 = time.time()
    hourly_path = PROJECT_ROOT / "data" / "processed" / "sp500_multisector_1hour.parquet"
    earn_path = PROJECT_ROOT / "data" / "processed" / "sp500_earnings_pit.parquet"

    print("[1/5] Cargando dataset horario y calculando retornos de Mercado (SPY) y Sectoriales...")
    df = pd.read_parquet(hourly_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values(["ticker", "timestamp"]).reset_index(drop=True)
    df["hour_key"] = df["timestamp"].dt.floor("h")

    # Retornos de SPY en 1d (7h), 3d (21h) y 5d (35h)
    spy = df[df["ticker"] == "SPY"].sort_values("timestamp").drop_duplicates("hour_key").set_index("hour_key")
    spy_close_map = spy["close"]
    spy_ret_3d_map = spy["close"].pct_change(21)
    spy_ret_5d_map = spy["close"].pct_change(35)

    stocks = df[~df["ticker"].isin(["SPY", "QQQ"])].copy().reset_index(drop=True)
    stocks["spy_close"] = stocks["hour_key"].map(spy_close_map).ffill()
    stocks["spy_ret_3d"] = stocks["hour_key"].map(spy_ret_3d_map).fillna(0.0)
    stocks["spy_ret_5d"] = stocks["hour_key"].map(spy_ret_5d_map).fillna(0.0)

    g = stocks.groupby("ticker", group_keys=False)
    stocks["ret_1h"] = g["close"].pct_change(1)
    stocks["ret_3d"] = g["close"].pct_change(21)
    stocks["ret_5d"] = g["close"].pct_change(35)
    stocks["ret_60d"] = g["close"].pct_change(420)
    stocks["vol_3d"] = g["ret_3d"].transform(lambda s: s.rolling(420, min_periods=70).std())
    stocks["vol_5d"] = g["ret_5d"].transform(lambda s: s.rolling(420, min_periods=70).std())
    stocks["z_3d"] = stocks["ret_3d"] / stocks["vol_3d"].replace(0, np.nan)
    stocks["z_5d"] = stocks["ret_5d"] / stocks["vol_5d"].replace(0, np.nan)

    # Retorno del Sector en tiempo real (promedio de todas las acciones del mismo sector en esa hora)
    sec_hourly = stocks.groupby(["sector", "hour_key"]).agg(
        sector_ret_1h=("ret_1h", "mean"),
        sector_ret_3d=("ret_3d", "mean"),
        sector_ret_5d=("ret_5d", "mean"),
    ).reset_index()
    # Construir índice acumulado del sector para medir cuánto sube el sector durante la recuperación
    sec_hourly = sec_hourly.sort_values(["sector", "hour_key"]).reset_index(drop=True)
    sec_hourly["sector_idx"] = sec_hourly.groupby("sector")["sector_ret_1h"].transform(
        lambda s: (1.0 + s.fillna(0.0)).cumprod()
    )

    stocks = stocks.merge(
        sec_hourly[["sector", "hour_key", "sector_ret_3d", "sector_ret_5d", "sector_idx"]],
        on=["sector", "hour_key"],
        how="left",
    )

    # Descomposición de la caída: ¿Cuánto es culpa propia (Idiosincrática) vs Sector y vs Mercado?
    stocks["excess_vs_spy_3d"] = stocks["ret_3d"] - stocks["spy_ret_3d"]
    stocks["excess_vs_sector_3d"] = stocks["ret_3d"] - stocks["sector_ret_3d"]
    stocks["sector_vs_spy_3d"] = stocks["sector_ret_3d"] - stocks["spy_ret_3d"]
    # Proporción idiosincrática: qué % de la caída de 3d NO se explica por el sector (0 = todo sectorial, 1 = 100% propia)
    stocks["idiosyncratic_ratio_3d"] = (
        (stocks["ret_3d"] - stocks["sector_ret_3d"]) / stocks["ret_3d"].clip(upper=-0.005)
    ).clip(0.0, 1.5)

    high_1d = g["high"].transform(lambda s: s.rolling(7, min_periods=3).max())
    low_1d = g["low"].transform(lambda s: s.rolling(7, min_periods=3).min())
    stocks["atr_daily_pct"] = ((high_1d - low_1d) / stocks["close"] * 100.0).groupby(stocks["ticker"]).transform(
        lambda s: s.rolling(98, min_periods=21).mean()
    )

    valid_tickers = stocks.groupby("ticker")["atr_daily_pct"].max()
    eligible_tickers = valid_tickers[valid_tickers >= 2.2].index.tolist()
    stocks = stocks[stocks["ticker"].isin(eligible_tickers)].copy().reset_index(drop=True)

    print(f"[2/5] Calculando features en {len(eligible_tickers)} empresas de alto ATR y extrayendo golpes...")
    ticker_dfs = {}
    shock_candidates = []

    for tk, sub in stocks.groupby("ticker"):
        sub_feat = compute_core_user_features_per_ticker(sub)
        rng = (sub_feat["high"] - sub_feat["low"]).replace(0, np.nan)
        sub_feat["upper_wick_ratio"] = (sub_feat["high"] - np.maximum(sub_feat["open"], sub_feat["close"])) / rng
        sub_feat["rsi_slope_3h"] = sub_feat["RSI_14"].diff(3)
        sub_feat = sub_feat.reset_index(drop=True)
        ticker_dfs[tk] = sub_feat

        mask = (
            (sub_feat["atr_daily_pct"] >= 2.2)
            & ((sub_feat["z_3d"] <= -2.0) | (sub_feat["z_5d"] <= -2.0))
            & ((sub_feat["close_loc_1h"] >= 0.55) | (sub_feat["lower_wick_ratio"] >= 0.25))
        )
        sc = sub_feat[mask].copy()
        if not sc.empty:
            sc["bar_idx"] = sc.index
            shock_candidates.append(sc)

    shocks_raw = pd.concat(shock_candidates, ignore_index=True)

    if earn_path.exists():
        earn_df = pd.read_parquet(earn_path)
        earn_df["earnings_date"] = pd.to_datetime(earn_df["earnings_date"], utc=True)
        earn_df = earn_df.sort_values(["ticker", "earnings_date"]).reset_index(drop=True)
        earn_df["eps_beat_4q_ratio"] = earn_df.groupby("ticker")["eps_beat"].transform(lambda s: s.rolling(4, min_periods=1).mean())
        earn_df["eps_surprise_4q_mean"] = earn_df.groupby("ticker")["eps_surprise_pct"].transform(lambda s: s.rolling(4, min_periods=1).mean().clip(-100, 200))

        shocks_raw = shocks_raw.sort_values("timestamp").reset_index(drop=True)
        earn_sorted = earn_df[["ticker", "earnings_date", "eps_surprise_pct", "eps_positive", "eps_beat", "eps_beat_4q_ratio", "eps_surprise_4q_mean"]].sort_values("earnings_date").reset_index(drop=True)
        shocks_raw = pd.merge_asof(shocks_raw, earn_sorted, left_on="timestamp", right_on="earnings_date", by="ticker", direction="backward")
        shocks_raw["days_since_earnings"] = ((shocks_raw["timestamp"] - shocks_raw["earnings_date"]).dt.total_seconds() / 86400.0).clip(0, 120).fillna(45.0)
        shocks_raw["eps_surprise_pct"] = shocks_raw["eps_surprise_pct"].clip(-100, 200).fillna(0.0)
        shocks_raw["eps_positive"] = shocks_raw["eps_positive"].fillna(1.0)
        shocks_raw["eps_beat_4q_ratio"] = shocks_raw["eps_beat_4q_ratio"].fillna(0.5)
        shocks_raw["eps_surprise_4q_mean"] = shocks_raw["eps_surprise_4q_mean"].fillna(0.0)

    shocks = deduplicate_events(shocks_raw, "timestamp", min_gap_hours=48)

    # Variables de interacción explícitas sobre el tipo de golpe
    shocks["is_earnings_window"] = (shocks["days_since_earnings"] <= 5.0).astype(float)
    shocks["earnings_window_x_surprise"] = shocks["is_earnings_window"] * shocks["eps_surprise_pct"]
    shocks["is_bad_earnings_shock"] = ((shocks["days_since_earnings"] <= 5.0) & (shocks["eps_surprise_pct"] < 0)).astype(float)
    shocks["is_good_earnings_shock"] = ((shocks["days_since_earnings"] <= 5.0) & (shocks["eps_surprise_pct"] >= 0)).astype(float)
    shocks["shock_origin_type"] = shocks.apply(classify_shock_origin, axis=1)

    print("[3/5] Simulando trayectorias de recuperación y extrayendo snapshots para el Modelo 1B en cascada...")
    valid_shocks = []
    meli_snapshots = []

    for event_id, row in shocks.iterrows():
        tk = row["ticker"]
        i0 = int(row["bar_idx"])
        tdf = ticker_dfs[tk]
        if i0 + 110 >= len(tdf):
            continue

        atr_d = row["atr_daily_pct"] / 100.0
        shock_drop = abs(min(row["ret_3d"], row["ret_5d"], -0.04))
        entry_p = tdf.loc[i0 + 1, "open"] * 1.0005
        sl_p = entry_p * (1.0 - 1.5 * atr_d)
        spy_entry = tdf.loc[i0 + 1, "spy_close"]
        sec_entry = tdf.loc[i0 + 1, "sector_idx"]

        fut = tdf.iloc[i0 + 1 : i0 + 106]
        highs = fut["high"].to_numpy()
        lows = fut["low"].to_numpy()
        closes = fut["close"].to_numpy()

        # Max recovery y simulación estática / candado / trailing
        max_rec = 0.0
        reaches_50 = 0
        reaches_85 = 0

        # Simulación Estática (75% TP)
        tp_75 = entry_p * (1.0 + 0.75 * shock_drop)
        exit_static = closes[-1]
        for step in range(len(fut)):
            rec = ((highs[step] - entry_p) / entry_p) / shock_drop
            if rec > max_rec:
                max_rec = rec
            if (step + 1) % 7 == 0 and closes[step] <= sl_p:
                exit_static = closes[step]
                break
            if highs[step] >= tp_75:
                exit_static = tp_75
                break
        ret_static = ((exit_static * 0.9995 - entry_p) / entry_p) * 100.0
        reaches_50 = 1 if max_rec >= 0.50 else 0
        reaches_85 = 1 if max_rec >= 0.85 else 0

        # Simulación Candado Progresivo (Cobro Conservador)
        tp_80 = entry_p * (1.0 + max(0.80 * shock_drop, 1.8 * atr_d))
        floor_candado = None
        peak_c = entry_p
        exit_candado = closes[-1]
        for step in range(len(fut)):
            if highs[step] > peak_c:
                peak_c = highs[step]
            m_rec = ((peak_c - entry_p) / entry_p) / shock_drop
            if m_rec >= 0.60:
                floor_candado = max(floor_candado or 0.0, entry_p * (1.0 + 0.45 * shock_drop))
            elif m_rec >= 0.40:
                floor_candado = max(floor_candado or 0.0, entry_p * 1.002)

            if floor_candado is not None and closes[step] <= floor_candado and step > 0:
                exit_candado = floor_candado
                break
            if (step + 1) % 7 == 0 and closes[step] <= sl_p:
                exit_candado = closes[step]
                break
            if highs[step] >= tp_80:
                exit_candado = tp_80
                break
        ret_candado = ((exit_candado * 0.9995 - entry_p) / entry_p) * 100.0

        # Simulación Trailing Stop Dinámico desde el 50%
        floor_trail = None
        peak_t = entry_p
        exit_trail = closes[-1]
        for step in range(len(fut)):
            if highs[step] > peak_t:
                peak_t = highs[step]
            m_rec = ((peak_t - entry_p) / entry_p) / shock_drop
            if m_rec >= 0.50:
                floor_trail = max(floor_trail or 0.0, peak_t * (1.0 - 0.8 * atr_d))
            if floor_trail is not None and closes[step] <= floor_trail and step > 0:
                exit_trail = floor_trail
                break
            if (step + 1) % 7 == 0 and closes[step] <= sl_p:
                exit_trail = closes[step]
                break
        ret_trail = ((exit_trail * 0.9995 - entry_p) / entry_p) * 100.0

        # Snapshot en el Hito del 50% de recuperación para el Modelo 1B en cascada
        for step in range(len(fut) - 7):
            if (step + 1) % 7 == 0 and closes[step] <= sl_p:
                break
            cur_h = highs[step]
            rec_here = ((cur_h - entry_p) / entry_p) / shock_drop
            if rec_here >= 0.50:
                rem_highs = highs[step + 1 :]
                rem_closes = closes[step + 1 :]
                if len(rem_highs) == 0:
                    break
                hit_85_after = np.max(rem_highs) >= entry_p * (1.0 + 0.85 * shock_drop)
                crashed_after = np.min(rem_closes) <= entry_p * (1.0 + 0.20 * shock_drop)
                keep_running = 1 if (hit_85_after and not crashed_after) else 0

                cur_bar = tdf.loc[i0 + 1 + step]
                spy_rebound = (cur_bar["spy_close"] - spy_entry) / spy_entry if spy_entry > 0 else 0.0
                sec_rebound = (cur_bar["sector_idx"] - sec_entry) / sec_entry if sec_entry > 0 else 0.0
                stock_rebound = 0.50 * shock_drop

                meli_snapshots.append({
                    "event_id": event_id,
                    "timestamp": cur_bar["timestamp"],
                    # --- A. Variables de la Velocidad y Geometría de la Recuperación ---
                    "steps_to_reach_50": step + 1,
                    "velocity_pct_per_bar": (stock_rebound * 100.0) / max(step + 1, 1),
                    "RSI_14_at_50": cur_bar["RSI_14"],
                    "rsi_slope_3h_at_50": cur_bar["rsi_slope_3h"],
                    "rvol_1h_at_50": cur_bar["rvol_1h"],
                    "close_loc_1h_at_50": cur_bar["close_loc_1h"],
                    "upper_wick_at_50": cur_bar["upper_wick_ratio"],
                    "sr_dist_res_atr_at_50": cur_bar["sr_dist_resistance_atr"],
                    "sr_channel_pos_at_50": cur_bar["sr_channel_position"],
                    "close_diff_SMA_100_at_50": cur_bar["close_diff_SMA_100"],
                    "BB_percent_b_at_50": cur_bar["BB_20_20_percent_b"],
                    # --- B. Variables del Viento de Cola durante el Rebote ---
                    "spy_rebound_since_entry": spy_rebound,
                    "sector_rebound_since_entry": sec_rebound,
                    "rebound_excess_vs_sector": stock_rebound - sec_rebound,
                    # --- C. Variables heredadas de la CAÍDA DE LA QUE VIENE ---
                    "shock_drop": shock_drop,
                    "entry_z_3d": row["z_3d"],
                    "entry_z_5d": row["z_5d"],
                    "entry_spy_ret_3d": row["spy_ret_3d"],
                    "entry_sector_ret_3d": row["sector_ret_3d"],
                    "entry_excess_vs_sector_3d": row["excess_vs_sector_3d"],
                    "entry_idiosyncratic_ratio": row["idiosyncratic_ratio_3d"],
                    "entry_days_since_earnings": row["days_since_earnings"],
                    "entry_eps_surprise_pct": row["eps_surprise_pct"],
                    "entry_eps_surprise_4q_mean": row["eps_surprise_4q_mean"],
                    "entry_is_bad_earnings": row["is_bad_earnings_shock"],
                    "entry_is_good_earnings": row["is_good_earnings_shock"],
                    "entry_SMA_100_momentum": row["SMA_100_pct_momentum"],
                    "entry_close_diff_SMA_100": row["close_diff_SMA_100"],
                    # Retornos según decisión
                    "ret_if_hold_trail": ret_trail,
                    "ret_if_candado": ret_candado,
                    "ret_if_take_now": (0.50 * shock_drop - 0.001) * 100.0,
                    "keep_running": keep_running,
                })
                break

        row_dict = row.to_dict()
        row_dict["event_id"] = event_id
        row_dict["reaches_50"] = reaches_50
        row_dict["reaches_85"] = reaches_85
        row_dict["max_rec"] = max_rec
        row_dict["target_1a"] = 1 if (ret_trail > 1.0 and max_rec >= 0.55) else 0
        row_dict["ret_static"] = ret_static
        row_dict["ret_candado"] = ret_candado
        row_dict["ret_trail"] = ret_trail
        valid_shocks.append(row_dict)

    df_1a = pd.DataFrame(valid_shocks).sort_values("timestamp").reset_index(drop=True)

    # =========================================================================
    # PARTE 1: RADIOGRAFÍA EMPÍRICA SEGÚN EL TIPO DE CAÍDA
    # =========================================================================
    print("\n" + "=" * 118)
    print("1. RADIOGRAFÍA EMPÍRICA POR TIPO DE CAÍDA (7,180 GOLPES EN 314 EMPRESAS DE ALTO ATR):")
    print("=" * 118)
    origin_rows = []
    for orig, grp in df_1a.groupby("shock_origin_type"):
        origin_rows.append({
            "Tipo_de_Caida": orig,
            "Eventos": len(grp),
            "Caida_Media %": round(grp["ret_3d"].mean() * 100, 2),
            "Llega_al_50% (MeLi) %": round(grp["reaches_50"].mean() * 100, 1),
            "Llega_al_85% (Completa) %": round(grp["reaches_85"].mean() * 100, 1),
            "WR_Candado %": round((grp["ret_candado"] > 0).mean() * 100, 1),
            "Ret_Candado %": round(grp["ret_candado"].mean(), 2),
            "WR_Trailing %": round((grp["ret_trail"] > 0).mean() * 100, 1),
            "Ret_Trailing %": round(grp["ret_trail"].mean(), 2),
        })
    print(pd.DataFrame(origin_rows).to_string(index=False))

    # =========================================================================
    # PARTE 2: ENTRENAMIENTO EN CASCADA (MODELO 1A -> MODELO 1B)
    # =========================================================================
    split_idx = int(len(df_1a) * 0.70)
    split_date = df_1a["timestamp"].iloc[split_idx]
    tr_1a = df_1a.iloc[:split_idx].copy().reset_index(drop=True)
    te_1a = df_1a.iloc[split_idx:].copy().reset_index(drop=True)

    feats_1a_base = [
        "SMA_100_pct_momentum", "SMA_40_pct_momentum", "close_diff_SMA_100", "close_diff_SMA_40",
        "ratio_SMA_40_SMA_100", "ratio_SMA_100_SMA_40", "ratio_SMA_100_SMA_10",
        "norm_diff_SMA_40_SMA_100", "norm_diff_SMA_10_SMA_100", "slope_SMA_100",
        "sr_dist_support_pct", "sr_dist_resistance_pct", "sr_dist_support_atr", "sr_dist_resistance_atr",
        "sr_support_touches", "sr_resistance_touches", "sr_channel_position",
        "ATR_14_pct", "ATR_21_pct", "ATR_21_relative", "BB_20_20_bandwidth", "BB_20_20_percent_b",
        "RSI_14", "close_loc_1h", "lower_wick_ratio", "body_ratio", "rvol_1h",
        "z_3d", "z_5d", "ret_60d", "spy_ret_3d",
        "days_since_earnings", "eps_surprise_pct", "eps_positive", "eps_beat_4q_ratio", "eps_surprise_4q_mean"
    ]
    feats_1a_enriched = feats_1a_base + [
        "spy_ret_5d", "sector_ret_3d", "sector_ret_5d",
        "excess_vs_spy_3d", "excess_vs_sector_3d", "sector_vs_spy_3d",
        "idiosyncratic_ratio_3d", "is_earnings_window", "earnings_window_x_surprise",
        "is_bad_earnings_shock", "is_good_earnings_shock"
    ]
    feats_1a_enriched = [f for f in feats_1a_enriched if f in df_1a.columns]

    # Generar probabilidades Out-of-Fold en Train para pasarlas limpias al Modelo 1B sin data leakage
    oof_probs_1a = np.zeros(len(tr_1a))
    kf = KFold(n_splits=5, shuffle=False)
    for idx_tr, idx_val in kf.split(tr_1a):
        m_fold = lgb.LGBMClassifier(
            n_estimators=150, learning_rate=0.03, max_depth=4, num_leaves=12,
            min_child_samples=40, subsample=0.8, colsample_bytree=0.7,
            reg_alpha=2.0, reg_lambda=8.0, random_state=42, verbose=-1
        )
        m_fold.fit(tr_1a.iloc[idx_tr][feats_1a_enriched].fillna(0.0), tr_1a.iloc[idx_tr]["target_1a"].values)
        oof_probs_1a[idx_val] = m_fold.predict_proba(tr_1a.iloc[idx_val][feats_1a_enriched].fillna(0.0))[:, 1]

    clf_1a = lgb.LGBMClassifier(
        n_estimators=150, learning_rate=0.03, max_depth=4, num_leaves=12,
        min_child_samples=40, subsample=0.8, colsample_bytree=0.7,
        reg_alpha=2.0, reg_lambda=8.0, random_state=42, verbose=-1
    )
    clf_1a.fit(tr_1a[feats_1a_enriched].fillna(0.0), tr_1a["target_1a"].values)
    probs_te_1a = clf_1a.predict_proba(te_1a[feats_1a_enriched].fillna(0.0))[:, 1]
    auc_1a_oos = roc_auc_score(te_1a["target_1a"].values, probs_te_1a)

    tr_1a["prob_1a"] = oof_probs_1a
    te_1a["prob_1a"] = probs_te_1a
    te_1a["decil"] = pd.qcut(te_1a["prob_1a"], q=10, labels=False, duplicates="drop")

    # Mapear `prob_1a` a cada snapshot del Modelo 1B
    prob_1a_map = {}
    for _, r in tr_1a.iterrows():
        prob_1a_map[int(r["event_id"])] = float(r["prob_1a"])
    for _, r in te_1a.iterrows():
        prob_1a_map[int(r["event_id"])] = float(r["prob_1a"])

    df_meli = pd.DataFrame(meli_snapshots).sort_values("timestamp").reset_index(drop=True)
    df_meli["model_1a_prob"] = df_meli["event_id"].map(prob_1a_map).fillna(0.5)

    feats_1b_old = [
        "steps_to_reach_50", "velocity_pct_per_bar", "RSI_14_at_50", "rsi_slope_3h_at_50",
        "rvol_1h_at_50", "close_loc_1h_at_50", "upper_wick_at_50",
        "sr_dist_res_atr_at_50", "sr_channel_pos_at_50", "BB_percent_b_at_50", "shock_drop"
    ]
    feats_1b_cascaded = feats_1b_old + [
        "model_1a_prob",
        "close_diff_SMA_100_at_50",
        "spy_rebound_since_entry", "sector_rebound_since_entry", "rebound_excess_vs_sector",
        "entry_z_3d", "entry_z_5d", "entry_spy_ret_3d", "entry_sector_ret_3d",
        "entry_excess_vs_sector_3d", "entry_idiosyncratic_ratio",
        "entry_days_since_earnings", "entry_eps_surprise_pct", "entry_eps_surprise_4q_mean",
        "entry_is_bad_earnings", "entry_is_good_earnings",
        "entry_SMA_100_momentum", "entry_close_diff_SMA_100"
    ]

    tr_m = df_meli[df_meli["timestamp"] < split_date].copy()
    te_m = df_meli[df_meli["timestamp"] >= split_date].copy()

    clf_1b_old = lgb.LGBMClassifier(
        n_estimators=120, learning_rate=0.03, max_depth=3, num_leaves=8,
        min_child_samples=30, subsample=0.8, colsample_bytree=0.8,
        reg_alpha=2.0, reg_lambda=8.0, random_state=42, verbose=-1
    )
    clf_1b_old.fit(tr_m[feats_1b_old].fillna(0.0), tr_m["keep_running"].values)
    auc_1b_old = roc_auc_score(te_m["keep_running"].values, clf_1b_old.predict_proba(te_m[feats_1b_old].fillna(0.0))[:, 1])

    clf_1b_casc = lgb.LGBMClassifier(
        n_estimators=140, learning_rate=0.03, max_depth=3, num_leaves=8,
        min_child_samples=30, subsample=0.8, colsample_bytree=0.8,
        reg_alpha=2.0, reg_lambda=8.0, random_state=42, verbose=-1
    )
    clf_1b_casc.fit(tr_m[feats_1b_cascaded].fillna(0.0), tr_m["keep_running"].values)
    probs_1b_casc = clf_1b_casc.predict_proba(te_m[feats_1b_cascaded].fillna(0.0))[:, 1]
    auc_1b_casc = roc_auc_score(te_m["keep_running"].values, probs_1b_casc)
    te_m["prob_keep"] = probs_1b_casc

    print("\n" + "=" * 118)
    print("2. IMPACTO EN CAPACIDAD PREDICTIVA OOS AL INYECTAR EL ORIGEN DE LA CAÍDA + CASCADA 1A -> 1B:")
    print(f"   * Modelo 1A (Entrada con Origen de Caída Sectorial/Macro/Earnings): ROC-AUC OOS = {auc_1a_oos:.4f}")
    print(f"   * Modelo 1B Antiguo (Ciego al origen de la caída y sin Modelo 1A):  ROC-AUC OOS = {auc_1b_old:.4f}")
    print(f"   * Modelo 1B EN CASCADA (Con prob_1A + ADN de la Caída + Rebote):    ROC-AUC OOS = {auc_1b_casc:.4f}")
    print("=" * 118)

    # Top 12 Features del Modelo 1B en Cascada
    imp_1b = pd.DataFrame({
        "Feature_Modelo_1B": feats_1b_cascaded,
        "Gain_Importance": clf_1b_casc.booster_.feature_importance(importance_type="gain")
    }).sort_values("Gain_Importance", ascending=False).head(12)
    print("\nTOP 12 VARIABLES QUE USA EL MODELO 1B (SALIDA) PARA DECIDIR SI COBRAR O DEJAR CORRER:")
    print(imp_1b.to_string(index=False))

    # Decisión inteligente del Modelo 1B en cascada:
    # - Si prob_keep es baja (< 0.40, ej. rebote de gato muerto en mal earnings o sin viento de cola): Cobra inmediato en el 50%
    # - Si prob_keep es media (0.40 a 0.52): Usa Candado Progresivo (asegura ganancia pero le da aire al 80%)
    # - Si prob_keep es alta (>= 0.52, rebote respaldado por Modelo 1A + sector/mercado): Usa Trailing Dinámico sin techo!
    casc_ret_map = {}
    for _, r in te_m.iterrows():
        ev_id = int(r["event_id"])
        pk = float(r["prob_keep"])
        if pk < 0.38:
            casc_ret_map[ev_id] = float(r["ret_if_take_now"])
        elif pk < 0.48:
            casc_ret_map[ev_id] = float(r["ret_if_candado"])
        else:
            casc_ret_map[ev_id] = float(r["ret_if_hold_trail"])

    te_1a["ret_cascaded_ml"] = [
        casc_ret_map.get(int(r["event_id"]), float(r["ret_candado"]))
        for _, r in te_1a.iterrows()
    ]

    print("\n" + "=" * 118)
    print("3. TABLA OOS POR DECILES DEL MODELO 1A CON SALIDA EN CASCADA (1A -> 1B):")
    print("=" * 118)
    dec_rows = []
    for dec, grp in te_1a.groupby("decil"):
        r_st = grp["ret_static"].to_numpy()
        r_ca = grp["ret_candado"].to_numpy()
        r_tr = grp["ret_trail"].to_numpy()
        r_cs = grp["ret_cascaded_ml"].to_numpy()
        w_cs = r_cs[r_cs > 0].sum()
        l_cs = abs(r_cs[r_cs <= 0].sum())
        pf_cs = w_cs / l_cs if l_cs > 0 else 99.0

        dec_rows.append({
            "Decil_1A": dec,
            "Trades": len(grp),
            "WR_Estático %": round((r_st > 0).mean() * 100, 1),
            "Ret_Estático %": round(r_st.mean(), 2),
            "WR_Candado %": round((r_ca > 0).mean() * 100, 1),
            "Ret_Candado %": round(r_ca.mean(), 2),
            "WR_Trailing %": round((r_tr > 0).mean() * 100, 1),
            "Ret_Trailing %": round(r_tr.mean(), 2),
            "WR_CASCADA_1A+1B %": round((r_cs > 0).mean() * 100, 1),
            "Ret_CASCADA_1A+1B %": round(r_cs.mean(), 2),
            "PF_CASCADA": round(pf_cs, 2),
        })
    print(pd.DataFrame(dec_rows).to_string(index=False))
    print("=" * 118)
    print(f"Tiempo total: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
