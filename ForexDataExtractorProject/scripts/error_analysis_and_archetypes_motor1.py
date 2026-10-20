"""
error_analysis_and_archetypes_motor1.py -
1. Error Analysis de la Opción A (prob_1a >= 0.61):
   - Falsos Positivos (El 11.5% de trades con prob >= 0.61 que perdieron: ¿qué tenían en común?).
   - Falsos Negativos ("Joyas Perdidas": trades que ganaron >= +4.0% neto pero el modelo les puso prob < 0.61: ¿por qué los ignoró?).
2. División del Motor 1 en Arquetipos Especializados según el Momento de Ingreso:
   - Arquetipo 1.1 (Caída por Arrastre Macro / Sectorial): Cuando cae el SPY o el Sector.
   - Arquetipo 1.2 (Caída Idiosincrática / Descuento Aislado en Mercado Tranquilo): Cuando el SPY y el Sector están calmos, pero una acción sufre un golpe de 3-5 días. Entrenado con su propio Modelo de Entrada y su propio Modelo de Salida.
3. Simulación de Portafolio combinando Motor 1.1 + Motor 1.2.
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
from scripts.backtest_portfolio_motor1_dual_ml import (
    SHOCK_TYPE_MAP,
    classify_shock_origin,
    simulate_adaptive_exit_by_shock_and_1b,
    run_portfolio_simulation,
)


def main():
    t0 = time.time()
    hourly_path = PROJECT_ROOT / "data" / "processed" / "sp500_multisector_1hour.parquet"
    earn_path = PROJECT_ROOT / "data" / "processed" / "sp500_earnings_pit.parquet"

    print("[1/4] Cargando datos y reconstruyendo universo OOS para Error Analysis...")
    df = pd.read_parquet(hourly_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values(["ticker", "timestamp"]).reset_index(drop=True)
    df["hour_key"] = df["timestamp"].dt.floor("h")

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
    stocks["ret_10d"] = g["close"].pct_change(70)
    stocks["ret_60d"] = g["close"].pct_change(420)
    stocks["vol_3d"] = g["ret_3d"].transform(lambda s: s.rolling(420, min_periods=70).std())
    stocks["vol_5d"] = g["ret_5d"].transform(lambda s: s.rolling(420, min_periods=70).std())
    stocks["z_3d"] = stocks["ret_3d"] / stocks["vol_3d"].replace(0, np.nan)
    stocks["z_5d"] = stocks["ret_5d"] / stocks["vol_5d"].replace(0, np.nan)

    sec_hourly = stocks.groupby(["sector", "hour_key"]).agg(
        sector_ret_1h=("ret_1h", "mean"),
        sector_ret_3d=("ret_3d", "mean"),
        sector_ret_5d=("ret_5d", "mean"),
    ).reset_index().sort_values(["sector", "hour_key"]).reset_index(drop=True)
    sec_hourly["sector_idx"] = sec_hourly.groupby("sector")["sector_ret_1h"].transform(
        lambda s: (1.0 + s.fillna(0.0)).cumprod()
    )

    stocks = stocks.merge(
        sec_hourly[["sector", "hour_key", "sector_ret_3d", "sector_ret_5d", "sector_idx"]],
        on=["sector", "hour_key"],
        how="left",
    )

    stocks["excess_vs_spy_3d"] = stocks["ret_3d"] - stocks["spy_ret_3d"]
    stocks["excess_vs_sector_3d"] = stocks["ret_3d"] - stocks["sector_ret_3d"]
    stocks["sector_vs_spy_3d"] = stocks["sector_ret_3d"] - stocks["spy_ret_3d"]
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

    ticker_dfs = {}
    shock_candidates = []
    for tk, sub in stocks.groupby("ticker"):
        sub_feat = compute_core_user_features_per_ticker(sub)
        rng = (sub_feat["high"] - sub_feat["low"]).replace(0, np.nan)
        sub_feat["upper_wick_ratio"] = (sub_feat["high"] - np.maximum(sub_feat["open"], sub_feat["close"])) / rng
        sub_feat["rsi_slope_3h"] = sub_feat["RSI_14"].diff(3)
        # Nuevas features candidatas sugeridas por el Error Analysis:
        # 1. ¿El golpe rompió el soporte o frenó encima? (interacción distancia soporte x toques)
        sub_feat["support_strength_score"] = sub_feat["sr_support_touches"] / (1.0 + sub_feat["sr_dist_support_atr"].clip(lower=0.0))
        # 2. ¿La tendencia estructural de 60d/SMA_100 venía alcista antes del golpe de 3d? (Pullback en tendencia alcista vs cuchillo cayendo)
        sub_feat["pre_shock_trend_55d"] = sub_feat["ret_60d"] - sub_feat["ret_5d"]
        # 3. Agotamiento de volumen vendedor en las últimas 3 barras vs caída previa
        sub_feat["rvol_3h_mean"] = sub_feat["rvol_1h"].rolling(3, min_periods=1).mean()
        sub_feat["absorption_power"] = (sub_feat["close_loc_1h"] + sub_feat["lower_wick_ratio"]) * sub_feat["rvol_1h"].clip(0.5, 4.0)

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
    shocks["is_earnings_window"] = (shocks["days_since_earnings"] <= 5.0).astype(float)
    shocks["earnings_window_x_surprise"] = shocks["is_earnings_window"] * shocks["eps_surprise_pct"]
    shocks["is_bad_earnings_shock"] = ((shocks["days_since_earnings"] <= 5.0) & (shocks["eps_surprise_pct"] < 0)).astype(float)
    shocks["is_good_earnings_shock"] = ((shocks["days_since_earnings"] <= 5.0) & (shocks["eps_surprise_pct"] >= 0)).astype(float)
    shocks["shock_origin_type"] = shocks.apply(classify_shock_origin, axis=1)
    shocks["shock_type_code"] = shocks["shock_origin_type"].map(SHOCK_TYPE_MAP)

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
        entry_ts = pd.Timestamp(tdf.loc[i0 + 1, "timestamp"])
        sl_p = entry_p * (1.0 - 1.5 * atr_d)
        spy_entry = tdf.loc[i0 + 1, "spy_close"]
        sec_entry = tdf.loc[i0 + 1, "sector_idx"]

        fut = tdf.iloc[i0 + 1 : i0 + 106]
        highs = fut["high"].to_numpy()
        closes = fut["close"].to_numpy()

        max_rec = 0.0
        floor_trail = None
        peak_t = entry_p
        exit_trail = closes[-1]
        for step in range(len(fut)):
            if highs[step] > peak_t:
                peak_t = highs[step]
            m_rec = ((peak_t - entry_p) / entry_p) / shock_drop
            if m_rec > max_rec:
                max_rec = m_rec
            if m_rec >= 0.50:
                floor_trail = max(floor_trail or 0.0, peak_t * (1.0 - 0.8 * atr_d))
            if floor_trail is not None and closes[step] <= floor_trail and step > 0:
                exit_trail = floor_trail
                break
            if (step + 1) % 7 == 0 and closes[step] <= sl_p:
                exit_trail = closes[step]
                break
        ret_trail = ((exit_trail * 0.9995 - entry_p) / entry_p) * 100.0

        for step in range(len(fut) - 7):
            if (step + 1) % 7 == 0 and closes[step] <= sl_p:
                break
            cur_h = highs[step]
            rec_here = ((cur_h - entry_p) / entry_p) / shock_drop
            if rec_here >= 0.45:
                rem_highs = highs[step + 1 :]
                rem_closes = closes[step + 1 :]
                if len(rem_highs) == 0:
                    break
                hit_85_after = np.max(rem_highs) >= entry_p * (1.0 + 0.85 * shock_drop)
                crashed_after = np.min(rem_closes) <= entry_p * (1.0 + 0.15 * shock_drop)
                keep_running = 1 if (hit_85_after and not crashed_after) else 0

                cur_bar = tdf.loc[i0 + 1 + step]
                spy_rebound = (cur_bar["spy_close"] - spy_entry) / spy_entry if spy_entry > 0 else 0.0
                sec_rebound = (cur_bar["sector_idx"] - sec_entry) / sec_entry if sec_entry > 0 else 0.0
                stock_rebound = 0.45 * shock_drop

                meli_snapshots.append({
                    "event_id": event_id,
                    "timestamp": cur_bar["timestamp"],
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
                    "spy_rebound_since_entry": spy_rebound,
                    "sector_rebound_since_entry": sec_rebound,
                    "rebound_excess_vs_sector": stock_rebound - sec_rebound,
                    "shock_drop": shock_drop,
                    "shock_type_code": row["shock_type_code"],
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
                    "keep_running": keep_running,
                })
                break

        row_dict = row.to_dict()
        row_dict["event_id"] = event_id
        row_dict["entry_ts"] = entry_ts
        row_dict["entry_p"] = entry_p
        row_dict["sl_p"] = sl_p
        row_dict["atr_d"] = atr_d
        row_dict["shock_drop"] = shock_drop
        row_dict["max_rec"] = max_rec
        row_dict["ret_trail"] = ret_trail
        row_dict["target_1a"] = 1 if (ret_trail > 1.0 and max_rec >= 0.55) else 0
        valid_shocks.append(row_dict)

    df_1a = pd.DataFrame(valid_shocks).sort_values("timestamp").reset_index(drop=True)
    df_meli = pd.DataFrame(meli_snapshots).sort_values("timestamp").reset_index(drop=True)

    feats_1a_old = [
        "SMA_100_pct_momentum", "SMA_40_pct_momentum", "close_diff_SMA_100", "close_diff_SMA_40",
        "ratio_SMA_40_SMA_100", "ratio_SMA_100_SMA_40", "ratio_SMA_100_SMA_10",
        "norm_diff_SMA_40_SMA_100", "norm_diff_SMA_10_SMA_100", "slope_SMA_100",
        "sr_dist_support_pct", "sr_dist_resistance_pct", "sr_dist_support_atr", "sr_dist_resistance_atr",
        "sr_support_touches", "sr_resistance_touches", "sr_channel_position",
        "ATR_14_pct", "ATR_21_pct", "ATR_21_relative", "BB_20_20_bandwidth", "BB_20_20_percent_b",
        "RSI_14", "close_loc_1h", "lower_wick_ratio", "body_ratio", "rvol_1h",
        "z_3d", "z_5d", "ret_60d", "spy_ret_3d",
        "days_since_earnings", "eps_surprise_pct", "eps_positive", "eps_beat_4q_ratio", "eps_surprise_4q_mean",
        "spy_ret_5d", "sector_ret_3d", "sector_ret_5d",
        "excess_vs_spy_3d", "excess_vs_sector_3d", "sector_vs_spy_3d",
        "idiosyncratic_ratio_3d", "is_earnings_window", "earnings_window_x_surprise",
        "is_bad_earnings_shock", "is_good_earnings_shock", "shock_type_code"
    ]
    feats_1a_old = [f for f in feats_1a_old if f in df_1a.columns]

    split_idx = int(len(df_1a) * 0.70)
    split_date = df_1a["timestamp"].iloc[split_idx]
    tr_1a = df_1a.iloc[:split_idx].copy().reset_index(drop=True)
    te_1a = df_1a.iloc[split_idx:].copy().reset_index(drop=True)

    clf_1a = lgb.LGBMClassifier(
        n_estimators=150, learning_rate=0.03, max_depth=4, num_leaves=12,
        min_child_samples=40, subsample=0.8, colsample_bytree=0.7,
        reg_alpha=2.0, reg_lambda=8.0, random_state=42, verbose=-1
    )
    clf_1a.fit(tr_1a[feats_1a_old].fillna(0.0), tr_1a["target_1a"].values)
    te_1a["prob_1a"] = clf_1a.predict_proba(te_1a[feats_1a_old].fillna(0.0))[:, 1]

    # =========================================================================
    # PARTE 1: ERROR ANALYSIS DE OPCIÓN A (Falsos Positivos y Falsos Negativos)
    # =========================================================================
    high_prob = te_1a[te_1a["prob_1a"] >= 0.61].copy()
    tp_group = high_prob[high_prob["ret_trail"] > 0]
    fp_group = high_prob[high_prob["ret_trail"] <= 0]

    low_prob = te_1a[te_1a["prob_1a"] < 0.61].copy()
    fn_gems = low_prob[low_prob["ret_trail"] >= 4.0]  # Joyas ocultas que ganaron >= +4.0% pero prob_1a < 0.61
    tn_group = low_prob[low_prob["ret_trail"] <= 0]

    print("\n" + "=" * 120)
    print("1. ERROR ANALYSIS DEL MODELO 1A EN OUT-OF-SAMPLE (2,175 GOLPES):")
    print(f"   * Aciertos en prob >= 0.61 (True Positives):  {len(tp_group)} trades (Retorno medio = +{tp_group['ret_trail'].mean():.2f}%)")
    print(f"   * Errores en prob >= 0.61 (False Positives):  {len(fp_group)} trades (Pérdida media = {fp_group['ret_trail'].mean():.2f}%)")
    print(f"   * JOYAS OCULTAS (False Negatives con Retorno >= +4.0% que tuvieron prob < 0.61): ¡{len(fn_gems)} operaciones!")
    print("=" * 120)

    print("\n¿POR QUÉ PERDIERON LOS FALSOS POSITIVOS (prob >= 0.61 que fallaron) VS LOS QUE GANARON?")
    diag_cols = [
        "pre_shock_trend_55d", "slope_SMA_100", "sr_dist_support_atr", "sr_support_touches",
        "support_strength_score", "absorption_power", "rvol_1h", "eps_surprise_4q_mean", "spy_ret_3d"
    ]
    comp_fp = pd.DataFrame({
        "Variable": diag_cols,
        "Media_Ganadores_0.61 (TP)": [round(tp_group[c].mean(), 4) for c in diag_cols],
        "Media_Perdedores_0.61 (FP)": [round(fp_group[c].mean(), 4) for c in diag_cols],
        "Media_Joyas_Perdidas_FN (>=+4%)": [round(fn_gems[c].mean(), 4) for c in diag_cols],
        "Media_Perdedores_Rechazados (TN)": [round(tn_group[c].mean(), 4) for c in diag_cols],
    })
    print(comp_fp.to_string(index=False))

    print("\n¿EN QUÉ TIPO DE CAÍDA ESTÁN LAS JOYAS OCULTAS (LOS 500+ TRADES GANADORES >= +4% QUE NO LLEGARON A 0.61)?")
    fn_by_type = fn_gems.groupby("shock_origin_type").agg(
        Joyas_Perdidas=("ret_trail", "count"),
        Prob_1A_Media=("prob_1a", lambda s: round(s.mean(), 3)),
        Retorno_Medio_Real_pct=("ret_trail", lambda s: round(s.mean(), 2)),
    ).reset_index()
    print(fn_by_type.to_string(index=False))

    # =========================================================================
    # PARTE 2: ARQUETIPOS ESPECIALIZADOS (MOTOR 1.1 vs MOTOR 1.2)
    # =========================================================================
    # Motor 1.1: Golpes con Arrastre Macro o Sectorial (spy_ret_3d <= -0.015 o sector_ret_3d <= -0.018)
    # Motor 1.2: Golpes en Mercado Tranquilo / Descuento Individual (spy_ret_3d > -0.015 y sector_ret_3d > -0.018 y sin mal balance)
    df_1a["is_archetype_1_1"] = ((df_1a["spy_ret_3d"] <= -0.015) | (df_1a["sector_ret_3d"] <= -0.018)).astype(int)

    feats_1_1 = feats_1a_old + ["support_strength_score", "pre_shock_trend_55d", "absorption_power", "rvol_3h_mean"]
    # Para el Arquetipo 1.2 (Descuento Individual en Mercado Tranquilo), quitamos spy_ret_3d/spy_ret_5d como atajo dominante
    # para obligar al modelo a aprender la anatomía de soporte, tendencia previa, fundamentales y absorción propia de la acción!
    feats_1_2 = [
        "SMA_100_pct_momentum", "SMA_40_pct_momentum", "close_diff_SMA_100", "close_diff_SMA_40",
        "ratio_SMA_40_SMA_100", "ratio_SMA_100_SMA_40", "ratio_SMA_100_SMA_10",
        "norm_diff_SMA_40_SMA_100", "norm_diff_SMA_10_SMA_100", "slope_SMA_100",
        "sr_dist_support_pct", "sr_dist_resistance_pct", "sr_dist_support_atr", "sr_dist_resistance_atr",
        "sr_support_touches", "sr_resistance_touches", "sr_channel_position",
        "support_strength_score", "pre_shock_trend_55d", "absorption_power", "rvol_3h_mean",
        "ATR_14_pct", "ATR_21_pct", "ATR_21_relative", "BB_20_20_bandwidth", "BB_20_20_percent_b",
        "RSI_14", "close_loc_1h", "lower_wick_ratio", "body_ratio", "rvol_1h",
        "z_3d", "z_5d", "ret_60d", "excess_vs_sector_3d", "idiosyncratic_ratio_3d",
        "days_since_earnings", "eps_surprise_pct", "eps_positive", "eps_beat_4q_ratio", "eps_surprise_4q_mean",
        "is_good_earnings_shock"
    ]

    df_1a["prob_1a"] = clf_1a.predict_proba(df_1a[feats_1a_old].fillna(0.0))[:, 1]
    tr_full = df_1a.iloc[:split_idx].copy().reset_index(drop=True)
    te_full = df_1a.iloc[split_idx:].copy().reset_index(drop=True)

    # Entrenar Arquetipo 1.1 (Caída Macro / Sectorial)
    tr_11 = tr_full[tr_full["is_archetype_1_1"] == 1].copy().reset_index(drop=True)
    te_11 = te_full[te_full["is_archetype_1_1"] == 1].copy().reset_index(drop=True)

    clf_11 = lgb.LGBMClassifier(
        n_estimators=150, learning_rate=0.03, max_depth=4, num_leaves=12,
        min_child_samples=35, subsample=0.8, colsample_bytree=0.7,
        reg_alpha=2.0, reg_lambda=8.0, random_state=42, verbose=-1
    )
    clf_11.fit(tr_11[feats_1_1].fillna(0.0), tr_11["target_1a"].values)
    te_11["prob_arch"] = clf_11.predict_proba(te_11[feats_1_1].fillna(0.0))[:, 1]
    q80_11 = np.quantile(clf_11.predict_proba(tr_11[feats_1_1].fillna(0.0))[:, 1], 0.80)

    # Entrenar Arquetipo 1.2 (Caída Individual / Mercado Tranquilo, excluyendo castigos de balance)
    tr_12 = tr_full[(tr_full["is_archetype_1_1"] == 0) & (tr_full["is_bad_earnings_shock"] == 0)].copy().reset_index(drop=True)
    te_12 = te_full[(te_full["is_archetype_1_1"] == 0) & (te_full["is_bad_earnings_shock"] == 0)].copy().reset_index(drop=True)

    clf_12 = lgb.LGBMClassifier(
        n_estimators=150, learning_rate=0.03, max_depth=4, num_leaves=12,
        min_child_samples=35, subsample=0.8, colsample_bytree=0.7,
        reg_alpha=2.0, reg_lambda=8.0, random_state=42, verbose=-1
    )
    clf_12.fit(tr_12[feats_1_2].fillna(0.0), tr_12["target_1a"].values)
    te_12["prob_arch"] = clf_12.predict_proba(te_12[feats_1_2].fillna(0.0))[:, 1]
    q85_12 = np.quantile(clf_12.predict_proba(tr_12[feats_1_2].fillna(0.0))[:, 1], 0.85)
    auc_12_old = roc_auc_score(te_12["target_1a"].values, te_12["prob_1a"].values)
    auc_12_new = roc_auc_score(te_12["target_1a"].values, te_12["prob_arch"].values)

    print("\n" + "=" * 120)
    print("2. RESULTADO DE SEPARAR EN ARQUETIPO 1.1 (CAÍDA MACRO/SECTORIAL) Y ARQUETIPO 1.2 (CAÍDA EN MERCADO TRANQUILO):")
    print(f"   * Arquetipo 1.2 (Caídas en Mercado Tranquilo): ROC-AUC con Modelo Único = {auc_12_old:.4f} -> Con Modelo Propio 1.2 = {auc_12_new:.4f}")
    sel_11 = te_11[te_11["prob_arch"] >= 0.60]
    sel_12 = te_12[te_12["prob_arch"] >= q85_12]
    print(f"   * Señales Élite Arquetipo 1.1 (Tormenta Macro/Sector): {len(sel_11)} señales | WR = {(sel_11['ret_trail']>0).mean()*100:.1f}% | Ret Medio = +{sel_11['ret_trail'].mean():.2f}%")
    print(f"   * Señales Élite Arquetipo 1.2 (Descuento en Calma):    {len(sel_12)} señales | WR = {(sel_12['ret_trail']>0).mean()*100:.1f}% | Ret Medio = +{sel_12['ret_trail'].mean():.2f}%")
    print("=" * 120)

    # Top 8 features que usa el Arquetipo 1.2 cuando el mercado está tranquilo
    imp_12 = pd.DataFrame({
        "Feature_Arquetipo_1.2": feats_1_2,
        "Gain": clf_12.booster_.feature_importance(importance_type="gain")
    }).sort_values("Gain", ascending=False).head(8)
    print("\nQUÉ MIRA EL ARQUETIPO 1.2 (MERCADO TRANQUILO) PARA ELEGIR UN DESCUENTO INDIVIDUAL:")
    print(imp_12.to_string(index=False))

    # Simular portafolio combinando Arquetipo 1.1 + Arquetipo 1.2 con sus salidas adaptativas
    # Entrenar Modelo 1B para las salidas
    prob_1a_map = {int(r["event_id"]): float(r["prob_1a"]) for _, r in df_1a.iterrows()}
    df_meli["model_1a_prob"] = df_meli["event_id"].map(prob_1a_map).fillna(0.5)
    feats_1b = [
        "model_1a_prob", "shock_type_code",
        "steps_to_reach_50", "velocity_pct_per_bar", "RSI_14_at_50", "rsi_slope_3h_at_50",
        "rvol_1h_at_50", "close_loc_1h_at_50", "upper_wick_at_50",
        "sr_dist_res_atr_at_50", "sr_channel_pos_at_50", "BB_percent_b_at_50", "shock_drop",
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
    clf_1b = lgb.LGBMClassifier(
        n_estimators=140, learning_rate=0.03, max_depth=3, num_leaves=8,
        min_child_samples=30, subsample=0.8, colsample_bytree=0.8,
        reg_alpha=2.0, reg_lambda=8.0, random_state=42, verbose=-1
    )
    clf_1b.fit(tr_m[feats_1b].fillna(0.0), tr_m["keep_running"].values)
    te_m["prob_1b"] = clf_1b.predict_proba(te_m[feats_1b].fillna(0.0))[:, 1]
    prob_1b_map = {int(r["event_id"]): float(r["prob_1b"]) for _, r in te_m.iterrows()}

    combined_candidates = pd.concat([sel_11, sel_12], ignore_index=True).sort_values("entry_ts").reset_index(drop=True)
    sim_rows = []
    for _, row in combined_candidates.iterrows():
        ev_id = int(row["event_id"])
        tk = row["ticker"]
        i0 = int(row["bar_idx"])
        tdf = ticker_dfs[tk]
        fut_df = tdf.iloc[i0 + 1 : i0 + 106]
        p_1b = prob_1b_map.get(ev_id, 0.50)
        net_ret, exit_ts, reason = simulate_adaptive_exit_by_shock_and_1b(
            fut_df, float(row["entry_p"]), float(row["sl_p"]), float(row["atr_d"]),
            float(row["shock_drop"]), int(row["shock_type_code"]), p_1b
        )
        r_dict = row.to_dict()
        r_dict["prob_1a"] = float(row["prob_arch"])
        r_dict["prob_1b"] = p_1b
        r_dict["net_ret_pct"] = net_ret
        r_dict["exit_ts"] = exit_ts
        r_dict["exit_reason"] = reason
        sim_rows.append(r_dict)

    comb_df = pd.DataFrame(sim_rows).sort_values("entry_ts").reset_index(drop=True)
    only_11_df = comb_df[comb_df["is_archetype_1_1"] == 1].copy()

    res_11_only = run_portfolio_simulation(only_11_df, initial_capital=25000.0, max_slots=5)
    res_comb_5 = run_portfolio_simulation(comb_df, initial_capital=25000.0, max_slots=5)
    res_comb_6 = run_portfolio_simulation(comb_df, initial_capital=25000.0, max_slots=6)

    print("\n" + "=" * 120)
    print("3. COMPARATIVA DE PORTAFOLIO: SOLO ARQUETIPO 1.1 VS COMBINACIÓN (ARQUETIPO 1.1 + ARQUETIPO 1.2):")
    print("=" * 120)
    comp_port = pd.DataFrame([
        {
            "Portafolio": "Solo Arquetipo 1.1 (Tormentas Macro/Sector, 5 Slots)",
            "Señales_Disp": len(only_11_df),
            "Trades_Ejecutados": res_11_only["trades_executed"],
            "Trades_Mes": round(res_11_only["trades_executed"] / 11.0, 1),
            "Win_Rate %": round(res_11_only["win_rate_pct"], 1),
            "Ret_Medio %": round(res_11_only["avg_trade_ret_pct"], 2),
            "PF": round(res_11_only["profit_factor"], 2),
            "Capital_Final": f"${res_11_only['final_capital']:,.0f}",
            "Retorno_Total": f"+{res_11_only['total_return_pct']:.1f}%",
            "Max_DD %": f"{res_11_only['max_dd_pct']:.2f}%",
            "Sharpe": round(res_11_only["sharpe"], 2),
        },
        {
            "Portafolio": "Motor 1.1 + Motor 1.2 Combinados (5 Slots)",
            "Señales_Disp": len(comb_df),
            "Trades_Ejecutados": res_comb_5["trades_executed"],
            "Trades_Mes": round(res_comb_5["trades_executed"] / 11.0, 1),
            "Win_Rate %": round(res_comb_5["win_rate_pct"], 1),
            "Ret_Medio %": round(res_comb_5["avg_trade_ret_pct"], 2),
            "PF": round(res_comb_5["profit_factor"], 2),
            "Capital_Final": f"${res_comb_5['final_capital']:,.0f}",
            "Retorno_Total": f"+{res_comb_5['total_return_pct']:.1f}%",
            "Max_DD %": f"{res_comb_5['max_dd_pct']:.2f}%",
            "Sharpe": round(res_comb_5["sharpe"], 2),
        },
        {
            "Portafolio": "Motor 1.1 + Motor 1.2 Combinados (6 Slots)",
            "Señales_Disp": len(comb_df),
            "Trades_Ejecutados": res_comb_6["trades_executed"],
            "Trades_Mes": round(res_comb_6["trades_executed"] / 11.0, 1),
            "Win_Rate %": round(res_comb_6["win_rate_pct"], 1),
            "Ret_Medio %": round(res_comb_6["avg_trade_ret_pct"], 2),
            "PF": round(res_comb_6["profit_factor"], 2),
            "Capital_Final": f"${res_comb_6['final_capital']:,.0f}",
            "Retorno_Total": f"+{res_comb_6['total_return_pct']:.1f}%",
            "Max_DD %": f"{res_comb_6['max_dd_pct']:.2f}%",
            "Sharpe": round(res_comb_6["sharpe"], 2),
        },
    ])
    print(comp_port.to_string(index=False))
    print("=" * 120)


if __name__ == "__main__":
    main()
