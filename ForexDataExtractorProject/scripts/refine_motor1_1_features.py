"""
refine_motor1_1_features.py -
Implementa las mejoras de Feature Engineering descubiertas en el Error Analysis para el Motor 1.1:
1. Normalización Vol-Adjusted del Arrastre Sectorial y Macro (`sector_z_3d`, `spy_z_3d`, y `systemic_pull_z = min(spy_z_3d, sector_z_3d)`):
   Corrige el sesgo por el cual el modelo ignoraba las 219 Joyas Sectoriales (+9.82% retorno medio) cuando el SPY no caía tanto.
2. Score de Absorción Institucional (`inst_absorption_score`) y Penalización de Rebote sin Volumen (`weak_volume_flag`):
   Filtra los Falsos Positivos que rebotaban con `rvol_1h` bajo (< 1.1x).
3. Escudo Fundamental Compuesto (`fundamental_quality_score` x `systemic_pull_z`):
   Premia a las empresas con sorpresas positivas acumuladas en los últimos 4 trimestres cuando caen arrastradas por su sector.
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

    print("[1/3] Calculando nuevas features del Error Analysis para Motor 1.1 en las 314 acciones...")
    df = pd.read_parquet(hourly_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values(["ticker", "timestamp"]).reset_index(drop=True)
    df["hour_key"] = df["timestamp"].dt.floor("h")

    spy = df[df["ticker"] == "SPY"].sort_values("timestamp").drop_duplicates("hour_key").set_index("hour_key")
    spy_close_map = spy["close"]
    spy_ret_3d = spy["close"].pct_change(21)
    spy_ret_5d = spy["close"].pct_change(35)
    spy_vol_3d = spy_ret_3d.rolling(420, min_periods=70).std().replace(0, np.nan)
    spy_z_3d_map = (spy_ret_3d / spy_vol_3d).clip(-6.0, 6.0)

    stocks = df[~df["ticker"].isin(["SPY", "QQQ"])].copy().reset_index(drop=True)
    stocks["spy_close"] = stocks["hour_key"].map(spy_close_map).ffill()
    stocks["spy_ret_3d"] = stocks["hour_key"].map(spy_ret_3d).fillna(0.0)
    stocks["spy_ret_5d"] = stocks["hour_key"].map(spy_ret_5d).fillna(0.0)
    stocks["spy_z_3d"] = stocks["hour_key"].map(spy_z_3d_map).fillna(0.0)

    g = stocks.groupby("ticker", group_keys=False)
    stocks["ret_1h"] = g["close"].pct_change(1)
    stocks["ret_3d"] = g["close"].pct_change(21)
    stocks["ret_5d"] = g["close"].pct_change(35)
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
    sec_hourly["sector_vol_3d"] = sec_hourly.groupby("sector")["sector_ret_3d"].transform(
        lambda s: s.rolling(420, min_periods=70).std().replace(0, np.nan)
    )
    sec_hourly["sector_z_3d"] = (sec_hourly["sector_ret_3d"] / sec_hourly["sector_vol_3d"]).clip(-6.0, 6.0).fillna(0.0)

    stocks = stocks.merge(
        sec_hourly[["sector", "hour_key", "sector_ret_3d", "sector_ret_5d", "sector_idx", "sector_z_3d"]],
        on=["sector", "hour_key"],
        how="left",
    )

    # Nuevas variables sistémicas/sectoriales agnósticas
    stocks["systemic_pull_ret_3d"] = np.minimum(stocks["spy_ret_3d"], stocks["sector_ret_3d"])
    stocks["systemic_pull_z_3d"] = np.minimum(stocks["spy_z_3d"], stocks["sector_z_3d"])
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

        # Features del Error Analysis:
        sub_feat["pre_shock_trend_55d"] = sub_feat["ret_60d"] - sub_feat["ret_5d"]
        sub_feat["rvol_3h_max"] = sub_feat["rvol_1h"].rolling(3, min_periods=1).max()
        sub_feat["inst_absorption_score"] = (
            sub_feat["rvol_3h_max"].clip(0.5, 4.0) * (sub_feat["close_loc_1h"] + sub_feat["lower_wick_ratio"])
        )
        sub_feat["weak_volume_flag"] = ((sub_feat["rvol_1h"] < 1.10) & (sub_feat["rvol_3h_max"] < 1.25)).astype(float)
        sub_feat["support_touch_density"] = sub_feat["sr_support_touches"] / (1.0 + sub_feat["sr_dist_support_atr"].clip(lower=0.0))

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

    # Interacción Fundamental x Arrastre Sectorial/Macro + Tendencia Previa
    shocks["fundamental_shield"] = (
        shocks["eps_positive"]
        * (0.5 + shocks["eps_beat_4q_ratio"])
        * (1.0 + shocks["eps_surprise_4q_mean"].clip(-20.0, 60.0) / 50.0)
        * (1.0 - shocks["is_bad_earnings_shock"])
    )
    shocks["sector_pull_x_fundamentals"] = (-shocks["systemic_pull_z_3d"].clip(upper=0.0)) * shocks["fundamental_shield"]
    shocks["sector_pull_x_absorption"] = (-shocks["systemic_pull_z_3d"].clip(upper=0.0)) * shocks["inst_absorption_score"]

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

    feats_old = [
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

    feats_new_1_1 = feats_old + [
        "spy_z_3d", "sector_z_3d", "systemic_pull_ret_3d", "systemic_pull_z_3d",
        "pre_shock_trend_55d", "rvol_3h_max", "inst_absorption_score", "weak_volume_flag",
        "support_touch_density", "fundamental_shield", "sector_pull_x_fundamentals", "sector_pull_x_absorption"
    ]

    split_idx = int(len(df_1a) * 0.70)
    split_date = df_1a["timestamp"].iloc[split_idx]
    tr_1a = df_1a.iloc[:split_idx].copy().reset_index(drop=True)
    te_1a = df_1a.iloc[split_idx:].copy().reset_index(drop=True)

    clf_old = lgb.LGBMClassifier(
        n_estimators=150, learning_rate=0.03, max_depth=4, num_leaves=12,
        min_child_samples=40, subsample=0.8, colsample_bytree=0.7,
        reg_alpha=2.0, reg_lambda=8.0, random_state=42, verbose=-1
    )
    clf_old.fit(tr_1a[feats_old].fillna(0.0), tr_1a["target_1a"].values)
    te_1a["prob_old"] = clf_old.predict_proba(te_1a[feats_old].fillna(0.0))[:, 1]
    auc_old = roc_auc_score(te_1a["target_1a"].values, te_1a["prob_old"].values)

    clf_new = lgb.LGBMClassifier(
        n_estimators=150, learning_rate=0.03, max_depth=4, num_leaves=12,
        min_child_samples=40, subsample=0.8, colsample_bytree=0.7,
        reg_alpha=2.0, reg_lambda=8.0, random_state=42, verbose=-1
    )
    clf_new.fit(tr_1a[feats_new_1_1].fillna(0.0), tr_1a["target_1a"].values)
    te_1a["prob_new"] = clf_new.predict_proba(te_1a[feats_new_1_1].fillna(0.0))[:, 1]
    auc_new = roc_auc_score(te_1a["target_1a"].values, te_1a["prob_new"].values)

    print(f"[2/3] ROC-AUC OOS Modelo 1A Antiguo = {auc_old:.4f} -> Modelo 1.1 Refinado = {auc_new:.4f}")

    # Entrenar Modelo 1B
    oof_probs_new = np.zeros(len(tr_1a))
    kf = KFold(n_splits=5, shuffle=False)
    for idx_tr, idx_val in kf.split(tr_1a):
        m_fold = lgb.LGBMClassifier(
            n_estimators=150, learning_rate=0.03, max_depth=4, num_leaves=12,
            min_child_samples=40, subsample=0.8, colsample_bytree=0.7,
            reg_alpha=2.0, reg_lambda=8.0, random_state=42, verbose=-1
        )
        m_fold.fit(tr_1a.iloc[idx_tr][feats_new_1_1].fillna(0.0), tr_1a.iloc[idx_tr]["target_1a"].values)
        oof_probs_new[idx_val] = m_fold.predict_proba(tr_1a.iloc[idx_val][feats_new_1_1].fillna(0.0))[:, 1]

    tr_1a["prob_new"] = oof_probs_new
    prob_1a_map = {int(r["event_id"]): float(r["prob_new"]) for _, r in pd.concat([tr_1a, te_1a]).iterrows()}
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

    oos_rows = []
    for _, row in te_1a.iterrows():
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
        r_dict["prob_1b"] = p_1b
        r_dict["net_ret_pct"] = net_ret
        r_dict["exit_ts"] = exit_ts
        r_dict["exit_reason"] = reason
        oos_rows.append(r_dict)

    oos_df = pd.DataFrame(oos_rows).sort_values("entry_ts").reset_index(drop=True)

    # Comparar cuántas señales superan 0.61 y 0.59 antes vs después
    old_61 = oos_df[oos_df["prob_old"] >= 0.61].copy()
    new_61 = oos_df[oos_df["prob_new"] >= 0.61].copy()
    new_59 = oos_df[(oos_df["prob_new"] >= 0.59) & (oos_df["weak_volume_flag"] == 0)].copy()

    print("\n" + "=" * 120)
    print("IMPACTO DEL REFINAMIENTO DE FEATURES EN LAS SEÑALES >= 0.61 (Y >= 0.59 SIN BAJO VOLUMEN):")
    print(f"  * Antiguo (prob_old >= 0.61):                 {len(old_61)} señales | WR = {(old_61['net_ret_pct']>0).mean()*100:.1f}% | Ret Medio = +{old_61['net_ret_pct'].mean():.2f}%")
    print(f"  * Nuevo Motor 1.1 (prob_new >= 0.61):         {len(new_61)} señales | WR = {(new_61['net_ret_pct']>0).mean()*100:.1f}% | Ret Medio = +{new_61['net_ret_pct'].mean():.2f}%")
    print(f"  * Nuevo Motor 1.1 (prob_new >= 0.59 + VolOK): {len(new_59)} señales | WR = {(new_59['net_ret_pct']>0).mean()*100:.1f}% | Ret Medio = +{new_59['net_ret_pct'].mean():.2f}%")
    print("=" * 120)

    old_61["prob_1a"] = old_61["prob_old"]
    new_61["prob_1a"] = new_61["prob_new"]
    new_59["prob_1a"] = new_59["prob_new"]

    cases = [
        ("1. Antiguo Motor 1A (prob >= 0.61, 5 Slots)", old_61, 5),
        ("2. Nuevo Motor 1.1 Refinado (prob >= 0.61, 5 Slots)", new_61, 5),
        ("3. Nuevo Motor 1.1 Refinado (prob >= 0.59 + VolOK, 5 Slots)", new_59, 5),
        ("4. Nuevo Motor 1.1 Refinado (prob >= 0.59 + VolOK, 6 Slots)", new_59, 6),
    ]
    rows = []
    for label, sub_c, slots in cases:
        res = run_portfolio_simulation(sub_c, initial_capital=25000.0, max_slots=slots)
        rows.append({
            "Configuración": label,
            "Señales_Disp": len(sub_c),
            "Trades_Ejecutados": res["trades_executed"],
            "Trades_Mes": round(res["trades_executed"] / 11.0, 1),
            "Win_Rate %": round(res["win_rate_pct"], 1),
            "Ret_Medio %": round(res["avg_trade_ret_pct"], 2),
            "PF": round(res["profit_factor"], 2),
            "Capital_Final": f"${res['final_capital']:,.0f}",
            "Retorno_Total": f"+{res['total_return_pct']:.1f}%",
            "Max_DD %": f"{res['max_dd_pct']:.2f}%",
            "Sharpe": round(res["sharpe"], 2),
        })
    print(pd.DataFrame(rows).to_string(index=False))
    print("=" * 120)


if __name__ == "__main__":
    main()
