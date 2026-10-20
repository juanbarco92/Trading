"""
Diagnóstico Forense en Producción Estricta (IBKR):
1. ¿Qué pasa con las señales que estamos dejando por fuera del tradeo (`prob_1a` entre 0.50 y 0.59)?
   - ¿Por qué bajaron las señales de 426 a 134 en el simulador estricto? (Efecto del cambio en la tasa base del Target 1A).
   - ¿Cuánto ganan realmente en producción estricta las señales entre `0.53` y `0.59` que dejamos afuera?
2. ¿Cuál modelo falla más en Producción Estricta: el de ENTRAR (Modelo 1A) o el de SALIR (Modelo 1B)?
   - Desglose matemático del embudo:
     a) Error de Entrada (1A): Operaciones que entran pero NUNCA llegan siquiera al hito del 45%-50% ("Momento MeLi").
     b) Error de Salida (1B): Operaciones que SÍ llegan al 45%-50% (1A acertó), pero al salir:
        - O devolvemos la ganancia por dejar correr cuando iba a revertir (Error 1B de Codicia / Gap),
        - O cortamos temprano con +2% cuando la acción siguió hasta +8% (Error 1B de Salida Prematura).
   - Comparación con Oráculo 1A vs Oráculo 1B para medir dónde está el mayor cuello de botella.
"""

import sys
import time
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.backtest_portfolio_motor1_dual_ml import run_portfolio_simulation
from scripts.forensic_audit_motor1_production import simulate_strict_production_exit


def main():
    t0 = time.time()
    cache_1a_path = PROJECT_ROOT / "data" / "processed" / "strict_prod_df_1a.parquet"
    cache_meli_path = PROJECT_ROOT / "data" / "processed" / "strict_prod_df_meli.parquet"

    hourly_path = PROJECT_ROOT / "data" / "processed" / "sp500_multisector_1hour.parquet"
    df_raw = pd.read_parquet(hourly_path)
    df_raw["timestamp"] = pd.to_datetime(df_raw["timestamp"], utc=True)
    df_raw = df_raw.sort_values(["ticker", "timestamp"]).reset_index(drop=True)

    # Si no existe caché de señales estrictas, importamos y corremos la extracción una vez
    if not (cache_1a_path.exists() and cache_meli_path.exists()):
        print("[1/3] Generando y guardando caché de señales de Producción Estricta...")
        from scripts.evaluate_multisector_motor1 import compute_core_user_features_per_ticker, deduplicate_events
        from scripts.backtest_portfolio_motor1_dual_ml import SHOCK_TYPE_MAP, classify_shock_origin
        earn_path = PROJECT_ROOT / "data" / "processed" / "sp500_earnings_pit.parquet"
        macro_path = PROJECT_ROOT / "data" / "processed" / "macro_perception_1hour.parquet"

        earn_df = pd.read_parquet(earn_path)
        earn_df["earnings_date"] = pd.to_datetime(earn_df["earnings_date"], utc=True)

        df = df_raw.copy()
        df["bar_close_ts"] = df["timestamp"] + pd.Timedelta(hours=1)
        df["hour_key"] = df["timestamp"].dt.floor("h")

        macro_df = pd.read_parquet(macro_path)
        macro_df["hour_key"] = pd.to_datetime(macro_df["hour_key"], utc=True)
        macro_df = macro_df.set_index("hour_key")

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

        for mcol in ["vix", "vix_ret_3d", "vix_slope_3h", "vix_z_20d", "vix_term_ratio", "tnx_ret_3d", "tnx_z_20d"]:
            stocks[mcol] = stocks["hour_key"].map(macro_df[mcol]).ffill().fillna(0.0)

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

        daily_closes = (
            stocks.groupby(["ticker", stocks["timestamp"].dt.floor("D")])["close"]
            .last()
            .reset_index()
            .rename(columns={"timestamp": "day_ts"})
            .sort_values(["ticker", "day_ts"])
            .reset_index(drop=True)
        )
        g_d = daily_closes.groupby("ticker", group_keys=False)
        daily_closes["close_prev_1d"] = g_d["close"].shift(1)
        daily_closes["close_next_1d"] = g_d["close"].shift(-1)
        daily_closes["close_next_3d"] = g_d["close"].shift(-3)
        daily_closes["react_1d"] = (daily_closes["close_next_1d"] - daily_closes["close_prev_1d"]) / daily_closes["close_prev_1d"]
        daily_closes["react_3d"] = (daily_closes["close_next_3d"] - daily_closes["close_prev_1d"]) / daily_closes["close_prev_1d"]

        earn_df = earn_df.sort_values(["ticker", "earnings_date"]).reset_index(drop=True)
        earn_df["eps_beat_4q_ratio"] = earn_df.groupby("ticker")["eps_beat"].transform(lambda s: s.rolling(4, min_periods=1).mean())
        earn_df["eps_surprise_4q_mean"] = earn_df.groupby("ticker")["eps_surprise_pct"].transform(lambda s: s.rolling(4, min_periods=1).mean().clip(-100, 200))
        earn_df["earnings_known_ts"] = earn_df["earnings_date"] + pd.Timedelta(hours=24)

        earn_sorted = earn_df.sort_values("earnings_date").reset_index(drop=True)
        daily_sorted = daily_closes[["ticker", "day_ts", "react_1d", "react_3d"]].sort_values("day_ts").reset_index(drop=True)
        merged_e = pd.merge_asof(earn_sorted, daily_sorted, left_on="earnings_date", right_on="day_ts", by="ticker", direction="backward")
        merged_e["last_earnings_reaction_1d"] = merged_e["react_1d"].clip(-0.40, 0.40).fillna(0.0)
        merged_e["last_earnings_reaction_3d"] = merged_e["react_3d"].clip(-0.50, 0.50).fillna(0.0)
        merged_e["guidance_disappointment_flag"] = ((merged_e["last_earnings_reaction_1d"] <= -0.03) | (merged_e["last_earnings_reaction_3d"] <= -0.04)).astype(float)
        merged_e["reaction_known_ts"] = merged_e["earnings_date"] + pd.Timedelta(days=5)

        ticker_dfs = {}
        shock_candidates = []
        for tk, sub in stocks.groupby("ticker"):
            if sub["atr_daily_pct"].max() < 2.2:
                continue
            sub_feat = compute_core_user_features_per_ticker(sub)
            rng = (sub_feat["high"] - sub_feat["low"]).replace(0, np.nan)
            sub_feat["upper_wick_ratio"] = (sub_feat["high"] - np.maximum(sub_feat["open"], sub_feat["close"])) / rng
            sub_feat["rsi_slope_3h"] = sub_feat["RSI_14"].diff(3)

            sub_feat["pre_shock_trend_55d"] = sub_feat["ret_60d"] - sub_feat["ret_5d"]
            sub_feat["rvol_3h_max"] = sub_feat["rvol_1h"].rolling(3, min_periods=1).max()
            sub_feat["inst_absorption_score"] = (
                sub_feat["rvol_3h_max"].clip(0.5, 4.0) * (sub_feat["close_loc_1h"] + sub_feat["lower_wick_ratio"])
            )
            sub_feat["weak_volume_flag"] = ((sub_feat["rvol_1h"] < 1.10) & (sub_feat["rvol_3h_max"] < 1.25)).astype(float)
            sub_feat["support_touch_density"] = sub_feat["sr_support_touches"] / (1.0 + sub_feat["sr_dist_support_atr"].clip(lower=0.0))

            mf_mult = (((sub_feat["close"] - sub_feat["low"]) - (sub_feat["high"] - sub_feat["close"])) / rng).fillna(0.0)
            mf_vol = mf_mult * sub_feat["volume"]
            sub_feat["cmf_15d"] = (
                mf_vol.rolling(105, min_periods=21).sum() / sub_feat["volume"].rolling(105, min_periods=21).sum().replace(0, np.nan)
            ).fillna(0.0)

            is_down = (sub_feat["close"] < sub_feat["open"]).astype(float)
            is_up = (sub_feat["close"] > sub_feat["open"]).astype(float)
            down_vol_15d = (sub_feat["volume"] * is_down).rolling(105, min_periods=21).sum()
            up_vol_15d = (sub_feat["volume"] * is_up).rolling(105, min_periods=21).sum().replace(0, np.nan)
            sub_feat["down_up_vol_ratio_15d"] = (down_vol_15d / up_vol_15d).clip(0.2, 4.0).fillna(1.0)

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

        shocks_raw = pd.concat(shock_candidates, ignore_index=True).sort_values("bar_close_ts").reset_index(drop=True)

        earn_basic = earn_df[["ticker", "earnings_known_ts", "eps_surprise_pct", "eps_positive", "eps_beat", "eps_beat_4q_ratio", "eps_surprise_4q_mean"]].sort_values("earnings_known_ts").reset_index(drop=True)
        shocks_raw = pd.merge_asof(shocks_raw, earn_basic, left_on="bar_close_ts", right_on="earnings_known_ts", by="ticker", direction="backward")
        shocks_raw["days_since_earnings"] = ((shocks_raw["bar_close_ts"] - shocks_raw["earnings_known_ts"]).dt.total_seconds() / 86400.0 + 1.0).clip(0, 120).fillna(45.0)
        shocks_raw["eps_surprise_pct"] = shocks_raw["eps_surprise_pct"].clip(-100, 200).fillna(0.0)
        shocks_raw["eps_positive"] = shocks_raw["eps_positive"].fillna(1.0)
        shocks_raw["eps_beat_4q_ratio"] = shocks_raw["eps_beat_4q_ratio"].fillna(0.5)
        shocks_raw["eps_surprise_4q_mean"] = shocks_raw["eps_surprise_4q_mean"].fillna(0.0)

        earn_react = merged_e[["ticker", "reaction_known_ts", "last_earnings_reaction_1d", "last_earnings_reaction_3d", "guidance_disappointment_flag"]].sort_values("reaction_known_ts").reset_index(drop=True)
        shocks_raw = pd.merge_asof(shocks_raw, earn_react, left_on="bar_close_ts", right_on="reaction_known_ts", by="ticker", direction="backward")
        shocks_raw["last_earnings_reaction_1d"] = shocks_raw["last_earnings_reaction_1d"].fillna(0.0)
        shocks_raw["last_earnings_reaction_3d"] = shocks_raw["last_earnings_reaction_3d"].fillna(0.0)
        shocks_raw["guidance_disappointment_flag"] = shocks_raw["guidance_disappointment_flag"].fillna(0.0)

        shocks = deduplicate_events(shocks_raw, "timestamp", min_gap_hours=48)
        shocks["is_earnings_window"] = (shocks["days_since_earnings"] <= 5.0).astype(float)
        shocks["earnings_window_x_surprise"] = shocks["is_earnings_window"] * shocks["eps_surprise_pct"]
        shocks["is_bad_earnings_shock"] = ((shocks["days_since_earnings"] <= 5.0) & (shocks["eps_surprise_pct"] < 0)).astype(float)
        shocks["is_good_earnings_shock"] = ((shocks["days_since_earnings"] <= 5.0) & (shocks["eps_surprise_pct"] >= 0)).astype(float)
        shocks["shock_origin_type"] = shocks.apply(classify_shock_origin, axis=1)
        shocks["shock_type_code"] = shocks["shock_origin_type"].map(SHOCK_TYPE_MAP)

        shocks["fundamental_shield"] = (
            shocks["eps_positive"]
            * (0.5 + shocks["eps_beat_4q_ratio"])
            * (1.0 + shocks["eps_surprise_4q_mean"].clip(-20.0, 60.0) / 50.0)
            * (1.0 - shocks["is_bad_earnings_shock"])
            * (1.0 - 0.4 * shocks["guidance_disappointment_flag"])
        )
        shocks["sector_pull_x_fundamentals"] = (-shocks["systemic_pull_z_3d"].clip(upper=0.0)) * shocks["fundamental_shield"]
        shocks["sector_pull_x_absorption"] = (-shocks["systemic_pull_z_3d"].clip(upper=0.0)) * shocks["inst_absorption_score"]

        cand_df = shocks.sort_values("timestamp").reset_index(drop=True)
        valid_shocks = []
        meli_snapshots = []

        for event_id, row in cand_df.iterrows():
            tk = row["ticker"]
            i0 = int(row["bar_idx"])
            tdf = ticker_dfs[tk]
            if i0 + 106 >= len(tdf):
                continue
            atr_d = row["atr_daily_pct"] / 100.0
            shock_drop = abs(min(row["ret_3d"], row["ret_5d"], -0.04))
            entry_p = tdf.loc[i0 + 1, "open"] * 1.0005
            entry_ts = pd.Timestamp(tdf.loc[i0 + 1, "timestamp"])
            sl_p = entry_p * (1.0 - 1.5 * atr_d)
            spy_entry = tdf.loc[i0 + 1, "spy_close"]
            sec_entry = tdf.loc[i0 + 1, "sector_idx"]

            fut = tdf.iloc[i0 + 1 : i0 + 106]
            opens = fut["open"].to_numpy()
            highs = fut["high"].to_numpy()
            lows = fut["low"].to_numpy()
            closes = fut["close"].to_numpy()

            max_rec = 0.0
            floor_trail = None
            peak_t = entry_p
            exit_trail = closes[-1]
            for step in range(len(fut)):
                if floor_trail is not None and step > 0:
                    if opens[step] <= floor_trail:
                        exit_trail = opens[step]
                        break
                    elif lows[step] <= floor_trail:
                        exit_trail = floor_trail
                        break
                if (step + 1) % 7 == 0 and closes[step] <= sl_p:
                    exit_trail = opens[step + 1] if (step + 1 < len(fut)) else closes[step]
                    break
                if highs[step] > peak_t:
                    peak_t = highs[step]
                m_rec = ((peak_t - entry_p) / entry_p) / shock_drop
                if m_rec > max_rec:
                    max_rec = m_rec
                if m_rec >= 0.50:
                    floor_trail = max(floor_trail or 0.0, peak_t * (1.0 - 0.8 * atr_d))
            ret_trail_strict = ((exit_trail * 0.9995 - entry_p) / entry_p) * 100.0

            # También guardamos el target original para comparar el efecto del desplazamiento de probabilidades
            max_rec_orig = 0.0
            floor_orig = None
            peak_orig = entry_p
            exit_orig = closes[-1]
            for step in range(len(fut)):
                if highs[step] > peak_orig:
                    peak_orig = highs[step]
                m_rec_o = ((peak_orig - entry_p) / entry_p) / shock_drop
                if m_rec_o > max_rec_orig:
                    max_rec_orig = m_rec_o
                if m_rec_o >= 0.50:
                    floor_orig = max(floor_orig or 0.0, peak_orig * (1.0 - 0.8 * atr_d))
                if (step + 1) % 7 == 0 and closes[step] <= sl_p:
                    exit_orig = closes[step]
                    break
                if floor_orig is not None and closes[step] <= floor_orig:
                    exit_orig = floor_orig
                    break
            ret_trail_orig = ((exit_orig * 0.9995 - entry_p) / entry_p) * 100.0

            meli_trigger_step = 999
            for step in range(len(fut) - 7):
                if (step + 1) % 7 == 0 and closes[step] <= sl_p:
                    break
                cur_h = highs[step]
                rec_here = ((cur_h - entry_p) / entry_p) / shock_drop
                if rec_here >= 0.45:
                    meli_trigger_step = step
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
                        "entry_last_earnings_reaction_3d": row["last_earnings_reaction_3d"],
                        "entry_cmf_15d": row["cmf_15d"],
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
            row_dict["meli_trigger_step"] = meli_trigger_step
            row_dict["max_rec_15d"] = max_rec
            row_dict["ret_trail_strict"] = ret_trail_strict
            row_dict["target_1a_strict"] = 1 if (ret_trail_strict > 1.0 and max_rec >= 0.55) else 0
            row_dict["target_1a_orig"] = 1 if (ret_trail_orig > 1.0 and max_rec_orig >= 0.55) else 0
            valid_shocks.append(row_dict)

        df_1a = pd.DataFrame(valid_shocks).sort_values("timestamp").reset_index(drop=True)
        df_meli = pd.DataFrame(meli_snapshots).sort_values("timestamp").reset_index(drop=True)
        df_1a.to_parquet(cache_1a_path, index=False)
        df_meli.to_parquet(cache_meli_path, index=False)
    else:
        print("[1/3] Cargando caché de señales de Producción Estricta...")
        df_1a = pd.read_parquet(cache_1a_path)
        df_meli = pd.read_parquet(cache_meli_path)

    # Indexar rápidamente ticker_dfs para simular salidas en segundos
    ticker_dfs = {
        tk: gdf.sort_values("timestamp").reset_index(drop=True)
        for tk, gdf in df_raw[~df_raw["ticker"].isin(["SPY", "QQQ"])].groupby("ticker")
    }

    feats_stock_perception = [
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
        "is_bad_earnings_shock", "is_good_earnings_shock", "shock_type_code",
        "spy_z_3d", "sector_z_3d", "systemic_pull_ret_3d", "systemic_pull_z_3d",
        "pre_shock_trend_55d", "rvol_3h_max", "inst_absorption_score", "weak_volume_flag",
        "support_touch_density", "fundamental_shield", "sector_pull_x_fundamentals", "sector_pull_x_absorption",
        "last_earnings_reaction_1d", "last_earnings_reaction_3d", "guidance_disappointment_flag",
        "cmf_15d", "down_up_vol_ratio_15d",
    ]

    test_start_ts = pd.Timestamp("2025-11-17", tz="UTC")
    train_end_ts = test_start_ts - pd.Timedelta(days=22)

    tr_1a = df_1a[df_1a["timestamp"] <= train_end_ts].copy().reset_index(drop=True)
    te_1a = df_1a[df_1a["timestamp"] >= test_start_ts].copy().reset_index(drop=True)

    target_col = "target_1a_strict" if "target_1a_strict" in tr_1a.columns else "target_1a"

    clf_1a = lgb.LGBMClassifier(
        n_estimators=160, learning_rate=0.03, max_depth=4, num_leaves=12,
        min_child_samples=45, subsample=0.8, colsample_bytree=0.7,
        reg_alpha=2.0, reg_lambda=8.0, random_state=42, verbose=-1
    )
    clf_1a.fit(tr_1a[feats_stock_perception].fillna(0.0), tr_1a[target_col].values)
    te_1a["prob_1a"] = clf_1a.predict_proba(te_1a[feats_stock_perception].fillna(0.0))[:, 1]

    oof_probs = np.zeros(len(tr_1a))
    kf = KFold(n_splits=5, shuffle=False)
    for idx_tr, idx_val in kf.split(tr_1a):
        m_fold = lgb.LGBMClassifier(
            n_estimators=160, learning_rate=0.03, max_depth=4, num_leaves=12,
            min_child_samples=45, subsample=0.8, colsample_bytree=0.7,
            reg_alpha=2.0, reg_lambda=8.0, random_state=42, verbose=-1
        )
        m_fold.fit(tr_1a.iloc[idx_tr][feats_stock_perception].fillna(0.0), tr_1a.iloc[idx_tr][target_col].values)
        oof_probs[idx_val] = m_fold.predict_proba(tr_1a.iloc[idx_val][feats_stock_perception].fillna(0.0))[:, 1]

    tr_1a["prob_1a"] = oof_probs
    prob_1a_map = {int(r["event_id"]): float(r["prob_1a"]) for _, r in pd.concat([tr_1a, te_1a]).iterrows()}
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
        "entry_SMA_100_momentum", "entry_close_diff_SMA_100",
        "entry_last_earnings_reaction_3d", "entry_cmf_15d"
    ]
    tr_m = df_meli[df_meli["timestamp"] <= train_end_ts].copy()
    te_m = df_meli[df_meli["timestamp"] >= test_start_ts].copy()
    clf_1b = lgb.LGBMClassifier(
        n_estimators=140, learning_rate=0.03, max_depth=3, num_leaves=8,
        min_child_samples=35, subsample=0.8, colsample_bytree=0.8,
        reg_alpha=2.0, reg_lambda=8.0, random_state=42, verbose=-1
    )
    clf_1b.fit(tr_m[feats_1b].fillna(0.0), tr_m["keep_running"].values)
    te_m["prob_1b"] = clf_1b.predict_proba(te_m[feats_1b].fillna(0.0))[:, 1]
    prob_1b_map = {int(r["event_id"]): float(r["prob_1b"]) for _, r in te_m.iterrows()}
    keep_running_oracle_map = {int(r["event_id"]): int(r["keep_running"]) for _, r in te_m.iterrows()}

    print("[2/3] Simulando todas las señales OOS en modo Producción Estricta (IBKR Stop-Market + Gaps)...")
    rows_strict = []
    for _, row in te_1a.iterrows():
        ev_id = int(row["event_id"])
        tk = row["ticker"]
        i0 = int(row["bar_idx"])
        tdf = ticker_dfs[tk]
        fut_df = tdf.iloc[i0 + 1 : i0 + 106]
        p_1b = prob_1b_map.get(ev_id, 0.50)
        oracle_1b = 0.85 if keep_running_oracle_map.get(ev_id, 0) == 1 else 0.20
        trig_step = int(row["meli_trigger_step"])

        # Salida Real 1B en IBKR (Stop desde t+1 con gaps)
        ret_ps, ts_ps, rsn_ps = simulate_strict_production_exit(
            fut_df, float(row["entry_p"]), float(row["sl_p"]), float(row["atr_d"]),
            float(row["shock_drop"]), int(row["shock_type_code"]), p_1b, trig_step,
            execution_mode="broker_stop_order_with_gap"
        )
        # Salida con Oráculo 1B (para medir cuánto dinero pierde 1B vs un 1B perfecto)
        ret_oracle_1b, _, _ = simulate_strict_production_exit(
            fut_df, float(row["entry_p"]), float(row["sl_p"]), float(row["atr_d"]),
            float(row["shock_drop"]), int(row["shock_type_code"]), oracle_1b, trig_step,
            execution_mode="broker_stop_order_with_gap"
        )
        # Máxima excursión favorable (MFE) que alcanzó el trade en los 15 días
        max_high = fut_df["high"].max()
        mfe_pct = ((max_high - float(row["entry_p"])) / float(row["entry_p"])) * 100.0
        max_rec_reached = (mfe_pct / 100.0) / float(row["shock_drop"])

        d = row.to_dict()
        d.update({
            "prob_1b": p_1b,
            "oracle_keep_running": keep_running_oracle_map.get(ev_id, -1),
            "net_ret_pct": ret_ps,
            "oracle_1b_ret_pct": ret_oracle_1b,
            "exit_ts": ts_ps,
            "exit_reason": rsn_ps,
            "mfe_pct": mfe_pct,
            "max_rec_reached": max_rec_reached,
            "reached_meli_45": int(trig_step < 900),
        })
        rows_strict.append(d)

    df_s = pd.DataFrame(rows_strict).sort_values("entry_ts").reset_index(drop=True)

    # 1. ¿QUÉ PASA CON LAS SEÑALES QUE ESTAMOS DEJANDO POR FUERA?
    # Comparemos los tramos de probabilidad de `prob_1a` en Producción Estricta
    base_shield = (
        (df_s["weak_volume_flag"] == 0)
        & ~((df_s["last_earnings_reaction_3d"] <= -0.05) & (df_s["cmf_15d"] <= -0.05))
        & ~((df_s["tnx_ret_3d"] >= 0.030) & (df_s["vix_slope_3h"] >= 0.04))
    )
    df_clean = df_s[base_shield].copy()

    print("\n" + "=" * 120)
    print("1. ¿QUÉ PASA CON LAS SEÑALES QUE DEJAMOS POR FUERA? (Tramos de `prob_1a` en Producción Estricta IBKR):")
    print(f"   * Tasa base de Target_1A en Train: {tr_1a[target_col].mean()*100:.1f}% (por eso la escala de prob_1a bajó ~0.04 puntos!)")
    print("=" * 120)
    bins = [
        ("Tramo 1: Top Francotirador (prob_1a >= 0.61)", (df_clean["prob_1a"] >= 0.61)),
        ("Tramo 2: Alta Convicción (0.59 <= prob_1a < 0.61)", (df_clean["prob_1a"] >= 0.59) & (df_clean["prob_1a"] < 0.61)),
        ("Tramo 3: DEJADAS AFUERA A (0.56 <= prob_1a < 0.59)", (df_clean["prob_1a"] >= 0.56) & (df_clean["prob_1a"] < 0.59)),
        ("Tramo 4: DEJADAS AFUERA B (0.53 <= prob_1a < 0.56)", (df_clean["prob_1a"] >= 0.53) & (df_clean["prob_1a"] < 0.56)),
        ("Tramo 5: DEJADAS AFUERA C (0.50 <= prob_1a < 0.53)", (df_clean["prob_1a"] >= 0.50) & (df_clean["prob_1a"] < 0.53)),
        ("Tramo 6: Cola Baja (prob_1a < 0.50)", (df_clean["prob_1a"] < 0.50)),
    ]
    b_rows = []
    for b_name, b_mask in bins:
        sub = df_clean[b_mask]
        if len(sub) == 0:
            continue
        w = sub[sub["net_ret_pct"] > 0]["net_ret_pct"].sum()
        l = abs(sub[sub["net_ret_pct"] <= 0]["net_ret_pct"].sum())
        b_rows.append({
            "Tramo_Probabilidad_1A": b_name,
            "Señales": len(sub),
            "% Llega_a_50%_MeLi": round(sub["reached_meli_45"].mean() * 100, 1),
            "Win_Rate_Real %": round((sub["net_ret_pct"] > 0).mean() * 100, 1),
            "Ret_Medio %": round(sub["net_ret_pct"].mean(), 2),
            "Profit_Factor": round(w / l if l > 0 else 9.9, 2),
            "%_De_Macro_o_Sector": round(sub["shock_type_code"].isin([3, 4]).mean() * 100, 1),
        })
    print(pd.DataFrame(b_rows).to_string(index=False))

    # ¿Y dentro de las DEJADAS AFUERA (0.53 <= prob_1a < 0.59), qué diferencia a las ganadoras de las perdedoras?
    missed_zone = df_clean[(df_clean["prob_1a"] >= 0.53) & (df_clean["prob_1a"] < 0.59)].copy()
    print("\n" + "-" * 120)
    print(f"1B. RADIOGRAFÍA DE LAS {len(missed_zone)} SEÑALES DEJADAS AFUERA (0.53 <= prob_1a < 0.59) SEGÚN ARQUETIPO:")
    print("-" * 120)
    m_arch = [
        ("En Zona 0.53-0.59: Caída Macro o Sectorial (Arquetipo 1.1)", missed_zone["shock_type_code"].isin([3, 4])),
        ("En Zona 0.53-0.59: Caída Aislada pero con Absorción Fuerte (rvol>=1.3)", (missed_zone["shock_type_code"] == 5) & (missed_zone["rvol_1h"] >= 1.3)),
        ("En Zona 0.53-0.59: Caída Aislada Débil o Post-Earnings", ~missed_zone["shock_type_code"].isin([3, 4]) & (missed_zone["rvol_1h"] < 1.3)),
    ]
    ma_rows = []
    for m_name, m_mask in m_arch:
        sub = missed_zone[m_mask]
        if len(sub) == 0:
            continue
        w = sub[sub["net_ret_pct"] > 0]["net_ret_pct"].sum()
        l = abs(sub[sub["net_ret_pct"] <= 0]["net_ret_pct"].sum())
        ma_rows.append({
            "Subgrupo_en_Zona_0.53_a_0.59": m_name,
            "Señales": len(sub),
            "% Llega_a_50%_MeLi": round(sub["reached_meli_45"].mean() * 100, 1),
            "Win_Rate_Real %": round((sub["net_ret_pct"] > 0).mean() * 100, 1),
            "Ret_Medio %": round(sub["net_ret_pct"].mean(), 2),
            "Profit_Factor": round(w / l if l > 0 else 9.9, 2),
        })
    print(pd.DataFrame(ma_rows).to_string(index=False))

    # 2. ¿CUÁL MODELO FALLA MÁS: EL DE ENTRAR (1A) O EL DE SALIR (1B)?
    print("\n" + "=" * 120)
    print("2. DIAGNÓSTICO FORENSE: ¿CUÁL FALLA MÁS, EL MODELO DE ENTRAR (1A) O EL DE SALIR (1B)?")
    print("=" * 120)
    traded_pool = df_clean[df_clean["prob_1a"] >= 0.55].copy()
    n_total = len(traded_pool)
    never_reached_meli = traded_pool[traded_pool["reached_meli_45"] == 0]
    reached_meli = traded_pool[traded_pool["reached_meli_45"] == 1]

    # Dentro de las que SÍ llegaron al 45%-50% ("Momento MeLi"):
    # ¿Cuántas terminaron perdiendo dinero o dejando >3% sobre la mesa?
    meli_turned_loser = reached_meli[reached_meli["net_ret_pct"] <= 0]
    meli_early_exit = reached_meli[(reached_meli["net_ret_pct"] > 0) & ((reached_meli["mfe_pct"] - reached_meli["net_ret_pct"]) >= 4.0)]
    meli_good_capture = reached_meli[(reached_meli["net_ret_pct"] > 0) & ((reached_meli["mfe_pct"] - reached_meli["net_ret_pct"]) < 4.0)]

    print(f"Sobre las {n_total} señales operables (`prob_1a >= 0.55`):")
    print(f"  A) FALLO DEL MODELO 1A (ENTRADA): {len(never_reached_meli)} señales ({len(never_reached_meli)/n_total*100:.1f}%) NUNCA llegaron siquiera al +45% de rebote:")
    print(f"     -> Win Rate de estas operaciones: {(never_reached_meli['net_ret_pct']>0).mean()*100:.1f}% | Retorno medio: {never_reached_meli['net_ret_pct'].mean():.2f}% | Suma de pérdidas: {never_reached_meli['net_ret_pct'].sum():.1f}%")
    print(f"  B) ÉXITO DE 1A -> PASA LA PELOTA AL MODELO 1B (SALIDA): {len(reached_meli)} señales ({len(reached_meli)/n_total*100:.1f}%) SÍ llegaron al hito del +45% ('Momento MeLi'):")
    print(f"     -> Excursión Máxima Promedio (MFE) que alcanzaron: +{reached_meli['mfe_pct'].mean():.2f}%")
    print(f"     -> Retorno Real Capturado al Salir (1B + Candado): +{reached_meli['net_ret_pct'].mean():.2f}% (Capturamos el {reached_meli['net_ret_pct'].mean()/reached_meli['mfe_pct'].mean()*100:.1f}% del pico máximo)")
    print(f"     -> Retorno si el Modelo 1B fuera un ORÁCULO PERFECTO: +{reached_meli['oracle_1b_ret_pct'].mean():.2f}%")
    print(f"     -> Desglose de lo que ocurre tras llegar al 45% de recuperación:")
    print(f"        1. Se dieron vuelta por Gap nocturno y terminaron <= 0%: {len(meli_turned_loser)} casos ({len(meli_turned_loser)/len(reached_meli)*100:.1f}%), Ret medio: {meli_turned_loser['net_ret_pct'].mean():.2f}%")
    print(f"        2. Salida prematura (dejaron >= 4% adicional sobre la mesa): {len(meli_early_exit)} casos ({len(meli_early_exit)/len(reached_meli)*100:.1f}%), Cobraron +{meli_early_exit['net_ret_pct'].mean():.2f}% vs Pico de +{meli_early_exit['mfe_pct'].mean():.2f}%")
    print(f"        3. Captura óptima cerca del techo: {len(meli_good_capture)} casos ({len(meli_good_capture)/len(reached_meli)*100:.1f}%), Cobraron +{meli_good_capture['net_ret_pct'].mean():.2f}% vs Pico de +{meli_good_capture['mfe_pct'].mean():.2f}%")

    # 3. SIMULACIÓN DE PORTAFOLIO ($25,000) RESCATANDO LAS SEÑALES DE 0.55-0.59 Y COMPARANDO ORÁCULOS
    print("\n" + "=" * 120)
    print("3. IMPACTO EN PORTAFOLIO ($25k, Producción Estricta IBKR): Rescatando Señales vs Oráculo 1A vs Oráculo 1B")
    print("=" * 120)
    # Crear un dataframe con `net_ret_pct = oracle_1b_ret_pct` para probar el techo de 1B
    df_oracle_1b = df_clean.copy()
    df_oracle_1b["net_ret_pct"] = df_oracle_1b["oracle_1b_ret_pct"]

    # Crear un Oráculo de 1A (solo entra si `reached_meli_45 == 1`)
    df_oracle_1a = df_clean[df_clean["reached_meli_45"] == 1].copy()

    port_cases = [
        ("1. Corte Antiguo Fijo (prob_1a >= 0.59, 5 Slots)", df_clean[df_clean["prob_1a"] >= 0.59], 5),
        ("2. Corte Ajustado a Tasa Base Estricta (prob_1a >= 0.56, 5 Slots)", df_clean[df_clean["prob_1a"] >= 0.56], 5),
        ("3. Corte Híbrido (prob_1a >= 0.59 O [prob_1a >= 0.54 & Macro/Sector], 5 Slots)",
         df_clean[(df_clean["prob_1a"] >= 0.59) | ((df_clean["prob_1a"] >= 0.54) & (df_clean["shock_type_code"].isin([3, 4])))], 5),
        ("4. ¿Qué pasa si 1B fuera PERFECTO (Oráculo 1B sobre Corte 0.56)?", df_oracle_1b[df_oracle_1b["prob_1a"] >= 0.56], 5),
        ("5. ¿Qué pasa si 1A fuera PERFECTO (Oráculo 1A con 1B actual, top prob_1a >= 0.56)?", df_oracle_1a[df_oracle_1a["prob_1a"] >= 0.56], 5),
    ]
    p_rows = []
    for label, sub_c, slots in port_cases:
        res = run_portfolio_simulation(sub_c, initial_capital=25000.0, max_slots=slots)
        p_rows.append({
            "Configuración_Auditada_IBKR": label,
            "Señales": len(sub_c),
            "Trades": res["trades_executed"],
            "Trades/Mes": round(res["trades_executed"] / 11.0, 1),
            "WR %": round(res["win_rate_pct"], 1),
            "Ret_Medio %": round(res["avg_trade_ret_pct"], 2),
            "PF": round(res["profit_factor"], 2),
            "Capital_Final": f"${res['final_capital']:,.0f}",
            "Retorno_11m": f"+{res['total_return_pct']:.1f}%",
            "Max_DD %": f"{res['max_dd_pct']:.2f}%",
            "Sharpe": round(res["sharpe"], 2),
        })
    print(pd.DataFrame(p_rows).to_string(index=False))
    print("=" * 120)
    print(f"Tiempo total: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
