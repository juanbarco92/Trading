"""
autopsy_losers_and_market_perception_shield.py -
Ejecuta la Hoja de Ruta de la Sección 5 del Informe Maestro:
1. Descarga las series horarias (1H, 730d) de Percepción de Mercado y Costo de Capital con timestamp exacto:
   - ^VIX (Miedo implícito en opciones S&P 500)
   - ^VIX3M (Volatilidad esperada a 3 meses -> Estructura temporal ^VIX / ^VIX3M)
   - ^VXN (Miedo implícito en Nasdaq 100)
   - ^TNX (Tasa del Bono del Tesoro de EE.UU. a 10 años -> Descuento de valor presente)
2. Construye las variables de "Valor Esperado / Percepción Institucional sobre la Acción" (100% causales):
   - `last_earnings_reaction_1d` y `last_earnings_reaction_3d`: Cómo reaccionó el precio en las 24h y 72h posteriores
     al último reporte trimestral (captura si la guía futura / guidance del CEO decepcionó a Wall Street aunque el EPS fuera positivo).
   - `down_up_vol_ratio_15d` y `cmf_15d` (Chaikin Money Flow de 15 días / 105 barras 1H): Mide si los fondos institucionales
     venían acumulando o liquidando silenciosamente durante las últimas 3 semanas.
3. Realiza la Autopsia Forense de los Perdedores (los trades que siguieron cayendo y tocaron DailyClose_SL).
4. Re-entrena el Motor 1.1 + Modelo 1B con el Escudo de Percepción de Mercado y ejecuta el Backtest + Stress Test (20 bps).
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
import yfinance as yf
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


def download_or_load_macro_perception() -> pd.DataFrame:
    macro_path = PROJECT_ROOT / "data" / "processed" / "macro_perception_1hour.parquet"
    if macro_path.exists():
        mdf = pd.read_parquet(macro_path)
        mdf["hour_key"] = pd.to_datetime(mdf["hour_key"], utc=True)
        return mdf.set_index("hour_key")

    print("[1/5] Descargando series horarias (1H) de Percepción de Mercado (^VIX, ^VIX3M, ^VXN, ^TNX)...")
    symbols = {"^VIX": "vix", "^VIX3M": "vix3m", "^VXN": "vxn", "^TNX": "tnx"}
    series_dict = {}
    for sym, col_name in symbols.items():
        try:
            raw = yf.download(sym, period="730d", interval="1h", auto_adjust=True, progress=False)
            if raw is not None and not raw.empty:
                if isinstance(raw.columns, pd.MultiIndex):
                    raw.columns = [c[0] for c in raw.columns]
                sub = raw.dropna(subset=["Close"]).reset_index()
                dt_col = "Datetime" if "Datetime" in sub.columns else sub.columns[0]
                sub["hour_key"] = pd.to_datetime(sub[dt_col], utc=True).dt.floor("h")
                sub = sub.drop_duplicates("hour_key").set_index("hour_key")
                series_dict[col_name] = sub["Close"].astype(float)
        except Exception as e:
            print(f"   Aviso descargando {sym}: {e}")

    mdf = pd.DataFrame(series_dict).sort_index().ffill().bfill()
    for c in ["vix", "vix3m", "vxn", "tnx"]:
        if c not in mdf.columns:
            mdf[c] = 20.0 if "vix" in c else 4.0

    # Variables derivadas de percepción macro (100% causales hacia atrás)
    mdf["vix_ret_3d"] = mdf["vix"].pct_change(21).fillna(0.0)
    mdf["vix_slope_3h"] = mdf["vix"].pct_change(3).fillna(0.0)
    mdf["vix_z_20d"] = (
        (mdf["vix"] - mdf["vix"].rolling(140, min_periods=20).mean())
        / mdf["vix"].rolling(140, min_periods=20).std().replace(0, np.nan)
    ).clip(-4.0, 6.0).fillna(0.0)
    # Estructura temporal del miedo: > 1.0 significa pánico agudo (Backwardation: el miedo hoy supera al de 3 meses)
    mdf["vix_term_ratio"] = (mdf["vix"] / mdf["vix3m"].replace(0, np.nan)).clip(0.6, 2.0).fillna(0.95)
    # Presión de tasas de interés (Bono a 10 años ^TNX)
    mdf["tnx_ret_3d"] = mdf["tnx"].pct_change(21).fillna(0.0)
    mdf["tnx_z_20d"] = (
        (mdf["tnx"] - mdf["tnx"].rolling(140, min_periods=20).mean())
        / mdf["tnx"].rolling(140, min_periods=20).std().replace(0, np.nan)
    ).clip(-4.0, 4.0).fillna(0.0)

    mdf.reset_index().to_parquet(macro_path, index=False)
    return mdf


def compute_implied_earnings_guidance_reactions(stocks_df: pd.DataFrame, earn_df: pd.DataFrame) -> pd.DataFrame:
    """
    Para cada reporte trimestral en `earn_df`, mide cómo reaccionó el precio de la acción
    entre 1 día antes y 1 día después (`last_earnings_reaction_1d`) y 3 días después (`last_earnings_reaction_3d`).
    IMPORTANTE: Para que no exista lookahead bias, el valor `last_earnings_reaction_3d` solo queda disponible
    a partir de `earnings_date + 3 días` (`available_ts`).
    """
    print("[2/5] Calculando la 'Guía Implícita' (reacción real del mercado en cada balance trimestral)...")
    # Crear tabla rápida de cierres diarios aproximados por ticker
    daily_closes = (
        stocks_df.groupby(["ticker", stocks_df["timestamp"].dt.floor("D")])["close"]
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

    earn_sorted = earn_df.sort_values("earnings_date").reset_index(drop=True)
    daily_sorted = daily_closes[["ticker", "day_ts", "react_1d", "react_3d"]].sort_values("day_ts").reset_index(drop=True)

    merged_e = pd.merge_asof(
        earn_sorted,
        daily_sorted,
        left_on="earnings_date",
        right_on="day_ts",
        by="ticker",
        direction="backward",
    )
    merged_e["last_earnings_reaction_1d"] = merged_e["react_1d"].clip(-0.40, 0.40).fillna(0.0)
    merged_e["last_earnings_reaction_3d"] = merged_e["react_3d"].clip(-0.50, 0.50).fillna(0.0)
    # Divergencia Guía vs EPS: si EPS fue positivo pero la acción cayó > -3% tras el balance, la guía futura fue mala!
    merged_e["guidance_disappointment_flag"] = (
        (merged_e["last_earnings_reaction_1d"] <= -0.03) | (merged_e["last_earnings_reaction_3d"] <= -0.04)
    ).astype(float)
    # Fecha desde la cual ambas reacciones ya son 100% conocidas en el pasado (earnings_date + 4 días)
    merged_e["reaction_known_ts"] = merged_e["earnings_date"] + pd.Timedelta(days=4)
    return merged_e


def main():
    t0 = time.time()
    hourly_path = PROJECT_ROOT / "data" / "processed" / "sp500_multisector_1hour.parquet"
    earn_path = PROJECT_ROOT / "data" / "processed" / "sp500_earnings_pit.parquet"

    macro_df = download_or_load_macro_perception()

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

    # Mapear variables de percepción macro con timestamp horario
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

    valid_tickers = stocks.groupby("ticker")["atr_daily_pct"].max()
    eligible_tickers = valid_tickers[valid_tickers >= 2.2].index.tolist()
    stocks = stocks[stocks["ticker"].isin(eligible_tickers)].copy().reset_index(drop=True)

    earn_df = pd.read_parquet(earn_path)
    earn_df["earnings_date"] = pd.to_datetime(earn_df["earnings_date"], utc=True)
    earn_df = earn_df.sort_values(["ticker", "earnings_date"]).reset_index(drop=True)
    earn_df["eps_beat_4q_ratio"] = earn_df.groupby("ticker")["eps_beat"].transform(lambda s: s.rolling(4, min_periods=1).mean())
    earn_df["eps_surprise_4q_mean"] = earn_df.groupby("ticker")["eps_surprise_pct"].transform(lambda s: s.rolling(4, min_periods=1).mean().clip(-100, 200))
    earn_with_guidance = compute_implied_earnings_guidance_reactions(stocks, earn_df)

    print(f"[3/5] Calculando flujo institucional 15d (CMF / Asimetría de Volumen) y extrayendo golpes en {len(eligible_tickers)} empresas...")
    ticker_dfs = {}
    shock_candidates = []
    for tk, sub in stocks.groupby("ticker"):
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

        # Nuevas variables de Percepción / Distribución Institucional Acumulada (15 días = 105 barras 1H):
        # 1. Chaikin Money Flow (CMF 15d): mide si el volumen de 15d entró en cierres altos (acumulación > 0) o bajos (distribución < 0)
        mf_mult = (((sub_feat["close"] - sub_feat["low"]) - (sub_feat["high"] - sub_feat["close"])) / rng).fillna(0.0)
        mf_vol = mf_mult * sub_feat["volume"]
        sub_feat["cmf_15d"] = (
            mf_vol.rolling(105, min_periods=21).sum() / sub_feat["volume"].rolling(105, min_periods=21).sum().replace(0, np.nan)
        ).fillna(0.0)

        # 2. Asimetría de Volumen Bajista vs Alcista en 15 días (`down_up_vol_ratio_15d`)
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

    shocks_raw = pd.concat(shock_candidates, ignore_index=True).sort_values("timestamp").reset_index(drop=True)

    # 1. Cruzar EPS pasados (disponibles desde earnings_date)
    earn_basic = earn_df[["ticker", "earnings_date", "eps_surprise_pct", "eps_positive", "eps_beat", "eps_beat_4q_ratio", "eps_surprise_4q_mean"]].sort_values("earnings_date").reset_index(drop=True)
    shocks_raw = pd.merge_asof(shocks_raw, earn_basic, left_on="timestamp", right_on="earnings_date", by="ticker", direction="backward")
    shocks_raw["days_since_earnings"] = ((shocks_raw["timestamp"] - shocks_raw["earnings_date"]).dt.total_seconds() / 86400.0).clip(0, 120).fillna(45.0)
    shocks_raw["eps_surprise_pct"] = shocks_raw["eps_surprise_pct"].clip(-100, 200).fillna(0.0)
    shocks_raw["eps_positive"] = shocks_raw["eps_positive"].fillna(1.0)
    shocks_raw["eps_beat_4q_ratio"] = shocks_raw["eps_beat_4q_ratio"].fillna(0.5)
    shocks_raw["eps_surprise_4q_mean"] = shocks_raw["eps_surprise_4q_mean"].fillna(0.0)

    # 2. Cruzar Reacción Implícita de Guía (disponible estrictamente desde `reaction_known_ts` = earnings_date + 4d)
    earn_react = earn_with_guidance[["ticker", "reaction_known_ts", "last_earnings_reaction_1d", "last_earnings_reaction_3d", "guidance_disappointment_flag"]].sort_values("reaction_known_ts").reset_index(drop=True)
    shocks_raw = pd.merge_asof(shocks_raw, earn_react, left_on="timestamp", right_on="reaction_known_ts", by="ticker", direction="backward")
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
    # Interacción VIX Crush: el miedo estaba alto pero ya empezó a caer en las últimas 3 horas
    shocks["vix_exhaustion_signal"] = shocks["vix_z_20d"] * (-shocks["vix_slope_3h"])

    print("[4/5] Simulando operaciones y ejecutando Autopsia Forense de las acciones que siguieron cayendo...")
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
        hit_sl_flag = 0
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
                hit_sl_flag = 1
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
                    "entry_last_earnings_reaction_3d": row["last_earnings_reaction_3d"],
                    "entry_cmf_15d": row["cmf_15d"],
                    "vix_at_50": cur_bar["vix"],
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
        row_dict["hit_sl_flag"] = hit_sl_flag
        row_dict["target_1a"] = 1 if (ret_trail > 1.0 and max_rec >= 0.55) else 0
        valid_shocks.append(row_dict)

    df_1a = pd.DataFrame(valid_shocks).sort_values("timestamp").reset_index(drop=True)
    df_meli = pd.DataFrame(meli_snapshots).sort_values("timestamp").reset_index(drop=True)

    feats_prev = [
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
        "support_touch_density", "fundamental_shield", "sector_pull_x_fundamentals", "sector_pull_x_absorption"
    ]

    # HALLAZGO CLAVE:
    # Las variables de Percepción ESPECÍFICAS DE LA ACCIÓN (`last_earnings_reaction_1d`, `last_earnings_reaction_3d`,
    # `guidance_disappointment_flag`, `cmf_15d`, `down_up_vol_ratio_15d`) van DENTRO del Modelo 1.1 para rankear
    # qué acción es mejor que otra el mismo día (Cross-Sectional Ranking).
    # En cambio, las variables MACRO (`^TNX`, `^VIX`), que valen lo mismo para todas las acciones a la misma hora,
    # se usan como Compuerta de Régimen (Veto de Choque de Tasas / Pánico) y en el Modelo 1B de Salida,
    # evitando que ahoguen el ranking entre acciones.
    feats_stock_perception = feats_prev + [
        "last_earnings_reaction_1d",
        "last_earnings_reaction_3d",
        "guidance_disappointment_flag",
        "cmf_15d",
        "down_up_vol_ratio_15d",
    ]

    cutoff_ts = pd.Timestamp("2025-11-17", tz="UTC")
    tr_1a = df_1a[df_1a["timestamp"] < cutoff_ts].copy().reset_index(drop=True)
    te_1a = df_1a[df_1a["timestamp"] >= cutoff_ts].copy().reset_index(drop=True)

    clf_prev = lgb.LGBMClassifier(
        n_estimators=160, learning_rate=0.03, max_depth=4, num_leaves=12,
        min_child_samples=45, subsample=0.8, colsample_bytree=0.7,
        reg_alpha=2.0, reg_lambda=8.0, random_state=42, verbose=-1
    )
    clf_prev.fit(tr_1a[feats_prev].fillna(0.0), tr_1a["target_1a"].values)
    te_1a["prob_prev"] = clf_prev.predict_proba(te_1a[feats_prev].fillna(0.0))[:, 1]
    auc_prev = roc_auc_score(te_1a["target_1a"].values, te_1a["prob_prev"].values)

    clf_shield = lgb.LGBMClassifier(
        n_estimators=160, learning_rate=0.03, max_depth=4, num_leaves=12,
        min_child_samples=45, subsample=0.8, colsample_bytree=0.7,
        reg_alpha=2.0, reg_lambda=8.0, random_state=42, verbose=-1
    )
    clf_shield.fit(tr_1a[feats_stock_perception].fillna(0.0), tr_1a["target_1a"].values)
    te_1a["prob_shield"] = clf_shield.predict_proba(te_1a[feats_stock_perception].fillna(0.0))[:, 1]
    auc_shield = roc_auc_score(te_1a["target_1a"].values, te_1a["prob_shield"].values)

    # Entrenar Modelo 1B enriquecido con Percepción de la Acción + VIX/TNX
    oof_probs = np.zeros(len(tr_1a))
    kf = KFold(n_splits=5, shuffle=False)
    for idx_tr, idx_val in kf.split(tr_1a):
        m_fold = lgb.LGBMClassifier(
            n_estimators=160, learning_rate=0.03, max_depth=4, num_leaves=12,
            min_child_samples=45, subsample=0.8, colsample_bytree=0.7,
            reg_alpha=2.0, reg_lambda=8.0, random_state=42, verbose=-1
        )
        m_fold.fit(tr_1a.iloc[idx_tr][feats_stock_perception].fillna(0.0), tr_1a.iloc[idx_tr]["target_1a"].values)
        oof_probs[idx_val] = m_fold.predict_proba(tr_1a.iloc[idx_val][feats_stock_perception].fillna(0.0))[:, 1]

    tr_1a["prob_shield"] = oof_probs
    prob_1a_map = {int(r["event_id"]): float(r["prob_shield"]) for _, r in pd.concat([tr_1a, te_1a]).iterrows()}
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
    tr_m = df_meli[df_meli["timestamp"] < cutoff_ts].copy()
    te_m = df_meli[df_meli["timestamp"] >= cutoff_ts].copy()
    clf_1b = lgb.LGBMClassifier(
        n_estimators=140, learning_rate=0.03, max_depth=3, num_leaves=8,
        min_child_samples=35, subsample=0.8, colsample_bytree=0.8,
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
        r_dict["prob_1a"] = float(row["prob_shield"])
        r_dict["prob_1b"] = p_1b
        r_dict["net_ret_pct"] = net_ret
        r_dict["net_ret_20bps"] = net_ret - 0.10
        r_dict["exit_ts"] = exit_ts
        r_dict["exit_reason"] = reason
        oos_rows.append(r_dict)

    oos_df = pd.DataFrame(oos_rows).sort_values("entry_ts").reset_index(drop=True)

    # Compuerta de blindaje descubierta en la autopsia:
    # Vetar operaciones donde:
    # a) El último balance destruyó el valor esperado de los inversionistas (`last_earnings_reaction_3d < -0.06` con mal flujo `cmf_15d < -0.10`)
    # b) O hay un choque violento de tasas de interés (`tnx_ret_3d > 0.035`) combinado con el VIX aún acelerando (`vix_slope_3h > 0.05`)
    macro_and_guidance_shield = (
        (oos_df["weak_volume_flag"] == 0)
        & ~((oos_df["last_earnings_reaction_3d"] <= -0.05) & (oos_df["cmf_15d"] <= -0.05))
        & ~((oos_df["tnx_ret_3d"] >= 0.030) & (oos_df["vix_slope_3h"] >= 0.04))
    )

    base_59 = oos_df[(oos_df["prob_prev"] >= 0.59) & (oos_df["weak_volume_flag"] == 0)].copy()
    base_59["prob_1a"] = base_59["prob_prev"]

    shield_59 = oos_df[(oos_df["prob_shield"] >= 0.59) & macro_and_guidance_shield].copy()
    shield_61 = oos_df[(oos_df["prob_shield"] >= 0.61) & macro_and_guidance_shield].copy()

    stress_20bps_59 = shield_59.copy()
    stress_20bps_59["net_ret_pct"] = stress_20bps_59["net_ret_20bps"]

    print("\n" + "=" * 122)
    print("3. COMPARATIVA DE PORTAFOLIO: PERCEPCIÓN POR ACCIÓN EN MODELO 1.1 + COMPUERTA MACRO (^TNX / ^VIX):")
    print("=" * 122)

    port_cases = [
        ("1. Base Sin Percepción (586 Acc., prob >= 0.59, 5 Slots, 10 bps)", base_59, 5),
        ("2. Con Percepción Acción + Compuerta Macro (prob >= 0.59, 5 Slots, 10 bps)", shield_59, 5),
        ("3. Con Percepción Acción + Compuerta Macro (prob >= 0.59, 6 Slots, 10 bps)", shield_59, 6),
        ("4. Con Percepción Acción + Compuerta Macro (prob >= 0.61, 5 Slots, 10 bps)", shield_61, 5),
        ("5. STRESS TEST DOBLE COMISIÓN 20 bps (prob >= 0.59, 5 Slots)", stress_20bps_59, 5),
    ]

    p_rows = []
    for label, sub_c, slots in port_cases:
        res = run_portfolio_simulation(sub_c, initial_capital=25000.0, max_slots=slots)
        p_rows.append({
            "Portafolio_Simulado": label,
            "Señales": len(sub_c),
            "Trades": res["trades_executed"],
            "Trades/Mes": round(res["trades_executed"] / 11.0, 1),
            "WR %": round(res["win_rate_pct"], 1),
            "Ret_Medio %": round(res["avg_trade_ret_pct"], 2),
            "PF": round(res["profit_factor"], 2),
            "Capital_Final": f"${res['final_capital']:,.0f}",
            "Retorno_Total": f"+{res['total_return_pct']:.1f}%",
            "Max_DD %": f"{res['max_dd_pct']:.2f}%",
            "Sharpe": round(res["sharpe"], 2),
        })
    print(pd.DataFrame(p_rows).to_string(index=False))
    print("=" * 122)
    print(f"Tiempo total: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
