"""
expand_to_550_high_atr_stocks.py -
1. Descarga ~240 acciones adicionales líquidas de mediana/gran capitalización (S&P 400 MidCap / Russell 1000 / Nasdaq)
   con alta elasticidad (ATR Diario >= 2.2%) en los 10 sectores económicos + sus fechas y sorpresas de Earnings Point-in-Time.
2. Consolida el dataset horario (`sp500_multisector_1hour.parquet`) y de balances (`sp500_earnings_pit.parquet`)
   llevando el universo elegible de alto ATR de 314 a ~520+ empresas.
3. Ejecuta el entrenamiento y simulación de portafolio Out-of-Sample del Motor 1.1 Refinado + Modelo 1B en Cascada.
"""

import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
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

# ~245 acciones líquidas adicionales (Russell 1000 / S&P 400 MidCap / Nasdaq) con alto ATR por sector
WAVE_2_HIGH_ATR_TICKERS = {
    "Technology": [
        "SMCI", "ARM", "APP", "PLTR", "IOT", "MSTR", "COIN", "MNDY", "GTLB", "DOCU",
        "OKTA", "TWLO", "PATH", "U", "BILL", "PCTY", "PAYC", "DT", "ESTC", "CFLT",
        "TOST", "SHOP", "MELI", "SE", "GRAB", "LOGI", "JBL", "FLEX", "CIEN", "LITE",
        "COHR", "MKSI", "LSCC", "RMBS", "SLAB", "DIOD", "WOLF", "AMKR", "ACLS", "ONTO",
        "CYBR", "TENB", "QLYS", "VRNS", "GWRE", "MANH", "BSY", "AZPN", "PEGA", "FIVN"
    ],
    "Consumer_Discretionary": [
        "CAVA", "SHAK", "WING", "TXRH", "BROS", "DPZ", "WEN", "BLMN", "EAT", "CAKE",
        "ANF", "AEO", "URBN", "GPS", "M", "KSS", "JWN", "DDS", "BURL", "FIVE",
        "OLLI", "ASO", "DKS", "YETI", "MAT", "HAS", "PLNT", "PENN", "CZR", "BYD",
        "CHDN", "VAIL", "WH", "H", "HLT", "MAR", "ABNB", "DASH", "LYFT", "UBER",
        "RIVN", "LCID", "THO", "WGO", "PII", "HOG", "BC", "pag", "LAD", "AN", "GPI"
    ],
    "Financials": [
        "HOOD", "UPST", "LC", "SLM", "NAVI", "OMF", "COOP", "PFSI", "RKT", "UWMC",
        "FHN", "ZION", "CMA", "WTFC", "SNV", "VLY", "COLB", "BOKF", "PNFP", "EWBC",
        "WAL", "CFR", "PB", "SSB", "ONB", "FNB", "SF", "EVR", "LAZ", "PIPR",
        "JEF", "HLI", "RJF", "LPLA", "SEIC", "VOYA", "LNC", "UNM", "PRU", "MET", "EQH", "RGA"
    ],
    "Industrials": [
        "VERT", "VRT", "EME", "FIX", "MTZ", "PRIM", "DY", "ACM", "STANTEC", "KBR",
        "FLR", "J", "BLD", "IBP", "SSD", "TREX", "AZEK", "AWI", "MAS", "OC",
        "FBIN", "AOS", "LII", "WSO", "SITE", "CNM", "FERG", "WCC", "AIT", "MSM",
        "SAIA", "XPO", "RXO", "KNX", "WERN", "LSTR", "CHRW", "EXPD", "GXO", "R",
        "AL", "AER", "FTAI", "HEI", "CW", "BWXT", "HII", "KTOS", "AVAV"
    ],
    "Healthcare": [
        "VRTX", "REGN", "ALNY", "IONS", "BMRN", "SRPT", "UTHR", "NBIX", "HALO", "EXEL",
        "JAZZ", "NTRA", "GH", "TXG", "TWST", "BEAM", "CRSP", "NTLA", "PCVX", "VKTX",
        "INSP", "SWAV", "PEN", "GMED", "NVST", "XRAY", "TFX", "MASI", "GKOS", "IRTC",
        "TNDM", "LNTH", "MEDP", "ICLR", "DOCS", "HQY", "THC", "UHS", "EHC", "ACHC"
    ],
    "Energy": [
        "OVV", "PR", "MTDR", "CHRD", "MGY", "SM", "MUR", "CNX", "RRC", "AR",
        "SWN", "CHK", "PBF", "DINO", "DK", "CVI", "VAL", "NE", "HP", "PTEN",
        "LBRT", "NEX", "CHX", "FTI", "NOV", "WHD", "AM", "HESM", "ENLC", "PAGP"
    ],
    "Materials": [
        "RS", "CMC", "ATI", "CRS", "MP", "LAC", "SQM", "TECK", "PAAS", "AG",
        "HL", "CDE", "RGLD", "WPM", "FNV", "AEM", "KGC", "AU", "GFI", "HUN",
        "OLN", "WLK", "CC", "TROX", "AXTA", "RPM", "SEE", "BERY", "SON", "CCK"
    ],
    "Communication": [
        "RDDT", "TTD", "APP", "ZG", "Z", "CARG", "YELP", "TRIP", "BMBL", "IAC",
        "NYT", "NWSA", "FOXA", "NXST", "TGNA", "SIRI", "EDR", "TKO", "WMG", "MSGS"
    ],
}


def fetch_single_ticker_earnings(tk: str):
    try:
        t_obj = yf.Ticker(tk)
        ed = t_obj.get_earnings_dates(limit=28)
        if ed is None or ed.empty:
            return None
        ed = ed.reset_index()
        dt_col = ed.columns[0]
        ed["earnings_date"] = pd.to_datetime(ed[dt_col], utc=True)
        ed = ed.rename(columns={
            "EPS Estimate": "eps_estimate",
            "Reported EPS": "eps_actual",
            "Surprise(%)": "eps_surprise_pct",
        })
        ed = ed.dropna(subset=["eps_actual"])
        if ed.empty:
            return None
        ed["ticker"] = tk
        ed["eps_surprise_pct"] = pd.to_numeric(ed["eps_surprise_pct"], errors="coerce").fillna(0.0)
        ed["eps_actual"] = pd.to_numeric(ed["eps_actual"], errors="coerce").fillna(0.0)
        ed["eps_estimate"] = pd.to_numeric(ed["eps_estimate"], errors="coerce").fillna(0.0)
        ed["eps_positive"] = (ed["eps_actual"] > 0).astype(float)
        ed["eps_beat"] = (ed["eps_actual"] >= ed["eps_estimate"]).astype(float)
        return ed[["ticker", "earnings_date", "eps_estimate", "eps_actual", "eps_surprise_pct", "eps_positive", "eps_beat"]]
    except Exception:
        return None


def download_wave_2_and_earnings():
    hourly_path = PROJECT_ROOT / "data" / "processed" / "sp500_multisector_1hour.parquet"
    earn_path = PROJECT_ROOT / "data" / "processed" / "sp500_earnings_pit.parquet"

    existing_df = pd.read_parquet(hourly_path)
    existing_df["timestamp"] = pd.to_datetime(existing_df["timestamp"], utc=True)
    existing_tickers = set(existing_df["ticker"].unique())

    new_ticker_to_sec = {}
    new_tickers = []
    for sec, tks in WAVE_2_HIGH_ATR_TICKERS.items():
        for t in tks:
            t_up = t.upper()
            if t_up not in existing_tickers and t_up not in new_ticker_to_sec:
                new_ticker_to_sec[t_up] = sec
                new_tickers.append(t_up)

    if new_tickers:
        print(f"[1/4] Descargando {len(new_tickers)} nuevas empresas líquidas de alto ATR (1H, 730d)...")
        batch_size = 35
        chunks = []
        for i in range(0, len(new_tickers), batch_size):
            batch = new_tickers[i : i + batch_size]
            print(f"      * Lote {i//batch_size + 1}/{(len(new_tickers)-1)//batch_size + 1}: {batch[:6]}...")
            try:
                raw = yf.download(
                    tickers=batch,
                    period="730d",
                    interval="1h",
                    group_by="ticker",
                    auto_adjust=True,
                    threads=True,
                    progress=False,
                )
                if raw is None or raw.empty:
                    continue
                for tk in batch:
                    try:
                        sub = raw[tk].copy() if len(batch) > 1 else raw.copy()
                        sub = sub.dropna(subset=["Open", "High", "Low", "Close"])
                        if len(sub) < 500:
                            continue
                        sub = sub.reset_index()
                        dt_col = "Datetime" if "Datetime" in sub.columns else sub.columns[0]
                        sub["timestamp"] = pd.to_datetime(sub[dt_col], utc=True)
                        sub = sub.rename(columns={
                            "Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"
                        })
                        sub["ticker"] = tk
                        sub["sector"] = new_ticker_to_sec[tk]
                        sub = sub[["timestamp", "open", "high", "low", "close", "volume", "ticker", "sector"]]
                        chunks.append(sub)
                    except Exception:
                        continue
            except Exception:
                continue

        if chunks:
            new_df = pd.concat(chunks, ignore_index=True)
            added_tickers = sorted(new_df["ticker"].unique().tolist())
            print(f"      -> Añadidas {len(added_tickers)} empresas con historial 1H completo ({len(new_df):,} barras).")
            existing_df = pd.concat([existing_df, new_df], ignore_index=True)
            existing_df = existing_df.drop_duplicates(subset=["ticker", "timestamp"]).sort_values(["ticker", "timestamp"]).reset_index(drop=True)
            existing_df.to_parquet(hourly_path, index=False)

            # Descargar Earnings Point-in-Time en paralelo para los nuevos tickers
            earn_df = pd.read_parquet(earn_path) if earn_path.exists() else pd.DataFrame()
            existing_earn_tks = set(earn_df["ticker"].unique()) if not earn_df.empty else set()
            missing_earn_tks = [t for t in added_tickers if t not in existing_earn_tks]

            if missing_earn_tks:
                print(f"[2/4] Descargando Point-in-Time Earnings en paralelo para {len(missing_earn_tks)} nuevas empresas...")
                earn_chunks = []
                with ThreadPoolExecutor(max_workers=12) as ex:
                    futs = {ex.submit(fetch_single_ticker_earnings, tk): tk for tk in missing_earn_tks}
                    for f in as_completed(futs):
                        res_e = f.result()
                        if res_e is not None and not res_e.empty:
                            earn_chunks.append(res_e)
                if earn_chunks:
                    new_earn_df = pd.concat(earn_chunks, ignore_index=True)
                    earn_df = pd.concat([earn_df, new_earn_df], ignore_index=True)
                    earn_df = earn_df.drop_duplicates(subset=["ticker", "earnings_date"]).sort_values(["ticker", "earnings_date"]).reset_index(drop=True)
                    earn_df.to_parquet(earn_path, index=False)
                    print(f"      -> Consolidados {len(earn_df):,} reportes trimestrales Point-in-Time.")
    else:
        print(f"[1/4] Universo ya expandido con {existing_df['ticker'].nunique()} activos.")

    return existing_df


def main():
    t0 = time.time()
    df = download_wave_2_and_earnings()
    earn_path = PROJECT_ROOT / "data" / "processed" / "sp500_earnings_pit.parquet"

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
    print(f"[3/4] Universo Elegible de Alto ATR (>= 2.2%): ¡{len(eligible_tickers)} empresas activas!")

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
    print(f"[4/4] Total Golpes 3-5d en las {len(eligible_tickers)} empresas: {len(df_1a):,} eventos.")

    feats_new_1_1 = [
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

    # Usar exactamente la misma fecha de corte OOS (~2025-11-17) para comparar manzanas con manzanas contra las 314 acciones
    cutoff_ts = pd.Timestamp("2025-11-17", tz="UTC")
    tr_1a = df_1a[df_1a["timestamp"] < cutoff_ts].copy().reset_index(drop=True)
    te_1a = df_1a[df_1a["timestamp"] >= cutoff_ts].copy().reset_index(drop=True)

    clf_new = lgb.LGBMClassifier(
        n_estimators=160, learning_rate=0.03, max_depth=4, num_leaves=12,
        min_child_samples=45, subsample=0.8, colsample_bytree=0.7,
        reg_alpha=2.0, reg_lambda=8.0, random_state=42, verbose=-1
    )
    clf_new.fit(tr_1a[feats_new_1_1].fillna(0.0), tr_1a["target_1a"].values)
    te_1a["prob_1a"] = clf_new.predict_proba(te_1a[feats_new_1_1].fillna(0.0))[:, 1]
    auc_oos = roc_auc_score(te_1a["target_1a"].values, te_1a["prob_1a"].values)

    oof_probs = np.zeros(len(tr_1a))
    kf = KFold(n_splits=5, shuffle=False)
    for idx_tr, idx_val in kf.split(tr_1a):
        m_fold = lgb.LGBMClassifier(
            n_estimators=160, learning_rate=0.03, max_depth=4, num_leaves=12,
            min_child_samples=45, subsample=0.8, colsample_bytree=0.7,
            reg_alpha=2.0, reg_lambda=8.0, random_state=42, verbose=-1
        )
        m_fold.fit(tr_1a.iloc[idx_tr][feats_new_1_1].fillna(0.0), tr_1a.iloc[idx_tr]["target_1a"].values)
        oof_probs[idx_val] = m_fold.predict_proba(tr_1a.iloc[idx_val][feats_new_1_1].fillna(0.0))[:, 1]

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
        "entry_SMA_100_momentum", "entry_close_diff_SMA_100"
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
    auc_1b = roc_auc_score(te_m["keep_running"].values, te_m["prob_1b"].values)
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

    c_61 = oos_df[oos_df["prob_1a"] >= 0.61].copy()
    c_59 = oos_df[(oos_df["prob_1a"] >= 0.59) & (oos_df["weak_volume_flag"] == 0)].copy()
    c_57 = oos_df[(oos_df["prob_1a"] >= 0.57) & (oos_df["weak_volume_flag"] == 0)].copy()

    print("\n" + "=" * 120)
    print(f"RESULTADOS EN EL UNIVERSO EXPANDIDO DE {len(eligible_tickers)} EMPRESAS DE ALTO ATR ({len(oos_df):,} GOLPES EN OOS):")
    print(f"  * ROC-AUC OOS Modelo 1.1 (Entrada): {auc_oos:.4f} | ROC-AUC OOS Modelo 1B (Salida): {auc_1b:.4f}")
    print(f"  * Señales con prob >= 0.61:         {len(c_61)} señales | WR = {(c_61['net_ret_pct']>0).mean()*100:.1f}% | Ret Medio = +{c_61['net_ret_pct'].mean():.2f}%")
    print(f"  * Señales con prob >= 0.59 + VolOK: {len(c_59)} señales | WR = {(c_59['net_ret_pct']>0).mean()*100:.1f}% | Ret Medio = +{c_59['net_ret_pct'].mean():.2f}%")
    print(f"  * Señales con prob >= 0.57 + VolOK: {len(c_57)} señales | WR = {(c_57['net_ret_pct']>0).mean()*100:.1f}% | Ret Medio = +{c_57['net_ret_pct'].mean():.2f}%")
    print("=" * 120)

    cases = [
        (f"1. Universo {len(eligible_tickers)} Acciones (prob >= 0.61, 5 Slots)", c_61, 5),
        (f"2. Universo {len(eligible_tickers)} Acciones (prob >= 0.61, 6 Slots)", c_61, 6),
        (f"3. Universo {len(eligible_tickers)} Acciones (prob >= 0.59 + VolOK, 5 Slots)", c_59, 5),
        (f"4. Universo {len(eligible_tickers)} Acciones (prob >= 0.59 + VolOK, 6 Slots)", c_59, 6),
        (f"5. Universo {len(eligible_tickers)} Acciones (prob >= 0.57 + VolOK, 6 Slots)", c_57, 6),
    ]
    rows = []
    for label, sub_c, slots in cases:
        res = run_portfolio_simulation(sub_c, initial_capital=25000.0, max_slots=slots)
        rows.append({
            "Configuración_Portafolio_Expandido": label,
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
    print(f"Tiempo total: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
