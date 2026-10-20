"""
Verificación Exacta del Drawdown Mark-to-Market (sobre las 74 operaciones exactas de `run_portfolio_simulation`),
Demostración de Dilución de Árboles vs Selección Ortogonal de Features (Mina A: `sector_breadth_oversold_3d` y Mina B: `vxn_vix_ratio`),
y Cómo Ahorrar Drawdown en la Fila 3 (+65.7%) sin perder su potencia ofensiva.
"""

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.backtest_portfolio_motor1_dual_ml import run_portfolio_simulation
from scripts.upgrade_model_1b_first_passage_and_perception import simulate_calibrated_1b_ibkr_exit


def compute_exact_mtm_on_simulation(
    signals_df: pd.DataFrame,
    ticker_day_prices: dict,
    initial_capital: float = 25000.0,
    max_slots: int = 5,
    size_scale_col: str = None,
):
    """
    Replica EXACTAMENTE la lógica evento por evento de `run_portfolio_simulation`
    y además calcula día a día:
    - `max_dd_closed`: El drawdown exacto de `run_portfolio_simulation` (posiciones abiertas al costo).
    - `max_dd_mtm_eod`: El drawdown Mark-to-Market al cierre de cada día en IBKR.
    - `max_dd_mtm_low`: El peor drawdown flotante intradía (usando el `Low` diario de cada posición abierta).
    - `mc_p50_dd`, `mc_p95_dd`: Bootstrap Monte Carlo de 2,000 secuencias.
    """
    df_sorted = signals_df.sort_values(["entry_ts", "prob_1a"], ascending=[True, False]).reset_index(drop=True)
    if df_sorted.empty:
        return {}

    cash = initial_capital
    open_positions = []
    closed_trades = []
    equity_curve = []

    for _, row in df_sorted.iterrows():
        cur_ts = pd.Timestamp(row["entry_ts"])

        still_open = []
        for pos in open_positions:
            if pos["exit_ts"] <= cur_ts:
                cash += pos["returned_cash"]
                closed_trades.append(pos)
            else:
                still_open.append(pos)
        open_positions = still_open

        cur_eq = cash + sum(p["invested"] for p in open_positions)
        equity_curve.append({"ts": cur_ts, "equity": cur_eq})

        if len(open_positions) >= max_slots:
            continue
        if any(p["ticker"] == row["ticker"] for p in open_positions):
            continue

        slots_free = max_slots - len(open_positions)
        base_alloc = min(cash / slots_free, cur_eq / max_slots)
        if size_scale_col is not None and size_scale_col in row:
            alloc = base_alloc * float(row[size_scale_col])
        else:
            alloc = base_alloc

        if alloc < 500:
            continue

        cash -= alloc
        ret_pct = float(row["net_ret_pct"]) / 100.0
        returned = alloc * (1.0 + ret_pct)

        open_positions.append({
            "ticker": row["ticker"],
            "sector": row["sector"],
            "entry_ts": cur_ts,
            "exit_ts": pd.Timestamp(row["exit_ts"]),
            "entry_p": float(row["entry_p"]),
            "sl_p": float(row["sl_p"]),
            "invested": alloc,
            "returned_cash": returned,
            "net_ret_pct": float(row["net_ret_pct"]),
            "prob_1a": float(row["prob_1a"]),
        })

    for pos in sorted(open_positions, key=lambda x: x["exit_ts"]):
        cash += pos["returned_cash"]
        closed_trades.append(pos)
        cur_eq = cash + sum(p["invested"] for p in open_positions if p["exit_ts"] > pos["exit_ts"])
        equity_curve.append({"ts": pos["exit_ts"], "equity": cur_eq})

    trades_df = pd.DataFrame(closed_trades)
    eq_df = pd.DataFrame(equity_curve).sort_values("ts").reset_index(drop=True)
    eq_df["peak"] = eq_df["equity"].cummax()
    eq_df["dd_pct"] = (eq_df["equity"] - eq_df["peak"]) / eq_df["peak"] * 100.0
    max_dd_closed = float(eq_df["dd_pct"].min())

    eq_df["date"] = eq_df["ts"].dt.floor("D")
    daily_eq = eq_df.groupby("date")["equity"].last()
    all_bdays = pd.date_range(daily_eq.index.min(), daily_eq.index.max(), freq="B", tz="UTC")
    daily_eq = daily_eq.reindex(all_bdays).ffill().fillna(initial_capital)
    daily_rets = daily_eq.pct_change().dropna()
    sharpe_closed = float((daily_rets.mean() / daily_rets.std()) * np.sqrt(252)) if len(daily_rets) > 10 and daily_rets.std() > 0 else 0.0

    # Ahora reconstruir el valor Mark-to-Market exacto en cada día real de mercado (`nyse_days`)
    nyse_days = sorted({k[1] for k in ticker_day_prices.keys() if daily_eq.index.min() <= k[1] <= daily_eq.index.max()})
    mtm_eod_series = []
    mtm_low_series = []
    max_same_sec_open = 0

    for bday in nyse_days:
        day_end = bday + pd.Timedelta(hours=23, minutes=59)
        realized_pnl = trades_df[trades_df["exit_ts"] <= day_end].apply(lambda r: r["returned_cash"] - r["invested"], axis=1).sum() if len(trades_df) > 0 else 0.0
        active = trades_df[(trades_df["entry_ts"] <= day_end) & (trades_df["exit_ts"] > day_end)]

        if len(active) > 0:
            sec_counts = active["sector"].value_counts().max()
            if sec_counts > max_same_sec_open:
                max_same_sec_open = int(sec_counts)

        invested_open = active["invested"].sum() if len(active) > 0 else 0.0
        cash_eod = initial_capital + realized_pnl - invested_open

        val_eod = cash_eod
        val_low = cash_eod
        for _, apos in active.iterrows():
            tk = apos["ticker"]
            ep = float(apos["entry_p"])
            slp = float(apos["sl_p"])
            inv = float(apos["invested"])
            dpx = ticker_day_prices.get((tk, bday), None)
            if dpx is not None:
                c_px, l_px = dpx
                l_px_eff = max(l_px, slp * 0.995)
                val_eod += inv * (c_px / ep)
                val_low += inv * (l_px_eff / ep)
            else:
                val_eod += inv
                val_low += inv

        mtm_eod_series.append(val_eod)
        mtm_low_series.append(val_low)

    mtm_eod_arr = np.array(mtm_eod_series)
    mtm_low_arr = np.array(mtm_low_series)
    mtm_peak = np.maximum.accumulate(np.insert(mtm_eod_arr, 0, initial_capital))[1:]
    dd_mtm_eod = float(np.min((mtm_eod_arr - mtm_peak) / mtm_peak * 100.0))
    dd_mtm_low = float(np.min((mtm_low_arr - mtm_peak) / mtm_peak * 100.0))

    worst_day_idx = int(np.argmin((mtm_eod_arr - mtm_peak) / mtm_peak))
    worst_day_str = str(nyse_days[worst_day_idx].date())

    # Bootstrap Monte Carlo (2,000 permutaciones)
    rng = np.random.default_rng(42)
    t_PnL_frac = (trades_df["returned_cash"] - trades_df["invested"]) / initial_capital
    # Mejor: usar el retorno relativo al portafolio de cada trade (`net_ret_pct / 100 * (invested / equity_at_entry)`)
    # En 5 slots equivalentes es ~1/5 de `net_ret_pct`
    t_rets = trades_df["net_ret_pct"].to_numpy() / 100.0
    w_mults = (trades_df["invested"] / trades_df["invested"].mean()).to_numpy()
    step_impacts = (t_rets * w_mults) / max_slots
    mc_dds = []
    for _ in range(2000):
        perm = rng.choice(step_impacts, size=len(step_impacts), replace=True)
        eq_p = initial_capital * np.cumprod(1.0 + np.insert(perm, 0, 0.0))
        pk_p = np.maximum.accumulate(eq_p)
        mc_dds.append(float(np.min((eq_p - pk_p) / pk_p * 100.0)))

    wins = trades_df[trades_df["net_ret_pct"] > 0]["net_ret_pct"].sum()
    losses = abs(trades_df[trades_df["net_ret_pct"] <= 0]["net_ret_pct"].sum())
    pf = wins / losses if losses > 0 else 9.99

    return {
        "trades_executed": len(trades_df),
        "win_rate_pct": float((trades_df["net_ret_pct"] > 0).mean() * 100.0),
        "avg_trade_ret_pct": float(trades_df["net_ret_pct"].mean()),
        "profit_factor": float(pf),
        "final_capital": float(cash),
        "total_return_pct": float((cash - initial_capital) / initial_capital * 100.0),
        "max_dd_closed": max_dd_closed,
        "max_dd_mtm_eod": dd_mtm_eod,
        "max_dd_mtm_low": dd_mtm_low,
        "worst_mtm_date": worst_day_str,
        "mc_p50_dd": float(np.percentile(mc_dds, 50)),
        "mc_p95_dd": float(np.percentile(mc_dds, 5)),
        "sharpe": sharpe_closed,
        "max_same_sec": max_same_sec_open,
        "trades_df": trades_df,
    }


def main():
    t0 = time.time()
    cache_1a_path = PROJECT_ROOT / "data" / "processed" / "strict_prod_df_1a.parquet"
    cache_meli_path = PROJECT_ROOT / "data" / "processed" / "strict_prod_df_meli.parquet"
    hourly_path = PROJECT_ROOT / "data" / "processed" / "sp500_multisector_1hour.parquet"
    macro_path = PROJECT_ROOT / "data" / "processed" / "macro_perception_1hour.parquet"

    df_1a = pd.read_parquet(cache_1a_path)
    df_meli = pd.read_parquet(cache_meli_path)

    df_raw = pd.read_parquet(hourly_path)
    df_raw["timestamp"] = pd.to_datetime(df_raw["timestamp"], utc=True)
    df_raw["hour_key"] = df_raw["timestamp"].dt.floor("h")
    df_raw["day_key"] = df_raw["timestamp"].dt.floor("D")

    macro_df = pd.read_parquet(macro_path)
    macro_df["hour_key"] = pd.to_datetime(macro_df["hour_key"], utc=True)
    macro_df = macro_df.set_index("hour_key")

    for mcol in ["vix", "vix_ret_3d", "vix_slope_3h", "tnx_ret_3d", "vxn", "vxn_ret_3d"]:
        if mcol in macro_df.columns:
            df_raw[mcol] = df_raw["hour_key"].map(macro_df[mcol]).ffill().fillna(0.0)
        else:
            df_raw[mcol] = 0.0

    test_start_ts = pd.Timestamp("2025-11-17", tz="UTC")
    raw_test_days = df_raw[df_raw["timestamp"] >= test_start_ts - pd.Timedelta(days=5)]
    daily_agg = (
        raw_test_days.groupby(["ticker", "day_key"])
        .agg(day_close=("close", "last"), day_low=("low", "min"))
        .reset_index()
    )
    ticker_day_prices = {
        (r["ticker"], r["day_key"]): (float(r["day_close"]), float(r["day_low"]))
        for _, r in daily_agg.iterrows()
    }

    spy_df = df_raw[df_raw["ticker"] == "SPY"].sort_values("timestamp").drop_duplicates("hour_key").set_index("hour_key")
    spy_ret_60d_map = spy_df["close"].pct_change(420).fillna(0.0)
    spy_ret_2h_map = spy_df["close"].pct_change(2).fillna(0.0)

    vxn_map = macro_df["vxn"].ffill().fillna(20.0)
    vix_map = macro_df["vix"].ffill().fillna(20.0)

    stocks_only = df_raw[~df_raw["ticker"].isin(["SPY", "QQQ"])].copy()
    stocks_only = stocks_only.sort_values(["ticker", "timestamp"]).reset_index(drop=True)
    stocks_only["ret_1h"] = stocks_only.groupby("ticker")["close"].pct_change(1)
    stocks_only["ret_3d_st"] = stocks_only.groupby("ticker")["close"].pct_change(21)
    stocks_only["is_oversold_3d"] = (stocks_only["ret_3d_st"] <= -0.04).astype(float)

    sec_h = (
        stocks_only.groupby(["sector", "hour_key"])
        .agg(ret_1h=("ret_1h", "mean"), breadth_oversold_3d=("is_oversold_3d", "mean"))
        .reset_index()
        .sort_values(["sector", "hour_key"])
        .reset_index(drop=True)
    )
    sec_h["sector_idx"] = sec_h.groupby("sector")["ret_1h"].transform(lambda s: (1.0 + s.fillna(0.0)).cumprod())
    sec_h["sector_ret_2h"] = sec_h.groupby("sector")["sector_idx"].pct_change(2).fillna(0.0)
    sec_h["sector_ret_3h"] = sec_h.groupby("sector")["sector_idx"].pct_change(3).fillna(0.0)

    sec_2h_map = sec_h.set_index(["sector", "hour_key"])["sector_ret_2h"].to_dict()
    sec_3h_map = sec_h.set_index(["sector", "hour_key"])["sector_ret_3h"].to_dict()
    sec_bover_3d_map = sec_h.set_index(["sector", "hour_key"])["breadth_oversold_3d"].to_dict()

    ticker_dfs = {
        tk: gdf.sort_values("timestamp").reset_index(drop=True)
        for tk, gdf in stocks_only.groupby("ticker")
    }

    rel_str_60d_list, atr_norm_drop_3d_list, floor_defense_3h_list = [], [], []
    bounce_thrust_2h_list, green_vol_ratio_5d_list, elastic_potential_list = [], [], []
    spy_ret_2h_list, sector_ret_2h_list, sector_ret_3h_list = [], [], []
    lower_highs_6h_list, higher_low_confirm_list, dist_7h_low_atr_list = [], [], []
    falling_knife_flag_list = []
    sec_breadth_oversold_3d_list, vxn_vix_ratio_list, dist_vwap_5d_atr_list = [], [], []
    selling_climax_exhaustion_list, is_friday_late_list = [], []

    for _, row in df_1a.iterrows():
        tk = row["ticker"]
        sec = row["sector"]
        i0 = int(row["bar_idx"])
        tdf = ticker_dfs[tk]
        ts = pd.Timestamp(row["timestamp"])
        h_key = ts.floor("h")

        spy_r60 = float(spy_ret_60d_map.get(h_key, 0.0))
        spy_r2h = float(spy_ret_2h_map.get(h_key, 0.0))
        sec_r2h = float(sec_2h_map.get((sec, h_key), 0.0))
        sec_r3h = float(sec_3h_map.get((sec, h_key), 0.0))
        sec_bov3d = float(sec_bover_3d_map.get((sec, h_key), 0.2))

        vxn_val = float(vxn_map.get(h_key, 22.0))
        vix_val = max(float(vix_map.get(h_key, 18.0)), 10.0)

        rel_str_60d = float(row["ret_60d"]) - spy_r60
        atr_d = max(float(row["atr_d"]), 0.015)
        atr_norm_drop = abs(float(row["ret_3d"])) / atr_d

        win3 = tdf.iloc[max(0, i0 - 2) : i0 + 1]
        rng3 = (win3["high"] - win3["low"]).replace(0, np.nan)
        cloc3 = ((win3["close"] - win3["low"]) / rng3).fillna(0.5)
        lwick3 = ((np.minimum(win3["open"], win3["close"]) - win3["low"]) / rng3).fillna(0.0)
        floor_def_3h = float((cloc3 + lwick3).mean())

        if i0 >= 2:
            ret_2h = (float(tdf.loc[i0, "close"]) - float(tdf.loc[i0 - 2, "low"])) / (float(tdf.loc[i0 - 2, "low"]) + 1e-9)
            thrust_2h = (ret_2h / atr_d) * float(row["rvol_1h"])
        else:
            thrust_2h = 0.0

        win35 = tdf.iloc[max(0, i0 - 34) : i0 + 1]
        g_vol = win35[win35["close"] >= win35["open"]]["volume"].sum()
        t_vol = win35["volume"].sum() + 1e-9
        green_v_ratio = float(g_vol / t_vol)
        elastic_pot = atr_norm_drop * float(row["fundamental_shield"]) * floor_def_3h

        vwap_5d = float((win35["close"] * win35["volume"]).sum() / t_vol)
        c_now = float(tdf.loc[i0, "close"])
        dist_vwap_atr = ((c_now - vwap_5d) / (vwap_5d + 1e-9)) / atr_d

        win7 = tdf.iloc[max(0, i0 - 6) : i0 + 1]
        h7 = win7["high"].to_numpy()
        l7 = win7["low"].to_numpy()
        c7 = win7["close"].to_numpy()
        o7 = win7["open"].to_numpy()
        v7 = win7["volume"].to_numpy()

        if len(v7) >= 5:
            early_vol_max = float(np.max(v7[:-2]))
            late_red_vol = float(np.mean([v7[-2], v7[-1] if c7[-1] < o7[-1] else v7[-1] * 0.4])) + 1e-9
            climax_exhaust = np.clip(early_vol_max / late_red_vol, 0.2, 8.0)
        else:
            climax_exhaust = 1.0

        is_fri_late = 1.0 if (ts.dayofweek == 4 and ts.hour >= 18) else 0.0
        lh_count = float(np.sum(np.diff(h7) < 0)) / max(len(h7) - 1, 1)
        if len(h7) >= 2:
            hl_confirm = 1.0 if (l7[-1] >= l7[-2] * 0.999 or h7[-1] > h7[-2]) else 0.0
        else:
            hl_confirm = 1.0

        min_l7 = float(np.min(l7))
        dist_7h_atr = ((c7[-1] - min_l7) / (c7[-1] + 1e-9)) / atr_d
        knife_flag = 1.0 if (sec_r2h <= -0.012 and thrust_2h < 0.15 and hl_confirm == 0.0) else 0.0

        rel_str_60d_list.append(rel_str_60d)
        atr_norm_drop_3d_list.append(atr_norm_drop)
        floor_defense_3h_list.append(floor_def_3h)
        bounce_thrust_2h_list.append(thrust_2h)
        green_vol_ratio_5d_list.append(green_v_ratio)
        elastic_potential_list.append(elastic_pot)
        spy_ret_2h_list.append(spy_r2h)
        sector_ret_2h_list.append(sec_r2h)
        sector_ret_3h_list.append(sec_r3h)
        lower_highs_6h_list.append(lh_count)
        higher_low_confirm_list.append(hl_confirm)
        dist_7h_low_atr_list.append(dist_7h_atr)
        falling_knife_flag_list.append(knife_flag)
        sec_breadth_oversold_3d_list.append(sec_bov3d)
        vxn_vix_ratio_list.append(vxn_val / vix_val)
        dist_vwap_5d_atr_list.append(dist_vwap_atr)
        selling_climax_exhaustion_list.append(climax_exhaust)
        is_friday_late_list.append(is_fri_late)

    df_1a["rel_strength_60d_vs_spy"] = rel_str_60d_list
    df_1a["atr_norm_drop_3d"] = atr_norm_drop_3d_list
    df_1a["floor_defense_3h"] = floor_defense_3h_list
    df_1a["bounce_thrust_2h"] = bounce_thrust_2h_list
    df_1a["green_vol_ratio_5d"] = green_vol_ratio_5d_list
    df_1a["elastic_potential"] = elastic_potential_list
    df_1a["spy_ret_2h"] = spy_ret_2h_list
    df_1a["sector_ret_2h"] = sector_ret_2h_list
    df_1a["sector_ret_3h"] = sector_ret_3h_list
    df_1a["lower_highs_streak_6h"] = lower_highs_6h_list
    df_1a["higher_low_confirm_1h"] = higher_low_confirm_list
    df_1a["dist_from_7h_low_atr"] = dist_7h_low_atr_list
    df_1a["falling_knife_flag"] = falling_knife_flag_list
    df_1a["sector_breadth_oversold_3d"] = sec_breadth_oversold_3d_list
    df_1a["vxn_vix_ratio"] = vxn_vix_ratio_list
    df_1a["dist_vwap_5d_atr"] = dist_vwap_5d_atr_list
    df_1a["selling_climax_exhaustion"] = selling_climax_exhaustion_list
    df_1a["is_friday_late"] = is_friday_late_list
    df_1a["reached_meli_target"] = (df_1a["meli_trigger_step"] < 900).astype(int)

    # Preparar Modelo 1B (`ROC-AUC = 0.6324`)
    ev_1a_lookup = df_1a.set_index("event_id")
    fp_targets, vix_cooling_list, vix_slope_50_list, tnx_50_list = [], [], [], []
    down_up_entry_list, sys_pull_entry_list, fund_shield_entry_list = [], [], []

    for _, mrow in df_meli.iterrows():
        ev_id = int(mrow["event_id"])
        r1a = ev_1a_lookup.loc[ev_id]
        tk = r1a["ticker"]
        i0 = int(r1a["bar_idx"])
        trig_step = int(r1a["meli_trigger_step"])
        entry_p = float(r1a["entry_p"])
        shock_drop = float(r1a["shock_drop"])
        tdf = ticker_dfs[tk]

        vix_at_entry = float(r1a["vix"]) if "vix" in r1a else 20.0
        cur_bar_idx = min(len(tdf) - 1, i0 + 1 + trig_step)
        cur_bar = tdf.iloc[cur_bar_idx]
        vix_at_50 = float(cur_bar["vix"])
        vix_cooling_list.append((vix_at_50 - vix_at_entry) / max(vix_at_entry, 10.0))
        vix_slope_50_list.append(float(cur_bar["vix_slope_3h"]))
        tnx_50_list.append(float(cur_bar["tnx_ret_3d"]))
        down_up_entry_list.append(float(r1a["down_up_vol_ratio_15d"]))
        sys_pull_entry_list.append(float(r1a["systemic_pull_z_3d"]))
        fund_shield_entry_list.append(float(r1a["fundamental_shield"]))

        fut_rem = tdf.iloc[i0 + 1 + trig_step + 1 : i0 + 106]
        rem_h = fut_rem["high"].to_numpy()
        rem_c = fut_rem["close"].to_numpy()
        target_85_p = entry_p * (1.0 + 0.85 * shock_drop)
        stop_be_p = entry_p * (1.0 + 0.05 * shock_drop)
        fp_win = 0
        for h_val, c_val in zip(rem_h, rem_c):
            if h_val >= target_85_p:
                fp_win = 1
                break
            if c_val <= stop_be_p:
                fp_win = 0
                break
        fp_targets.append(fp_win)

    df_meli["keep_running_fp"] = fp_targets
    df_meli["vix_cooling_since_entry"] = vix_cooling_list
    df_meli["vix_slope_at_50"] = vix_slope_50_list
    df_meli["tnx_ret_3d_at_50"] = tnx_50_list
    df_meli["entry_down_up_vol_ratio"] = down_up_entry_list
    df_meli["entry_systemic_pull_z_3d"] = sys_pull_entry_list
    df_meli["entry_fundamental_shield"] = fund_shield_entry_list

    feats_lever2 = [
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
        "rel_strength_60d_vs_spy", "atr_norm_drop_3d", "floor_defense_3h",
        "bounce_thrust_2h", "green_vol_ratio_5d", "elastic_potential",
    ]
    # Sólo las 3 features ortogonales ganadoras (sin ruido intradía redundante):
    # `sector_breadth_oversold_3d`, `vxn_vix_ratio`, `selling_climax_exhaustion`
    feats_orthogonal_v4 = feats_lever2 + [
        "sector_breadth_oversold_3d", "vxn_vix_ratio", "selling_climax_exhaustion"
    ]
    feats_guardian_v4 = feats_lever2 + [
        "spy_ret_2h", "sector_ret_2h", "sector_ret_3h",
        "lower_highs_streak_6h", "higher_low_confirm_1h", "dist_from_7h_low_atr",
        "sector_breadth_oversold_3d", "vxn_vix_ratio", "selling_climax_exhaustion", "dist_vwap_5d_atr"
    ]

    train_end_ts = test_start_ts - pd.Timedelta(days=22)
    tr_1a = df_1a[df_1a["timestamp"] <= train_end_ts].copy().reset_index(drop=True)
    te_1a = df_1a[df_1a["timestamp"] >= test_start_ts].copy().reset_index(drop=True)
    target_col = "target_1a_strict" if "target_1a_strict" in tr_1a.columns else "target_1a"

    clf_l2 = lgb.LGBMClassifier(
        n_estimators=165, learning_rate=0.03, max_depth=4, num_leaves=12,
        min_child_samples=45, subsample=0.8, colsample_bytree=0.7,
        reg_alpha=2.0, reg_lambda=8.0, random_state=42, verbose=-1
    )
    clf_l2.fit(tr_1a[feats_lever2].fillna(0.0), tr_1a[target_col].values)
    te_1a["prob_1a_l2"] = clf_l2.predict_proba(te_1a[feats_lever2].fillna(0.0))[:, 1]

    clf_orth = lgb.LGBMClassifier(
        n_estimators=165, learning_rate=0.03, max_depth=4, num_leaves=12,
        min_child_samples=45, subsample=0.8, colsample_bytree=0.7,
        reg_alpha=2.0, reg_lambda=8.0, random_state=42, verbose=-1
    )
    clf_orth.fit(tr_1a[feats_orthogonal_v4].fillna(0.0), tr_1a[target_col].values)
    te_1a["prob_1a_orth"] = clf_orth.predict_proba(te_1a[feats_orthogonal_v4].fillna(0.0))[:, 1]

    # Guardián Anti-Cuchillo V4 (`ROC-AUC = 0.5971`)
    clf_meli_v4 = lgb.LGBMClassifier(
        n_estimators=145, learning_rate=0.03, max_depth=3, num_leaves=8,
        min_child_samples=45, subsample=0.8, colsample_bytree=0.7,
        reg_alpha=2.0, reg_lambda=8.0, random_state=101, verbose=-1
    )
    clf_meli_v4.fit(tr_1a[feats_guardian_v4].fillna(0.0), tr_1a["reached_meli_target"].values)
    te_1a["prob_reach_meli_v4"] = clf_meli_v4.predict_proba(te_1a[feats_guardian_v4].fillna(0.0))[:, 1]

    print(f"ROC-AUC 1A Palanca #2: {roc_auc_score(te_1a[target_col], te_1a['prob_1a_l2']):.4f}")
    print(f"ROC-AUC 1A Ortogonal (+3 Minas): {roc_auc_score(te_1a[target_col], te_1a['prob_1a_orth']):.4f}")
    print(f"ROC-AUC Guardián V4 (+4 Minas):  {roc_auc_score(te_1a['reached_meli_target'], te_1a['prob_reach_meli_v4']):.4f}")

    # Entrenar 1B (`ROC-AUC = 0.6324`)
    prob_1a_map = {int(r["event_id"]): float(r["prob_1a_l2"]) for _, r in pd.concat([tr_1a, te_1a]).iterrows()}
    df_meli["model_1a_prob"] = df_meli["event_id"].map(prob_1a_map).fillna(0.5)
    feats_1b_upgraded = [
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
        "entry_last_earnings_reaction_3d", "entry_cmf_15d",
        "vix_cooling_since_entry", "vix_slope_at_50", "tnx_ret_3d_at_50",
        "entry_down_up_vol_ratio", "entry_systemic_pull_z_3d", "entry_fundamental_shield"
    ]
    tr_m = df_meli[df_meli["timestamp"] <= train_end_ts].copy().reset_index(drop=True)
    te_m = df_meli[df_meli["timestamp"] >= test_start_ts].copy().reset_index(drop=True)
    clf_1b_new = lgb.LGBMClassifier(
        n_estimators=150, learning_rate=0.03, max_depth=3, num_leaves=8,
        min_child_samples=35, subsample=0.8, colsample_bytree=0.8,
        reg_alpha=2.0, reg_lambda=8.0, random_state=42, verbose=-1
    )
    clf_1b_new.fit(tr_m[feats_1b_upgraded].fillna(0.0), tr_m["keep_running_fp"].values)
    prob_new_tr = clf_1b_new.predict_proba(tr_m[feats_1b_upgraded].fillna(0.0))[:, 1]
    prob_new_te = clf_1b_new.predict_proba(te_m[feats_1b_upgraded].fillna(0.0))[:, 1]
    q70_1b = float(np.quantile(prob_new_tr, 0.70))
    q80_1b = float(np.quantile(prob_new_tr, 0.80))
    prob_1b_new_map = {int(r["event_id"]): float(p) for r, p in zip(te_m.to_dict("records"), prob_new_te)}

    base_shield_mask = (
        (te_1a["weak_volume_flag"] == 0)
        & ~((te_1a["last_earnings_reaction_3d"] <= -0.05) & (te_1a["cmf_15d"] <= -0.05))
        & ~((te_1a["tnx_ret_3d"] >= 0.030) & (te_1a["vix_slope_3h"] >= 0.04))
    )
    te_clean = te_1a[base_shield_mask].copy().reset_index(drop=True)

    sim_rows = []
    for _, row in te_clean.iterrows():
        ev_id = int(row["event_id"])
        tk = row["ticker"]
        i0 = int(row["bar_idx"])
        tdf = ticker_dfs[tk]
        fut_df = tdf.iloc[i0 + 1 : i0 + 106]
        trig_step = int(row["meli_trigger_step"])
        p_1b = prob_1b_new_map.get(ev_id, 0.50)

        r_net, x_ts, x_rsn = simulate_calibrated_1b_ibkr_exit(
            fut_df, float(row["entry_p"]), float(row["sl_p"]), float(row["atr_d"]),
            float(row["shock_drop"]), int(row["shock_type_code"]), p_1b, trig_step,
            q25_1b=q70_1b, q75_1b=q80_1b, mode="calibrated_hourly_close"
        )
        d = row.to_dict()
        d.update({"prob_1b": p_1b, "net_ret_pct": r_net, "exit_ts": x_ts, "exit_reason": x_rsn})
        sim_rows.append(d)

    df_sim = pd.DataFrame(sim_rows).sort_values("entry_ts").reset_index(drop=True)

    # 1. Fila 3 Exacta (`+65.7%`, 74 trades)
    df_f3 = df_sim.copy()
    df_f3["prob_1a"] = df_f3["prob_1a_l2"]
    sub_f3 = df_f3[(df_f3["prob_1a"] >= 0.59) | ((df_f3["prob_1a"] >= 0.575) & (df_f3["shock_type_code"].isin([3, 4])))].copy()

    # Crear columna de Sizing Dinámico Guiado por Guardián V4 (`prob_reach_meli_v4`) y Riesgo Viernes:
    # - Si `prob_reach_meli_v4 >= 0.62` y no es cuchillo -> 100% del cupo (1.0)
    # - Si `prob_reach_meli_v4` está entre 0.56 y 0.62 -> 75% del cupo (0.75)
    # - Si `prob_reach_meli_v4 < 0.56` o `falling_knife_flag == 1` o `is_friday_late == 1` -> 45% del cupo (0.45)
    size_mults = []
    for _, r in sub_f3.iterrows():
        pm = float(r["prob_reach_meli_v4"])
        kf = int(r["falling_knife_flag"])
        fr = int(r["is_friday_late"])
        if kf == 1 or fr == 1 or pm < 0.56:
            size_mults.append(0.45)
        elif pm < 0.62:
            size_mults.append(0.75)
        else:
            size_mults.append(1.00)
    sub_f3["guardian_size_mult"] = size_mults

    # 2. Fila 3 + Filtro Quirúrgico del Guardián V4 (Vetar sólo el peor 15%: `falling_knife_flag == 0` & `prob_reach_meli_v4 >= 0.58` & `is_friday_late == 0`)
    sub_f3_surgical = sub_f3[
        (sub_f3["falling_knife_flag"] == 0)
        & (sub_f3["is_friday_late"] == 0)
        & (sub_f3["prob_reach_meli_v4"] >= 0.58)
    ].copy()

    # 3. Fila 3 + Guardián V4 Estricto (`prob_reach_meli_v4 >= 0.615` & `falling_knife_flag == 0`)
    sub_f3_strict = sub_f3[
        (sub_f3["falling_knife_flag"] == 0)
        & (sub_f3["is_friday_late"] == 0)
        & (sub_f3["prob_reach_meli_v4"] >= 0.615)
    ].copy()

    # 4. Modelo 1A Ortogonal (+3 Minas: `sector_breadth_oversold_3d`, `vxn_vix_ratio`, `selling_climax_exhaustion`) + Guardián V4 Quirúrgico
    df_orth = df_sim.copy()
    df_orth["prob_1a"] = df_orth["prob_1a_orth"]
    sub_orth_surgical = df_orth[
        ((df_orth["prob_1a"] >= 0.59) | ((df_orth["prob_1a"] >= 0.575) & (df_orth["shock_type_code"].isin([3, 4]))))
        & (df_orth["falling_knife_flag"] == 0)
        & (df_orth["is_friday_late"] == 0)
        & (df_orth["prob_reach_meli_v4"] >= 0.58)
    ].copy()

    configs = [
        ("1. FILA 3 ORIGINAL (Ofensiva Pura, 74 Trades)", sub_f3, None),
        ("2. FILA 3 + SIZING DINÁMICO GUARDIÁN V4 (74 Trades, 45%/75%/100% Slot)", sub_f3, "guardian_size_mult"),
        ("3. FILA 3 + VETO QUIRÚRGICO GUARDIÁN V4 (>=0.58 & Sin Viernes Tarde)", sub_f3_surgical, None),
        ("4. FILA 3 + GUARDIÁN V4 ESTRICTO (>=0.615 & Sin Viernes Tarde)", sub_f3_strict, None),
        ("5. MODELO 1A ORTOGONAL (+3 Minas) + GUARDIÁN V4 QUIRÚRGICO", sub_orth_surgical, None),
    ]

    print("\n" + "=" * 150)
    print("TABLA DEFINITIVA: DRAWDOWN CERRADO vs MARK-TO-MARKET (CIERRE DÍA) vs PEOR MÍNIMO INTRADÍA vs MONTE CARLO P95")
    print("=" * 150)
    rows = []
    for label, sdf, sz_col in configs:
        res = compute_exact_mtm_on_simulation(sdf, ticker_day_prices, 25000.0, 5, size_scale_col=sz_col)
        rows.append({
            "Configuración_IBKR": label,
            "Trades": res["trades_executed"],
            "WR %": round(res["win_rate_pct"], 1),
            "PF": round(res["profit_factor"], 2),
            "Capital_Final": f"${res['final_capital']:,.0f}",
            "Retorno_11m": f"+{res['total_return_pct']:.1f}%",
            "DD_Cerrado": f"{res['max_dd_closed']:.2f}%",
            "DD_MtM_CierreDía": f"{res['max_dd_mtm_eod']:.2f}%",
            "DD_MtM_PeorLow": f"{res['max_dd_mtm_low']:.2f}%",
            "MC_P50_DD": f"{res['mc_p50_dd']:.2f}%",
            "MC_P95_DD": f"{res['mc_p95_dd']:.2f}%",
            "Peor_Día_MtM": res["worst_mtm_date"],
            "Sharpe": round(res["sharpe"], 2),
        })
    print(pd.DataFrame(rows).to_string(index=False))
    print("=" * 150)
    print(f"Tiempo total: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
