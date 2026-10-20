"""
Auditoría Forense de Drawdown Mark-to-Market (Flotante vs Cerrado) en IBKR,
Nuevas Minas de Feature Engineering (Amplitud Sectorial en Tiempo Real, VWAP Institucional 5D,
Clímax de Agotamiento de Volumen y Rotación QQQ/VXN) y Reducción del Drawdown de la Fila 3 (+65.7%).
"""

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.upgrade_model_1b_first_passage_and_perception import simulate_calibrated_1b_ibkr_exit


def run_ibkr_portfolio_with_full_mtm_audit(
    signals_df: pd.DataFrame,
    ticker_day_prices: dict,
    initial_capital: float = 25000.0,
    max_slots: int = 5,
    max_per_sector: int = 99,
    dynamic_sizing: bool = False,
):
    """
    Simulador de Portafolio IBKR que registra:
    1. `closed_max_dd_pct`: Drawdown tradicional sobre operaciones cerradas (valuando abiertas al costo).
    2. `mtm_eod_max_dd_pct`: Drawdown Mark-to-Market al cierre de cada día hábil (valuando todas las posiciones
       abiertas al precio `Close` real de ese día en IBKR).
    3. `mtm_intraday_low_max_dd_pct`: Peor Drawdown Flotante Intradía (valuando todas las posiciones abiertas
       al mínimo `Low` simultáneo del día en IBKR).
    4. `mc_p50_dd_pct` y `mc_p95_dd_pct`: Distribución de Drawdown por Bootstrap Monte Carlo (2,000 caminos).
    """
    df_sorted = signals_df.sort_values(["entry_ts", "prob_1a"], ascending=[True, False]).reset_index(drop=True)
    if df_sorted.empty:
        return {
            "trades_executed": 0, "win_rate_pct": 0.0, "avg_trade_ret_pct": 0.0, "profit_factor": 0.0,
            "final_capital": initial_capital, "total_return_pct": 0.0,
            "closed_max_dd_pct": 0.0, "mtm_eod_max_dd_pct": 0.0, "mtm_intraday_low_max_dd_pct": 0.0,
            "mc_p50_dd_pct": 0.0, "mc_p95_dd_pct": 0.0, "sharpe_closed": 0.0, "sharpe_mtm": 0.0,
        }

    all_days = pd.date_range(
        df_sorted["entry_ts"].min().floor("D"),
        df_sorted["exit_ts"].max().ceil("D"),
        freq="B",
        tz="UTC",
    )

    cash = initial_capital
    open_positions = []
    closed_trades = []
    daily_records = []

    sig_idx = 0
    n_sigs = len(df_sorted)

    for d_ts in all_days:
        day_key = d_ts.floor("D")
        day_end = d_ts + pd.Timedelta(hours=23, minutes=59)

        while sig_idx < n_sigs and pd.Timestamp(df_sorted.loc[sig_idx, "entry_ts"]) <= day_end:
            row = df_sorted.loc[sig_idx]
            sig_idx += 1
            cur_ts = pd.Timestamp(row["entry_ts"])

            # Cerrar posiciones vencidas hasta `cur_ts`
            still_open = []
            for pos in open_positions:
                if pos["exit_ts"] <= cur_ts:
                    cash += pos["returned_cash"]
                    closed_trades.append(pos)
                else:
                    still_open.append(pos)
            open_positions = still_open

            if len(open_positions) >= max_slots:
                continue
            if any(p["ticker"] == row["ticker"] for p in open_positions):
                continue

            # Regla de diversificación sectorial (`max_per_sector`)
            sec = row["sector"]
            if sum(1 for p in open_positions if p["sector"] == sec) >= max_per_sector:
                continue

            total_eq_now = cash + sum(p["invested"] for p in open_positions)
            base_slot = total_eq_now / max_slots

            if dynamic_sizing:
                # Escalar el tamaño del slot según la certeza del Guardián MeLi y la sincronía sectorial
                p_meli = float(row.get("prob_reach_meli", 0.65))
                sec_green = float(row.get("sector_breadth_green_1h", 0.50))
                if p_meli >= 0.66 and sec_green >= 0.45:
                    size_mult = 1.00
                elif p_meli >= 0.61:
                    size_mult = 0.85
                else:
                    size_mult = 0.65
                alloc = min(cash, base_slot * size_mult)
            else:
                alloc = min(cash, base_slot)

            if alloc < 500:
                continue

            cash -= alloc
            ret_net = float(row["net_ret_pct"]) / 100.0
            returned = alloc * (1.0 + ret_net)

            open_positions.append({
                "ticker": row["ticker"],
                "sector": sec,
                "entry_ts": cur_ts,
                "exit_ts": pd.Timestamp(row["exit_ts"]),
                "entry_p": float(row["entry_p"]),
                "sl_p": float(row["sl_p"]),
                "invested": alloc,
                "returned_cash": returned,
                "net_ret_pct": float(row["net_ret_pct"]),
            })

        # Cerrar posiciones que salieron antes de `day_end`
        still_open = []
        for pos in open_positions:
            if pos["exit_ts"] <= day_end:
                cash += pos["returned_cash"]
                closed_trades.append(pos)
            else:
                still_open.append(pos)
        open_positions = still_open

        # 1. Equidad Contable Cerrada (al costo)
        eq_closed = cash + sum(p["invested"] for p in open_positions)

        # 2. Equidad Mark-to-Market al Cierre del Día (`mtm_eod`) y al Mínimo Intradía (`mtm_low`)
        mtm_eod_val = cash
        mtm_low_val = cash
        for pos in open_positions:
            tk = pos["ticker"]
            ep = pos["entry_p"]
            slp = pos["sl_p"]
            inv = pos["invested"]
            day_px = ticker_day_prices.get((tk, day_key), None)
            if day_px is not None:
                c_px, l_px = day_px
                # El mínimo intradía no baja más allá del stop loss de emergencia si seguía abierta al cierre
                l_px_eff = max(l_px, slp * 0.995)
                mtm_eod_val += inv * (c_px / ep)
                mtm_low_val += inv * (l_px_eff / ep)
            else:
                mtm_eod_val += inv
                mtm_low_val += inv

        daily_records.append({
            "day": day_key,
            "eq_closed": eq_closed,
            "eq_mtm_eod": mtm_eod_val,
            "eq_mtm_low": mtm_low_val,
            "n_open": len(open_positions),
        })

    for pos in sorted(open_positions, key=lambda x: x["exit_ts"]):
        cash += pos["returned_cash"]
        closed_trades.append(pos)

    tdf = pd.DataFrame(closed_trades)
    wins = tdf[tdf["net_ret_pct"] > 0]["net_ret_pct"].sum()
    losses = abs(tdf[tdf["net_ret_pct"] <= 0]["net_ret_pct"].sum())
    pf = wins / losses if losses > 0 else 9.99

    eq_df = pd.DataFrame(daily_records)
    eq_df["peak_closed"] = eq_df["eq_closed"].cummax()
    eq_df["dd_closed"] = (eq_df["eq_closed"] - eq_df["peak_closed"]) / eq_df["peak_closed"] * 100.0

    eq_df["peak_mtm"] = eq_df["eq_mtm_eod"].cummax()
    eq_df["dd_mtm_eod"] = (eq_df["eq_mtm_eod"] - eq_df["peak_mtm"]) / eq_df["peak_mtm"] * 100.0
    eq_df["dd_mtm_low"] = (eq_df["eq_mtm_low"] - eq_df["peak_mtm"]) / eq_df["peak_mtm"] * 100.0

    closed_max_dd = float(eq_df["dd_closed"].min())
    mtm_eod_max_dd = float(eq_df["dd_mtm_eod"].min())
    mtm_low_max_dd = float(eq_df["dd_mtm_low"].min())

    d_rets_closed = eq_df["eq_closed"].pct_change().dropna()
    sharpe_closed = float((d_rets_closed.mean() / d_rets_closed.std()) * np.sqrt(252)) if len(d_rets_closed) > 10 and d_rets_closed.std() > 0 else 0.0

    d_rets_mtm = eq_df["eq_mtm_eod"].pct_change().dropna()
    sharpe_mtm = float((d_rets_mtm.mean() / d_rets_mtm.std()) * np.sqrt(252)) if len(d_rets_mtm) > 10 and d_rets_mtm.std() > 0 else 0.0

    # 3. Bootstrap / Monte Carlo de 2,000 secuencias (reasignando el orden de los trades en bloques)
    rng = np.random.default_rng(42)
    trade_rets = tdf["net_ret_pct"].to_numpy() / 100.0
    n_tr = len(trade_rets)
    mc_dds = []
    if n_tr >= 5:
        for _ in range(2000):
            sampled = rng.choice(trade_rets, size=n_tr, replace=True)
            # Cada trade impacta 1/max_slots del portafolio
            port_step_rets = sampled / max_slots
            eq_path = initial_capital * np.cumprod(1.0 + np.insert(port_step_rets, 0, 0.0))
            pk = np.maximum.accumulate(eq_path)
            dd_path = (eq_path - pk) / pk * 100.0
            mc_dds.append(float(np.min(dd_path)))
        mc_p50_dd = float(np.percentile(mc_dds, 50))
        mc_p95_dd = float(np.percentile(mc_dds, 5))  # Percentil 5 más negativo = P95 de severidad
    else:
        mc_p50_dd, mc_p95_dd = closed_max_dd, closed_max_dd

    return {
        "trades_executed": len(tdf),
        "win_rate_pct": float((tdf["net_ret_pct"] > 0).mean() * 100.0),
        "avg_trade_ret_pct": float(tdf["net_ret_pct"].mean()),
        "profit_factor": float(pf),
        "final_capital": float(cash),
        "total_return_pct": float((cash - initial_capital) / initial_capital * 100.0),
        "closed_max_dd_pct": closed_max_dd,
        "mtm_eod_max_dd_pct": mtm_eod_max_dd,
        "mtm_intraday_low_max_dd_pct": mtm_low_max_dd,
        "mc_p50_dd_pct": mc_p50_dd,
        "mc_p95_dd_pct": mc_p95_dd,
        "sharpe_closed": sharpe_closed,
        "sharpe_mtm": sharpe_mtm,
        "trades_df": tdf,
        "eq_df": eq_df,
    }


def main():
    t0 = time.time()
    cache_1a_path = PROJECT_ROOT / "data" / "processed" / "strict_prod_df_1a.parquet"
    cache_meli_path = PROJECT_ROOT / "data" / "processed" / "strict_prod_df_meli.parquet"
    hourly_path = PROJECT_ROOT / "data" / "processed" / "sp500_multisector_1hour.parquet"
    macro_path = PROJECT_ROOT / "data" / "processed" / "macro_perception_1hour.parquet"

    print("[1/4] Cargando cachés de producción y construyendo las 4 Nuevas Minas de Feature Engineering...")
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

    # Diccionario de precios diarios (Close y Low del día) para auditoría Mark-to-Market exacta
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

    # Mapas de SPY y QQQ (Mina B: Rotación QQQ vs SPY)
    spy_df = df_raw[df_raw["ticker"] == "SPY"].sort_values("timestamp").drop_duplicates("hour_key").set_index("hour_key")
    qqq_df = df_raw[df_raw["ticker"] == "QQQ"].sort_values("timestamp").drop_duplicates("hour_key").set_index("hour_key")

    spy_ret_60d_map = spy_df["close"].pct_change(420).fillna(0.0)
    spy_ret_3d_map = spy_df["close"].pct_change(21).fillna(0.0)
    spy_ret_2h_map = spy_df["close"].pct_change(2).fillna(0.0)
    qqq_ret_3d_map = qqq_df["close"].pct_change(21).fillna(0.0)
    qqq_ret_2h_map = qqq_df["close"].pct_change(2).fillna(0.0)

    vxn_map = macro_df["vxn"].ffill().fillna(20.0) if "vxn" in macro_df.columns else pd.Series(dtype=float)
    vix_map = macro_df["vix"].ffill().fillna(20.0) if "vix" in macro_df.columns else pd.Series(dtype=float)

    # Mina A: Amplitud / Sincronía Interna del Sector en Tiempo Real (`sector_breadth_green_1h`, `sector_breadth_oversold_3d`)
    stocks_only = df_raw[~df_raw["ticker"].isin(["SPY", "QQQ"])].copy()
    stocks_only = stocks_only.sort_values(["ticker", "timestamp"]).reset_index(drop=True)
    stocks_only["ret_1h"] = stocks_only.groupby("ticker")["close"].pct_change(1)
    stocks_only["ret_3d_st"] = stocks_only.groupby("ticker")["close"].pct_change(21)
    stocks_only["is_green_1h"] = (stocks_only["ret_1h"] > 0.0005).astype(float)
    stocks_only["is_oversold_3d"] = (stocks_only["ret_3d_st"] <= -0.04).astype(float)

    sec_h = (
        stocks_only.groupby(["sector", "hour_key"])
        .agg(
            ret_1h=("ret_1h", "mean"),
            breadth_green_1h=("is_green_1h", "mean"),
            breadth_oversold_3d=("is_oversold_3d", "mean"),
            disp_1h=("ret_1h", "std"),
        )
        .reset_index()
        .sort_values(["sector", "hour_key"])
        .reset_index(drop=True)
    )
    sec_h["sector_idx"] = sec_h.groupby("sector")["ret_1h"].transform(lambda s: (1.0 + s.fillna(0.0)).cumprod())
    sec_h["sector_ret_2h"] = sec_h.groupby("sector")["sector_idx"].pct_change(2).fillna(0.0)
    sec_h["sector_ret_3h"] = sec_h.groupby("sector")["sector_idx"].pct_change(3).fillna(0.0)
    sec_h["breadth_green_2h"] = sec_h.groupby("sector")["breadth_green_1h"].transform(lambda s: s.rolling(2, min_periods=1).mean())

    sec_2h_map = sec_h.set_index(["sector", "hour_key"])["sector_ret_2h"].to_dict()
    sec_3h_map = sec_h.set_index(["sector", "hour_key"])["sector_ret_3h"].to_dict()
    sec_bgreen_1h_map = sec_h.set_index(["sector", "hour_key"])["breadth_green_1h"].to_dict()
    sec_bgreen_2h_map = sec_h.set_index(["sector", "hour_key"])["breadth_green_2h"].to_dict()
    sec_bover_3d_map = sec_h.set_index(["sector", "hour_key"])["breadth_oversold_3d"].to_dict()

    ticker_dfs = {
        tk: gdf.sort_values("timestamp").reset_index(drop=True)
        for tk, gdf in stocks_only.groupby("ticker")
    }

    # Calcular features de Palanca #2, Palanca #3 y las 4 Nuevas Minas de Feature Engineering
    rel_str_60d_list, atr_norm_drop_3d_list, floor_defense_3h_list = [], [], []
    bounce_thrust_2h_list, green_vol_ratio_5d_list, elastic_potential_list = [], [], []
    spy_ret_2h_list, sector_ret_2h_list, sector_ret_3h_list = [], [], []
    lower_highs_6h_list, higher_low_confirm_list, dist_7h_low_atr_list = [], [], []
    falling_knife_flag_list = []

    # Listas para las 4 Nuevas Minas
    sec_breadth_green_1h_list, sec_breadth_green_2h_list, sec_breadth_oversold_3d_list = [], [], []
    qqq_vs_spy_3d_list, qqq_ret_2h_list, vxn_vix_ratio_list = [], [], []
    dist_vwap_5d_atr_list, selling_climax_exhaustion_list = [], []
    is_friday_late_list, is_power_hour_list = [], []

    for _, row in df_1a.iterrows():
        tk = row["ticker"]
        sec = row["sector"]
        i0 = int(row["bar_idx"])
        tdf = ticker_dfs[tk]
        ts = pd.Timestamp(row["timestamp"])
        h_key = ts.floor("h")

        spy_r60 = float(spy_ret_60d_map.get(h_key, 0.0))
        spy_r3d = float(spy_ret_3d_map.get(h_key, 0.0))
        spy_r2h = float(spy_ret_2h_map.get(h_key, 0.0))
        qqq_r3d = float(qqq_ret_3d_map.get(h_key, 0.0))
        qqq_r2h = float(qqq_ret_2h_map.get(h_key, 0.0))

        vxn_val = float(vxn_map.get(h_key, 22.0))
        vix_val = max(float(vix_map.get(h_key, 18.0)), 10.0)

        sec_r2h = float(sec_2h_map.get((sec, h_key), 0.0))
        sec_r3h = float(sec_3h_map.get((sec, h_key), 0.0))
        sec_bg1h = float(sec_bgreen_1h_map.get((sec, h_key), 0.5))
        sec_bg2h = float(sec_bgreen_2h_map.get((sec, h_key), 0.5))
        sec_bov3d = float(sec_bover_3d_map.get((sec, h_key), 0.2))

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

        # Mina D1: Distancia al VWAP Institucional de 5 Días (`dist_vwap_5d_atr`)
        vwap_5d = float((win35["close"] * win35["volume"]).sum() / t_vol)
        c_now = float(tdf.loc[i0, "close"])
        dist_vwap_atr = ((c_now - vwap_5d) / (vwap_5d + 1e-9)) / atr_d

        # Mina C: Clímax de Venta Previo + Secado de Volumen Vendedor (`selling_climax_exhaustion`)
        win7 = tdf.iloc[max(0, i0 - 6) : i0 + 1]
        h7 = win7["high"].to_numpy()
        l7 = win7["low"].to_numpy()
        c7 = win7["close"].to_numpy()
        o7 = win7["open"].to_numpy()
        v7 = win7["volume"].to_numpy()

        if len(v7) >= 5:
            # Volumen máximo en las horas previas de liquidación vs volumen en las últimas 2 horas de piso
            early_vol_max = float(np.max(v7[:-2]))
            late_red_vol = float(np.mean([v7[-2], v7[-1] if c7[-1] < o7[-1] else v7[-1] * 0.4])) + 1e-9
            climax_exhaust = np.clip(early_vol_max / late_red_vol, 0.2, 8.0)
        else:
            climax_exhaust = 1.0

        # Mina D2: Efecto Calendario / Hora Institucional NY
        is_fri_late = 1.0 if (ts.dayofweek == 4 and ts.hour >= 18) else 0.0
        is_pwr_hr = 1.0 if (ts.hour in [14, 15, 19, 20]) else 0.0

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

        sec_breadth_green_1h_list.append(sec_bg1h)
        sec_breadth_green_2h_list.append(sec_bg2h)
        sec_breadth_oversold_3d_list.append(sec_bov3d)
        qqq_vs_spy_3d_list.append(qqq_r3d - spy_r3d)
        qqq_ret_2h_list.append(qqq_r2h)
        vxn_vix_ratio_list.append(vxn_val / vix_val)
        dist_vwap_5d_atr_list.append(dist_vwap_atr)
        selling_climax_exhaustion_list.append(climax_exhaust)
        is_friday_late_list.append(is_fri_late)
        is_power_hour_list.append(is_pwr_hr)

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

    # Nuevas Minas
    df_1a["sector_breadth_green_1h"] = sec_breadth_green_1h_list
    df_1a["sector_breadth_green_2h"] = sec_breadth_green_2h_list
    df_1a["sector_breadth_oversold_3d"] = sec_breadth_oversold_3d_list
    df_1a["qqq_vs_spy_3d"] = qqq_vs_spy_3d_list
    df_1a["qqq_ret_2h"] = qqq_ret_2h_list
    df_1a["vxn_vix_ratio"] = vxn_vix_ratio_list
    df_1a["dist_vwap_5d_atr"] = dist_vwap_5d_atr_list
    df_1a["selling_climax_exhaustion"] = selling_climax_exhaustion_list
    df_1a["is_friday_late"] = is_friday_late_list
    df_1a["is_power_hour"] = is_power_hour_list

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
    feats_lever3 = feats_lever2 + [
        "spy_ret_2h", "sector_ret_2h", "sector_ret_3h",
        "lower_highs_streak_6h", "higher_low_confirm_1h", "dist_from_7h_low_atr"
    ]
    feats_next_goldmine = feats_lever3 + [
        "sector_breadth_green_1h", "sector_breadth_green_2h", "sector_breadth_oversold_3d",
        "qqq_vs_spy_3d", "qqq_ret_2h", "vxn_vix_ratio",
        "dist_vwap_5d_atr", "selling_climax_exhaustion",
        "is_friday_late", "is_power_hour",
    ]

    train_end_ts = test_start_ts - pd.Timedelta(days=22)
    tr_1a = df_1a[df_1a["timestamp"] <= train_end_ts].copy().reset_index(drop=True)
    te_1a = df_1a[df_1a["timestamp"] >= test_start_ts].copy().reset_index(drop=True)
    target_col = "target_1a_strict" if "target_1a_strict" in tr_1a.columns else "target_1a"

    # Diagnóstico en Train de las 4 Nuevas Minas
    w_tr = tr_1a[tr_1a[target_col] == 1]
    l_tr = tr_1a[tr_1a[target_col] == 0]
    print("\n" + "=" * 126)
    print("1. HALLAZGOS EN TRAIN DE LAS 4 NUEVAS MINAS DE FEATURE ENGINEERING (100% CAUSALES EN IBKR):")
    print("=" * 126)
    new_mine_diag = [
        ("sector_breadth_green_1h", "Mina A: % de acciones del mismo sector en verde en la hora t"),
        ("sector_breadth_green_2h", "Mina A: % de acciones del mismo sector girando en verde (2H)"),
        ("sector_breadth_oversold_3d", "Mina A: % de acciones del sector en sobreventa simultánea (3D)"),
        ("qqq_vs_spy_3d", "Mina B: Rotación Growth/Tech (QQQ) vs S&P 500 (SPY) en 3 días"),
        ("vxn_vix_ratio", "Mina B: Ratio de Miedo Nasdaq (^VXN) vs Miedo S&P 500 (^VIX)"),
        ("selling_climax_exhaustion", "Mina C: Ratio Clímax de Volumen Previo / Secado Reciente"),
        ("dist_vwap_5d_atr", "Mina D: Descuento respecto al VWAP Institucional de 5 días (en ATRs)"),
        ("is_friday_late", "Mina D: Entrada en Viernes tarde (riesgo de gap de fin de semana)"),
    ]
    m_rows = []
    for col, desc in new_mine_diag:
        mw = float(w_tr[col].mean())
        ml = float(l_tr[col].mean())
        m_rows.append({
            "Nueva_Feature": col,
            "Descripción": desc,
            "Media_Ganadores_Train": round(mw, 4),
            "Media_Perdedores_Train": round(ml, 4),
        })
    print(pd.DataFrame(m_rows).to_string(index=False))

    # Entrenar modelos 1A: Palanca #2, Palanca #3 y Nueva Mina V4 (`feats_next_goldmine`)
    clf_l2 = lgb.LGBMClassifier(
        n_estimators=165, learning_rate=0.03, max_depth=4, num_leaves=12,
        min_child_samples=45, subsample=0.8, colsample_bytree=0.7,
        reg_alpha=2.0, reg_lambda=8.0, random_state=42, verbose=-1
    )
    clf_l2.fit(tr_1a[feats_lever2].fillna(0.0), tr_1a[target_col].values)
    te_1a["prob_1a_l2"] = clf_l2.predict_proba(te_1a[feats_lever2].fillna(0.0))[:, 1]

    clf_l3 = lgb.LGBMClassifier(
        n_estimators=165, learning_rate=0.03, max_depth=4, num_leaves=12,
        min_child_samples=45, subsample=0.8, colsample_bytree=0.7,
        reg_alpha=2.0, reg_lambda=8.0, random_state=42, verbose=-1
    )
    clf_l3.fit(tr_1a[feats_lever3].fillna(0.0), tr_1a[target_col].values)
    te_1a["prob_1a_l3"] = clf_l3.predict_proba(te_1a[feats_lever3].fillna(0.0))[:, 1]

    clf_v4 = lgb.LGBMClassifier(
        n_estimators=175, learning_rate=0.03, max_depth=4, num_leaves=12,
        min_child_samples=45, subsample=0.8, colsample_bytree=0.7,
        reg_alpha=2.0, reg_lambda=8.0, random_state=42, verbose=-1
    )
    clf_v4.fit(tr_1a[feats_next_goldmine].fillna(0.0), tr_1a[target_col].values)
    te_1a["prob_1a_v4"] = clf_v4.predict_proba(te_1a[feats_next_goldmine].fillna(0.0))[:, 1]

    # Guardián MeLi enriquecido con las nuevas minas (`clf_meli_v4`)
    clf_meli_v4 = lgb.LGBMClassifier(
        n_estimators=145, learning_rate=0.03, max_depth=3, num_leaves=8,
        min_child_samples=45, subsample=0.8, colsample_bytree=0.7,
        reg_alpha=2.0, reg_lambda=8.0, random_state=101, verbose=-1
    )
    clf_meli_v4.fit(tr_1a[feats_next_goldmine].fillna(0.0), tr_1a["reached_meli_target"].values)
    te_1a["prob_reach_meli"] = clf_meli_v4.predict_proba(te_1a[feats_next_goldmine].fillna(0.0))[:, 1]

    auc_l2 = roc_auc_score(te_1a[target_col].values, te_1a["prob_1a_l2"].values)
    auc_l3 = roc_auc_score(te_1a[target_col].values, te_1a["prob_1a_l3"].values)
    auc_v4 = roc_auc_score(te_1a[target_col].values, te_1a["prob_1a_v4"].values)
    auc_meli = roc_auc_score(te_1a["reached_meli_target"].values, te_1a["prob_reach_meli"].values)

    print("\n" + "=" * 126)
    print(f"2. EVOLUCIÓN DEL ROC-AUC FUERA DE MUESTRA (OOS) AL AGREGAR LAS 4 NUEVAS MINAS:")
    print(f"   * Modelo 1A Palanca #2:                         {auc_l2:.4f}")
    print(f"   * Modelo 1A Palanca #3:                         {auc_l3:.4f}")
    print(f"   * Modelo 1A V4 (+ Amplitud Sector + VWAP + QQQ):{auc_v4:.4f}")
    print(f"   * Guardián Anti-Cuchillo V4 (Llegar al +45%):   {auc_meli:.4f}")
    print("=" * 126)

    # Entrenar Modelo 1B (`ROC-AUC = 0.6324`) y simular salidas en IBKR
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

    # Preparar datasets de las configuraciones a auditar
    # A) Fila 3 Original (+65.7%): Modelo 1A Palanca #2, Corte >= 0.59 o [>=0.575 & Macro/Sector]
    df_fila3 = df_sim.copy()
    df_fila3["prob_1a"] = df_fila3["prob_1a_l2"]
    sub_fila3 = df_fila3[(df_fila3["prob_1a"] >= 0.59) | ((df_fila3["prob_1a"] >= 0.575) & (df_fila3["shock_type_code"].isin([3, 4])))].copy()

    # B) Fila 4 Original (+47.0%): Palanca #3 con Guardián Duro >= 0.62
    df_fila4 = df_sim.copy()
    df_fila4["prob_1a"] = df_fila4["prob_1a_l3"]
    sub_fila4 = df_fila4[
        (df_fila4["falling_knife_flag"] == 0)
        & (df_fila4["prob_reach_meli"] >= 0.62)
        & ((df_fila4["prob_1a"] >= 0.59) | ((df_fila4["prob_1a"] >= 0.575) & (df_fila4["shock_type_code"].isin([3, 4]))))
    ].copy()

    # C) Fila 3 + Tope de Concentración Sectorial (`max_per_sector = 2`)
    # D) Nuevo Modelo V4 (4 Nuevas Minas) + Guardián Suave (`prob_reach_meli >= 0.60`, `falling_knife_flag == 0`, `is_friday_late == 0`)
    df_v4 = df_sim.copy()
    df_v4["prob_1a"] = df_v4["prob_1a_v4"]
    sub_v4_off = df_v4[
        (df_v4["falling_knife_flag"] == 0)
        & (df_v4["is_friday_late"] == 0)
        & (df_v4["prob_reach_meli"] >= 0.59)
        & ((df_v4["prob_1a"] >= 0.59) | ((df_v4["prob_1a"] >= 0.575) & (df_v4["shock_type_code"].isin([3, 4]))))
    ].copy()

    sub_v4_guarded = df_v4[
        (df_v4["falling_knife_flag"] == 0)
        & (df_v4["is_friday_late"] == 0)
        & (df_v4["prob_reach_meli"] >= 0.62)
        & ((df_v4["prob_1a"] >= 0.59) | ((df_v4["prob_1a"] >= 0.575) & (df_v4["shock_type_code"].isin([3, 4]))))
    ].copy()

    audit_configs = [
        ("1. FILA 3 ORIGINAL (Ofensiva +65.7%, Sin Tope Sectorial)", sub_fila3, 99, False),
        ("2. FILA 3 + TOPE 2 POR SECTOR (Máx 2 del mismo sector a la vez)", sub_fila3, 2, False),
        ("3. FILA 3 + SIZING DINÁMICO + TOPE 2 POR SECTOR", sub_fila3, 2, True),
        ("4. FILA 4 ORIGINAL (Defensiva Guardián >= 0.62)", sub_fila4, 99, False),
        ("5. NUEVO V4 (4 Minas Nuevas + Guardián >= 0.59 + Sin Viernes Tarde)", sub_v4_off, 99, False),
        ("6. NUEVO V4 + TOPE 2 POR SECTOR (Ofensivo Blindado)", sub_v4_off, 2, False),
        ("7. NUEVO V4 + GUARDIÁN >= 0.62 + TOPE 2 POR SECTOR (Ultra-Blindado)", sub_v4_guarded, 2, False),
    ]

    print("\n" + "=" * 145)
    print("3. AUDITORÍA FORENSE DEL DRAWDOWN EN IBKR: CERRADO vs MARK-TO-MARKET (CIERRE DÍA) vs PEOR MÍNIMO INTRADÍA vs MONTE CARLO P95")
    print("=" * 145)

    out_rows = []
    for label, sdf, max_sec, dyn_sz in audit_configs:
        res = run_ibkr_portfolio_with_full_mtm_audit(
            sdf, ticker_day_prices, initial_capital=25000.0,
            max_slots=5, max_per_sector=max_sec, dynamic_sizing=dyn_sz
        )
        out_rows.append({
            "Configuración_IBKR": label,
            "Trades": res["trades_executed"],
            "WR %": round(res["win_rate_pct"], 1),
            "PF": round(res["profit_factor"], 2),
            "Retorno_11m": f"+{res['total_return_pct']:.1f}%",
            "DD_Cerrado": f"{res['closed_max_dd_pct']:.2f}%",
            "DD_MtM_CierreDía": f"{res['mtm_eod_max_dd_pct']:.2f}%",
            "DD_MtM_PeorLow": f"{res['mtm_intraday_low_max_dd_pct']:.2f}%",
            "MC_P50_DD": f"{res['mc_p50_dd_pct']:.2f}%",
            "MC_P95_DD": f"{res['mc_p95_dd_pct']:.2f}%",
            "Sharpe_MtM": round(res["sharpe_mtm"], 2),
        })

    print(pd.DataFrame(out_rows).to_string(index=False))
    print("=" * 145)
    print(f"Tiempo total: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
