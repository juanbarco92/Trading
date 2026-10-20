"""
Palanca #3: Autopsia de los Perdedores que Nunca Llegan al +45% ('Momento MeLi'),
Optimización del Target del Modelo 1A y Confirmación de Giro Sectorial / Microestructural en IBKR.

Hipótesis de Microestructura en Tiempo Real (100% Causal al Cierre de la Vela `i0`):
¿Por qué ~22%-25% de las entradas de alta convicción (incluso de Arquetipo 1.1 Macro/Sectorial) siguen cayendo
después de comprar en `Open[t+1]` y nunca llegan siquiera al +45% de recuperación?
1. `sector_ret_2h` y `spy_ret_2h` (¿Ya frenó la hemorragia del sector/mercado en las últimas 2 horas?):
   - Sabemos que comprar caídas de 3 días por arrastre sectorial (`sector_z_3d <= -1.5`) es nuestra mayor ventaja.
   - PERO si en las últimas 2 horas el sector y el SPY siguen desplomándose en caída libre, estamos comprando
     en el día 3 de una caída de 5 días! Si en cambio el sector ya frenó o giró en las últimas 2 horas, entramos en el piso.
2. `higher_low_or_high_break_1h` (Confirmación de Ruptura de Escalera Bajista):
   - En vez de entrar en una vela que sigue haciendo nuevos mínimos y máximos decrecientes cada hora ("Escalera Sangrante"),
     medimos si la vela `i0` ya dejó de perforar el piso (`low[i0] >= low[i0-1]`) o rompió el máximo de la hora previa (`high[i0] > high[i0-1]`).
3. `dist_from_7h_low_atr` (Colchón sobre el mínimo del día):
   - Cuántos ATRs por encima del mínimo de las últimas 7 horas cierra la vela `i0`.
4. Comparación del Target del Modelo 1A:
   - Target Compuesto Actual (`target_1a_strict`) vs Target First-Passage al Momento MeLi (`reached_meli_fp`)
     vs Ensamble Dual 1A (`Prob(Target_1A) * Prob(Llegar_a_MeLi_Sin_SL)`).
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

from scripts.backtest_portfolio_motor1_dual_ml import run_portfolio_simulation
from scripts.upgrade_model_1b_first_passage_and_perception import simulate_calibrated_1b_ibkr_exit


def run_tiered_ibkr_portfolio_exact_sharpe(
    signals_df: pd.DataFrame,
    initial_capital: float = 25000.0,
    max_total_slots: int = 5,
    sniper_reserved_slots: int = 1,
    sniper_prob_thresh: float = 0.605,
):
    """
    Simulador de Portafolio IBKR con Cupos Escalonados y cálculo exacto de Sharpe diario
    (rellenando todos los días hábiles del calendario igual que `run_portfolio_simulation`).
    """
    df_sorted = signals_df.sort_values(["entry_ts", "prob_1a"], ascending=[True, False]).reset_index(drop=True)
    if df_sorted.empty:
        return {"trades_executed": 0, "win_rate_pct": 0.0, "avg_trade_ret_pct": 0.0, "profit_factor": 0.0,
                "final_capital": initial_capital, "total_return_pct": 0.0, "max_dd_pct": 0.0, "sharpe": 0.0}

    all_days = pd.date_range(
        df_sorted["entry_ts"].min().floor("D"),
        df_sorted["exit_ts"].max().ceil("D"),
        freq="B",
        tz="UTC",
    )

    cash = initial_capital
    open_positions = []
    closed_trades = []
    daily_equity_records = []

    sig_idx = 0
    n_sigs = len(df_sorted)

    for d_ts in all_days:
        day_end = d_ts + pd.Timedelta(hours=23, minutes=59)

        # Procesar todas las señales e hitos de salida que ocurren hasta `day_end`
        while sig_idx < n_sigs and pd.Timestamp(df_sorted.loc[sig_idx, "entry_ts"]) <= day_end:
            row = df_sorted.loc[sig_idx]
            sig_idx += 1
            cur_ts = pd.Timestamp(row["entry_ts"])

            # Cerrar posiciones terminadas hasta `cur_ts`
            still_open = []
            for pos in open_positions:
                if pos["exit_ts"] <= cur_ts:
                    cash += pos["returned_cash"]
                    closed_trades.append(pos)
                else:
                    still_open.append(pos)
            open_positions = still_open

            is_sniper = float(row["prob_1a"]) >= sniper_prob_thresh
            allowed_slots = max_total_slots if is_sniper else (max_total_slots - sniper_reserved_slots)

            if len(open_positions) >= allowed_slots:
                continue
            if any(p["ticker"] == row["ticker"] for p in open_positions):
                continue

            slots_free = max_total_slots - len(open_positions)
            if slots_free <= 0:
                continue
            total_eq_now = cash + sum(p["invested"] for p in open_positions)
            alloc = min(cash, total_eq_now / max_total_slots)
            if alloc < 500:
                continue

            cash -= alloc
            ret_net = float(row["net_ret_pct"]) / 100.0
            returned = alloc * (1.0 + ret_net)

            open_positions.append({
                "ticker": row["ticker"],
                "entry_ts": cur_ts,
                "exit_ts": pd.Timestamp(row["exit_ts"]),
                "invested": alloc,
                "returned_cash": returned,
                "net_ret_pct": float(row["net_ret_pct"]),
                "prob_1a": float(row["prob_1a"]),
            })

        # Cerrar posiciones que vencieron antes de `day_end`
        still_open = []
        for pos in open_positions:
            if pos["exit_ts"] <= day_end:
                cash += pos["returned_cash"]
                closed_trades.append(pos)
            else:
                still_open.append(pos)
        open_positions = still_open

        # Registrar equity al cierre del día hábil `d_ts`
        eq_eod = cash + sum(p["invested"] for p in open_positions)
        daily_equity_records.append({"day": d_ts, "equity": eq_eod})

    for pos in sorted(open_positions, key=lambda x: x["exit_ts"]):
        cash += pos["returned_cash"]
        closed_trades.append(pos)

    tdf = pd.DataFrame(closed_trades)
    wins = tdf[tdf["net_ret_pct"] > 0]["net_ret_pct"].sum()
    losses = abs(tdf[tdf["net_ret_pct"] <= 0]["net_ret_pct"].sum())
    pf = wins / losses if losses > 0 else 9.99

    eq_df = pd.DataFrame(daily_equity_records)
    eq_df["peak"] = eq_df["equity"].cummax()
    eq_df["dd"] = (eq_df["equity"] - eq_df["peak"]) / eq_df["peak"] * 100.0
    max_dd = float(eq_df["dd"].min())

    d_rets = eq_df["equity"].pct_change().dropna()
    sharpe = float((d_rets.mean() / d_rets.std()) * np.sqrt(252)) if len(d_rets) > 10 and d_rets.std() > 0 else 0.0

    return {
        "trades_executed": len(tdf),
        "win_rate_pct": float((tdf["net_ret_pct"] > 0).mean() * 100.0),
        "avg_trade_ret_pct": float(tdf["net_ret_pct"].mean()),
        "profit_factor": float(pf),
        "final_capital": float(cash),
        "total_return_pct": float((cash - initial_capital) / initial_capital * 100.0),
        "max_dd_pct": max_dd,
        "sharpe": sharpe,
        "trades_df": tdf,
    }


def main():
    t0 = time.time()
    cache_1a_path = PROJECT_ROOT / "data" / "processed" / "strict_prod_df_1a.parquet"
    cache_meli_path = PROJECT_ROOT / "data" / "processed" / "strict_prod_df_meli.parquet"
    hourly_path = PROJECT_ROOT / "data" / "processed" / "sp500_multisector_1hour.parquet"
    macro_path = PROJECT_ROOT / "data" / "processed" / "macro_perception_1hour.parquet"

    print("[1/4] Cargando datos y calculando variables de Freno Sectorial y Ruptura de Escalera Bajista...")
    df_1a = pd.read_parquet(cache_1a_path)
    df_meli = pd.read_parquet(cache_meli_path)

    df_raw = pd.read_parquet(hourly_path)
    df_raw["timestamp"] = pd.to_datetime(df_raw["timestamp"], utc=True)
    df_raw["hour_key"] = df_raw["timestamp"].dt.floor("h")

    macro_df = pd.read_parquet(macro_path)
    macro_df["hour_key"] = pd.to_datetime(macro_df["hour_key"], utc=True)
    macro_df = macro_df.set_index("hour_key")

    for mcol in ["vix", "vix_ret_3d", "vix_slope_3h", "tnx_ret_3d"]:
        df_raw[mcol] = df_raw["hour_key"].map(macro_df[mcol]).ffill().fillna(0.0)

    spy_df = df_raw[df_raw["ticker"] == "SPY"].sort_values("timestamp").drop_duplicates("hour_key").set_index("hour_key")
    spy_ret_60d_map = spy_df["close"].pct_change(420).fillna(0.0)
    spy_ret_2h_map = spy_df["close"].pct_change(2).fillna(0.0)
    spy_ret_3h_map = spy_df["close"].pct_change(3).fillna(0.0)

    # Calcular el retorno de las últimas 2 horas y 3 horas de cada SECTOR (`sector_ret_2h`, `sector_ret_3h`)
    stocks_only = df_raw[~df_raw["ticker"].isin(["SPY", "QQQ"])].copy()
    stocks_only["ret_1h"] = stocks_only.groupby("ticker")["close"].pct_change(1)
    sec_h = (
        stocks_only.groupby(["sector", "hour_key"])["ret_1h"]
        .mean()
        .reset_index()
        .sort_values(["sector", "hour_key"])
        .reset_index(drop=True)
    )
    sec_h["sector_idx"] = sec_h.groupby("sector")["ret_1h"].transform(lambda s: (1.0 + s.fillna(0.0)).cumprod())
    sec_h["sector_ret_2h"] = sec_h.groupby("sector")["sector_idx"].pct_change(2).fillna(0.0)
    sec_h["sector_ret_3h"] = sec_h.groupby("sector")["sector_idx"].pct_change(3).fillna(0.0)
    sec_2h_map = sec_h.set_index(["sector", "hour_key"])["sector_ret_2h"].to_dict()
    sec_3h_map = sec_h.set_index(["sector", "hour_key"])["sector_ret_3h"].to_dict()

    ticker_dfs = {
        tk: gdf.sort_values("timestamp").reset_index(drop=True)
        for tk, gdf in stocks_only.groupby("ticker")
    }

    # Calcular variables de Palanca #2 + Palanca #3 (Anti-Cuchillo / Confirmación de Giro)
    rel_str_60d_list, atr_norm_drop_3d_list, floor_defense_3h_list = [], [], []
    bounce_thrust_2h_list, green_vol_ratio_5d_list, elastic_potential_list = [], [], []

    spy_ret_2h_list, sector_ret_2h_list, sector_ret_3h_list = [], [], []
    lower_highs_6h_list, higher_low_confirm_list, dist_7h_low_atr_list = [], [], []
    falling_knife_flag_list = []

    for _, row in df_1a.iterrows():
        tk = row["ticker"]
        sec = row["sector"]
        i0 = int(row["bar_idx"])
        tdf = ticker_dfs[tk]
        h_key = pd.Timestamp(row["timestamp"]).floor("h")

        spy_r60 = float(spy_ret_60d_map.get(h_key, 0.0))
        spy_r2h = float(spy_ret_2h_map.get(h_key, 0.0))
        sec_r2h = float(sec_2h_map.get((sec, h_key), 0.0))
        sec_r3h = float(sec_3h_map.get((sec, h_key), 0.0))

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

        # Nuevas Variables Palanca #3 (Anti-Cuchillo / Freno de Escalera Bajista)
        win7 = tdf.iloc[max(0, i0 - 6) : i0 + 1]
        h7 = win7["high"].to_numpy()
        l7 = win7["low"].to_numpy()
        c7 = win7["close"].to_numpy()

        # ¿Cuántas de las últimas 6 horas hicieron un máximo inferior (`high[t] < high[t-1]`)?
        lh_count = float(np.sum(np.diff(h7) < 0)) / max(len(h7) - 1, 1)

        # ¿La vela actual dejó de hacer un nuevo mínimo (`low[i0] >= low[i0-1]`) o rompió el máximo anterior (`high[i0] > high[i0-1]`)?
        if len(h7) >= 2:
            hl_confirm = 1.0 if (l7[-1] >= l7[-2] * 0.999 or h7[-1] > h7[-2]) else 0.0
        else:
            hl_confirm = 1.0

        # Distancia desde el mínimo de las últimas 7 horas en unidades de ATR diario
        min_l7 = float(np.min(l7))
        dist_7h_atr = ((c7[-1] - min_l7) / (c7[-1] + 1e-9)) / atr_d

        # Bandera de "Cuchillo Sectorial Aún en Caída Libre" (`falling_knife_flag`):
        # El sector sigue desplomándose fuerte en las últimas 2-3 horas (`sec_r2h <= -0.012`) Y la acción sigue en escalera bajista sin giro
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

    # Target puro de Primera Etapa (`reached_meli_45`): ¿Llega al +45% de recuperación antes de tocar el Stop Loss diario?
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

    test_start_ts = pd.Timestamp("2025-11-17", tz="UTC")
    train_end_ts = test_start_ts - pd.Timedelta(days=22)

    tr_1a = df_1a[df_1a["timestamp"] <= train_end_ts].copy().reset_index(drop=True)
    te_1a = df_1a[df_1a["timestamp"] >= test_start_ts].copy().reset_index(drop=True)
    target_col = "target_1a_strict" if "target_1a_strict" in tr_1a.columns else "target_1a"

    # 1. Modelo 1A de Palanca #2 (`feats_lever2`)
    clf_l2 = lgb.LGBMClassifier(
        n_estimators=165, learning_rate=0.03, max_depth=4, num_leaves=12,
        min_child_samples=45, subsample=0.8, colsample_bytree=0.7,
        reg_alpha=2.0, reg_lambda=8.0, random_state=42, verbose=-1
    )
    clf_l2.fit(tr_1a[feats_lever2].fillna(0.0), tr_1a[target_col].values)
    te_1a["prob_1a_l2"] = clf_l2.predict_proba(te_1a[feats_lever2].fillna(0.0))[:, 1]

    # Out-of-Fold en Train para hacer la Autopsia de los Perdedores de Alta Convicción SIN FUGA!
    kf = KFold(n_splits=5, shuffle=False)
    oof_l2 = np.zeros(len(tr_1a))
    for idx_tr, idx_val in kf.split(tr_1a):
        m_f = lgb.LGBMClassifier(
            n_estimators=165, learning_rate=0.03, max_depth=4, num_leaves=12,
            min_child_samples=45, subsample=0.8, colsample_bytree=0.7,
            reg_alpha=2.0, reg_lambda=8.0, random_state=42, verbose=-1
        )
        m_f.fit(tr_1a.iloc[idx_tr][feats_lever2].fillna(0.0), tr_1a.iloc[idx_tr][target_col].values)
        oof_l2[idx_val] = m_f.predict_proba(tr_1a.iloc[idx_val][feats_lever2].fillna(0.0))[:, 1]
    tr_1a["prob_1a_l2"] = oof_l2

    # Autopsia en Train OOF de las señales de alta convicción (`prob_1a_l2 >= 0.575`) que NO llegan al Momento MeLi vs las que SÍ llegan
    hi_train = tr_1a[tr_1a["prob_1a_l2"] >= 0.575].copy()
    hi_meli_ok = hi_train[hi_train["reached_meli_target"] == 1]
    hi_meli_fail = hi_train[hi_train["reached_meli_target"] == 0]

    print("\n" + "=" * 124)
    print(f"1. AUTOPSIA EN TRAIN OOF DE LAS SEÑALES DE ALTA CONVICCIÓN (`prob_1a >= 0.575`: {len(hi_train)} casos):")
    print(f"   ¿Qué diferencia a las {len(hi_meli_ok)} que SÍ llegan al +45% (MeLi) de las {len(hi_meli_fail)} que FALLAN y tocan Stop Loss?")
    print("=" * 124)
    l3_diag = [
        ("sector_ret_2h", "Retorno del Sector en las últimas 2 horas"),
        ("spy_ret_2h", "Retorno del SPY en las últimas 2 horas"),
        ("dist_from_7h_low_atr", "Colchón sobre el mínimo del día en unidades ATR"),
        ("higher_low_confirm_1h", "% que ya dejó de hacer mínimos menores (Giro 1H)"),
        ("lower_highs_streak_6h", "Proporción de últimas 6H en escalera bajista"),
        ("tnx_ret_3d", "Cambio de 3 días en Tasa de 10 Años (^TNX)"),
        ("vix_slope_3h", "Aceleración del ^VIX en las últimas 3 horas"),
    ]
    d_rows = []
    for col, desc in l3_diag:
        m_ok = float(hi_meli_ok[col].mean())
        m_fl = float(hi_meli_fail[col].mean())
        d_rows.append({
            "Variable_Anti_Cuchillo": col,
            "Descripción": desc,
            "Media_Llegan_a_45%": round(m_ok, 4),
            "Media_Fallan_StopLoss": round(m_fl, 4),
        })
    print(pd.DataFrame(d_rows).to_string(index=False))

    # 2. Entrenar Modelo 1A de Palanca #3:
    #    a) Cabeza 1 (`clf_l3_main`): predice `target_1a_strict` con `feats_lever3`
    #    b) Cabeza 2 Guardián (`clf_l3_meli`): predice `reached_meli_target` (¿llegará al +45% sin morir en el intento?)
    clf_l3_main = lgb.LGBMClassifier(
        n_estimators=165, learning_rate=0.03, max_depth=4, num_leaves=12,
        min_child_samples=45, subsample=0.8, colsample_bytree=0.7,
        reg_alpha=2.0, reg_lambda=8.0, random_state=42, verbose=-1
    )
    clf_l3_main.fit(tr_1a[feats_lever3].fillna(0.0), tr_1a[target_col].values)
    p_main_te = clf_l3_main.predict_proba(te_1a[feats_lever3].fillna(0.0))[:, 1]

    clf_l3_meli = lgb.LGBMClassifier(
        n_estimators=140, learning_rate=0.03, max_depth=3, num_leaves=8,
        min_child_samples=45, subsample=0.8, colsample_bytree=0.7,
        reg_alpha=2.0, reg_lambda=8.0, random_state=101, verbose=-1
    )
    clf_l3_meli.fit(tr_1a[feats_lever3].fillna(0.0), tr_1a["reached_meli_target"].values)
    p_meli_te = clf_l3_meli.predict_proba(te_1a[feats_lever3].fillna(0.0))[:, 1]
    p_meli_tr = clf_l3_meli.predict_proba(tr_1a[feats_lever3].fillna(0.0))[:, 1]

    te_1a["prob_1a_l3"] = p_main_te
    te_1a["prob_reach_meli"] = p_meli_te
    # Score conjunto calibrado: combina `prob_1a_l3` con la certeza de llegar al Momento MeLi (`prob_reach_meli`)
    te_1a["prob_1a_dual_head"] = 0.75 * p_main_te + 0.25 * (p_meli_te - float(np.mean(p_meli_tr)) + 0.50)

    auc_l2 = roc_auc_score(te_1a[target_col].values, te_1a["prob_1a_l2"].values)
    auc_l3 = roc_auc_score(te_1a[target_col].values, te_1a["prob_1a_l3"].values)
    auc_dual = roc_auc_score(te_1a[target_col].values, te_1a["prob_1a_dual_head"].values)
    auc_meli_head = roc_auc_score(te_1a["reached_meli_target"].values, te_1a["prob_reach_meli"].values)

    print("\n" + "=" * 124)
    print(f"2. PODER PREDICTIVO FUERA DE MUESTRA (ROC-AUC OOS):")
    print(f"   * Modelo 1A Palanca #2:              {auc_l2:.4f}")
    print(f"   * Modelo 1A Palanca #3 (Freno 2H):   {auc_l3:.4f}")
    print(f"   * Cabeza Guardián (Llegar al +45%):  {auc_meli_head:.4f}")
    print(f"   * Ensamble Dual-Head (1A + Guardián):{auc_dual:.4f}")
    print("=" * 124)

    # Entrenar Modelo 1B (`ROC-AUC = 0.6324`) y simular salidas estrictas en IBKR
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

    # Preparar las versiones para evaluar el impacto acumulativo de Palanca #1 -> Palanca #2 -> Palanca #3
    df_l2 = df_sim.copy()
    df_l2["prob_1a"] = df_l2["prob_1a_l2"]

    df_l3 = df_sim.copy()
    df_l3["prob_1a"] = df_l3["prob_1a_l3"]
    # Filtro Anti-Cuchillo de Palanca #3: No comprar si el sector sigue en caída libre en las últimas 2H y no hay giro (`falling_knife_flag == 0`)
    # y exigir que la Cabeza Guardián confirme probabilidad de llegar al Momento MeLi (`prob_reach_meli >= 0.62`)
    df_l3_guarded = df_l3[(df_l3["falling_knife_flag"] == 0) & (df_l3["prob_reach_meli"] >= 0.62)].copy()

    df_dual = df_sim.copy()
    df_dual["prob_1a"] = df_dual["prob_1a_dual_head"]
    df_dual_guarded = df_dual[(df_dual["falling_knife_flag"] == 0) & (df_dual["prob_reach_meli"] >= 0.62)].copy()

    print("\n" + "=" * 132)
    print("3. EVOLUCIÓN ACUMULADA EN PRODUCCIÓN ESTRICTA IBKR ($25,000, 10 bps, Cero Fugas): Palanca #1 -> Palanca #2 -> Palanca #3")
    print("=" * 132)

    cases = [
        ("1. PALANCA #1 (Nuevo 1B AUC 0.6324, 1A Base >= 0.59, 5 Slots)",
         run_portfolio_simulation(df_sim[df_sim["prob_1a_l2"] >= 0.99], 25000.0, 5), 0),  # placeholder replaced below
    ]

    # Construir tabla comparativa exacta
    df_p1 = df_sim.copy()
    # Re-evaluar 1A base original para fila 1
    eval_list = [
        ("1. Fin Palanca #2: Nuevo 1A (Corte >= 0.59, 5 Slots)",
         df_l2[df_l2["prob_1a"] >= 0.59], 5, False),
        ("2. Fin Palanca #2: Nuevo 1A (Corte >= 0.59 o [>=0.575 & Macro/Sector], 5 Slots)",
         df_l2[(df_l2["prob_1a"] >= 0.59) | ((df_l2["prob_1a"] >= 0.575) & (df_l2["shock_type_code"].isin([3, 4])))], 5, False),
        ("3. PALANCA #3 (1A con Freno Sectorial 2H + Guardián MeLi, Corte >= 0.59, 5 Slots)",
         df_l3_guarded[df_l3_guarded["prob_1a"] >= 0.59], 5, False),
        ("4. PALANCA #3 (1A + Guardián MeLi, Corte >= 0.59 o [>=0.575 & Macro/Sector], 5 Slots)",
         df_l3_guarded[(df_l3_guarded["prob_1a"] >= 0.59) | ((df_l3_guarded["prob_1a"] >= 0.575) & (df_l3_guarded["shock_type_code"].isin([3, 4])))], 5, False),
        ("5. PALANCA #3 ENSAMBLE DUAL-HEAD (Corte >= 0.585 + Guardián MeLi, 5 Slots)",
         df_dual_guarded[(df_dual_guarded["prob_1a"] >= 0.59) | ((df_dual_guarded["prob_1a"] >= 0.575) & (df_dual_guarded["shock_type_code"].isin([3, 4])))], 5, False),
        ("6. PALANCA #3 + 6 CUPOS ESCALONADOS (4 Estándar + 2 Francotirador >= 0.60, Sharpe Exacto)",
         df_l3_guarded[(df_l3_guarded["prob_1a"] >= 0.59) | ((df_l3_guarded["prob_1a"] >= 0.575) & (df_l3_guarded["shock_type_code"].isin([3, 4])))], 6, True),
    ]

    t_rows = []
    for label, sub_df, slots, use_tiered in eval_list:
        if use_tiered:
            res = run_tiered_ibkr_portfolio_exact_sharpe(sub_df, 25000.0, max_total_slots=slots, sniper_reserved_slots=2, sniper_prob_thresh=0.60)
        else:
            res = run_portfolio_simulation(sub_df, 25000.0, max_slots=slots)
        # Calcular cuántos de los candidatos tenían `reached_meli_target == 1`
        meli_hit_rate = sub_df["reached_meli_target"].mean() * 100.0 if len(sub_df) > 0 else 0.0
        t_rows.append({
            "Configuración_Auditada_IBKR": label,
            "Señales": len(sub_df),
            "%_Llega_MeLi": f"{meli_hit_rate:.1f}%",
            "Trades": res["trades_executed"],
            "WR %": round(res["win_rate_pct"], 1),
            "Ret_Medio %": round(res["avg_trade_ret_pct"], 2),
            "PF": round(res["profit_factor"], 2),
            "Capital_Final": f"${res['final_capital']:,.0f}",
            "Retorno_11m": f"+{res['total_return_pct']:.1f}%",
            "Max_DD %": f"{res['max_dd_pct']:.2f}%",
            "Sharpe_Real": round(res["sharpe"], 2),
        })
    print(pd.DataFrame(t_rows).to_string(index=False))
    print("=" * 132)
    print(f"Tiempo total: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
