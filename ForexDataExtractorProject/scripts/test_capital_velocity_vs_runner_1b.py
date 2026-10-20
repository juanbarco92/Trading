"""
Prueba de la Hipótesis Microestructural del 'Momento MeLi' (Palanca #1 en IBKR):

Hallazgo de `inspect_217_premature_exits.py`:
1. Cuando una acción llega al +45%-50% de recuperación ('Momento MeLi'), ya acumula una ganancia media de +5.0%.
2. En el 37.9% de los casos que NO son grandes corredores (se estancan y al día 15 terminan en -1.05%),
   nuestro código antiguo NO vendía en el +50%: ponía un trailing stop más abajo (+35%-38% de recuperación),
   devolviendo ~1.5% de ganancia ya ganada y ocupando el cupo (slot) días extra.
3. En el 62.1% de los casos que SÍ son grandes corredores (llegan a +16.36% MFE en ~8 días),
   ANTES de despegar hacen un pullback natural cuya mediana baja hasta el +24.6% de la caída (0.55 a 1.02 ATR).
   Como nuestro código subía el piso inmediatamente al +38% de la caída apenas tocaba el 52%, ¡ese pullback natural
   al +24.6% nos sacaba en el 69% de los ganadores!

Por lo tanto, probamos la separación limpia del trabajo del Modelo 1B en el Momento MeLi:
- Cuando `prob_1b < umbral` (1B detecta que NO hay gasolina para segunda ola, o es Post-Earnings):
  -> COBRO DIRECTO (`Market-on-Open` en `opens[step + 1]` o `Limit` inmediato) al cerrar la vela del 50% de recuperación!
     Asegura el +5.0% íntegro sin devolver ganancia y libera el slot del portafolio en ~2-3 días para rotar el capital.
- Cuando `prob_1b >= umbral` (1B detecta que SÍ hay segunda ola institucional):
  -> Se protege el trade en `Break-Even + 12% de la caída` (por debajo del pullback mediano de 24.6% de la caída,
     dándole 1.15 ATR de respiro al cierre horario) para sobrevivir el retest natural, y recién cuando cruza el +75% de
     recuperación (confirmando la segunda ola) se sube el candado al +55% / +75% de la caída!
"""

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.backtest_portfolio_motor1_dual_ml import run_portfolio_simulation


def simulate_bimodal_1b_ibkr_exit(
    fut_df: pd.DataFrame,
    entry_p: float,
    sl_p: float,
    atr_d: float,
    shock_drop: float,
    shock_type_code: int,
    prob_1b: float,
    meli_trigger_step: int,
    runner_prob_thresh: float = 0.52,
    milestone_rec_target: float = 0.50,
    runner_second_wave_rec: float = 0.80,
):
    """
    Ejecución 100% estricta en Interactive Brokers (sin lookahead intra-barra):
    - Si `prob_1b < runner_prob_thresh` (o Post-Earnings):
      Al cerrar la vela donde alcanza `milestone_rec_target` (ej. 50% de la caída),
      COBRA COMPLETO en la apertura siguiente `opens[step + 1]` (sin devolver un centavo al pullback
      y liberando el slot del portafolio de inmediato).
    - Si `prob_1b >= runner_prob_thresh` (Corredor de 2da Ola):
      1) En el Momento MeLi pone piso de emergencia en `entry_p * (1 + 0.12 * shock_drop)` (sobrevive el pullback a 24.6%)
         y trailing amplio al cierre horario (`peak_close_p * (1 - 1.25 * atr_d)`).
      2) Cuando rompe la 2da ola (`max_rec >= runner_second_wave_rec`, ej. 80% de recuperación),
         sube el piso garantizado a `0.60 * shock_drop` y ciñe el trailing a `0.55 * atr_d` (o cobra en Limit 105%).
    """
    opens = fut_df["open"].to_numpy()
    highs = fut_df["high"].to_numpy()
    lows = fut_df["low"].to_numpy()
    closes = fut_df["close"].to_numpy()
    timestamps = fut_df["timestamp"].to_numpy()
    n_bars = len(fut_df)

    floor_stop_broker = None
    floor_close_eval = None
    tp_limit_broker = None
    cash_out_next_open = False

    peak_p = entry_p
    peak_close_p = entry_p
    exit_p = closes[-1]
    exit_ts = timestamps[-1]
    exit_reason = "TimeStop_15d"

    is_earnings_shock = shock_type_code in (1, 2)

    for step in range(n_bars):
        # 1. DURANTE LA BARRA `step`: Ejecutar órdenes decididas al cierre de `step - 1`
        if step > 0:
            if cash_out_next_open:
                exit_p = opens[step]
                exit_ts = timestamps[step]
                exit_reason = "MeLi_FastCashOut_NextOpen"
                break

            if floor_stop_broker is not None:
                if opens[step] <= floor_stop_broker:
                    exit_p = opens[step]
                    exit_ts = timestamps[step]
                    exit_reason = "IBKR_Stop_GapFill"
                    break
                elif lows[step] <= floor_stop_broker:
                    exit_p = floor_stop_broker
                    exit_ts = timestamps[step]
                    exit_reason = "IBKR_Stop_TouchFill"
                    break

            if tp_limit_broker is not None:
                if opens[step] >= tp_limit_broker:
                    exit_p = opens[step]
                    exit_ts = timestamps[step]
                    exit_reason = "IBKR_TP_GapFill"
                    break
                elif highs[step] >= tp_limit_broker:
                    exit_p = tp_limit_broker
                    exit_ts = timestamps[step]
                    exit_reason = "IBKR_TP_LimitFill"
                    break

            if floor_close_eval is not None and closes[step] <= floor_close_eval:
                exit_p = opens[step + 1] if (step + 1 < n_bars) else closes[step]
                exit_ts = timestamps[step + 1] if (step + 1 < n_bars) else timestamps[step]
                exit_reason = "HourlyClose_Trail_NextOpen"
                break

        # 2. AL CIERRE DEL DÍA (cada 7 barras): Stop Loss Estructural Diario (`sl_p`)
        if (step + 1) % 7 == 0 and closes[step] <= sl_p:
            exit_p = opens[step + 1] if (step + 1 < n_bars) else closes[step]
            exit_ts = timestamps[step + 1] if (step + 1 < n_bars) else timestamps[step]
            exit_reason = "DailyClose_SL_NextOpen"
            break

        # 3. AL CIERRE DE LA BARRA `step`: Actualizar picos y decidir órdenes para `step + 1`
        if highs[step] > peak_p:
            peak_p = highs[step]
        if closes[step] > peak_close_p:
            peak_close_p = closes[step]

        max_rec = ((peak_p - entry_p) / entry_p) / shock_drop
        close_rec = ((closes[step] - entry_p) / entry_p) / shock_drop
        model_1b_is_known = step >= meli_trigger_step

        if model_1b_is_known:
            # RAMA A: Modelo 1B dice "Cobra rápido, no arriesgues al pullback" (`prob_1b < runner_prob_thresh`)
            if is_earnings_shock or prob_1b < runner_prob_thresh:
                if close_rec >= milestone_rec_target:
                    # La vela cerró en o por encima del objetivo (ej. 50% de la caída): vendemos en `opens[step + 1]`!
                    cash_out_next_open = True
                elif max_rec >= milestone_rec_target:
                    # Tocó el 50% intrahora pero cerró un poco más abajo: ponemos piso ceñido en 38% y Limit en 50%
                    tp_limit_broker = entry_p * (1.0 + milestone_rec_target * shock_drop)
                    floor_close_eval = max(
                        floor_close_eval or 0.0,
                        peak_close_p * (1.0 - 0.35 * atr_d),
                        entry_p * (1.0 + 0.36 * shock_drop),
                    )
                    floor_stop_broker = max(floor_stop_broker or 0.0, entry_p * (1.0 + 0.25 * shock_drop))
                else:
                    floor_stop_broker = max(floor_stop_broker or 0.0, entry_p * (1.0 + 0.18 * shock_drop))

            # RAMA B: Modelo 1B dice "¡CORREDOR DE 2DA OLA!" (`prob_1b >= runner_prob_thresh`)
            else:
                # 1) Piso de supervivencia durante el pullback natural (12% de la caída, debajo del 24.6% mediano)
                floor_stop_broker = max(floor_stop_broker or 0.0, entry_p * (1.0 + 0.12 * shock_drop))
                floor_close_eval = max(
                    floor_close_eval or 0.0,
                    peak_close_p * (1.0 - 1.25 * atr_d),
                    entry_p * (1.0 + 0.15 * shock_drop),
                )
                # 2) Si ya superó la 2da ola (`runner_second_wave_rec`, ej. 80% de la caída), blindar ganancia grande!
                if max_rec >= runner_second_wave_rec:
                    floor_close_eval = max(
                        floor_close_eval or 0.0,
                        peak_close_p * (1.0 - 0.55 * atr_d),
                        entry_p * (1.0 + 0.62 * shock_drop),
                    )
                    floor_stop_broker = max(floor_stop_broker or 0.0, entry_p * (1.0 + 0.52 * shock_drop))
                # 3) Take Profit Limit generoso en 110% de recuperación de la caída
                tp_limit_broker = entry_p * (1.0 + 1.10 * shock_drop)

    net_ret_pct = ((exit_p * 0.9995 - entry_p) / entry_p) * 100.0
    return net_ret_pct, pd.Timestamp(exit_ts), exit_reason


def main():
    t0 = time.time()
    cache_1a_path = PROJECT_ROOT / "data" / "processed" / "strict_prod_df_1a.parquet"
    cache_meli_path = PROJECT_ROOT / "data" / "processed" / "strict_prod_df_meli.parquet"
    hourly_path = PROJECT_ROOT / "data" / "processed" / "sp500_multisector_1hour.parquet"

    df_1a = pd.read_parquet(cache_1a_path)
    df_meli = pd.read_parquet(cache_meli_path)
    df_raw = pd.read_parquet(hourly_path)
    df_raw["timestamp"] = pd.to_datetime(df_raw["timestamp"], utc=True)
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
    tr_m = df_meli[df_meli["timestamp"] <= train_end_ts].copy().reset_index(drop=True)
    te_m = df_meli[df_meli["timestamp"] >= test_start_ts].copy().reset_index(drop=True)

    clf_1b = lgb.LGBMClassifier(
        n_estimators=140, learning_rate=0.03, max_depth=3, num_leaves=8,
        min_child_samples=35, subsample=0.8, colsample_bytree=0.8,
        reg_alpha=2.0, reg_lambda=8.0, random_state=42, verbose=-1
    )
    clf_1b.fit(tr_m[feats_1b].fillna(0.0), tr_m["keep_running"].values)
    te_m["prob_1b"] = clf_1b.predict_proba(te_m[feats_1b].fillna(0.0))[:, 1]
    prob_1b_map = {int(r["event_id"]): float(r["prob_1b"]) for _, r in te_m.iterrows()}

    def apply_shield(d: pd.DataFrame, p_min: float):
        m = (
            (d["prob_1a"] >= p_min)
            & (d["weak_volume_flag"] == 0)
            & ~((d["last_earnings_reaction_3d"] <= -0.05) & (d["cmf_15d"] <= -0.05))
            & ~((d["tnx_ret_3d"] >= 0.030) & (d["vix_slope_3h"] >= 0.04))
        )
        return d[m].copy()

    test_cand = apply_shield(te_1a, 0.55)

    # Probar distintos umbrales de `runner_prob_thresh` del Modelo 1B:
    # - 0.99 = Cobrar SIEMPRE rápido en el 50% (Alta Rotación pura)
    # - 0.58 = Solo dejar correr al Top 25% más fuerte según Modelo 1B; el otro 75% cobra en el 50% y libera cupo!
    # - 0.54 = Dejar correr al Top 45% según Modelo 1B; el otro 55% cobra rápido en el 50%
    # - 0.48 = Dejar correr a casi todos (75%) con espacio para el pullback
    configs = [
        ("A. Cobro Rápido en 50% salvo Top 1B (prob_1b >= 0.58 deja correr con respiro)", 0.58, 0.48, 0.80),
        ("B. Cobro Rápido en 50% salvo 1B Fuerte (prob_1b >= 0.54 deja correr con respiro)", 0.54, 0.48, 0.80),
        ("C. Cobro Rápido en 55% salvo 1B Fuerte (prob_1b >= 0.54, target 55%)", 0.54, 0.54, 0.85),
        ("D. Cobro Rápido Universal en el 50% (Alta Rotación de Capital pura)", 0.99, 0.48, 0.80),
        ("E. Dejar Correr Casi Todo con Respiro (prob_1b >= 0.45 deja correr)", 0.45, 0.48, 0.80),
    ]

    rows_059 = []
    rows_061 = []

    for label, p1b_th, m_rec, w2_rec in configs:
        sim_list = []
        for _, row in test_cand.iterrows():
            ev_id = int(row["event_id"])
            tk = row["ticker"]
            i0 = int(row["bar_idx"])
            tdf = ticker_dfs[tk]
            fut_df = tdf.iloc[i0 + 1 : i0 + 106]
            p_1b = prob_1b_map.get(ev_id, 0.50)
            trig_step = int(row["meli_trigger_step"])

            r_net, x_ts, x_rsn = simulate_bimodal_1b_ibkr_exit(
                fut_df, float(row["entry_p"]), float(row["sl_p"]), float(row["atr_d"]),
                float(row["shock_drop"]), int(row["shock_type_code"]), p_1b, trig_step,
                runner_prob_thresh=p1b_th, milestone_rec_target=m_rec, runner_second_wave_rec=w2_rec
            )
            hold_days = (x_ts - pd.Timestamp(row["entry_ts"])).total_seconds() / 86400.0
            d = row.to_dict()
            d.update({"prob_1b": p_1b, "net_ret_pct": r_net, "exit_ts": x_ts, "exit_reason": x_rsn, "hold_days": hold_days})
            sim_list.append(d)

        df_sim = pd.DataFrame(sim_list).sort_values("entry_ts").reset_index(drop=True)

        sub_059 = df_sim[df_sim["prob_1a"] >= 0.59].copy()
        res_059 = run_portfolio_simulation(sub_059, initial_capital=25000.0, max_slots=5)
        rows_059.append({
            "Estrategia_Bimodal_1B_IBKR (prob_1a >= 0.59)": label,
            "Señales": len(sub_059),
            "Trades_Ejecutados": res_059["trades_executed"],
            "Días_Medio_Trade": round(sub_059["hold_days"].mean(), 1),
            "WR %": round(res_059["win_rate_pct"], 1),
            "Ret_Medio %": round(res_059["avg_trade_ret_pct"], 2),
            "PF": round(res_059["profit_factor"], 2),
            "Capital_Final": f"${res_059['final_capital']:,.0f}",
            "Retorno_11m": f"+{res_059['total_return_pct']:.1f}%",
            "Max_DD %": f"{res_059['max_dd_pct']:.2f}%",
            "Sharpe": round(res_059["sharpe"], 2),
        })

        sub_061 = df_sim[df_sim["prob_1a"] >= 0.61].copy()
        res_061 = run_portfolio_simulation(sub_061, initial_capital=25000.0, max_slots=5)
        rows_061.append({
            "Estrategia_Bimodal_1B_IBKR (Francotirador >= 0.61)": label,
            "Señales": len(sub_061),
            "Trades_Ejecutados": res_061["trades_executed"],
            "Días_Medio_Trade": round(sub_061["hold_days"].mean(), 1),
            "WR %": round(res_061["win_rate_pct"], 1),
            "Ret_Medio %": round(res_061["avg_trade_ret_pct"], 2),
            "PF": round(res_061["profit_factor"], 2),
            "Capital_Final": f"${res_061['final_capital']:,.0f}",
            "Retorno_11m": f"+{res_061['total_return_pct']:.1f}%",
            "Max_DD %": f"{res_061['max_dd_pct']:.2f}%",
            "Sharpe": round(res_061["sharpe"], 2),
        })

    print("\n" + "=" * 130)
    print("1. RESULTADOS PORTAFOLIO IBKR ($25k, 5 Slots, 10 bps, `prob_1a >= 0.59`): Velocidad de Capital vs Segunda Ola 1B")
    print("=" * 130)
    print(pd.DataFrame(rows_059).to_string(index=False))

    print("\n" + "=" * 130)
    print("2. RESULTADOS PORTAFOLIO FRANCOTIRADOR IBKR ($25k, 5 Slots, 10 bps, `prob_1a >= 0.61`):")
    print("=" * 130)
    print(pd.DataFrame(rows_061).to_string(index=False))
    print("=" * 130)


if __name__ == "__main__":
    main()
