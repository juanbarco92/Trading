"""
Reparación Estructural del Modelo 1B (Palanca #1 en Producción Estricta IBKR):

Dos hallazgos críticos al inspeccionar por qué el Modelo 1B y el Trailing Stop dejaban el 68% de la ganancia en la mesa:
1. BUG EN EL TARGET DE ENTRENAMIENTO DE 1B (`keep_running`):
   - Antes se definía como: `np.max(rem_highs) >= +85% AND np.min(rem_closes) > +15%` mirando TODAS las barras hasta el día 15.
   - Si una acción llegaba al +100% de recuperación el Día 4 (¡éxito total!) y el Día 13 bajaba al +10%,
     `np.min(rem_closes)` la marcaba como FRACASO (`keep_running = 0`)!
   - Eso hundió la tasa base de `keep_running` a ~22% y comprimió todas las probabilidades `prob_1b` por debajo de 0.45
     (por eso ningún trade superaba `prob_1b >= 0.54`).
   - SOLUCIÓN: Target First-Passage Real ("¿Toca primero el +85% de recuperación ANTES de caer a Break-Even?").

2. VARIABLES DE PERCEPCIÓN DE MERCADO EN EL 'MOMENTO MELI' PARA EL MODELO 1B:
   - Cuando la acción toca el +45% de recuperación, agregamos qué pasó desde que abrimos la operación hasta ese instante con:
     * `vix_cooling_since_entry`: ¿Se desinfló el miedo (`^VIX`) mientras la acción rebotaba, o el VIX sigue subiendo (rebote falso)?
     * `cmf_15d_at_50` y `down_up_vol_ratio_at_50`: Flujo de dinero institucional acumulado hasta el Momento MeLi.
     * `tnx_ret_3d_at_50`: Presión de tasas a 10 años en el momento de decidir si mantener o cobrar.
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


def simulate_calibrated_1b_ibkr_exit(
    fut_df: pd.DataFrame,
    entry_p: float,
    sl_p: float,
    atr_d: float,
    shock_drop: float,
    shock_type_code: int,
    prob_1b: float,
    meli_trigger_step: int,
    q25_1b: float,
    q75_1b: float,
    mode: str = "calibrated_hourly_close",
):
    """
    Misma mecánica limpia sin tope (sin Limit TP que corte a los grandes ganadores),
    alimentada por el Nuevo Modelo 1B (`ROC-AUC = 0.6324`) calibrado con cuantiles de Train:
    - Cola Inferior (`prob_1b < q25_1b` o Post-Earnings, éxito < 32%): Candado rápido `0.35 * atr_d` y piso `0.42 * shock_drop`.
    - Zona Media (`q25_1b <= prob_1b < q75_1b`): Candado estándar `0.65 * atr_d` y piso `0.38 * shock_drop`.
    - Top 25% Súper-Corredor (`prob_1b >= q75_1b` y Macro/Sector, éxito > 75%):
      Respiro amplio `1.05 * atr_d` (sin techo Limit TP que corte las subidas de +15% a +20%) y piso `0.38 * shock_drop`.
    """
    opens = fut_df["open"].to_numpy()
    highs = fut_df["high"].to_numpy()
    lows = fut_df["low"].to_numpy()
    closes = fut_df["close"].to_numpy()
    timestamps = fut_df["timestamp"].to_numpy()
    n_bars = len(fut_df)

    floor_p = None
    peak_p = entry_p
    exit_p = closes[-1]
    exit_ts = timestamps[-1]
    exit_reason = "TimeStop_15d"

    is_earnings_shock = shock_type_code in (1, 2)
    is_macro_or_sector = shock_type_code in (3, 4)

    for step in range(n_bars):
        # 1. DURANTE LA BARRA `step`: Evaluar `floor_p` vigente desde `step - 1`
        if floor_p is not None and step > 0:
            if mode == "calibrated_intraday_stop":
                if opens[step] <= floor_p:
                    exit_p = opens[step]
                    exit_ts = timestamps[step]
                    exit_reason = "StopOrder_GapFill"
                    break
                elif lows[step] <= floor_p:
                    exit_p = floor_p
                    exit_ts = timestamps[step]
                    exit_reason = "StopOrder_TouchFill"
                    break
            else:
                if closes[step] <= floor_p:
                    exit_p = opens[step + 1] if (step + 1 < n_bars) else closes[step]
                    exit_ts = timestamps[step + 1] if (step + 1 < n_bars) else timestamps[step]
                    exit_reason = "CloseTrigger_NextOpenFill"
                    break

        # 2. AL CIERRE DEL DÍA (cada 7 barras): Stop Loss Estructural Diario (`sl_p`)
        if (step + 1) % 7 == 0 and closes[step] <= sl_p:
            exit_p = opens[step + 1] if (step + 1 < n_bars) else closes[step]
            exit_ts = timestamps[step + 1] if (step + 1 < n_bars) else timestamps[step]
            exit_reason = "DailyClose_SL_NextOpen"
            break

        # 3. AL CIERRE DE LA BARRA `step`: Actualizar `peak_p` y `floor_p` para `step + 1`
        if highs[step] > peak_p:
            peak_p = highs[step]
        max_rec = ((peak_p - entry_p) / entry_p) / shock_drop
        model_1b_is_known = step >= meli_trigger_step

        if is_earnings_shock or (model_1b_is_known and prob_1b < q25_1b):
            if max_rec >= 0.52:
                trail_dist = 0.35 * atr_d
                floor_p = max(floor_p or 0.0, peak_p * (1.0 - trail_dist), entry_p * (1.0 + 0.42 * shock_drop))
            elif max_rec >= 0.38:
                floor_p = max(floor_p or 0.0, entry_p * (1.0 + 0.22 * shock_drop))
        elif is_macro_or_sector and model_1b_is_known and prob_1b >= q75_1b:
            if max_rec >= 0.55:
                trail_dist = 1.05 * atr_d
                floor_p = max(floor_p or 0.0, peak_p * (1.0 - trail_dist), entry_p * (1.0 + 0.38 * shock_drop))
            elif max_rec >= 0.42:
                floor_p = max(floor_p or 0.0, entry_p * 1.003)
        else:
            if max_rec >= 0.52:
                trail_dist = 0.65 * atr_d
                floor_p = max(floor_p or 0.0, peak_p * (1.0 - trail_dist), entry_p * (1.0 + 0.38 * shock_drop))
            elif max_rec >= 0.40:
                floor_p = max(floor_p or 0.0, entry_p * 1.003)

    net_ret_pct = ((exit_p * 0.9995 - entry_p) / entry_p) * 100.0
    return net_ret_pct, pd.Timestamp(exit_ts), exit_reason


def main():
    t0 = time.time()
    cache_1a_path = PROJECT_ROOT / "data" / "processed" / "strict_prod_df_1a.parquet"
    cache_meli_path = PROJECT_ROOT / "data" / "processed" / "strict_prod_df_meli.parquet"
    hourly_path = PROJECT_ROOT / "data" / "processed" / "sp500_multisector_1hour.parquet"
    macro_path = PROJECT_ROOT / "data" / "processed" / "macro_perception_1hour.parquet"

    print("[1/3] Cargando caché de señales 1A, Momento MeLi y enriqueciendo 1B con First-Passage Target + Percepción...")
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

    ticker_dfs = {
        tk: gdf.sort_values("timestamp").reset_index(drop=True)
        for tk, gdf in df_raw[~df_raw["ticker"].isin(["SPY", "QQQ"])].groupby("ticker")
    }

    # Calcular para cada evento en `df_meli`:
    # 1) El nuevo Target First-Passage (`keep_running_fp`): ¿Alcanza >= 85% de recuperación ANTES de tocar Break-Even (`closes <= entry_p`)?
    # 2) Las nuevas variables de Percepción en el Momento MeLi (`vix_cooling_since_entry`, `vix_slope_at_50`, `tnx_ret_3d_at_50`, `down_up_vol_at_entry`)
    ev_1a_lookup = df_1a.set_index("event_id")

    fp_targets = []
    vix_cooling_list = []
    vix_slope_50_list = []
    tnx_50_list = []
    down_up_entry_list = []
    sys_pull_entry_list = []
    fund_shield_entry_list = []

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
        vix_cooling = (vix_at_50 - vix_at_entry) / max(vix_at_entry, 10.0)

        vix_cooling_list.append(vix_cooling)
        vix_slope_50_list.append(float(cur_bar["vix_slope_3h"]))
        tnx_50_list.append(float(cur_bar["tnx_ret_3d"]))
        down_up_entry_list.append(float(r1a["down_up_vol_ratio_15d"]))
        sys_pull_entry_list.append(float(r1a["systemic_pull_z_3d"]))
        fund_shield_entry_list.append(float(r1a["fundamental_shield"]))

        # First-Passage Target desde `trig_step + 1` hasta `i0 + 106`
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

    # Comparar Modelo 1B Antiguo vs Modelo 1B Mejorado (First-Passage + Percepción)
    clf_1b_old = lgb.LGBMClassifier(
        n_estimators=140, learning_rate=0.03, max_depth=3, num_leaves=8,
        min_child_samples=35, subsample=0.8, colsample_bytree=0.8,
        reg_alpha=2.0, reg_lambda=8.0, random_state=42, verbose=-1
    )
    clf_1b_old.fit(tr_m[feats_1b_upgraded[:-6]].fillna(0.0), tr_m["keep_running"].values)
    prob_old_te = clf_1b_old.predict_proba(te_m[feats_1b_upgraded[:-6]].fillna(0.0))[:, 1]
    auc_old = roc_auc_score(te_m["keep_running"].values, prob_old_te)

    clf_1b_new = lgb.LGBMClassifier(
        n_estimators=150, learning_rate=0.03, max_depth=3, num_leaves=8,
        min_child_samples=35, subsample=0.8, colsample_bytree=0.8,
        reg_alpha=2.0, reg_lambda=8.0, random_state=42, verbose=-1
    )
    clf_1b_new.fit(tr_m[feats_1b_upgraded].fillna(0.0), tr_m["keep_running_fp"].values)
    prob_new_tr = clf_1b_new.predict_proba(tr_m[feats_1b_upgraded].fillna(0.0))[:, 1]
    prob_new_te = clf_1b_new.predict_proba(te_m[feats_1b_upgraded].fillna(0.0))[:, 1]
    auc_new = roc_auc_score(te_m["keep_running_fp"].values, prob_new_te)

    # Cuantiles estrictamente calculados en TRAIN (`prob_new_tr`)
    q35_1b = float(np.quantile(prob_new_tr, 0.35))
    q50_1b = float(np.quantile(prob_new_tr, 0.50))
    q70_1b = float(np.quantile(prob_new_tr, 0.70))
    q80_1b = float(np.quantile(prob_new_tr, 0.80))

    print("\n" + "=" * 124)
    print("1. REPARACIÓN DEL TARGET Y VARIABLES DEL MODELO 1B (En el 'Momento MeLi'):")
    print("=" * 124)
    print(f"   * Tasa base de `keep_running` ANTIGUA (contaminada con caídas en día 14): {tr_m['keep_running'].mean()*100:.1f}% (Mediana prob_1b = {np.median(prob_old_te):.3f}, Max = {np.max(prob_old_te):.3f})")
    print(f"   * Tasa base de `keep_running_fp` NUEVA (First-Passage +85% antes de BE): {tr_m['keep_running_fp'].mean()*100:.1f}% (Mediana prob_1b = {np.median(prob_new_te):.3f}, P50 = {q50_1b:.3f}, P80 = {q80_1b:.3f})")
    print(f"   * ROC-AUC Fuera de Muestra (Antiguo = {auc_old:.4f} -> Nuevo First-Passage + Percepción = {auc_new:.4f})")

    # Verificar separación por deciles/terciles del Nuevo Modelo 1B en Test OOS
    te_m["prob_1b_new"] = prob_new_te
    print("\n   Separación Real Fuera de Muestra del Nuevo Modelo 1B en las 2,604 señales que llegan al Momento MeLi:")
    for t_name, t_mask in [
        ("Tercil Bajo 1B (Alerta Rebote Frágil / Gato Muerto, < P35)", te_m["prob_1b_new"] < q35_1b),
        ("Tercil Medio 1B (Continuación Normal, P35 a P80)", (te_m["prob_1b_new"] >= q35_1b) & (te_m["prob_1b_new"] < q80_1b)),
        ("Top 20% 1B (Súper-Corredor Institucional de 2da Ola, >= P80)", te_m["prob_1b_new"] >= q80_1b),
    ]:
        sub_t = te_m[t_mask]
        print(f"     - {t_name:60s}: {len(sub_t):4d} casos | Éxito First-Passage al +85% = {sub_t['keep_running_fp'].mean()*100:.1f}%")

    prob_1b_new_map = {int(r["event_id"]): float(r["prob_1b_new"]) for _, r in te_m.iterrows()}
    prob_1b_old_map = {int(r["event_id"]): float(p) for r, p in zip(te_m.to_dict("records"), prob_old_te)}

    def apply_shield(d: pd.DataFrame, p_min: float):
        m = (
            (d["prob_1a"] >= p_min)
            & (d["weak_volume_flag"] == 0)
            & ~((d["last_earnings_reaction_3d"] <= -0.05) & (d["cmf_15d"] <= -0.05))
            & ~((d["tnx_ret_3d"] >= 0.030) & (d["vix_slope_3h"] >= 0.04))
        )
        return d[m].copy()

    test_cand = apply_shield(te_1a, 0.55)

    from scripts.forensic_audit_motor1_production import simulate_strict_production_exit

    rows_base_stop = []
    rows_base_close = []
    rows_cal_stop = []
    rows_cal_close = []

    for _, row in test_cand.iterrows():
        ev_id = int(row["event_id"])
        tk = row["ticker"]
        i0 = int(row["bar_idx"])
        tdf = ticker_dfs[tk]
        fut_df = tdf.iloc[i0 + 1 : i0 + 106]
        trig_step = int(row["meli_trigger_step"])
        p_old = prob_1b_old_map.get(ev_id, 0.30)
        p_new = prob_1b_new_map.get(ev_id, 0.50)

        # 1. Base Antiguo (Stop-Market con prob_1b descalibrada)
        r1, ts1, rsn1 = simulate_strict_production_exit(
            fut_df, float(row["entry_p"]), float(row["sl_p"]), float(row["atr_d"]),
            float(row["shock_drop"]), int(row["shock_type_code"]), p_old, trig_step,
            execution_mode="broker_stop_order_with_gap"
        )
        d1 = row.to_dict()
        d1.update({"prob_1b": p_old, "net_ret_pct": r1, "exit_ts": ts1, "exit_reason": rsn1})
        rows_base_stop.append(d1)

        # 2. Base Antiguo (Cierre Horario con prob_1b descalibrada)
        r2, ts2, rsn2 = simulate_strict_production_exit(
            fut_df, float(row["entry_p"]), float(row["sl_p"]), float(row["atr_d"]),
            float(row["shock_drop"]), int(row["shock_type_code"]), p_old, trig_step,
            execution_mode="close_triggered_next_open"
        )
        d2 = row.to_dict()
        d2.update({"prob_1b": p_old, "net_ret_pct": r2, "exit_ts": ts2, "exit_reason": rsn2})
        rows_base_close.append(d2)

        # 3. Nuevo 1B Calibrado EV+ (Stop-Market en Broker)
        r3, ts3, rsn3 = simulate_calibrated_1b_ibkr_exit(
            fut_df, float(row["entry_p"]), float(row["sl_p"]), float(row["atr_d"]),
            float(row["shock_drop"]), int(row["shock_type_code"]), p_new, trig_step,
            q25_1b=q70_1b, q75_1b=q80_1b, mode="calibrated_intraday_stop"
        )
        d3 = row.to_dict()
        d3.update({"prob_1b": p_new, "net_ret_pct": r3, "exit_ts": ts3, "exit_reason": rsn3})
        rows_cal_stop.append(d3)

        # 4. Nuevo 1B Calibrado EV+ (Cierre Horario -> Open[t+1])
        r4, ts4, rsn4 = simulate_calibrated_1b_ibkr_exit(
            fut_df, float(row["entry_p"]), float(row["sl_p"]), float(row["atr_d"]),
            float(row["shock_drop"]), int(row["shock_type_code"]), p_new, trig_step,
            q25_1b=q70_1b, q75_1b=q80_1b, mode="calibrated_hourly_close"
        )
        d4 = row.to_dict()
        d4.update({"prob_1b": p_new, "net_ret_pct": r4, "exit_ts": ts4, "exit_reason": rsn4})
        rows_cal_close.append(d4)

    df_b_stop = pd.DataFrame(rows_base_stop).sort_values("entry_ts").reset_index(drop=True)
    df_b_close = pd.DataFrame(rows_base_close).sort_values("entry_ts").reset_index(drop=True)
    df_c_stop = pd.DataFrame(rows_cal_stop).sort_values("entry_ts").reset_index(drop=True)
    df_c_close = pd.DataFrame(rows_cal_close).sort_values("entry_ts").reset_index(drop=True)

    print("\n" + "=" * 124)
    print("2. IMPACTO EN PORTAFOLIO ESTRICTO IBKR ($25,000, 5 Slots, 10 bps): 1B Antiguo vs 1B Reparado (First-Passage + Percepción)")
    print("=" * 124)
    comparisons = [
        ("1. Base Antiguo: Stop-Market IBKR (prob_1a >= 0.59, 5 Slots)", df_b_stop[df_b_stop["prob_1a"] >= 0.59], 5),
        ("2. NUEVO 1B CALIBRADO: Stop-Market IBKR (prob_1a >= 0.59, 5 Slots)", df_c_stop[df_c_stop["prob_1a"] >= 0.59], 5),
        ("3. Base Antiguo: Cierre Horario -> Open[t+1] (prob_1a >= 0.59, 5 Slots)", df_b_close[df_b_close["prob_1a"] >= 0.59], 5),
        ("4. NUEVO 1B CALIBRADO: Cierre Horario -> Open[t+1] (prob_1a >= 0.59, 5 Slots)", df_c_close[df_c_close["prob_1a"] >= 0.59], 5),
        ("5. Base Antiguo Francotirador: Cierre Horario (prob_1a >= 0.61, 5 Slots)", df_b_close[df_b_close["prob_1a"] >= 0.61], 5),
        ("6. NUEVO 1B CALIBRADO Francotirador: Cierre Horario (prob_1a >= 0.61, 5 Slots)", df_c_close[df_c_close["prob_1a"] >= 0.61], 5),
    ]
    c_rows = []
    for label, sub_df, slots in comparisons:
        res = run_portfolio_simulation(sub_df, initial_capital=25000.0, max_slots=slots)
        c_rows.append({
            "Configuración_Portafolio_IBKR": label,
            "Señales": len(sub_df),
            "Trades": res["trades_executed"],
            "WR %": round(res["win_rate_pct"], 1),
            "Ret_Medio %": round(res["avg_trade_ret_pct"], 2),
            "PF": round(res["profit_factor"], 2),
            "Capital_Final": f"${res['final_capital']:,.0f}",
            "Retorno_11m": f"+{res['total_return_pct']:.1f}%",
            "Max_DD %": f"{res['max_dd_pct']:.2f}%",
            "Sharpe": round(res["sharpe"], 2),
        })
    print(pd.DataFrame(c_rows).to_string(index=False))
    print("=" * 124)
    print(f"Tiempo total: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
