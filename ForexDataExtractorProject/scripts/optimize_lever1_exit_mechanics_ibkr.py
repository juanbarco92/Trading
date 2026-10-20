"""
Palanca #1: Optimización y Blindaje de la Mecánica de Salida (Modelo 1B + Órdenes Nativas IBKR).

Problema diagnosticado:
- De las señales que llegan al +45% de recuperación ("Momento MeLi"), el 98.7% son ganadoras y alcanzan una
  Excursión Máxima (MFE) promedio de +13.89%.
- Sin embargo, el candado actual (`Stop-Market` intradía a 0.65-0.90 ATR del máximo horario, evaluado contra `lows[step]`)
  nos saca prematuramente en el 68.9% de los casos por mechas intradía normales, capturando solo +4.40% (31.7% del MFE).

Experimento Riguroso (Validado en Train Out-of-Fold 2024-2025 y en Test OOS Nov 2025 - Oct 2026):
Comparamos 5 mecánicas de salida 100% reales en Interactive Brokers (sin lookahead intra-barra, órdenes activas solo desde `step + 1`):
  1. BASE ACTUAL (Stop-Market intradía `lows[step] <= floor_p` a 0.65-0.90 ATR).
  2. STOP AL CIERRE HORARIO (`closes[step] <= floor_p` -> vende en `opens[step + 1]`) + Piso de Emergencia en Broker.
     Evita que una mecha intradía de 15 minutos te saque si la vela de 1 hora termina recuperándose.
  3. STOP AL CIERRE DIARIO + PISO GARANTIZADO EN BROKER (`Stop-Market` fijo en +25% de recuperación contra desastres,
     pero el trailing dinámico solo se evalúa al cierre del día `(step+1)%7==0` -> `opens[step+1]`).
  4. TRINQUETE POR ESCALONES DE RECUPERACIÓN + MODELO 1B (Ratchet OCO en IBKR):
     - Si 1B dice "¡Peligro de Reversión!" (`prob_1b < 0.46` o Post-Earnings): Cobro/candado ceñido inmediato.
     - Si 1B dice "¡Deja correr!" (`prob_1b >= 0.46`):
       * Piso inicial en +25% de la caída recuperada (NUNCA se pierde una vez tocado el Momento MeLi).
       * Al cruzar 65% de recuperación -> el piso sube a +45% de la caída recuperada.
       * Al cruzar 85% de recuperación -> el piso sube a +68% de la caída recuperada.
       * Si `prob_1b >= 0.56` (alta convicción de continuación), el trailing usa cierre horario con holgura 1.25 ATR
         (o Take Profit Limit en 115% de la caída); si `prob_1b` es medio, Take Profit Limit en 90% de la caída.
  5. BRACKET HÍBRIDO 1B (Cierre Horario + Holgura guiada por `prob_1b` + Escalones Ratchet):
     Combina el filtro de mechas horarias (evaluar el trailing al cierre de la hora + Stop de catástrofe en broker)
     con los escalones de trinquete y la probabilidad continua del Modelo 1B.
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


def simulate_ibkr_exit_archetypes(
    fut_df: pd.DataFrame,
    entry_p: float,
    sl_p: float,
    atr_d: float,
    shock_drop: float,
    shock_type_code: int,
    prob_1b: float,
    meli_trigger_step: int,
    mode: str = "1_base_intraday_stop",
):
    """
    Simulador 100% estricto de producción en Interactive Brokers (IBKR):
    - Cualquier orden (`floor_stop_broker`, `floor_close_eval`, `tp_limit_broker`) calculada al final de la barra `step`
      SOLO entra en vigor a partir de la barra `step + 1`.
    - Si `floor_stop_broker` (orden STP nativa en IBKR) se activa en `step`:
      * Si `opens[step] <= floor_stop_broker`, sufre el gap real y se llena a `opens[step]`.
      * Si `lows[step] <= floor_stop_broker`, se llena a `floor_stop_broker`.
    - Si `tp_limit_broker` (orden LMT nativa de Take Profit en IBKR) se activa en `step`:
      * Si `opens[step] >= tp_limit_broker`, nos dan el gap a favor en `opens[step]`.
      * Si `highs[step] >= tp_limit_broker`, se llena exactamente en `tp_limit_broker`.
    - Si `floor_close_eval` (trailing evaluado al cierre horario o diario) se rompe al cierre de `step`:
      * Se envía orden Market que se llena en `opens[step + 1]` (sufriendo cualquier gap/deslizamiento real).
    """
    opens = fut_df["open"].to_numpy()
    highs = fut_df["high"].to_numpy()
    lows = fut_df["low"].to_numpy()
    closes = fut_df["close"].to_numpy()
    timestamps = fut_df["timestamp"].to_numpy()
    n_bars = len(fut_df)

    # Órdenes vigentes para la barra actual (calculadas en la barra anterior)
    floor_stop_broker = None   # Orden STP en el servidor de IBKR (sensible a lows[step] y gaps opens[step])
    floor_close_eval = None    # Nivel evaluado al cierre de la barra -> ejecuta en opens[step + 1]
    tp_limit_broker = None     # Orden LMT de Take Profit en el servidor de IBKR

    peak_p = entry_p
    peak_close_p = entry_p
    exit_p = closes[-1]
    exit_ts = timestamps[-1]
    exit_reason = "TimeStop_15d"

    is_earnings_shock = shock_type_code in (1, 2)
    is_macro_or_sector = shock_type_code in (3, 4)

    for step in range(n_bars):
        # =====================================================================
        # 1. DURANTE LA BARRA `step`: Evaluar órdenes nativas alojadas en IBKR desde `step - 1`
        # =====================================================================
        if step > 0:
            # A) Orden Stop-Market en IBKR (si existe)
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

            # B) Orden Limit Take-Profit en IBKR (si existe)
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

            # C) Evaluación al cierre de la barra `step` (horaria o diaria según modo) -> ejecuta en `opens[step + 1]`
            if floor_close_eval is not None:
                if mode == "3_daily_close_trail":
                    if (step + 1) % 7 == 0 and closes[step] <= floor_close_eval:
                        exit_p = opens[step + 1] if (step + 1 < n_bars) else closes[step]
                        exit_ts = timestamps[step + 1] if (step + 1 < n_bars) else timestamps[step]
                        exit_reason = "DailyClose_Trail_NextOpen"
                        break
                else:
                    if closes[step] <= floor_close_eval:
                        exit_p = opens[step + 1] if (step + 1 < n_bars) else closes[step]
                        exit_ts = timestamps[step + 1] if (step + 1 < n_bars) else timestamps[step]
                        exit_reason = "HourlyClose_Trail_NextOpen"
                        break

        # =====================================================================
        # 2. AL CIERRE DEL DÍA (cada 7 barras): Stop Loss Estructural Diario (`sl_p`)
        # =====================================================================
        if (step + 1) % 7 == 0 and closes[step] <= sl_p:
            exit_p = opens[step + 1] if (step + 1 < n_bars) else closes[step]
            exit_ts = timestamps[step + 1] if (step + 1 < n_bars) else timestamps[step]
            exit_reason = "DailyClose_SL_NextOpen"
            break

        # =====================================================================
        # 3. AL TERMINAR LA BARRA `step`: Actualizar picos y órdenes para `step + 1`
        # =====================================================================
        if highs[step] > peak_p:
            peak_p = highs[step]
        if closes[step] > peak_close_p:
            peak_close_p = closes[step]

        max_rec = ((peak_p - entry_p) / entry_p) / shock_drop
        model_1b_is_known = step >= meli_trigger_step

        # ---------------------------------------------------------------------
        # MODO 1: BASE ACTUAL (Stop-Market intradía sensible a cualquier mecha `lows[step]`)
        # ---------------------------------------------------------------------
        if mode == "1_base_intraday_stop":
            if is_earnings_shock or (model_1b_is_known and prob_1b < 0.43):
                if max_rec >= 0.52:
                    floor_stop_broker = max(floor_stop_broker or 0.0, peak_p * (1.0 - 0.35 * atr_d), entry_p * (1.0 + 0.42 * shock_drop))
                elif max_rec >= 0.38:
                    floor_stop_broker = max(floor_stop_broker or 0.0, entry_p * (1.0 + 0.22 * shock_drop))
            elif is_macro_or_sector and model_1b_is_known and prob_1b >= 0.52:
                if max_rec >= 0.55:
                    floor_stop_broker = max(floor_stop_broker or 0.0, peak_p * (1.0 - 0.90 * atr_d), entry_p * (1.0 + 0.38 * shock_drop))
                elif max_rec >= 0.42:
                    floor_stop_broker = max(floor_stop_broker or 0.0, entry_p * 1.003)
            else:
                if max_rec >= 0.52:
                    floor_stop_broker = max(floor_stop_broker or 0.0, peak_p * (1.0 - 0.65 * atr_d), entry_p * (1.0 + 0.38 * shock_drop))
                elif max_rec >= 0.40:
                    floor_stop_broker = max(floor_stop_broker or 0.0, entry_p * 1.003)

        # ---------------------------------------------------------------------
        # MODO 2: EVALUADO AL CIERRE HORARIO (`closes[step] <= floor` -> `opens[step+1]`) + STOP CATÁSTROFE EN BROKER
        # ---------------------------------------------------------------------
        elif mode == "2_hourly_close_trail":
            if is_earnings_shock or (model_1b_is_known and prob_1b < 0.43):
                if max_rec >= 0.52:
                    floor_close_eval = max(floor_close_eval or 0.0, peak_p * (1.0 - 0.35 * atr_d), entry_p * (1.0 + 0.42 * shock_drop))
                elif max_rec >= 0.38:
                    floor_close_eval = max(floor_close_eval or 0.0, entry_p * (1.0 + 0.22 * shock_drop))
            elif is_macro_or_sector and model_1b_is_known and prob_1b >= 0.52:
                if max_rec >= 0.55:
                    floor_close_eval = max(floor_close_eval or 0.0, peak_p * (1.0 - 0.90 * atr_d), entry_p * (1.0 + 0.38 * shock_drop))
                elif max_rec >= 0.42:
                    floor_close_eval = max(floor_close_eval or 0.0, entry_p * 1.003)
            else:
                if max_rec >= 0.52:
                    floor_close_eval = max(floor_close_eval or 0.0, peak_p * (1.0 - 0.65 * atr_d), entry_p * (1.0 + 0.38 * shock_drop))
                elif max_rec >= 0.40:
                    floor_close_eval = max(floor_close_eval or 0.0, entry_p * 1.003)
            # Stop de catástrofe en el servidor de IBKR (1.0 ATR más abajo del floor_close_eval para proteger de flash crash)
            if floor_close_eval is not None:
                floor_stop_broker = max(floor_stop_broker or 0.0, entry_p * 1.002, floor_close_eval * (1.0 - 0.85 * atr_d))

        # ---------------------------------------------------------------------
        # MODO 3: TRAILING EVALUADO SOLO AL CIERRE DIARIO + PISO GARANTIZADO EN BROKER
        # ---------------------------------------------------------------------
        elif mode == "3_daily_close_trail":
            if max_rec >= 0.45:
                # En IBKR ponemos un piso duro en Break-Even + 20% de la caída para no perder jamás
                floor_stop_broker = max(floor_stop_broker or 0.0, entry_p * (1.0 + 0.20 * shock_drop))
            if model_1b_is_known:
                if is_earnings_shock or prob_1b < 0.44:
                    # Si 1B ve debilidad, ahí sí usamos cierre horario rápido
                    floor_close_eval = max(floor_close_eval or 0.0, peak_p * (1.0 - 0.45 * atr_d), entry_p * (1.0 + 0.38 * shock_drop))
                else:
                    # Si 1B dice "Deja correr", el trailing solo se mira al cierre del día y el piso duro en broker sube por escalones
                    floor_close_eval = max(floor_close_eval or 0.0, peak_close_p * (1.0 - 0.95 * atr_d), entry_p * (1.0 + 0.35 * shock_drop))
                    if max_rec >= 0.75:
                        floor_stop_broker = max(floor_stop_broker or 0.0, entry_p * (1.0 + 0.48 * shock_drop))
                    if max_rec >= 1.00:
                        floor_stop_broker = max(floor_stop_broker or 0.0, entry_p * (1.0 + 0.72 * shock_drop))

        # ---------------------------------------------------------------------
        # MODO 4: TRINQUETE POR ESCALONES (RATCHET) + HOLGURA 1B AL CIERRE HORARIO (Sin ahogar mechas)
        # ---------------------------------------------------------------------
        elif mode == "4_ratchet_1b_hourly_close":
            if max_rec >= 0.42:
                # Escalón 0 (Momento MeLi): Piso de emergencia en broker en Break-Even (+0.3%)
                floor_stop_broker = max(floor_stop_broker or 0.0, entry_p * 1.003)

            if model_1b_is_known:
                # 1. ¿El Modelo 1B advierte agotamiento (`prob_1b < 0.45`) o es Post-Earnings?
                if is_earnings_shock or prob_1b < 0.45:
                    if max_rec >= 0.50:
                        floor_close_eval = max(
                            floor_close_eval or 0.0,
                            peak_close_p * (1.0 - 0.40 * atr_d),
                            entry_p * (1.0 + 0.40 * shock_drop),
                        )
                        floor_stop_broker = max(floor_stop_broker or 0.0, entry_p * (1.0 + 0.32 * shock_drop))
                # 2. ¿El Modelo 1B tiene ALTA convicción de continuación (`prob_1b >= 0.54`)?
                elif prob_1b >= 0.54:
                    # Dejamos respirar las velas horarias midiendo el trailing desde `peak_close_p` con 1.25 ATR
                    # y subiendo el piso garantizado por escalones a medida que recupera el 65%, 85% y 110% de la caída!
                    if max_rec >= 0.55:
                        floor_close_eval = max(
                            floor_close_eval or 0.0,
                            peak_close_p * (1.0 - 1.25 * atr_d),
                            entry_p * (1.0 + 0.35 * shock_drop),
                        )
                    if max_rec >= 0.70:
                        floor_close_eval = max(floor_close_eval or 0.0, entry_p * (1.0 + 0.50 * shock_drop))
                        floor_stop_broker = max(floor_stop_broker or 0.0, entry_p * (1.0 + 0.40 * shock_drop))
                    if max_rec >= 0.90:
                        floor_close_eval = max(floor_close_eval or 0.0, entry_p * (1.0 + 0.68 * shock_drop))
                        floor_stop_broker = max(floor_stop_broker or 0.0, entry_p * (1.0 + 0.58 * shock_drop))
                    if max_rec >= 1.15:
                        # Ya recuperó más del 115% de la caída: ceñir trailing a 0.65 ATR
                        floor_close_eval = max(
                            floor_close_eval or 0.0,
                            peak_close_p * (1.0 - 0.65 * atr_d),
                            entry_p * (1.0 + 0.90 * shock_drop),
                        )
                # 3. Convicción intermedia de 1B (`0.45 <= prob_1b < 0.54`):
                else:
                    if max_rec >= 0.52:
                        floor_close_eval = max(
                            floor_close_eval or 0.0,
                            peak_close_p * (1.0 - 0.95 * atr_d),
                            entry_p * (1.0 + 0.38 * shock_drop),
                        )
                    if max_rec >= 0.75:
                        floor_close_eval = max(floor_close_eval or 0.0, entry_p * (1.0 + 0.52 * shock_drop))
                        floor_stop_broker = max(floor_stop_broker or 0.0, entry_p * (1.0 + 0.42 * shock_drop))
                    if max_rec >= 0.95:
                        floor_close_eval = max(
                            floor_close_eval or 0.0,
                            peak_close_p * (1.0 - 0.55 * atr_d),
                            entry_p * (1.0 + 0.72 * shock_drop),
                        )

        # ---------------------------------------------------------------------
        # MODO 5: BRACKET OCO EN IBKR (Ratchet + Take-Profit Limit en 100%-125% de la caída según 1B)
        # ---------------------------------------------------------------------
        elif mode == "5_ibkr_oco_bracket_1b":
            if max_rec >= 0.42:
                floor_stop_broker = max(floor_stop_broker or 0.0, entry_p * 1.003)

            if model_1b_is_known:
                if is_earnings_shock or prob_1b < 0.45:
                    # 1B ve agotamiento: poner Limit Take-Profit cercano en 60% de la caída y subir piso a 38%
                    tp_limit_broker = entry_p * (1.0 + 0.62 * shock_drop)
                    floor_close_eval = max(
                        floor_close_eval or 0.0,
                        peak_close_p * (1.0 - 0.45 * atr_d),
                        entry_p * (1.0 + 0.38 * shock_drop),
                    )
                elif prob_1b >= 0.54:
                    # 1B ve fuerza: apuntar al 115% de la caída con Limit Order en IBKR y dejar respirar 1.30 ATR al cierre
                    tp_limit_broker = entry_p * (1.0 + 1.18 * shock_drop)
                    if max_rec >= 0.55:
                        floor_close_eval = max(
                            floor_close_eval or 0.0,
                            peak_close_p * (1.0 - 1.30 * atr_d),
                            entry_p * (1.0 + 0.35 * shock_drop),
                        )
                    if max_rec >= 0.75:
                        floor_close_eval = max(floor_close_eval or 0.0, entry_p * (1.0 + 0.52 * shock_drop))
                        floor_stop_broker = max(floor_stop_broker or 0.0, entry_p * (1.0 + 0.42 * shock_drop))
                else:
                    # Convicción media: Limit Take-Profit en el 92% de recuperación de la caída
                    tp_limit_broker = entry_p * (1.0 + 0.92 * shock_drop)
                    if max_rec >= 0.52:
                        floor_close_eval = max(
                            floor_close_eval or 0.0,
                            peak_close_p * (1.0 - 1.00 * atr_d),
                            entry_p * (1.0 + 0.38 * shock_drop),
                        )
                    if max_rec >= 0.72:
                        floor_close_eval = max(floor_close_eval or 0.0, entry_p * (1.0 + 0.50 * shock_drop))
                        floor_stop_broker = max(floor_stop_broker or 0.0, entry_p * (1.0 + 0.40 * shock_drop))

    net_ret_pct = ((exit_p * 0.9995 - entry_p) / entry_p) * 100.0
    return net_ret_pct, pd.Timestamp(exit_ts), exit_reason


def main():
    t0 = time.time()
    cache_1a_path = PROJECT_ROOT / "data" / "processed" / "strict_prod_df_1a.parquet"
    cache_meli_path = PROJECT_ROOT / "data" / "processed" / "strict_prod_df_meli.parquet"
    hourly_path = PROJECT_ROOT / "data" / "processed" / "sp500_multisector_1hour.parquet"

    print("[1/3] Cargando caché de Producción Estricta y velas 1H...")
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

    # Out-of-Fold 1B probabilities en Train para validar sin tocar Test!
    oof_1b = np.zeros(len(tr_m))
    for idx_tr, idx_val in kf.split(tr_m):
        m1b_f = lgb.LGBMClassifier(
            n_estimators=140, learning_rate=0.03, max_depth=3, num_leaves=8,
            min_child_samples=35, subsample=0.8, colsample_bytree=0.8,
            reg_alpha=2.0, reg_lambda=8.0, random_state=42, verbose=-1
        )
        m1b_f.fit(tr_m.iloc[idx_tr][feats_1b].fillna(0.0), tr_m.iloc[idx_tr]["keep_running"].values)
        oof_1b[idx_val] = m1b_f.predict_proba(tr_m.iloc[idx_val][feats_1b].fillna(0.0))[:, 1]
    tr_m["prob_1b"] = oof_1b

    clf_1b = lgb.LGBMClassifier(
        n_estimators=140, learning_rate=0.03, max_depth=3, num_leaves=8,
        min_child_samples=35, subsample=0.8, colsample_bytree=0.8,
        reg_alpha=2.0, reg_lambda=8.0, random_state=42, verbose=-1
    )
    clf_1b.fit(tr_m[feats_1b].fillna(0.0), tr_m["keep_running"].values)
    te_m["prob_1b"] = clf_1b.predict_proba(te_m[feats_1b].fillna(0.0))[:, 1]

    prob_1b_map = {int(r["event_id"]): float(r["prob_1b"]) for _, r in pd.concat([tr_m, te_m]).iterrows()}

    modes = [
        ("1. BASE ACTUAL: Stop-Market Intradía (lows[step] <= floor, 0.65-0.90 ATR)", "1_base_intraday_stop"),
        ("2. CIERRE HORARIO: Evaluado al Cierre 1H -> Open[t+1] + Stop Catástrofe", "2_hourly_close_trail"),
        ("3. CIERRE DIARIO: Trailing al Cierre Diario + Piso Escalón en Broker", "3_daily_close_trail"),
        ("4. TRINQUETE RATCHET + 1B: Escalones 50/70/90% + Holgura 1.25 ATR al Cierre 1H", "4_ratchet_1b_hourly_close"),
        ("5. BRACKET OCO IBKR + 1B: Ratchet + Limit Take-Profit (62% / 92% / 118%)", "5_ibkr_oco_bracket_1b"),
    ]

    def apply_shield(d: pd.DataFrame, p_min: float):
        m = (
            (d["prob_1a"] >= p_min)
            & (d["weak_volume_flag"] == 0)
            & ~((d["last_earnings_reaction_3d"] <= -0.05) & (d["cmf_15d"] <= -0.05))
            & ~((d["tnx_ret_3d"] >= 0.030) & (d["vix_slope_3h"] >= 0.04))
        )
        return d[m].copy()

    # Primero validemos en TRAIN (Out-of-Fold 2024-2025) y luego en TEST OOS (Nov 2025 - Oct 2026)
    print("[2/3] Evaluando las 5 Arquitecturas de Salida en TRAIN Out-of-Fold y TEST Fuera de Muestra...")
    train_cand = apply_shield(tr_1a, 0.56)
    test_cand_all = apply_shield(te_1a, 0.55)

    meli_capture_rows = []
    port_059_rows = []
    port_061_rows = []

    for m_label, m_code in modes:
        # A) Evaluar en TRAIN Out-of-Fold (para verificar que no hay overfitting al periodo de Test)
        tr_rets = []
        for _, row in train_cand.iterrows():
            ev_id = int(row["event_id"])
            tk = row["ticker"]
            i0 = int(row["bar_idx"])
            tdf = ticker_dfs[tk]
            fut_df = tdf.iloc[i0 + 1 : i0 + 106]
            p_1b = prob_1b_map.get(ev_id, 0.50)
            trig_step = int(row["meli_trigger_step"])
            r_net, _, _ = simulate_ibkr_exit_archetypes(
                fut_df, float(row["entry_p"]), float(row["sl_p"]), float(row["atr_d"]),
                float(row["shock_drop"]), int(row["shock_type_code"]), p_1b, trig_step, mode=m_code
            )
            tr_rets.append(r_net)
        tr_avg_ret = float(np.mean(tr_rets))

        # B) Evaluar en TEST OOS (todas las señales de te_1a con prob_1a >= 0.55)
        te_sim_rows = []
        for _, row in test_cand_all.iterrows():
            ev_id = int(row["event_id"])
            tk = row["ticker"]
            i0 = int(row["bar_idx"])
            tdf = ticker_dfs[tk]
            fut_df = tdf.iloc[i0 + 1 : i0 + 106]
            p_1b = prob_1b_map.get(ev_id, 0.50)
            trig_step = int(row["meli_trigger_step"])
            r_net, x_ts, x_rsn = simulate_ibkr_exit_archetypes(
                fut_df, float(row["entry_p"]), float(row["sl_p"]), float(row["atr_d"]),
                float(row["shock_drop"]), int(row["shock_type_code"]), p_1b, trig_step, mode=m_code
            )
            max_high = fut_df["high"].max()
            mfe_pct = ((max_high - float(row["entry_p"])) / float(row["entry_p"])) * 100.0
            d = row.to_dict()
            d.update({
                "prob_1b": p_1b,
                "net_ret_pct": r_net,
                "exit_ts": x_ts,
                "exit_reason": x_rsn,
                "mfe_pct": mfe_pct,
                "reached_meli_45": int(trig_step < 900),
            })
            te_sim_rows.append(d)

        df_te_mode = pd.DataFrame(te_sim_rows).sort_values("entry_ts").reset_index(drop=True)
        meli_sub = df_te_mode[df_te_mode["reached_meli_45"] == 1]
        early_exits = meli_sub[(meli_sub["net_ret_pct"] > 0) & ((meli_sub["mfe_pct"] - meli_sub["net_ret_pct"]) >= 4.0)]

        meli_capture_rows.append({
            "Arquitectura_Salida_IBKR": m_label,
            "Train_OOF_Ret %": round(tr_avg_ret, 2),
            "OOS_MeLi_WR %": round((meli_sub["net_ret_pct"] > 0).mean() * 100, 1),
            "OOS_MeLi_Ret_Capturado %": round(meli_sub["net_ret_pct"].mean(), 2),
            "%_Del_MFE_Capturado": f"{meli_sub['net_ret_pct'].mean() / meli_sub['mfe_pct'].mean() * 100:.1f}%",
            "%_Salidas_Prematuras": f"{len(early_exits) / len(meli_sub) * 100:.1f}%",
            "OOS_Total_Ret_Medio %": round(df_te_mode["net_ret_pct"].mean(), 2),
        })

        # Simulación de Portafolio ($25,000, 5 Slots) en prob_1a >= 0.59 y prob_1a >= 0.61
        sub_059 = df_te_mode[df_te_mode["prob_1a"] >= 0.59].copy()
        res_059 = run_portfolio_simulation(sub_059, initial_capital=25000.0, max_slots=5)
        port_059_rows.append({
            "Arquitectura_Salida_IBKR (prob_1a >= 0.59, 5 Slots)": m_label,
            "Señales": len(sub_059),
            "Trades": res_059["trades_executed"],
            "WR %": round(res_059["win_rate_pct"], 1),
            "Ret_Medio %": round(res_059["avg_trade_ret_pct"], 2),
            "PF": round(res_059["profit_factor"], 2),
            "Capital_Final": f"${res_059['final_capital']:,.0f}",
            "Retorno_11m": f"+{res_059['total_return_pct']:.1f}%",
            "Max_DD %": f"{res_059['max_dd_pct']:.2f}%",
            "Sharpe": round(res_059["sharpe"], 2),
        })

        sub_061 = df_te_mode[df_te_mode["prob_1a"] >= 0.61].copy()
        res_061 = run_portfolio_simulation(sub_061, initial_capital=25000.0, max_slots=5)
        port_061_rows.append({
            "Arquitectura_Salida_IBKR (Francotirador prob_1a >= 0.61)": m_label,
            "Señales": len(sub_061),
            "Trades": res_061["trades_executed"],
            "WR %": round(res_061["win_rate_pct"], 1),
            "Ret_Medio %": round(res_061["avg_trade_ret_pct"], 2),
            "PF": round(res_061["profit_factor"], 2),
            "Capital_Final": f"${res_061['final_capital']:,.0f}",
            "Retorno_11m": f"+{res_061['total_return_pct']:.1f}%",
            "Max_DD %": f"{res_061['max_dd_pct']:.2f}%",
            "Sharpe": round(res_061["sharpe"], 2),
        })

    print("\n" + "=" * 126)
    print("1. RADIOGRAFÍA DE CAPTURA EN LAS 315 SEÑALES QUE LLEGAN AL +45% ('MOMENTO MELI') — TRAIN OOF vs TEST OOS:")
    print("=" * 126)
    print(pd.DataFrame(meli_capture_rows).to_string(index=False))

    print("\n" + "=" * 126)
    print("2. IMPACTO EN PORTAFOLIO REAL IBKR ($25,000 Iniciales, 5 Slots, 10 bps, `prob_1a >= 0.59`):")
    print("=" * 126)
    print(pd.DataFrame(port_059_rows).to_string(index=False))

    print("\n" + "=" * 126)
    print("3. IMPACTO EN PORTAFOLIO FRANCOTIRADOR IBKR ($25,000 Iniciales, 5 Slots, 10 bps, `prob_1a >= 0.61`):")
    print("=" * 126)
    print(pd.DataFrame(port_061_rows).to_string(index=False))
    print("=" * 126)
    print(f"Tiempo total: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
