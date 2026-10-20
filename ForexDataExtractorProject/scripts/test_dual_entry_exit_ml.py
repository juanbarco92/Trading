"""
test_dual_entry_exit_ml.py - Prueba Empírica de la Arquitectura de 2 Modelos ML para el Motor 1:
  * MODELO 1A (Entry ML): Decide CUÁNDO INGRESAR ante un Golpe Duro de 3 a 5 días.
  * MODELO 1B (Exit ML):  Una vez dentro del trade, evalúa día a día las variables en tiempo real
                          (distancia actual a resistencia, agotamiento de RSI, mechas superiores de rechazo,
                          pérdida de volumen, porcentaje ya recuperado) y decide CUÁNDO SALIR
                          (¿Cobrar ganancia hoy o dejar correr porque aún tiene gasolina?).
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
from scripts.evaluate_multisector_motor1 import compute_core_user_features_per_ticker, deduplicate_events


def main():
    t0 = time.time()
    print("=" * 105)
    print("ARQUITECTURA DUAL DE MACHINE LEARNING (MOTOR 1):")
    print("  [MODELO 1A: CUÁNDO ENTRAR] + [MODELO 1B: CUÁNDO SALIR DINÁMICAMENTE]")
    print("=" * 105)

    hourly_path = PROJECT_ROOT / "data" / "processed" / "sp500_multisector_1hour.parquet"
    earn_path = PROJECT_ROOT / "data" / "processed" / "sp500_earnings_pit.parquet"

    df = pd.read_parquet(hourly_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values(["ticker", "timestamp"]).reset_index(drop=True)

    spy = df[df["ticker"] == "SPY"].sort_values("timestamp").copy()
    spy["hour_key"] = spy["timestamp"].dt.floor("h")
    spy = spy.drop_duplicates(subset=["hour_key"]).set_index("hour_key")
    spy_ret_3d = spy["close"].pct_change(21)

    stocks = df[~df["ticker"].isin(["SPY", "QQQ"])].copy().reset_index(drop=True)
    stocks["hour_key"] = stocks["timestamp"].dt.floor("h")
    stocks["spy_ret_3d"] = stocks["hour_key"].map(spy_ret_3d).fillna(0.0)

    g = stocks.groupby("ticker", group_keys=False)
    stocks["ret_3d"] = g["close"].pct_change(21)
    stocks["ret_5d"] = g["close"].pct_change(35)
    stocks["ret_60d"] = g["close"].pct_change(420)
    stocks["vol_3d"] = g["ret_3d"].transform(lambda s: s.rolling(420, min_periods=70).std())
    stocks["vol_5d"] = g["ret_5d"].transform(lambda s: s.rolling(420, min_periods=70).std())
    stocks["z_3d"] = stocks["ret_3d"] / stocks["vol_3d"].replace(0, np.nan)
    stocks["z_5d"] = stocks["ret_5d"] / stocks["vol_5d"].replace(0, np.nan)

    high_1d = g["high"].transform(lambda s: s.rolling(7, min_periods=3).max())
    low_1d = g["low"].transform(lambda s: s.rolling(7, min_periods=3).min())
    stocks["atr_daily_pct"] = ((high_1d - low_1d) / stocks["close"] * 100.0).groupby(stocks["ticker"]).transform(
        lambda s: s.rolling(98, min_periods=21).mean()
    )

    print("1. Calculando tus features estructurales en el universo de Alta Elasticidad (ATR >= 2.2%)...")
    # Filtrar primero activos que alcanzan ATR >= 2.2% para velocidad
    valid_tickers = stocks.groupby("ticker")["atr_daily_pct"].max()
    eligible_tickers = valid_tickers[valid_tickers >= 2.2].index.tolist()

    stocks = stocks[stocks["ticker"].isin(eligible_tickers)].copy().reset_index(drop=True)

    ticker_dfs = {}
    shock_candidates = []

    for i, (tk, sub) in enumerate(stocks.groupby("ticker")):
        sub_feat = compute_core_user_features_per_ticker(sub)
        # Añadir upper_wick_ratio y pendiente de RSI para el Modelo 1B de Salida
        rng = (sub_feat["high"] - sub_feat["low"]).replace(0, np.nan)
        sub_feat["upper_wick_ratio"] = (sub_feat["high"] - np.maximum(sub_feat["open"], sub_feat["close"])) / rng
        sub_feat["rsi_slope_3h"] = sub_feat["RSI_14"].diff(3)
        sub_feat["ret_1d_cur"] = sub_feat["close"].pct_change(7) * 100.0
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

    # Cruzar con Earnings Point-in-Time si están disponibles
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
    else:
        for col in ["days_since_earnings", "eps_surprise_pct", "eps_positive", "eps_beat_4q_ratio", "eps_surprise_4q_mean"]:
            shocks_raw[col] = 0.0

    shocks = deduplicate_events(shocks_raw, "timestamp", min_gap_hours=48)

    # 2. Construir los datasets para AMBOS modelos:
    #    - Dataset 1A (En la barra de entrada t=0): ¿El golpe logra recuperar al menos >=55% de la caída antes de tocar SL al cierre?
    #    - Dataset 1B (En cada cierre diario dentro del trade, días d=1..14, cuando la posición ya está en verde / recuperando):
    #      ¿Conviene MANTENER (1: en los próximos 5 días subirá > +1.5% adicional sin caer al piso) o SALIR HOY (0: ya hizo techo y va a retroceder)?
    valid_shocks = []
    exit_snapshots = []

    for event_id, row in shocks.iterrows():
        tk = row["ticker"]
        i0 = int(row["bar_idx"])
        tdf = ticker_dfs[tk]
        if i0 + 126 >= len(tdf):
            continue

        atr_d = row["atr_daily_pct"] / 100.0
        shock_drop = abs(min(row["ret_3d"], row["ret_5d"], -0.04))
        entry_p = tdf.loc[i0 + 1, "open"] * 1.0005
        sl_p = entry_p * (1.0 - 1.5 * atr_d)

        # Evaluar si este golpe es un "Buen Rebote" para el Modelo 1A (alcanza >= 55% de recuperación de la caída en <=15 días sin tocar SL antes)
        max_rec = 0.0
        hit_sl = False
        for step in range(105):
            c_val = tdf.loc[i0 + 1 + step, "close"]
            h_val = tdf.loc[i0 + 1 + step, "high"]
            rec = ((h_val - entry_p) / entry_p) / shock_drop
            if rec > max_rec:
                max_rec = rec
            if (step + 1) % 7 == 0 and c_val <= sl_p:
                hit_sl = True
                break

        target_1a = 1 if (max_rec >= 0.55) else 0
        row_copy = row.to_dict()
        row_copy["event_id"] = event_id
        row_copy["entry_p"] = entry_p
        row_copy["shock_drop"] = shock_drop
        row_copy["atr_d"] = atr_d
        row_copy["target_1a_good_entry"] = target_1a
        valid_shocks.append(row_copy)

        # Generar puntos de decisión día a día (días 1 a 14 = cada 7 barras) para entrenar/evaluar el Modelo 1B (Cuándo Salir)
        peak_so_far = entry_p
        for day_num in range(1, 15):
            step = day_num * 7 - 1
            cur_idx = i0 + 1 + step
            cur_row = tdf.loc[cur_idx]
            cur_close = cur_row["close"]
            cur_high_slice = tdf.loc[i0 + 1 : cur_idx, "high"].max()
            peak_so_far = max(peak_so_far, cur_high_slice)

            if cur_close <= sl_p:
                break  # Ya habría salido por SL

            unrealized_ret_pct = (cur_close - entry_p) / entry_p * 100.0
            recovery_ratio_now = ((cur_close - entry_p) / entry_p) / shock_drop
            drawdown_from_peak_atr = ((peak_so_far - cur_close) / entry_p) / atr_d

            # Solo le preguntamos al Modelo 1B de Salida cuando el trade ya recuperó >= 35% de la caída (o cuando ya lleva >= 5 días abierto)
            if recovery_ratio_now >= 0.35 or day_num >= 5:
                # Mirar qué pasa en los siguientes 4 días (28 barras) si NO cerramos hoy:
                # ¿Cuánto es el retorno al cierre dentro de 3-4 días vs cerrar hoy mismo?
                fut_end_idx = min(cur_idx + 28, i0 + 105)
                fut_closes = tdf.loc[cur_idx + 1 : fut_end_idx, "close"].to_numpy()
                fut_highs = tdf.loc[cur_idx + 1 : fut_end_idx, "high"].to_numpy()
                fut_lows = tdf.loc[cur_idx + 1 : fut_end_idx, "low"].to_numpy()

                if len(fut_closes) == 0:
                    continue

                # Retorno adicional si esperamos 3 días más (21 barras) o hasta el final
                look_idx = min(20, len(fut_closes) - 1)
                future_change_pct = (fut_closes[look_idx] - cur_close) / cur_close * 100.0
                max_further_upside_pct = (np.max(fut_highs) - cur_close) / cur_close * 100.0
                max_further_drop_pct = (np.min(fut_lows) - cur_close) / cur_close * 100.0

                # Target 1B (target_exit_now = 1 si CONVIENE SALIR HOY porque en los próximos 3-4 días cae o pierde momentum)
                # target_exit_now = 1 si el precio cae más de -1.0% en los próximos 3 días o si el upside restante es menor que la caída
                should_exit_now = 1 if (future_change_pct <= 0.3 or abs(max_further_drop_pct) > max_further_upside_pct) else 0

                exit_snapshots.append({
                    "event_id": event_id,
                    "timestamp": cur_row["timestamp"],
                    "ticker": tk,
                    "sector": row["sector"],
                    "day_in_trade": day_num,
                    "unrealized_ret_pct": unrealized_ret_pct,
                    "recovery_ratio_now": recovery_ratio_now,
                    "drawdown_from_peak_atr": drawdown_from_peak_atr,
                    "RSI_14_now": cur_row["RSI_14"],
                    "rsi_slope_3h_now": cur_row["rsi_slope_3h"],
                    "close_loc_1h_now": cur_row["close_loc_1h"],
                    "upper_wick_now": cur_row["upper_wick_ratio"],
                    "lower_wick_now": cur_row["lower_wick_ratio"],
                    "rvol_1h_now": cur_row["rvol_1h"],
                    "sr_dist_res_atr_now": cur_row["sr_dist_resistance_atr"],
                    "sr_dist_sup_atr_now": cur_row["sr_dist_support_atr"],
                    "sr_channel_pos_now": cur_row["sr_channel_position"],
                    "BB_percent_b_now": cur_row["BB_20_20_percent_b"],
                    "close_diff_SMA_40_now": cur_row["close_diff_SMA_40"],
                    "ret_1d_cur": cur_row["ret_1d_cur"],
                    "future_change_3d_pct": future_change_pct,
                    "should_exit_now": should_exit_now,
                })

    df_1a = pd.DataFrame(valid_shocks).sort_values("timestamp").reset_index(drop=True)
    df_1b = pd.DataFrame(exit_snapshots).sort_values("timestamp").reset_index(drop=True)

    print(f"Dataset Modelo 1A (Entradas): {len(df_1a):,} golpes únicos | Tasa base buen rebote (>=55% recup): {df_1a['target_1a_good_entry'].mean()*100:.1f}%")
    print(f"Dataset Modelo 1B (Decisiones de Salida en vivo): {len(df_1b):,} revisiones diarias dentro de trades abiertos.\n")

    # Split Cronológico 70% Train / 30% Test OOS Ciego (Misma fecha de corte para ambos modelos)
    split_date = df_1a["timestamp"].iloc[int(len(df_1a) * 0.70)]
    print(f"Fecha de Corte Estricta Out-of-Sample: {split_date.date()} (Train < {split_date.date()} | Test OOS >= {split_date.date()})")

    # =========================================================================
    # ENTRENAMIENTO Y EVALUACIÓN DEL MODELO 1A (CUÁNDO INGRESAR)
    # =========================================================================
    feats_1a = [
        "SMA_100_pct_momentum", "SMA_40_pct_momentum", "close_diff_SMA_100", "close_diff_SMA_40",
        "ratio_SMA_40_SMA_100", "ratio_SMA_100_SMA_40", "ratio_SMA_100_SMA_10",
        "norm_diff_SMA_40_SMA_100", "norm_diff_SMA_10_SMA_100", "slope_SMA_100",
        "sr_dist_support_pct", "sr_dist_resistance_pct", "sr_dist_support_atr", "sr_dist_resistance_atr",
        "sr_support_touches", "sr_resistance_touches", "sr_channel_position",
        "ATR_14_pct", "ATR_21_pct", "ATR_21_relative", "BB_20_20_bandwidth", "BB_20_20_percent_b",
        "RSI_14", "close_loc_1h", "lower_wick_ratio", "body_ratio", "rvol_1h",
        "z_3d", "z_5d", "ret_60d", "spy_ret_3d",
        "days_since_earnings", "eps_surprise_pct", "eps_positive", "eps_beat_4q_ratio", "eps_surprise_4q_mean"
    ]
    feats_1a = [f for f in feats_1a if f in df_1a.columns]

    tr_1a = df_1a[df_1a["timestamp"] < split_date].copy()
    te_1a = df_1a[df_1a["timestamp"] >= split_date].copy()

    clf_1a = lgb.LGBMClassifier(
        n_estimators=150, learning_rate=0.03, max_depth=4, num_leaves=12,
        min_child_samples=40, subsample=0.8, colsample_bytree=0.7,
        reg_alpha=2.0, reg_lambda=8.0, random_state=42, verbose=-1
    )
    clf_1a.fit(tr_1a[feats_1a].fillna(0.0), tr_1a["target_1a_good_entry"].values)
    probs_1a = clf_1a.predict_proba(te_1a[feats_1a].fillna(0.0))[:, 1]
    auc_1a = roc_auc_score(te_1a["target_1a_good_entry"].values, probs_1a)
    te_1a["prob_entry"] = probs_1a

    # =========================================================================
    # ENTRENAMIENTO Y EVALUACIÓN DEL MODELO 1B (CUÁNDO SALIR / COBRAR)
    # =========================================================================
    feats_1b = [
        "day_in_trade", "unrealized_ret_pct", "recovery_ratio_now", "drawdown_from_peak_atr",
        "RSI_14_now", "rsi_slope_3h_now", "close_loc_1h_now", "upper_wick_now", "lower_wick_now",
        "rvol_1h_now", "sr_dist_res_atr_now", "sr_dist_sup_atr_now", "sr_channel_pos_now",
        "BB_percent_b_now", "close_diff_SMA_40_now", "ret_1d_cur"
    ]
    tr_1b = df_1b[df_1b["timestamp"] < split_date].copy()
    te_1b = df_1b[df_1b["timestamp"] >= split_date].copy()

    clf_1b = lgb.LGBMClassifier(
        n_estimators=150, learning_rate=0.03, max_depth=4, num_leaves=12,
        min_child_samples=60, subsample=0.8, colsample_bytree=0.8,
        reg_alpha=2.0, reg_lambda=8.0, random_state=42, verbose=-1
    )
    clf_1b.fit(tr_1b[feats_1b].fillna(0.0), tr_1b["should_exit_now"].values)
    probs_1b = clf_1b.predict_proba(te_1b[feats_1b].fillna(0.0))[:, 1]
    auc_1b = roc_auc_score(te_1b["should_exit_now"].values, probs_1b)
    te_1b["prob_exit"] = probs_1b

    print("\n" + "=" * 105)
    print(f"DESEMPEÑO OUT-OF-SAMPLE DE AMBOS CEREBROS (TEST CIEGO: {len(te_1a):,} OPERACIONES):")
    print(f"  * MODELO 1A (Selector de Entrada): ROC-AUC = {auc_1a:.4f}")
    print(f"  * MODELO 1B (Detector de Salida):  ROC-AUC = {auc_1b:.4f}")
    print("=" * 105)

    # Mostrar qué pasa en la vida real cuando el Modelo 1B dice "SALIR YA" vs cuando dice "MANTENER ABIERTA"
    te_1b["quintil_salida"] = pd.qcut(te_1b["prob_exit"], q=5, labels=False, duplicates="drop")
    q1b_summary = te_1b.groupby("quintil_salida").agg(
        decisiones=("should_exit_now", "count"),
        prob_salir_min=("prob_exit", lambda s: round(s.min(), 3)),
        prob_salir_max=("prob_exit", lambda s: round(s.max(), 3)),
        se_freno_o_cayo_pct=("should_exit_now", lambda s: round(s.mean() * 100, 1)),
        retorno_siguientes_3d_pct=("future_change_3d_pct", lambda s: round(s.mean(), 2)),
    ).reset_index()
    print("\n--- ¿QUÉ PASA EN LOS SIGUIENTES 3 DÍAS SEGÚN EL MODELO 1B (CUÁNDO SALIR)? ---")
    print("  (Quintil 0 = Modelo 1B dice 'MANTENER ABIERTA, AÚN TIENE GASOLINA' | Quintil 4 = Modelo 1B dice '¡SALIR YA!')")
    print(q1b_summary.to_string(index=False))

    # =========================================================================
    # SIMULACIÓN CONJUNTA EN TEST OUT-OF-SAMPLE: MODELO 1A + MODELO 1B TRABAJANDO JUNTOS
    # =========================================================================
    print("\n" + "=" * 105)
    print("COMPARATIVA FINAL EN TEST OUT-OF-SAMPLE CIEGO: REGLA FIJA vs SISTEMA DE 2 MODELOS (1A + 1B):")
    print("=" * 105)

    # Indexar las probabilidades de salida del Modelo 1B por (event_id, day_in_trade)
    exit_prob_map = {
        (int(r["event_id"]), int(r["day_in_trade"])): float(r["prob_exit"])
        for _, r in te_1b.iterrows()
    }

    def run_oos_strategy(subset_1a: pd.DataFrame, use_ml_exit: bool, label: str):
        rets = []
        days_held = []
        for _, row in subset_1a.iterrows():
            ev_id = int(row["event_id"])
            tk = row["ticker"]
            i0 = int(row["bar_idx"])
            tdf = ticker_dfs[tk]
            entry_p = row["entry_p"]
            shock_drop = row["shock_drop"]
            atr_d = row["atr_d"]

            tp_static = entry_p * (1.0 + 0.75 * shock_drop)
            sl_p = entry_p * (1.0 - 1.5 * atr_d)

            exit_p = tdf.loc[i0 + 105, "close"]
            held = 15.0

            for step in range(105):
                c_val = tdf.loc[i0 + 1 + step, "close"]
                h_val = tdf.loc[i0 + 1 + step, "high"]

                if not use_ml_exit:
                    # Regla estática: espera el 75% TP o SL al cierre
                    if (step + 1) % 7 == 0 and c_val <= sl_p:
                        exit_p = c_val
                        held = (step + 1) / 7.0
                        break
                    if h_val >= tp_static:
                        exit_p = tp_static
                        held = (step + 1) / 7.0
                        break
                else:
                    # Sistema con Modelo 1B (Salir dinámicamente cuando el Modelo 1B detecta agotamiento)
                    if (step + 1) % 7 == 0:
                        day_num = (step + 1) // 7
                        if c_val <= sl_p:
                            exit_p = c_val
                            held = float(day_num)
                            break
                        rec_now = ((c_val - entry_p) / entry_p) / shock_drop
                        p_exit = exit_prob_map.get((ev_id, day_num), 0.0)
                        # Si ya va ganando (recuperó >= 35%) y el Modelo 1B ve agotamiento (prob_exit >= 0.56), cobra ganancia hoy!
                        if rec_now >= 0.35 and p_exit >= 0.56:
                            exit_p = c_val
                            held = float(day_num)
                            break
                        # O si ya superó el 100% de la caída y el Modelo 1B no está ultra alcista (p_exit >= 0.48), cobra!
                        if rec_now >= 1.00 and p_exit >= 0.48:
                            exit_p = c_val
                            held = float(day_num)
                            break

            net_ret = ((exit_p * 0.9995 - entry_p) / entry_p) * 100.0
            rets.append(net_ret)
            days_held.append(held)

        rets = np.array(rets)
        wins = rets[rets > 0]
        losses = rets[rets <= 0]
        pf = wins.sum() / abs(losses.sum()) if len(losses) > 0 and losses.sum() != 0 else 99.0
        return {
            "Estrategia en Test OOS Ciego": label,
            "Trades": len(rets),
            "Win_Rate_Neto %": round((rets > 0).mean() * 100, 1),
            "Retorno_Medio_Trade %": round(rets.mean(), 2),
            "Retorno_Acumulado_Simple %": round(rets.sum(), 1),
            "Profit_Factor": round(pf, 2),
            "Días_Medios": round(np.mean(days_held), 1),
        }

    th_top40 = np.percentile(te_1a["prob_entry"], 60)
    th_top25 = np.percentile(te_1a["prob_entry"], 75)

    comp = [
        run_oos_strategy(te_1a, False, "1. Sin ML (Todos los golpes + Salida Estática 75% TP)"),
        run_oos_strategy(te_1a, True,  "2. Solo Modelo 1B (Todos los golpes + Salida Dinámica con ML 1B)"),
        run_oos_strategy(te_1a[te_1a["prob_entry"] >= th_top40], False, "3. Solo Modelo 1A (Top 40% Entradas + Salida Estática)"),
        run_oos_strategy(te_1a[te_1a["prob_entry"] >= th_top40], True,  "4. DUAL ML (Top 40% Modelo 1A Entrada + Modelo 1B Salida Dinámica)"),
        run_oos_strategy(te_1a[te_1a["prob_entry"] >= th_top25], True,  "5. DUAL ML ALTA CONVICCIÓN (Top 25% Modelo 1A + Modelo 1B Salida)"),
    ]
    print(pd.DataFrame(comp).to_string(index=False))

    # Top Features del Modelo 1B de Salida
    imp_1b = pd.DataFrame({
        "Feature_Salida (Modelo 1B)": feats_1b,
        "Gain": clf_1b.booster_.feature_importance(importance_type="gain")
    }).sort_values("Gain", ascending=False).head(10)
    print("\n--- TOP 10 VARIABLES QUE MIRA EL MODELO 1B PARA DECIDIR CUÁNDO SALIR ---")
    print(imp_1b.to_string(index=False))
    print(f"\nCompletado en {time.time()-t0:.1f}s.")
    print("=" * 105)


if __name__ == "__main__":
    main()
