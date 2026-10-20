"""
compare_1a_deciles_and_exit_modes.py -
Evalúa en los 2,170 golpes Out-of-Sample (314 empresas de ATR >= 2.2%):
1. La Tabla de Deciles (0 a 9) del Modelo 1A (ROC-AUC = 0.5824).
2. Qué pasa cuando combinamos los mejores Deciles del Modelo 1A con:
   - Salida A: Candado Progresivo (Breakeven 40% / Asegurar ganancia al 60%).
   - Salida B: Trailing Stop Dinámico desde el 50% de recuperación.
   - Salida C: Modelo 1B especializado ÚNICAMENTE en el "Momento MeLi" (cuando la acción toca el 60% de
     recuperación, el Modelo 1B mira si chocó con resistencia o perdió volumen para decidir si
     COBRA EL +60% EN EL ACTO o ACTIVA TRAILING STOP PARA BUSCAR EL 100%+).
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

    valid_tickers = stocks.groupby("ticker")["atr_daily_pct"].max()
    eligible_tickers = valid_tickers[valid_tickers >= 2.2].index.tolist()
    stocks = stocks[stocks["ticker"].isin(eligible_tickers)].copy().reset_index(drop=True)

    ticker_dfs = {}
    shock_candidates = []

    for tk, sub in stocks.groupby("ticker"):
        sub_feat = compute_core_user_features_per_ticker(sub)
        rng = (sub_feat["high"] - sub_feat["low"]).replace(0, np.nan)
        sub_feat["upper_wick_ratio"] = (sub_feat["high"] - np.maximum(sub_feat["open"], sub_feat["close"])) / rng
        sub_feat["rsi_slope_3h"] = sub_feat["RSI_14"].diff(3)
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

    # Simular para cada evento los 3 resultados netos:
    # 1) Retorno con Salida Estática (75% TP o Día 15)
    # 2) Retorno con Candado Progresivo (Breakeven al 40%, Candado 45% al tocar 60%, TP en 80%)
    # 3) Retorno con Trailing Stop Dinámico desde el 50% de recuperación
    # 4) Y guardar el snapshot exacto cuando un trade toca el 55% de recuperación ("Momento MeLi") para el Modelo 1B especializado
    valid_shocks = []
    meli_moment_snapshots = []

    for event_id, row in shocks.iterrows():
        tk = row["ticker"]
        i0 = int(row["bar_idx"])
        tdf = ticker_dfs[tk]
        if i0 + 110 >= len(tdf):
            continue

        atr_d = row["atr_daily_pct"] / 100.0
        shock_drop = abs(min(row["ret_3d"], row["ret_5d"], -0.04))
        entry_p = tdf.loc[i0 + 1, "open"] * 1.0005
        sl_p = entry_p * (1.0 - 1.5 * atr_d)

        fut = tdf.iloc[i0 + 1 : i0 + 106]
        highs = fut["high"].to_numpy()
        lows = fut["low"].to_numpy()
        closes = fut["close"].to_numpy()

        # --- Simulación 1: Estática ---
        tp_75 = entry_p * (1.0 + 0.75 * shock_drop)
        exit_static = closes[-1]
        max_rec = 0.0
        for step in range(len(fut)):
            rec = ((highs[step] - entry_p) / entry_p) / shock_drop
            if rec > max_rec:
                max_rec = rec
            if (step + 1) % 7 == 0 and closes[step] <= sl_p:
                exit_static = closes[step]
                break
            if highs[step] >= tp_75:
                exit_static = tp_75
                break
        ret_static = ((exit_static * 0.9995 - entry_p) / entry_p) * 100.0

        # --- Simulación 2: Candado Progresivo ---
        tp_80 = entry_p * (1.0 + max(0.80 * shock_drop, 1.8 * atr_d))
        floor_candado = None
        peak_c = entry_p
        exit_candado = closes[-1]
        for step in range(len(fut)):
            if highs[step] > peak_c:
                peak_c = highs[step]
            m_rec = ((peak_c - entry_p) / entry_p) / shock_drop
            if m_rec >= 0.60:
                floor_candado = max(floor_candado or 0.0, entry_p * (1.0 + 0.45 * shock_drop))
            elif m_rec >= 0.40:
                floor_candado = max(floor_candado or 0.0, entry_p * 1.002)

            if floor_candado is not None and closes[step] <= floor_candado and step > 0:
                exit_candado = floor_candado
                break
            if (step + 1) % 7 == 0 and closes[step] <= sl_p:
                exit_candado = closes[step]
                break
            if highs[step] >= tp_80:
                exit_candado = tp_80
                break
        ret_candado = ((exit_candado * 0.9995 - entry_p) / entry_p) * 100.0

        # --- Simulación 3: Trailing Stop desde el 50% de recuperación ---
        floor_trail = None
        peak_t = entry_p
        exit_trail = closes[-1]
        days_trail = 15.0
        for step in range(len(fut)):
            if highs[step] > peak_t:
                peak_t = highs[step]
            m_rec = ((peak_t - entry_p) / entry_p) / shock_drop
            if m_rec >= 0.50:
                floor_trail = max(floor_trail or 0.0, peak_t * (1.0 - 0.8 * atr_d))

            if floor_trail is not None and closes[step] <= floor_trail and step > 0:
                exit_trail = floor_trail
                days_trail = (step + 1) / 7.0
                break
            if (step + 1) % 7 == 0 and closes[step] <= sl_p:
                exit_trail = closes[step]
                days_trail = (step + 1) / 7.0
                break
        ret_trail = ((exit_trail * 0.9995 - entry_p) / entry_p) * 100.0

        # --- Capturar el "Momento MeLi": La primera barra donde la acción alcanza >= 50% de recuperación ---
        for step in range(len(fut) - 7):
            if (step + 1) % 7 == 0 and closes[step] <= sl_p:
                break
            cur_h = highs[step]
            rec_here = ((cur_h - entry_p) / entry_p) / shock_drop
            if rec_here >= 0.50:
                # ¡Llegó al 50% de recuperación!
                # ¿Qué pasa en el resto de los 15 días? ¿Alcanza el 85% de recuperación sin devolver la ganancia?
                price_at_50 = entry_p * (1.0 + 0.50 * shock_drop)
                rem_highs = highs[step + 1 :]
                rem_closes = closes[step + 1 :]
                if len(rem_highs) == 0:
                    break
                reaches_85 = np.max(rem_highs) >= entry_p * (1.0 + 0.85 * shock_drop)
                drops_back_below_25 = np.min(rem_closes) <= entry_p * (1.0 + 0.20 * shock_drop)
                # target_keep_running = 1 si sigue subiendo hasta >=85% de recuperación sin desplomarse antes
                keep_running = 1 if (reaches_85 and not drops_back_below_25) else 0

                cur_bar = tdf.loc[i0 + 1 + step]
                meli_moment_snapshots.append({
                    "event_id": event_id,
                    "timestamp": cur_bar["timestamp"],
                    "steps_to_reach_50": step + 1,
                    "velocity_pct_per_bar": (0.50 * shock_drop * 100.0) / max(step + 1, 1),
                    "RSI_14_at_50": cur_bar["RSI_14"],
                    "rsi_slope_3h_at_50": cur_bar["rsi_slope_3h"],
                    "rvol_1h_at_50": cur_bar["rvol_1h"],
                    "close_loc_1h_at_50": cur_bar["close_loc_1h"],
                    "upper_wick_at_50": cur_bar["upper_wick_ratio"],
                    "sr_dist_res_atr_at_50": cur_bar["sr_dist_resistance_atr"],
                    "sr_channel_pos_at_50": cur_bar["sr_channel_position"],
                    "BB_percent_b_at_50": cur_bar["BB_20_20_percent_b"],
                    "ATR_14_pct_at_50": cur_bar["ATR_14_pct"],
                    "shock_drop": shock_drop,
                    "ret_if_hold_trail": ret_trail,
                    "ret_if_take_now": (0.50 * shock_drop - 0.001) * 100.0,
                    "keep_running": keep_running,
                })
                break

        row_dict = row.to_dict()
        row_dict["event_id"] = event_id
        row_dict["target_1a"] = 1 if (ret_trail > 1.0 and max_rec >= 0.55) else 0
        row_dict["ret_static"] = ret_static
        row_dict["ret_candado"] = ret_candado
        row_dict["ret_trail"] = ret_trail
        row_dict["days_trail"] = days_trail
        valid_shocks.append(row_dict)

    df_1a = pd.DataFrame(valid_shocks).sort_values("timestamp").reset_index(drop=True)
    split_idx = int(len(df_1a) * 0.70)
    split_date = df_1a["timestamp"].iloc[split_idx]
    tr_1a = df_1a.iloc[:split_idx].copy()
    te_1a = df_1a.iloc[split_idx:].copy()

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

    clf_1a = lgb.LGBMClassifier(
        n_estimators=150, learning_rate=0.03, max_depth=4, num_leaves=12,
        min_child_samples=40, subsample=0.8, colsample_bytree=0.7,
        reg_alpha=2.0, reg_lambda=8.0, random_state=42, verbose=-1
    )
    clf_1a.fit(tr_1a[feats_1a].fillna(0.0), tr_1a["target_1a"].values)
    probs_te = clf_1a.predict_proba(te_1a[feats_1a].fillna(0.0))[:, 1]
    auc_oos = roc_auc_score(te_1a["target_1a"].values, probs_te)
    te_1a["prob"] = probs_te
    te_1a["decil"] = pd.qcut(te_1a["prob"], q=10, labels=False, duplicates="drop")

    # Entrenar Modelo 1B Especializado en el "Momento MeLi" (Cuando la acción toca el 50% de recuperación)
    df_meli = pd.DataFrame(meli_moment_snapshots).sort_values("timestamp").reset_index(drop=True)
    feats_meli = [
        "steps_to_reach_50", "velocity_pct_per_bar", "RSI_14_at_50", "rsi_slope_3h_at_50",
        "rvol_1h_at_50", "close_loc_1h_at_50", "upper_wick_at_50",
        "sr_dist_res_atr_at_50", "sr_channel_pos_at_50", "BB_percent_b_at_50", "ATR_14_pct_at_50", "shock_drop"
    ]
    tr_m = df_meli[df_meli["timestamp"] < split_date].copy()
    te_m = df_meli[df_meli["timestamp"] >= split_date].copy()

    clf_meli = lgb.LGBMClassifier(
        n_estimators=120, learning_rate=0.03, max_depth=3, num_leaves=8,
        min_child_samples=30, subsample=0.8, colsample_bytree=0.8,
        reg_alpha=2.0, reg_lambda=8.0, random_state=42, verbose=-1
    )
    clf_meli.fit(tr_m[feats_meli].fillna(0.0), tr_m["keep_running"].values)
    probs_m = clf_meli.predict_proba(te_m[feats_meli].fillna(0.0))[:, 1]
    auc_meli_1b = roc_auc_score(te_m["keep_running"].values, probs_m)
    te_m["prob_keep"] = probs_m

    # Mapear decisión del Modelo 1B Especializado al dataframe te_1a:
    # Si prob_keep >= 0.50 -> Deja correr con Trailing Stop; si prob_keep < 0.50 -> Cobra el 50% en el acto!
    meli_decision_ret = {}
    for _, r in te_m.iterrows():
        ev_id = int(r["event_id"])
        if r["prob_keep"] >= 0.48:
            meli_decision_ret[ev_id] = float(r["ret_if_hold_trail"])
        else:
            meli_decision_ret[ev_id] = float(r["ret_if_take_now"])

    te_1a["ret_dual_ml"] = [
        meli_decision_ret.get(int(r["event_id"]), float(r["ret_candado"]))
        for _, r in te_1a.iterrows()
    ]

    print("=" * 115)
    print(f"RESULTADOS OUT-OF-SAMPLE CIEGOS EN {len(te_1a):,} GOLPES (314 EMPRESAS DE ALTO ATR):")
    print(f"  * MODELO 1A (Cuándo Entrar en el Golpe 3-5d):                ROC-AUC OOS = {auc_oos:.4f}")
    print(f"  * MODELO 1B (En el 'Momento MeLi' al 50%: ¿Cobrar o Correr?): ROC-AUC OOS = {auc_meli_1b:.4f}")
    print("=" * 115)

    dec_rows = []
    for dec, grp in te_1a.groupby("decil"):
        r_st = grp["ret_static"].to_numpy()
        r_ca = grp["ret_candado"].to_numpy()
        r_tr = grp["ret_trail"].to_numpy()
        r_du = grp["ret_dual_ml"].to_numpy()

        w_du = r_du[r_du > 0].sum()
        l_du = abs(r_du[r_du <= 0].sum())
        pf_du = w_du / l_du if l_du > 0 else 99.0

        dec_rows.append({
            "Decil_1A": dec,
            "Trades": len(grp),
            "Prob_1A": f"{grp['prob'].min():.2f}-{grp['prob'].max():.2f}",
            "WR_Estático %": round((r_st > 0).mean() * 100, 1),
            "Ret_Estático %": round(r_st.mean(), 2),
            "WR_Candado %": round((r_ca > 0).mean() * 100, 1),
            "Ret_Candado %": round(r_ca.mean(), 2),
            "WR_Trailing %": round((r_tr > 0).mean() * 100, 1),
            "Ret_Trailing %": round(r_tr.mean(), 2),
            "WR_DUAL_1A+1B %": round((r_du > 0).mean() * 100, 1),
            "Ret_DUAL_1A+1B %": round(r_du.mean(), 2),
            "PF_DUAL_1A+1B": round(pf_du, 2),
        })

    print(pd.DataFrame(dec_rows).to_string(index=False))
    print("=" * 115)


if __name__ == "__main__":
    main()
