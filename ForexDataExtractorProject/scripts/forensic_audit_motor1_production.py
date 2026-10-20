"""
forensic_audit_motor1_production.py -
Auditoría Forense Rigurosa de Data Leakage y Simulación de Ambiente Productivo para el Motor 1 (1A + 1B).

Audita y cuantifica el impacto exacto de 5 posibles fuentes de inflación de resultados:
1. FUGA #1 (CRÍTICA - Simulación de Salida Intra-Barra y Precio de Fill `exit_p = floor_p`):
   - En el script anterior, durante la barra `step`:
     a) Se leía `highs[step]` para subir el candado `floor_p` y en la MISMA barra `step` se verificaba `closes[step] <= floor_p`.
     b) Cuando `closes[step] <= floor_p`, se asignaba `exit_p = floor_p` (¡como si hubieras vendido en el techo del candado aunque la vela cerrara mucho más abajo o abriera con gap bajista!).
     c) Además, `prob_1b` se usaba desde el inicio de la operación, antes de que ocurriera la barra donde el precio toca el 45%-50%.
   - CORRECCIÓN PRODUCTIVA ESTRICTA:
     - Cualquier actualización de `floor_p` o activación del Modelo 1B al cierre de la barra `step` SOLO puede proteger a partir de la barra `step + 1`.
     - Si se evalúa al cierre de la barra (`closes[step] <= floor_p`), la ejecución ocurre en `opens[step + 1]` (al precio real de mercado, sufriendo cualquier gap o caída).
     - Si se usa orden Stop en el broker desde `step + 1` (`lows[step] <= floor_p`), si hay gap bajista (`opens[step] < floor_p`) se ejecuta en `opens[step]`, no en `floor_p`.

2. FUGA #2 (Timestamps de Reportes Trimestrales `earnings_date`):
   - Verifica a qué hora vienen fechados los reportes en `sp500_earnings_pit.parquet` y añade un buffer de seguridad de +24 horas (`earnings_date + 1 día`) para garantizar que un reporte publicado a las 4:05 PM (After-Hours) jamás sea leído durante ese mismo día.

3. FUGA #3 (Purge Gap entre Train y Test):
   - Añade una brecha de purga de 21 días calendario entre el último evento de entrenamiento y el primer evento de prueba para que ninguna operación abierta en Train siga viva durante el período Test.

4. FUGA #4 (Pre-selección de Tickers con `ATR >= 2.2%`):
   - Evalúa `atr_daily_pct >= 2.2%` únicamente con el historial previo hasta la barra `t` (sin usar `.max()` global).
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
from sklearn.model_selection import KFold
from scripts.evaluate_multisector_motor1 import compute_core_user_features_per_ticker, deduplicate_events
from scripts.backtest_portfolio_motor1_dual_ml import (
    SHOCK_TYPE_MAP,
    classify_shock_origin,
    simulate_adaptive_exit_by_shock_and_1b,
    run_portfolio_simulation,
)


def simulate_strict_production_exit(
    fut_df: pd.DataFrame,
    entry_p: float,
    sl_p: float,
    atr_d: float,
    shock_drop: float,
    shock_type_code: int,
    prob_1b: float,
    meli_trigger_step: int,
    execution_mode: str = "close_triggered_next_open",
):
    """
    Simulación 100% fiel a un ambiente productivo en tiempo real:
    - Durante la barra `step`, solo está vigente el `floor_p` que fue calculado al FINAL de la barra `step - 1`.
    - El Modelo 1B (`prob_1b`) NO se conoce al entrar: solo se activa DESPUÉS de que termina la barra `meli_trigger_step`
      (cuando el precio realmente tocó el 45% de recuperación y cerró esa vela horaria).
    - Dos modos reales de ejecución en broker cuando se activa el candado `floor_p` (vigente desde `step - 1`):
      a) `close_triggered_next_open`: El bot revisa al final de la hora `step` si `closes[step] <= floor_p`.
         Si sí, manda orden Market que se ejecuta en `opens[step + 1]` (o `closes[step]` si es la última barra).
      b) `broker_stop_order_with_gap`: El bot deja una orden Stop-Market en IBKR en `floor_p` al terminar `step - 1`.
         Durante `step`, si abre con gap abajo (`opens[step] <= floor_p`), se llena a `opens[step]`.
         Si abre arriba y toca `lows[step] <= floor_p`, se llena a `floor_p`.
    - Stop Loss Estructural Diario (`sl_p`): Evaluado al cierre del día (`(step + 1) % 7 == 0 and closes[step] <= sl_p`),
      ejecutado en `opens[step + 1]` (sufriendo cualquier deslizamiento real).
    """
    opens = fut_df["open"].to_numpy()
    highs = fut_df["high"].to_numpy()
    lows = fut_df["low"].to_numpy()
    closes = fut_df["close"].to_numpy()
    timestamps = fut_df["timestamp"].to_numpy()
    n_bars = len(fut_df)

    floor_p = None  # Candado vigente para la barra actual (calculado al cierre de la barra anterior)
    peak_p = entry_p
    exit_p = closes[-1]
    exit_ts = timestamps[-1]
    exit_reason = "TimeStop_15d"

    is_earnings_shock = shock_type_code in (1, 2)
    is_macro_or_sector = shock_type_code in (3, 4)

    for step in range(n_bars):
        # ---------------------------------------------------------------------
        # 1. DURANTE LA BARRA `step`: Evaluar si se dispara el `floor_p` que ya estaba puesto desde `step - 1`
        # ---------------------------------------------------------------------
        if floor_p is not None and step > 0:
            if execution_mode == "broker_stop_order_with_gap":
                if opens[step] <= floor_p:
                    # Gap bajista en la apertura de la barra: nos llenan al peor precio (Open), NO en floor_p
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
                # Evaluado al cierre de la barra `step`, ejecutado en `opens[step + 1]`
                if closes[step] <= floor_p:
                    exit_p = opens[step + 1] if (step + 1 < n_bars) else closes[step]
                    exit_ts = timestamps[step + 1] if (step + 1 < n_bars) else timestamps[step]
                    exit_reason = "CloseTrigger_NextOpenFill"
                    break

        # ---------------------------------------------------------------------
        # 2. AL CIERRE DEL DÍA (cada 7 barras): Evaluar Stop Loss Estructural (`sl_p`)
        #    Si cierra por debajo de `sl_p`, se vende en `opens[step + 1]`
        # ---------------------------------------------------------------------
        if (step + 1) % 7 == 0 and closes[step] <= sl_p:
            exit_p = opens[step + 1] if (step + 1 < n_bars) else closes[step]
            exit_ts = timestamps[step + 1] if (step + 1 < n_bars) else timestamps[step]
            exit_reason = "DailyClose_SL_NextOpen"
            break

        # ---------------------------------------------------------------------
        # 3. AL TERMINAR LA BARRA `step`: Actualizar `peak_p` y calcular el nuevo `floor_p`
        #    que recién estará activo para la SIGUIENTE barra (`step + 1`)!
        # ---------------------------------------------------------------------
        if highs[step] > peak_p:
            peak_p = highs[step]
        max_rec = ((peak_p - entry_p) / entry_p) / shock_drop

        # ¿Ya se activó el Modelo 1B? Solo si `step >= meli_trigger_step`!
        # Antes de `meli_trigger_step`, solo existen reglas base sin `prob_1b`.
        model_1b_is_known = step >= meli_trigger_step

        if is_earnings_shock or (model_1b_is_known and prob_1b < 0.43):
            if max_rec >= 0.52:
                trail_dist = 0.35 * atr_d
                floor_p = max(floor_p or 0.0, peak_p * (1.0 - trail_dist), entry_p * (1.0 + 0.42 * shock_drop))
            elif max_rec >= 0.38:
                floor_p = max(floor_p or 0.0, entry_p * (1.0 + 0.22 * shock_drop))
        elif is_macro_or_sector and model_1b_is_known and prob_1b >= 0.52:
            if max_rec >= 0.55:
                trail_dist = 0.90 * atr_d
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
    hourly_path = PROJECT_ROOT / "data" / "processed" / "sp500_multisector_1hour.parquet"
    earn_path = PROJECT_ROOT / "data" / "processed" / "sp500_earnings_pit.parquet"
    macro_path = PROJECT_ROOT / "data" / "processed" / "macro_perception_1hour.parquet"

    print("[1/4] Auditando Timestamps en `sp500_earnings_pit.parquet`, `macro_perception_1hour.parquet` y velas 1H...")
    earn_df = pd.read_parquet(earn_path)
    earn_df["earnings_date"] = pd.to_datetime(earn_df["earnings_date"], utc=True)
    earn_hours = earn_df["earnings_date"].dt.hour.value_counts().head(5).to_dict()
    print(f"      * Distribución de horas UTC en `earnings_date`: {earn_hours}")

    df = pd.read_parquet(hourly_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values(["ticker", "timestamp"]).reset_index(drop=True)
    # IMPORTANTE EN PRODUCCIÓN:
    # Una vela horaria que en Yahoo Finance tiene timestamp `13:30 UTC` (09:30 NY) cubre de 09:30 a 10:30 NY.
    # Por lo tanto, el CIERRE de esa vela solo se conoce a las `14:30 UTC` (`bar_close_ts = timestamp + 1h`).
    df["bar_close_ts"] = df["timestamp"] + pd.Timedelta(hours=1)
    df["hour_key"] = df["timestamp"].dt.floor("h")

    macro_df = pd.read_parquet(macro_path)
    macro_df["hour_key"] = pd.to_datetime(macro_df["hour_key"], utc=True)
    macro_df = macro_df.set_index("hour_key")

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

    # Sin pre-filtro global `.max()`: filtramos causalmente barra por barra (`atr_daily_pct >= 2.2`)
    # Calcular la guía implícita con buffer estricto de +5 días después del reporte
    daily_closes = (
        stocks.groupby(["ticker", stocks["timestamp"].dt.floor("D")])["close"]
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

    earn_df = earn_df.sort_values(["ticker", "earnings_date"]).reset_index(drop=True)
    earn_df["eps_beat_4q_ratio"] = earn_df.groupby("ticker")["eps_beat"].transform(lambda s: s.rolling(4, min_periods=1).mean())
    earn_df["eps_surprise_4q_mean"] = earn_df.groupby("ticker")["eps_surprise_pct"].transform(lambda s: s.rolling(4, min_periods=1).mean().clip(-100, 200))
    # Buffer estricto de +24 horas sobre `earnings_date` para asegurar que un reporte After-Hours jamás se lea antes del día siguiente!
    earn_df["earnings_known_ts"] = earn_df["earnings_date"] + pd.Timedelta(hours=24)

    earn_sorted = earn_df.sort_values("earnings_date").reset_index(drop=True)
    daily_sorted = daily_closes[["ticker", "day_ts", "react_1d", "react_3d"]].sort_values("day_ts").reset_index(drop=True)
    merged_e = pd.merge_asof(earn_sorted, daily_sorted, left_on="earnings_date", right_on="day_ts", by="ticker", direction="backward")
    merged_e["last_earnings_reaction_1d"] = merged_e["react_1d"].clip(-0.40, 0.40).fillna(0.0)
    merged_e["last_earnings_reaction_3d"] = merged_e["react_3d"].clip(-0.50, 0.50).fillna(0.0)
    merged_e["guidance_disappointment_flag"] = ((merged_e["last_earnings_reaction_1d"] <= -0.03) | (merged_e["last_earnings_reaction_3d"] <= -0.04)).astype(float)
    # 5 días calendario completos después de `earnings_date` para garantizar que los 3 días hábiles posteriores ya cerraron!
    merged_e["reaction_known_ts"] = merged_e["earnings_date"] + pd.Timedelta(days=5)

    print("[2/4] Calculando features por acción y extrayendo candidatos de golpe de 3-5 días...")
    ticker_dfs = {}
    shock_candidates = []
    for tk, sub in stocks.groupby("ticker"):
        if sub["atr_daily_pct"].max() < 2.2:
            continue
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

        mf_mult = (((sub_feat["close"] - sub_feat["low"]) - (sub_feat["high"] - sub_feat["close"])) / rng).fillna(0.0)
        mf_vol = mf_mult * sub_feat["volume"]
        sub_feat["cmf_15d"] = (
            mf_vol.rolling(105, min_periods=21).sum() / sub_feat["volume"].rolling(105, min_periods=21).sum().replace(0, np.nan)
        ).fillna(0.0)

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

    shocks_raw = pd.concat(shock_candidates, ignore_index=True).sort_values("bar_close_ts").reset_index(drop=True)

    # Cruzar contra `earnings_known_ts` (earnings_date + 24h) usando `bar_close_ts`
    earn_basic = earn_df[["ticker", "earnings_known_ts", "eps_surprise_pct", "eps_positive", "eps_beat", "eps_beat_4q_ratio", "eps_surprise_4q_mean"]].sort_values("earnings_known_ts").reset_index(drop=True)
    shocks_raw = pd.merge_asof(shocks_raw, earn_basic, left_on="bar_close_ts", right_on="earnings_known_ts", by="ticker", direction="backward")
    shocks_raw["days_since_earnings"] = ((shocks_raw["bar_close_ts"] - shocks_raw["earnings_known_ts"]).dt.total_seconds() / 86400.0 + 1.0).clip(0, 120).fillna(45.0)
    shocks_raw["eps_surprise_pct"] = shocks_raw["eps_surprise_pct"].clip(-100, 200).fillna(0.0)
    shocks_raw["eps_positive"] = shocks_raw["eps_positive"].fillna(1.0)
    shocks_raw["eps_beat_4q_ratio"] = shocks_raw["eps_beat_4q_ratio"].fillna(0.5)
    shocks_raw["eps_surprise_4q_mean"] = shocks_raw["eps_surprise_4q_mean"].fillna(0.0)

    # Cruzar reacción de guía usando `reaction_known_ts` (earnings_date + 5d)
    earn_react = merged_e[["ticker", "reaction_known_ts", "last_earnings_reaction_1d", "last_earnings_reaction_3d", "guidance_disappointment_flag"]].sort_values("reaction_known_ts").reset_index(drop=True)
    shocks_raw = pd.merge_asof(shocks_raw, earn_react, left_on="bar_close_ts", right_on="reaction_known_ts", by="ticker", direction="backward")
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

    print("[3/4] Comparando Simulador Antiguo (con sesgo Intra-Barra) vs Simulador Productivo Estricto...")
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
        opens = fut["open"].to_numpy()
        highs = fut["high"].to_numpy()
        lows = fut["low"].to_numpy()
        closes = fut["close"].to_numpy()

        # Target de entrenamiento 100% causal (sin sesgo intra-barra)
        max_rec = 0.0
        floor_trail = None
        peak_t = entry_p
        exit_trail = closes[-1]
        for step in range(len(fut)):
            # 1. Evaluar candado vigente desde `step - 1`
            if floor_trail is not None and step > 0:
                if opens[step] <= floor_trail:
                    exit_trail = opens[step]
                    break
                elif lows[step] <= floor_trail:
                    exit_trail = floor_trail
                    break
            # 2. Evaluar SL al cierre diario -> ejecuta en `opens[step + 1]`
            if (step + 1) % 7 == 0 and closes[step] <= sl_p:
                exit_trail = opens[step + 1] if (step + 1 < len(fut)) else closes[step]
                break
            # 3. Al cierre de la barra `step`, actualizar `peak_t` y `floor_trail` para `step + 1`
            if highs[step] > peak_t:
                peak_t = highs[step]
            m_rec = ((peak_t - entry_p) / entry_p) / shock_drop
            if m_rec > max_rec:
                max_rec = m_rec
            if m_rec >= 0.50:
                floor_trail = max(floor_trail or 0.0, peak_t * (1.0 - 0.8 * atr_d))
        ret_trail_strict = ((exit_trail * 0.9995 - entry_p) / entry_p) * 100.0

        meli_trigger_step = 999
        for step in range(len(fut) - 7):
            if (step + 1) % 7 == 0 and closes[step] <= sl_p:
                break
            cur_h = highs[step]
            rec_here = ((cur_h - entry_p) / entry_p) / shock_drop
            if rec_here >= 0.45:
                meli_trigger_step = step
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
        row_dict["meli_trigger_step"] = meli_trigger_step
        row_dict["target_1a"] = 1 if (ret_trail_strict > 1.0 and max_rec >= 0.55) else 0
        valid_shocks.append(row_dict)

    df_1a = pd.DataFrame(valid_shocks).sort_values("timestamp").reset_index(drop=True)
    df_meli = pd.DataFrame(meli_snapshots).sort_values("timestamp").reset_index(drop=True)

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

    # PURGE GAP ESTRICTO DE 21 DÍAS ENTRE TRAIN Y TEST:
    # Ningún trade de Train puede estar aún abierto cuando arranca Test en `2025-11-17`
    test_start_ts = pd.Timestamp("2025-11-17", tz="UTC")
    train_end_ts = test_start_ts - pd.Timedelta(days=22)

    tr_1a = df_1a[df_1a["timestamp"] <= train_end_ts].copy().reset_index(drop=True)
    te_1a = df_1a[df_1a["timestamp"] >= test_start_ts].copy().reset_index(drop=True)

    clf_1a = lgb.LGBMClassifier(
        n_estimators=160, learning_rate=0.03, max_depth=4, num_leaves=12,
        min_child_samples=45, subsample=0.8, colsample_bytree=0.7,
        reg_alpha=2.0, reg_lambda=8.0, random_state=42, verbose=-1
    )
    clf_1a.fit(tr_1a[feats_stock_perception].fillna(0.0), tr_1a["target_1a"].values)
    te_1a["prob_1a"] = clf_1a.predict_proba(te_1a[feats_stock_perception].fillna(0.0))[:, 1]
    auc_1a_strict = roc_auc_score(te_1a["target_1a"].values, te_1a["prob_1a"].values)

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
    tr_m = df_meli[df_meli["timestamp"] <= train_end_ts].copy()
    te_m = df_meli[df_meli["timestamp"] >= test_start_ts].copy()
    clf_1b = lgb.LGBMClassifier(
        n_estimators=140, learning_rate=0.03, max_depth=3, num_leaves=8,
        min_child_samples=35, subsample=0.8, colsample_bytree=0.8,
        reg_alpha=2.0, reg_lambda=8.0, random_state=42, verbose=-1
    )
    clf_1b.fit(tr_m[feats_1b].fillna(0.0), tr_m["keep_running"].values)
    te_m["prob_1b"] = clf_1b.predict_proba(te_m[feats_1b].fillna(0.0))[:, 1]
    auc_1b_strict = roc_auc_score(te_m["keep_running"].values, te_m["prob_1b"].values)
    prob_1b_map = {int(r["event_id"]): float(r["prob_1b"]) for _, r in te_m.iterrows()}

    # Simular las 3 versiones de ejecución en el conjunto de prueba:
    # A) Simulador Antiguo (con lookahead intra-barra en el candado `exit_p = floor_p`)
    # B) Simulador Productivo Estricto Modo 1 (Orden Stop en Broker desde la barra siguiente, sufriendo gaps `opens[step] < floor_p`)
    # C) Simulador Productivo Estricto Modo 2 (Evaluado al cierre de la barra `closes[step] <= floor_p` y ejecutado en `opens[step + 1]`)
    rows_old = []
    rows_prod_stop = []
    rows_prod_close = []

    for _, row in te_1a.iterrows():
        ev_id = int(row["event_id"])
        tk = row["ticker"]
        i0 = int(row["bar_idx"])
        tdf = ticker_dfs[tk]
        fut_df = tdf.iloc[i0 + 1 : i0 + 106]
        p_1b = prob_1b_map.get(ev_id, 0.50)
        trig_step = int(row["meli_trigger_step"])

        # A) Antiguo
        ret_old, ts_old, rsn_old = simulate_adaptive_exit_by_shock_and_1b(
            fut_df, float(row["entry_p"]), float(row["sl_p"]), float(row["atr_d"]),
            float(row["shock_drop"]), int(row["shock_type_code"]), p_1b
        )
        d_old = row.to_dict()
        d_old.update({"prob_1b": p_1b, "net_ret_pct": ret_old, "exit_ts": ts_old, "exit_reason": rsn_old})
        rows_old.append(d_old)

        # B) Productivo Estricto: Orden Stop-Market en Broker (vigente desde step+1, sufre gaps)
        ret_ps, ts_ps, rsn_ps = simulate_strict_production_exit(
            fut_df, float(row["entry_p"]), float(row["sl_p"]), float(row["atr_d"]),
            float(row["shock_drop"]), int(row["shock_type_code"]), p_1b, trig_step,
            execution_mode="broker_stop_order_with_gap"
        )
        d_ps = row.to_dict()
        d_ps.update({"prob_1b": p_1b, "net_ret_pct": ret_ps, "exit_ts": ts_ps, "exit_reason": rsn_ps})
        rows_prod_stop.append(d_ps)

        # C) Productivo Estricto: Evaluado al cierre de barra, vendido en `Open[step+1]`
        ret_pc, ts_pc, rsn_pc = simulate_strict_production_exit(
            fut_df, float(row["entry_p"]), float(row["sl_p"]), float(row["atr_d"]),
            float(row["shock_drop"]), int(row["shock_type_code"]), p_1b, trig_step,
            execution_mode="close_triggered_next_open"
        )
        d_pc = row.to_dict()
        d_pc.update({"prob_1b": p_1b, "net_ret_pct": ret_pc, "exit_ts": ts_pc, "exit_reason": rsn_pc})
        rows_prod_close.append(d_pc)

    df_old = pd.DataFrame(rows_old).sort_values("entry_ts").reset_index(drop=True)
    df_ps = pd.DataFrame(rows_prod_stop).sort_values("entry_ts").reset_index(drop=True)
    df_pc = pd.DataFrame(rows_prod_close).sort_values("entry_ts").reset_index(drop=True)

    def apply_shield_mask(d: pd.DataFrame, p_thresh: float):
        m = (
            (d["prob_1a"] >= p_thresh)
            & (d["weak_volume_flag"] == 0)
            & ~((d["last_earnings_reaction_3d"] <= -0.05) & (d["cmf_15d"] <= -0.05))
            & ~((d["tnx_ret_3d"] >= 0.030) & (d["vix_slope_3h"] >= 0.04))
        )
        return d[m].copy()

    print("\n" + "=" * 124)
    print(f"4. VEREDICTO DE LA AUDITORÍA FORENSE EN LAS 586 EMPRESAS (ROC-AUC 1A Estricto = {auc_1a_strict:.4f} | 1B = {auc_1b_strict:.4f}):")
    print("   Comparando el Simulador Antiguo (con inflación intra-barra) vs Ejecución Productiva Estricta (sin trampa):")
    print("=" * 124)

    audit_cases = [
        ("1. SIMULADOR ANTIGUO (Con Inflación Intra-Barra en Candado, prob>=0.59, 5 Slots)", apply_shield_mask(df_old, 0.59), 5),
        ("2. PRODUCCIÓN ESTRICTA: Stop en Broker desde Barra t+1 + Gaps (prob>=0.59, 5 Slots)", apply_shield_mask(df_ps, 0.59), 5),
        ("3. PRODUCCIÓN ESTRICTA: Evaluado al Cierre -> Venta en Open[t+1] (prob>=0.59, 5 Slots)", apply_shield_mask(df_pc, 0.59), 5),
        ("4. PRODUCCIÓN ESTRICTA: Stop en Broker desde Barra t+1 + Gaps (prob>=0.61, 5 Slots)", apply_shield_mask(df_ps, 0.61), 5),
        ("5. PRODUCCIÓN ESTRICTA: Evaluado al Cierre -> Venta en Open[t+1] (prob>=0.61, 5 Slots)", apply_shield_mask(df_pc, 0.61), 5),
    ]

    a_rows = []
    for label, sub_c, slots in audit_cases:
        res = run_portfolio_simulation(sub_c, initial_capital=25000.0, max_slots=slots)
        a_rows.append({
            "Modo_de_Ejecución_Auditado": label,
            "Señales": len(sub_c),
            "Trades": res["trades_executed"],
            "WR_Real %": round(res["win_rate_pct"], 1),
            "Ret_Medio %": round(res["avg_trade_ret_pct"], 2),
            "PF": round(res["profit_factor"], 2),
            "Capital_Final": f"${res['final_capital']:,.0f}",
            "Retorno_Total": f"+{res['total_return_pct']:.1f}%",
            "Max_DD %": f"{res['max_dd_pct']:.2f}%",
            "Sharpe": round(res["sharpe"], 2),
        })
    print(pd.DataFrame(a_rows).to_string(index=False))
    print("=" * 124)
    print(f"Tiempo total de auditoría: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
