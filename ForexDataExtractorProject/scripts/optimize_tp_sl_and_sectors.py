"""
optimize_tp_sl_and_sectors.py - Estudio Empírico de TP, SL, Horizonte por Sector y Volatilidad en los 5,595 Golpes.

Responde con datos reales a:
1. ¿Cuánto tarda en recuperarse cada sector y nivel de volatilidad (3d, 5d, 10d, 15d, 20d)?
2. ¿Qué tipo de acciones/sectores vale la pena operar con el Motor 1 y cuáles descartar por baja elasticidad?
3. ¿Cuál es la combinación óptima de Take Profit (TP) y Stop Loss (SL) — Intradía vs Cierre Diario —
   calibrada en unidades de ATR Diario o % del Golpe recuperado?
"""

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from scripts.evaluate_multisector_motor1 import deduplicate_events


def main():
    hourly_path = PROJECT_ROOT / "data" / "processed" / "sp500_multisector_1hour.parquet"
    df = pd.read_parquet(hourly_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values(["ticker", "timestamp"]).reset_index(drop=True)

    spy = df[df["ticker"] == "SPY"].sort_values("timestamp").copy()
    spy["hour_key"] = spy["timestamp"].dt.floor("h")
    spy = spy.drop_duplicates(subset=["hour_key"]).set_index("hour_key")
    spy_open_next = spy["open"].shift(-1)
    for d, b in [(3, 21), (5, 35), (10, 70), (15, 105), (20, 140)]:
        spy[f"spy_fut_{d}d"] = (spy["close"].shift(-b) - spy_open_next) / spy_open_next * 100.0

    stocks = df[~df["ticker"].isin(["SPY", "QQQ"])].copy().reset_index(drop=True)
    stocks["hour_key"] = stocks["timestamp"].dt.floor("h")
    for d in [3, 5, 10, 15, 20]:
        stocks[f"spy_fut_{d}d"] = stocks["hour_key"].map(spy[f"spy_fut_{d}d"]).ffill()

    g = stocks.groupby("ticker", group_keys=False)
    stocks["ret_3d"] = g["close"].pct_change(21)
    stocks["ret_5d"] = g["close"].pct_change(35)
    stocks["vol_3d"] = g["ret_3d"].transform(lambda s: s.rolling(420, min_periods=70).std())
    stocks["vol_5d"] = g["ret_5d"].transform(lambda s: s.rolling(420, min_periods=70).std())
    stocks["z_3d"] = stocks["ret_3d"] / stocks["vol_3d"].replace(0, np.nan)
    stocks["z_5d"] = stocks["ret_5d"] / stocks["vol_5d"].replace(0, np.nan)

    # ATR Diario aproximado (rango de 7 barras = 1 día promediado en 14 días = 98 barras)
    high_1d = g["high"].transform(lambda s: s.rolling(7, min_periods=3).max())
    low_1d = g["low"].transform(lambda s: s.rolling(7, min_periods=3).min())
    stocks["atr_daily_pct"] = ((high_1d - low_1d) / stocks["close"] * 100.0).groupby(stocks["ticker"]).transform(
        lambda s: s.rolling(98, min_periods=21).mean()
    )

    # Vela 1H y RVOL
    rng_1h = (stocks["high"] - stocks["low"]).replace(0, np.nan)
    stocks["close_loc_1h"] = (stocks["close"] - stocks["low"]) / rng_1h
    stocks["lower_wick_ratio"] = (np.minimum(stocks["open"], stocks["close"]) - stocks["low"]) / rng_1h
    vol_sma50 = g["volume"].transform(lambda s: s.rolling(50, min_periods=10).mean())
    stocks["rvol_1h"] = (stocks["volume"] / vol_sma50.replace(0, np.nan)).clip(0, 15.0)

    # Desenlaces a 3, 5, 10, 15 y 20 días desde Open[t+1]
    open_next = g["open"].shift(-1)
    stocks["open_next"] = open_next
    for d, b in [(3, 21), (5, 35), (10, 70), (15, 105), (20, 140)]:
        stocks[f"fut_ret_{d}d"] = (g["close"].shift(-b) - open_next) / open_next * 100.0
        stocks[f"fut_alpha_{d}d"] = stocks[f"fut_ret_{d}d"] - stocks[f"spy_fut_{d}d"]

    # Filtrar Golpes de 3 a 5 días CON señal de absorción en vela de 1H (close_loc >= 0.55 o mecha >= 0.25)
    shock_mask = (
        ((stocks["z_3d"] <= -2.0) | (stocks["z_5d"] <= -2.0))
        & ((stocks["close_loc_1h"] >= 0.55) | (stocks["lower_wick_ratio"] >= 0.25))
    )
    shocks_raw = stocks[shock_mask].dropna(subset=["fut_ret_20d", "atr_daily_pct"]).copy()
    shocks = deduplicate_events(shocks_raw, "timestamp", min_gap_hours=48)

    print("=" * 105)
    print(f"1. CURVA DE VELOCIDAD DE RECUPERACIÓN POR SECTOR (3d, 5d, 10d, 15d, 20d) — {len(shocks):,} EVENTOS")
    print("=" * 105)
    sec_rows = []
    for sec, grp in shocks.groupby("sector"):
        sec_rows.append({
            "Sector": sec,
            "Eventos": len(grp),
            "ATR_Diario %": round(grp["atr_daily_pct"].mean(), 2),
            "Caída_3d %": round(grp["ret_3d"].mean() * 100, 2),
            "Ret_3d %": round(grp["fut_ret_3d"].mean(), 2),
            "Ret_5d %": round(grp["fut_ret_5d"].mean(), 2),
            "Ret_10d %": round(grp["fut_ret_10d"].mean(), 2),
            "Ret_15d %": round(grp["fut_ret_15d"].mean(), 2),
            "Ret_20d %": round(grp["fut_ret_20d"].mean(), 2),
            "Alpha_15d %": round(grp["fut_alpha_15d"].mean(), 2),
            "WR_15d %": round((grp["fut_ret_15d"] > 0).mean() * 100, 1),
        })
    df_sec = pd.DataFrame(sec_rows).sort_values("Ret_15d %", ascending=False)
    print(df_sec.to_string(index=False))

    # =========================================================================
    # 2. IMPACTO DE LA VOLATILIDAD DEL ACTIVO (ATR DIARIO %) Y DEL VOLUMEN (RVOL)
    # =========================================================================
    print("\n" + "=" * 105)
    print("2. ¿QUÉ TIPO DE ACCIONES CONVIENE OPERAR SEGÚN SU VOLATILIDAD DIARIA Y VOLUMEN DE ABSORCIÓN?")
    print("=" * 105)
    vol_buckets = [
        ("1. Acciones Muy Lentas (ATR Diario < 1.6%: KO, PG, VZ, DUK)", shocks["atr_daily_pct"] < 1.6),
        ("2. Acciones Elasticidad Media (ATR Diario 1.6% a 2.5%: JPM, UNH, CAT, AAPL)", (shocks["atr_daily_pct"] >= 1.6) & (shocks["atr_daily_pct"] < 2.5)),
        ("3. Acciones Dinámicas (ATR Diario 2.5% a 4.0%: META, AMZN, GS, LLY, MELI)", (shocks["atr_daily_pct"] >= 2.5) & (shocks["atr_daily_pct"] < 4.0)),
        ("4. Acciones Explosivas (ATR Diario >= 4.0%: NVDA, AMD, TSLA, COIN, PLTR)", shocks["atr_daily_pct"] >= 4.0),
        ("5. Dinámicas/Explosivas (ATR >= 2.0%) + Alto Volumen Absorción (RVOL_1h >= 1.5)", (shocks["atr_daily_pct"] >= 2.0) & (shocks["rvol_1h"] >= 1.5)),
    ]
    v_rows = []
    for label, m in vol_buckets:
        sub = shocks[m]
        if len(sub) == 0:
            continue
        v_rows.append({
            "Segmento de Volatilidad y Volumen": label,
            "Eventos": len(sub),
            "Caída_3d %": round(sub["ret_3d"].mean() * 100, 2),
            "Ret_5d %": round(sub["fut_ret_5d"].mean(), 2),
            "Ret_10d %": round(sub["fut_ret_10d"].mean(), 2),
            "Ret_15d %": round(sub["fut_ret_15d"].mean(), 2),
            "Ret_20d %": round(sub["fut_ret_20d"].mean(), 2),
            "Alpha_15d %": round(sub["fut_alpha_15d"].mean(), 2),
            "WR_15d %": round((sub["fut_ret_15d"] > 0).mean() * 100, 1),
        })
    print(pd.DataFrame(v_rows).to_string(index=False))

    # =========================================================================
    # 3. SIMULACIÓN DE REGLAS DE TAKE PROFIT (TP) Y STOP LOSS (SL) EN LAS VELAS FUTURAS
    # =========================================================================
    print("\n" + "=" * 105)
    print("3. CALIBRACIÓN DE TAKE PROFIT (TP) Y STOP LOSS (SL) EN UNIDADES DE ATR DIARIO:")
    print("   Compara Stop Intradiario (cualquier mecha te saca) vs Stop al Cierre del Día (ignora ruido intradía)")
    print("=" * 105)

    # Indexar el dataframe completo por ticker para simular barra a barra las siguientes 105 horas (15 días)
    ticker_groups = {tk: grp.sort_values("timestamp").reset_index(drop=True) for tk, grp in stocks.groupby("ticker")}

    # Filtrar activos con elasticidad suficiente (ATR Diario >= 1.8%)
    active_shocks = shocks[shocks["atr_daily_pct"] >= 1.8].copy()

    configs = [
        # (Nombre, k_tp_daily_atr, k_sl_daily_atr, horizon_bars, stop_on_daily_close_only)
        ("A. Stop Intradía Estrecho (TP=2.0x ATR_d, SL=1.0x ATR_d, H=10d)", 2.0, 1.0, 70, False),
        ("B. Stop Intradía Medio    (TP=2.5x ATR_d, SL=1.5x ATR_d, H=15d)", 2.5, 1.5, 105, False),
        ("C. Stop Intradía Amplio   (TP=3.0x ATR_d, SL=2.0x ATR_d, H=15d)", 3.0, 2.0, 105, False),
        ("D. Stop al CIERRE DIARIO  (TP=2.5x ATR_d, SL=1.5x ATR_d al cierre 7H, H=15d)", 2.5, 1.5, 105, True),
        ("E. Stop al CIERRE DIARIO  (TP=3.0x ATR_d, SL=1.8x ATR_d al cierre 7H, H=15d)", 3.0, 1.8, 105, True),
        ("F. Recuperar 75% de la Caída Sufrida + SL=1.5x ATR_d al Cierre (H=15d)", "75_SHOCK", 1.5, 105, True),
    ]

    cfg_rows = []
    for label, k_tp, k_sl, h_bars, close_only_sl in configs:
        rets = []
        outcomes = []  # 'TP', 'SL', 'TIME'
        bars_held = []

        for _, row in active_shocks.iterrows():
            tk = row["ticker"]
            ts = row["timestamp"]
            atr_d = row["atr_daily_pct"] / 100.0
            shock_drop = abs(min(row["ret_3d"], row["ret_5d"], -0.03))

            tdf = ticker_groups[tk]
            idx_arr = tdf.index[tdf["timestamp"] == ts].to_numpy()
            if len(idx_arr) == 0:
                continue
            i0 = idx_arr[0]
            if i0 + h_bars >= len(tdf):
                continue

            entry_p = tdf.loc[i0 + 1, "open"] * 1.0005  # Fricción entrada 5 bps
            if k_tp == "75_SHOCK":
                tp_p = entry_p * (1.0 + max(0.75 * shock_drop, 1.5 * atr_d))
            else:
                tp_p = entry_p * (1.0 + k_tp * atr_d)
            sl_p = entry_p * (1.0 - k_sl * atr_d)

            future_slice = tdf.iloc[i0 + 1 : i0 + 1 + h_bars]
            highs = future_slice["high"].to_numpy()
            lows = future_slice["low"].to_numpy()
            closes = future_slice["close"].to_numpy()

            exit_p = closes[-1]
            outcome = "TIME"
            held = h_bars

            for step in range(len(future_slice)):
                # En Stop al cierre diario, solo comprobamos el SL cada 7 barras (final del día)
                if not close_only_sl:
                    if lows[step] <= sl_p:
                        exit_p = sl_p
                        outcome = "SL"
                        held = step + 1
                        break
                else:
                    if (step + 1) % 7 == 0 and closes[step] <= sl_p:
                        exit_p = closes[step]
                        outcome = "SL"
                        held = step + 1
                        break

                # El Take Profit es una orden Limit en el broker (se ejecuta apenas el precio toca tp_p)
                if highs[step] >= tp_p:
                    exit_p = tp_p
                    outcome = "TP"
                    held = step + 1
                    break

            net_ret = ((exit_p * 0.9995 - entry_p) / entry_p) * 100.0
            rets.append(net_ret)
            outcomes.append(outcome)
            bars_held.append(held / 7.0)  # Días

        rets = np.array(rets)
        outcomes = np.array(outcomes)
        wins = rets[rets > 0]
        losses = rets[rets <= 0]
        pf = wins.sum() / abs(losses.sum()) if len(losses) > 0 and losses.sum() != 0 else 99.0

        cfg_rows.append({
            "Esquema de TP y SL": label,
            "TP_Hit %": round((outcomes == "TP").mean() * 100, 1),
            "SL_Hit %": round((outcomes == "SL").mean() * 100, 1),
            "Time_Exit %": round((outcomes == "TIME").mean() * 100, 1),
            "Win_Rate_Neto %": round((rets > 0).mean() * 100, 1),
            "Ret_Medio_Neto %": round(rets.mean(), 2),
            "Profit_Factor": round(pf, 2),
            "Días_Medios_En_Trade": round(np.mean(bars_held), 1),
        })

    print(pd.DataFrame(cfg_rows).to_string(index=False))
    print("=" * 105)


if __name__ == "__main__":
    main()
