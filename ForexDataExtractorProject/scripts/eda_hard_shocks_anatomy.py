"""
eda_hard_shocks_anatomy.py - Estudio Exploratorio Empírico de "Golpes Duros" (Hard Shocks).

Objetivo:
Descubrir con datos reales (sin suposiciones a priori) la anatomía exacta de las caídas
fuertes en acciones y cuándo se recuperan a mediano plazo (3, 5, 10 y 15 días),
evaluando:
1. Escala y velocidad del golpe (Intradía 3H, 1 Día, 3 Días, 5 Días).
2. Golpe Propio (Idiosincrático) vs Pánico General del Mercado (SPY).
3. Huella Institucional en Velas de 1 Hora (Cierre en mínimos vs Absorción con RVOL vs Giro t+1).
4. Contexto Estructural Previo (Tendencia alcista de fondo vs Tendencia bajista).
5. Excursión Adversa (MAE) vs Favorable (MFE) para diseñar el Stop Loss óptimo.
6. Estabilidad por Régimen Anual (2022 Bear Market vs 2023-2024 Bull vs 2025-2026).
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

pd.set_option("display.max_columns", 20)
pd.set_option("display.width", 140)


def main():
    print("=" * 95)
    print("ESTUDIO EXPLORATORIO (EDA): ANATOMÍA DE 'GOLPES DUROS' Y RECUPERACIÓN A MEDIANO PLAZO")
    print("=" * 95)

    # 1. Cargar datos horarios maestros
    data_path = PROJECT_ROOT / "data" / "processed" / "universe_1hour.parquet"
    df = pd.read_parquet(data_path)
    time_col = "timestamp" if "timestamp" in df.columns else "datetime"
    df[time_col] = pd.to_datetime(df[time_col], utc=True)
    df = df.sort_values(["ticker", time_col]).reset_index(drop=True)

    tickers = sorted(df["ticker"].unique())
    print(f"Universo cargado: {len(df):,} velas de 1H | {len(tickers)} activos: {tickers}")
    print(f"Rango temporal: {df[time_col].min().date()} a {df[time_col].max().date()}\n")

    # 2. Extraer serie del SPY para medir qué hacía el mercado general en cada hora y día
    spy = df[df["ticker"] == "SPY"].sort_values(time_col).set_index(time_col)
    spy_ret_3h = spy["close"].pct_change(3)
    spy_ret_1d = spy["close"].pct_change(7)    # ~1 día hábil (7 velas de 1H)
    spy_ret_3d = spy["close"].pct_change(21)   # ~3 días hábiles
    spy_ret_5d = spy["close"].pct_change(35)   # ~5 días hábiles

    # Retornos futuros del SPY (desde Open[t+1] para comparar Alpha limpio sin lookahead)
    spy_open_next = spy["open"].shift(-1)
    spy_fut_3d = (spy["close"].shift(-21) - spy_open_next) / spy_open_next * 100.0
    spy_fut_5d = (spy["close"].shift(-35) - spy_open_next) / spy_open_next * 100.0
    spy_fut_10d = (spy["close"].shift(-70) - spy_open_next) / spy_open_next * 100.0
    spy_fut_15d = (spy["close"].shift(-105) - spy_open_next) / spy_open_next * 100.0

    # Filtrar ETFs de índice para analizar solo acciones individuales
    stocks_df = df[~df["ticker"].isin(["SPY", "QQQ"])].copy().reset_index(drop=True)

    # Mapear variables del SPY por timestamp exacto
    stocks_df["spy_ret_3h"] = stocks_df[time_col].map(spy_ret_3h)
    stocks_df["spy_ret_1d"] = stocks_df[time_col].map(spy_ret_1d)
    stocks_df["spy_ret_3d"] = stocks_df[time_col].map(spy_ret_3d)
    stocks_df["spy_ret_5d"] = stocks_df[time_col].map(spy_ret_5d)

    stocks_df["spy_fut_3d"] = stocks_df[time_col].map(spy_fut_3d)
    stocks_df["spy_fut_5d"] = stocks_df[time_col].map(spy_fut_5d)
    stocks_df["spy_fut_10d"] = stocks_df[time_col].map(spy_fut_10d)
    stocks_df["spy_fut_15d"] = stocks_df[time_col].map(spy_fut_15d)

    # 3. Calcular métricas causales por acción (en velas 1H y agregados multimarco)
    g = stocks_df.groupby("ticker", group_keys=False)

    # Retornos pasados de la acción
    stocks_df["ret_1h"] = g["close"].pct_change(1)
    stocks_df["ret_3h"] = g["close"].pct_change(3)
    stocks_df["ret_1d"] = g["close"].pct_change(7)
    stocks_df["ret_3d"] = g["close"].pct_change(21)
    stocks_df["ret_5d"] = g["close"].pct_change(35)
    stocks_df["ret_20d"] = g["close"].pct_change(140)
    stocks_df["ret_60d"] = g["close"].pct_change(420)

    # Volatilidad histórica causal (desviación estándar rodante de retornos de 1 día = 7 barras, ventana 60 días = 420 barras)
    stocks_df["vol_3h"] = g["ret_3h"].transform(lambda s: s.rolling(420, min_periods=70).std())
    stocks_df["vol_1d"] = g["ret_1d"].transform(lambda s: s.rolling(420, min_periods=70).std())
    stocks_df["vol_3d"] = g["ret_3d"].transform(lambda s: s.rolling(420, min_periods=70).std())
    stocks_df["vol_5d"] = g["ret_5d"].transform(lambda s: s.rolling(420, min_periods=70).std())

    # Z-Scores del golpe (cuántas desviaciones estándar cayó la acción respecto a su propia normalidad)
    stocks_df["z_3h"] = stocks_df["ret_3h"] / stocks_df["vol_3h"].replace(0, np.nan)
    stocks_df["z_1d"] = stocks_df["ret_1d"] / stocks_df["vol_1d"].replace(0, np.nan)
    stocks_df["z_3d"] = stocks_df["ret_3d"] / stocks_df["vol_3d"].replace(0, np.nan)
    stocks_df["z_5d"] = stocks_df["ret_5d"] / stocks_df["vol_5d"].replace(0, np.nan)

    # Choque Idiosincrático (restando el retorno del SPY ajustado por ratio de volatilidad / Beta aproximada)
    spy_vol_1d = stocks_df["spy_ret_1d"].rolling(420, min_periods=70).std().bfill()
    beta_approx = (stocks_df["vol_1d"] / spy_vol_1d.replace(0, np.nan)).clip(0.5, 3.0).fillna(1.2)
    stocks_df["idiosyncratic_ret_1d"] = stocks_df["ret_1d"] - (beta_approx * stocks_df["spy_ret_1d"])
    stocks_df["z_idio_1d"] = stocks_df["idiosyncratic_ret_1d"] / stocks_df["vol_1d"].replace(0, np.nan)

    # RVOL Horario y Diario causal
    stocks_df["vol_sma50_1h"] = g["volume"].transform(lambda s: s.rolling(50, min_periods=10).mean())
    stocks_df["rvol_1h"] = (stocks_df["volume"] / stocks_df["vol_sma50_1h"].replace(0, np.nan)).clip(0, 15.0)

    stocks_df["vol_1d_sum"] = g["volume"].transform(lambda s: s.rolling(7, min_periods=3).sum())
    stocks_df["vol_1d_avg"] = g["vol_1d_sum"].transform(lambda s: s.rolling(350, min_periods=70).mean())
    stocks_df["rvol_1d"] = (stocks_df["vol_1d_sum"] / stocks_df["vol_1d_avg"].replace(0, np.nan)).clip(0, 15.0)

    # Anatomía de la vela 1H y de la ventana de 1 día (7 horas)
    rng_1h = (stocks_df["high"] - stocks_df["low"]).replace(0, np.nan)
    stocks_df["close_loc_1h"] = (stocks_df["close"] - stocks_df["low"]) / rng_1h
    stocks_df["lower_wick_1h"] = (np.minimum(stocks_df["open"], stocks_df["close"]) - stocks_df["low"]) / rng_1h

    high_1d = g["high"].transform(lambda s: s.rolling(7, min_periods=3).max())
    low_1d = g["low"].transform(lambda s: s.rolling(7, min_periods=3).min())
    rng_1d = (high_1d - low_1d).replace(0, np.nan)
    stocks_df["close_loc_1d"] = (stocks_df["close"] - low_1d) / rng_1d

    # Tendencia estructural previa (Media móvil de 100 días = ~700 barras 1H)
    stocks_df["sma_100d"] = g["close"].transform(lambda s: s.rolling(700, min_periods=200).mean())
    stocks_df["above_sma100d"] = (stocks_df["close"] > stocks_df["sma_100d"]).astype(int)
    # ¿Estaba en tendencia alcista ANTES del golpe (hace 5 días = 35 barras)?
    stocks_df["was_uptrend_pre_shock"] = g["above_sma100d"].shift(35).fillna(0)

    # Confirmación de giro en la vela actual (vela verde tras caída previa)
    stocks_df["is_green_reversal_bar"] = ((stocks_df["close"] > stocks_df["open"]) & (stocks_df["close_loc_1h"] >= 0.60)).astype(int)

    # 4. Calcular Desenlaces Futuros (100% Causal: Entrada en Open[t+1])
    open_next = g["open"].shift(-1)
    stocks_df["fut_ret_3d"] = (g["close"].shift(-21) - open_next) / open_next * 100.0
    stocks_df["fut_ret_5d"] = (g["close"].shift(-35) - open_next) / open_next * 100.0
    stocks_df["fut_ret_10d"] = (g["close"].shift(-70) - open_next) / open_next * 100.0
    stocks_df["fut_ret_15d"] = (g["close"].shift(-105) - open_next) / open_next * 100.0

    # Alpha Excedente Futuro (Retorno de la Acción - Retorno del SPY en el mismo período)
    stocks_df["fut_alpha_3d"] = stocks_df["fut_ret_3d"] - stocks_df["spy_fut_3d"]
    stocks_df["fut_alpha_5d"] = stocks_df["fut_ret_5d"] - stocks_df["spy_fut_5d"]
    stocks_df["fut_alpha_10d"] = stocks_df["fut_ret_10d"] - stocks_df["spy_fut_10d"]
    stocks_df["fut_alpha_15d"] = stocks_df["fut_ret_15d"] - stocks_df["spy_fut_15d"]

    # Excursión Máxima Adversa (MAE) y Favorable (MFE) en los próximos 10 días (70 barras)
    # Usando rolling inverso sobre las siguientes 70 barras
    fut_max_high_10d = g["high"].transform(lambda s: s.shift(-1).rolling(70, min_periods=35).max().shift(-69))
    fut_min_low_10d = g["low"].transform(lambda s: s.shift(-1).rolling(70, min_periods=35).min().shift(-69))
    stocks_df["mfe_10d_pct"] = (fut_max_high_10d - open_next) / open_next * 100.0
    stocks_df["mae_10d_pct"] = (fut_min_low_10d - open_next) / open_next * 100.0

    stocks_df["year"] = stocks_df[time_col].dt.year
    stocks_df["hour_utc"] = stocks_df[time_col].dt.hour

    valid = stocks_df.dropna(subset=["z_1d", "fut_ret_10d", "fut_alpha_10d"]).copy()

    # =========================================================================
    # EXPERIMENTO 1: ¿QUÉ MAGNITUD Y VENTANA DE GOLPE REBOTA MEJOR?
    # =========================================================================
    print("=" * 95)
    print("1. MAGNITUD Y VELOCIDAD DEL GOLPE (COMPARATIVA CONTRA LÍNEA BASE)")
    print("=" * 95)
    base_ret_10d = valid["fut_ret_10d"].mean()
    base_alpha_10d = valid["fut_alpha_10d"].mean()
    base_wr_10d = (valid["fut_ret_10d"] > 0).mean() * 100.0
    base_alpha_wr = (valid["fut_alpha_10d"] > 0).mean() * 100.0
    print(f"Línea Base (Cualquier vela al azar, N={len(valid):,}):")
    print(f"  * Retorno 10d: {base_ret_10d:+.2f}% (WR: {base_wr_10d:.1f}%) | Alpha vs SPY 10d: {base_alpha_10d:+.2f}% (Alpha WR: {base_alpha_wr:.1f}%)\n")

    shock_defs = [
        ("Caída Rápida 3H <= -2.0 SD", valid["z_3h"] <= -2.0),
        ("Caída Rápida 3H <= -3.0 SD", valid["z_3h"] <= -3.0),
        ("Caída Rápida 3H <= -4.0 SD", valid["z_3h"] <= -4.0),
        ("Caída 1 Día (7H) <= -1.5 SD", valid["z_1d"] <= -1.5),
        ("Caída 1 Día (7H) <= -2.0 SD", valid["z_1d"] <= -2.0),
        ("Caída 1 Día (7H) <= -2.5 SD", valid["z_1d"] <= -2.5),
        ("Caída 1 Día (7H) <= -3.0 SD", valid["z_1d"] <= -3.0),
        ("Caída 1 Día (7H) <= -4.0 SD", valid["z_1d"] <= -4.0),
        ("Caída 3 Días (21H) <= -2.0 SD", valid["z_3d"] <= -2.0),
        ("Caída 3 Días (21H) <= -3.0 SD", valid["z_3d"] <= -3.0),
        ("Caída 5 Días (35H) <= -2.0 SD", valid["z_5d"] <= -2.0),
        ("Caída 5 Días (35H) <= -3.0 SD", valid["z_5d"] <= -3.0),
    ]

    rows_1 = []
    for label, mask in shock_defs:
        sub = valid[mask]
        if len(sub) == 0:
            continue
        rows_1.append({
            "Definición del Golpe": label,
            "Eventos": len(sub),
            "Caída Media %": round(sub["ret_1d"].mean() * 100, 2),
            "Ret 5d %": round(sub["fut_ret_5d"].mean(), 2),
            "Ret 10d %": round(sub["fut_ret_10d"].mean(), 2),
            "Ret 15d %": round(sub["fut_ret_15d"].mean(), 2),
            "Alpha 10d vs SPY": round(sub["fut_alpha_10d"].mean(), 2),
            "WR 10d %": round((sub["fut_ret_10d"] > 0).mean() * 100, 1),
            "Alpha>0 %": round((sub["fut_alpha_10d"] > 0).mean() * 100, 1),
            "MFE 10d %": round(sub["mfe_10d_pct"].mean(), 2),
            "MAE 10d %": round(sub["mae_10d_pct"].mean(), 2),
        })
    print(pd.DataFrame(rows_1).to_string(index=False))

    # =========================================================================
    # EXPERIMENTO 2: ¿GOLPE PROPIO (IDIOSINCRÁTICO) VS GOLPE DEL MERCADO (SPY)?
    # =========================================================================
    print("\n" + "=" * 95)
    print("2. GOLPE PROPIO DE LA ACCIÓN VS CAÍDA ARRASTRADA POR EL MERCADO (En Golpes z_1d <= -2.0 SD)")
    print("=" * 95)
    shocks_2 = valid[valid["z_1d"] <= -2.0].copy()

    market_regimes = [
        ("A. Golpe 100% Propio (SPY alcista/estable > 0%)", shocks_2["spy_ret_1d"] > 0.0),
        ("B. Golpe Propio con SPY leve (-0.8% a 0%)", (shocks_2["spy_ret_1d"] <= 0.0) & (shocks_2["spy_ret_1d"] > -0.008)),
        ("C. Caída con Mercado Débil (SPY -1.8% a -0.8%)", (shocks_2["spy_ret_1d"] <= -0.008) & (shocks_2["spy_ret_1d"] > -0.018)),
        ("D. Pánico Sistémico / Crash (SPY <= -1.8%)", shocks_2["spy_ret_1d"] <= -0.018),
    ]

    rows_2 = []
    for label, mask in market_regimes:
        sub = shocks_2[mask]
        if len(sub) == 0:
            continue
        rows_2.append({
            "Escenario de Mercado": label,
            "Eventos": len(sub),
            "Caída Acción %": round(sub["ret_1d"].mean() * 100, 2),
            "Caída SPY %": round(sub["spy_ret_1d"].mean() * 100, 2),
            "Ret 5d %": round(sub["fut_ret_5d"].mean(), 2),
            "Ret 10d %": round(sub["fut_ret_10d"].mean(), 2),
            "Alpha 10d %": round(sub["fut_alpha_10d"].mean(), 2),
            "WR 10d %": round((sub["fut_ret_10d"] > 0).mean() * 100, 1),
            "Alpha>0 %": round((sub["fut_alpha_10d"] > 0).mean() * 100, 1),
            "MAE 10d (Dolor) %": round(sub["mae_10d_pct"].mean(), 2),
        })
    print(pd.DataFrame(rows_2).to_string(index=False))

    # =========================================================================
    # EXPERIMENTO 3: EL PODER DE LA VELA POR HORAS (1H) — ¿CÓMO DISTINGUIR EL PISO?
    # =========================================================================
    print("\n" + "=" * 95)
    print("3. MICROESTRUCTURA HORARIA (1H) EN EL GOLPE (En Golpes z_1d <= -2.0 SD)")
    print("=" * 95)

    hourly_filters = [
        ("1. Cuchillo Cayendo (Vela roja cerrando en mínimo < 20%)", (shocks_2["close"] < shocks_2["open"]) & (shocks_2["close_loc_1h"] < 0.20)),
        ("2. Vela de Absorción 1H (Mecha inf >= 30% + RVOL >= 1.5)", (shocks_2["lower_wick_1h"] >= 0.30) & (shocks_2["rvol_1h"] >= 1.5)),
        ("3. Vela Verde de Giro 1H (Cierre > Open y en tercio sup >= 65%)", (shocks_2["close"] > shocks_2["open"]) & (shocks_2["close_loc_1h"] >= 0.65)),
        ("4. Giro 1H + Alto Volumen (Vela verde >= 65% + RVOL_1d >= 1.8)", (shocks_2["close"] > shocks_2["open"]) & (shocks_2["close_loc_1h"] >= 0.65) & (shocks_2["rvol_1d"] >= 1.8)),
        ("5. Absorción de Tarde Institucional (17-20 UTC + Vela Verde)", (shocks_2["hour_utc"].isin([17, 18, 19, 20])) & (shocks_2["close"] > shocks_2["open"]) & (shocks_2["close_loc_1h"] >= 0.60)),
        ("6. Tendencia Alcista Previa + Giro 1H + RVOL >= 1.8", (shocks_2["was_uptrend_pre_shock"] == 1) & (shocks_2["close"] > shocks_2["open"]) & (shocks_2["rvol_1d"] >= 1.8)),
        ("7. Tendencia Bajista Crónica + Cuchillo Cayendo", (shocks_2["was_uptrend_pre_shock"] == 0) & (shocks_2["close"] < shocks_2["open"]) & (shocks_2["close_loc_1h"] < 0.25)),
    ]

    rows_3 = []
    for label, mask in hourly_filters:
        sub = shocks_2[mask]
        if len(sub) == 0:
            continue
        rows_3.append({
            "Comportamiento en Velas 1H": label,
            "Eventos": len(sub),
            "Ret 5d %": round(sub["fut_ret_5d"].mean(), 2),
            "Ret 10d %": round(sub["fut_ret_10d"].mean(), 2),
            "Ret 15d %": round(sub["fut_ret_15d"].mean(), 2),
            "Alpha 10d %": round(sub["fut_alpha_10d"].mean(), 2),
            "Alpha 15d %": round(sub["fut_alpha_15d"].mean(), 2),
            "WR 10d %": round((sub["fut_ret_10d"] > 0).mean() * 100, 1),
            "MFE 10d %": round(sub["mfe_10d_pct"].mean(), 2),
            "MAE 10d %": round(sub["mae_10d_pct"].mean(), 2),
        })
    print(pd.DataFrame(rows_3).to_string(index=False))

    # =========================================================================
    # EXPERIMENTO 4: COMPORTAMIENTO POR AÑO (PRUEBA ANTI-CRISIS 2022 VS BULL IA 2023-2026)
    # =========================================================================
    print("\n" + "=" * 95)
    print("4. ESTABILIDAD ANUAL (2022 BEAR MARKET vs 2023-2026) EN GOLPES CON GIRO 1H + TENDENCIA PREVIA")
    print("=" * 95)
    best_setup = shocks_2[
        (shocks_2["was_uptrend_pre_shock"] == 1) &
        (shocks_2["close"] > shocks_2["open"]) &
        (shocks_2["rvol_1d"] >= 1.5)
    ]

    rows_4 = []
    for yr, grp in best_setup.groupby("year"):
        rows_4.append({
            "Año": yr,
            "Eventos": len(grp),
            "Retorno Acción 10d %": round(grp["fut_ret_10d"].mean(), 2),
            "Retorno SPY 10d %": round(grp["spy_fut_10d"].mean(), 2),
            "Alpha Puro 10d %": round(grp["fut_alpha_10d"].mean(), 2),
            "Win Rate Absoluto %": round((grp["fut_ret_10d"] > 0).mean() * 100, 1),
            "Win Rate Alpha>0 %": round((grp["fut_alpha_10d"] > 0).mean() * 100, 1),
            "MFE 10d %": round(grp["mfe_10d_pct"].mean(), 2),
            "MAE 10d %": round(grp["mae_10d_pct"].mean(), 2),
        })
    print(pd.DataFrame(rows_4).to_string(index=False))

    # =========================================================================
    # EXPERIMENTO 5: DESGLOSE POR ACTIVO (TECH vs FINANZAS vs SALUD vs COMERCIO)
    # =========================================================================
    print("\n" + "=" * 95)
    print("5. COMPORTAMIENTO POR ACTIVO EN GOLPES CON ABSORCIÓN (¿ES SOLO BURBUJA IA O ES UNIVERSAL?)")
    print("=" * 95)
    rows_5 = []
    for tk, grp in best_setup.groupby("ticker"):
        if len(grp) < 5:
            continue
        rows_5.append({
            "Ticker": tk,
            "Eventos": len(grp),
            "Caída Media %": round(grp["ret_1d"].mean() * 100, 2),
            "Ret 10d %": round(grp["fut_ret_10d"].mean(), 2),
            "Alpha 10d %": round(grp["fut_alpha_10d"].mean(), 2),
            "WR 10d %": round((grp["fut_ret_10d"] > 0).mean() * 100, 1),
            "Alpha>0 %": round((grp["fut_alpha_10d"] > 0).mean() * 100, 1),
            "MFE 10d %": round(grp["mfe_10d_pct"].mean(), 2),
            "MAE 10d %": round(grp["mae_10d_pct"].mean(), 2),
        })
    df_5 = pd.DataFrame(rows_5).sort_values("Alpha 10d %", ascending=False)
    print(df_5.to_string(index=False))
    print("=" * 95)


if __name__ == "__main__":
    main()
