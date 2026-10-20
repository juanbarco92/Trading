"""
market_regime.py - TICK-04: Fuerza Relativa (RS) vs Benchmark (SPY) y Régimen de Mercado

Este módulo calcula los factores macroeconómicos y de fuerza relativa del universo:
1. Filtro de Régimen de Mercado (Macro Regime):
   - Tendencia del SPY sobre sus medias móviles exponenciales (EMA 50 y EMA 200 horarias).
   - Momentum del SPY en 1 día y 5 días.
   - Banderas de régimen alcista / bajista / sobre-extendido.
2. Fuerza Relativa (RS) vs. SPY (Mansfield Relative Strength & Excess Return):
   - Alpha / Rendimiento excedente vs SPY a 1 día (7 velas), 5 días (35 velas) y 20 días (140 velas).
   - Ratio RS = (Close_Stock / Close_SPY) normalizado frente a su media móvil.
   - Banderas de acumulación institucional relativa (la acción sube mientras el SPY cae o se mantiene).
"""

import pandas as pd
import numpy as np
from typing import Optional, List


def compute_benchmark_regime(spy_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula las variables de régimen macro sobre el benchmark (SPY).
    Requiere un DataFrame ordenado cronológicamente con columnas 'timestamp' y 'close'.
    """
    df = spy_df.sort_values(by="timestamp").copy()

    # 1. Medias Móviles Exponenciales (EMA 50: ~1 semana; EMA 200: ~1 mes)
    df["spy_ema_50"] = df["close"].ewm(span=50, min_periods=20).mean()
    df["spy_ema_200"] = df["close"].ewm(span=200, min_periods=50).mean()

    # 2. Distancia porcentual de SPY a sus EMAs (estacionaria)
    df["spy_dist_ema50_pct"] = ((df["close"] - df["spy_ema_50"]) / df["spy_ema_50"]) * 100.0
    df["spy_dist_ema200_pct"] = ((df["close"] - df["spy_ema_200"]) / df["spy_ema_200"]) * 100.0

    # 3. Rendimientos porcentuales de SPY en distintas ventanas
    # 7 barras = 1 día de mercado (6.5 horas); 35 barras = 1 semana hábil (5 días)
    df["spy_ret_1d"] = df["close"].pct_change(7) * 100.0
    df["spy_ret_5d"] = df["close"].pct_change(35) * 100.0

    # 4. Régimen de Mercado Binario
    # Mercado alcista sano: Precio > EMA 50 y EMA 50 > EMA 200
    df["is_market_bullish"] = (
        (df["close"] >= df["spy_ema_50"]) & (df["spy_ema_50"] >= df["spy_ema_200"])
    ).astype(int)

    # Mercado bajo riesgo / bajista: Precio < EMA 50
    df["is_market_bearish"] = (df["close"] < df["spy_ema_50"]).astype(int)

    # Columnas a conservar del benchmark para merge
    regime_cols = [
        "timestamp",
        "close",
        "spy_ema_50",
        "spy_ema_200",
        "spy_dist_ema50_pct",
        "spy_dist_ema200_pct",
        "spy_ret_1d",
        "spy_ret_5d",
        "is_market_bullish",
        "is_market_bearish",
    ]

    benchmark_regime = df[regime_cols].rename(columns={"close": "spy_close"})
    return benchmark_regime


def compute_relative_strength(
    df: pd.DataFrame,
    benchmark_regime: pd.DataFrame,
    benchmark_symbol: str = "SPY",
) -> pd.DataFrame:
    """
    Calcula la Fuerza Relativa (RS) y Rendimiento Excedente (Alpha) de cada acción frente al benchmark.
    """
    # 1. Unir los datos del benchmark con el dataframe del universo por timestamp
    merged = pd.merge(df, benchmark_regime, on="timestamp", how="left")

    # 2. Rendimientos porcentuales propios de la acción
    merged["stock_ret_1d"] = (
        merged.groupby("ticker")["close"].pct_change(7) * 100.0
    )
    merged["stock_ret_5d"] = (
        merged.groupby("ticker")["close"].pct_change(35) * 100.0
    )
    merged["stock_ret_20d"] = (
        merged.groupby("ticker")["close"].pct_change(140) * 100.0
    )

    # 3. Alpha / Rendimiento Excedente vs. SPY (Stock Return - SPY Return)
    merged["alpha_1d"] = merged["stock_ret_1d"] - merged["spy_ret_1d"]
    merged["alpha_5d"] = merged["stock_ret_5d"] - merged["spy_ret_5d"]

    # 4. Mansfield Relative Strength Ratio: RS = Stock_Price / SPY_Price
    merged["rs_ratio"] = merged["close"] / np.maximum(merged["spy_close"], 1e-4)

    # Normalizar el ratio RS contra su media móvil de 35 periodos (1 semana)
    def calc_rs_momentum(s: pd.Series) -> pd.Series:
        rs_base = s.rolling(window=35, min_periods=10).mean()
        return ((s / np.maximum(rs_base, 1e-4)) - 1.0) * 100.0

    merged["rs_momentum_5d"] = (
        merged.groupby("ticker")["rs_ratio"].transform(calc_rs_momentum).fillna(0.0)
    )

    # 5. Señales de Fuerza Relativa Clave:
    # - is_outperforming_spy: La acción tiene mayor momentum relativo que el SPY
    merged["is_outperforming_spy"] = (merged["rs_momentum_5d"] > 0).astype(int)

    # - divergence_bullish: La acción sube (o frena) mientras el SPY cae en 1D (Acumulación Institucional Oculta)
    merged["rs_divergence_bullish"] = (
        (merged["stock_ret_1d"] > 0) & (merged["spy_ret_1d"] < -0.3)
    ).astype(int)

    # Para el propio benchmark SPY, neutralizar alpha y banderas
    is_spy = merged["ticker"] == benchmark_symbol
    merged.loc[is_spy, "alpha_1d"] = 0.0
    merged.loc[is_spy, "alpha_5d"] = 0.0
    merged.loc[is_spy, "rs_momentum_5d"] = 0.0
    merged.loc[is_spy, "is_outperforming_spy"] = 0
    merged.loc[is_spy, "rs_divergence_bullish"] = 0

    # Rellenar valores iniciales de calentamiento (warm-up)
    fill_zero_cols = [
        "stock_ret_1d", "stock_ret_5d", "stock_ret_20d",
        "alpha_1d", "alpha_5d", "rs_momentum_5d",
        "spy_ret_1d", "spy_ret_5d", "spy_dist_ema50_pct", "spy_dist_ema200_pct"
    ]
    for col in fill_zero_cols:
        if col in merged.columns:
            merged[col] = merged[col].fillna(0.0)

    # Banderas binarias a 0 si tienen NaNs
    bin_cols = ["is_market_bullish", "is_market_bearish", "is_outperforming_spy", "rs_divergence_bullish"]
    for col in bin_cols:
        if col in merged.columns:
            merged[col] = merged[col].fillna(0).astype(int)

    return merged


def compute_market_regime_features(
    df: pd.DataFrame,
    benchmark_symbol: str = "SPY",
) -> pd.DataFrame:
    """
    Función de orquestación principal para calcular todas las variables de régimen y fuerza relativa.
    Busca automáticamente el benchmark SPY dentro del DataFrame.
    """
    if df.empty:
        return df

    req_cols = {"timestamp", "ticker", "close"}
    if not req_cols.issubset(df.columns):
        raise ValueError(f"El DataFrame debe contener las columnas: {req_cols}")

    # Verificar que el benchmark está presente en los datos
    if benchmark_symbol not in df["ticker"].unique():
        raise ValueError(f"El benchmark '{benchmark_symbol}' no se encuentra en el universo de datos.")

    # 1. Extraer los datos del benchmark
    spy_slice = df[df["ticker"] == benchmark_symbol].copy()
    benchmark_regime = compute_benchmark_regime(spy_slice)

    # 2. Calcular Fuerza Relativa cruzada para todo el universo
    result_df = compute_relative_strength(
        df,
        benchmark_regime,
        benchmark_symbol=benchmark_symbol,
    )

    return result_df
