"""
volatility_features.py - TICK-05: Volatilidad Dinámica y Distancia Estructural (ATR Normalizado)

Este módulo calcula las métricas de volatilidad y estructura de mercado estacionarias:
1. ATR Normalizado (% del Precio):
   - ATR(14) expresado como % de Close: (ATR / Close) * 100.
   - Permite comparar la volatilidad de activos de distinto valor nominal (MELI a $2000 vs NVDA a $100).
2. Régimen de Volatilidad (Ratio ATR 14 / ATR 100):
   - Mide compresión de volatilidad (preparación para ruptura) y expansión (tendencia activa).
3. Bandas de Bollinger Estacionarias (%B y Bandwidth):
   - %B: Posición relativa del precio entre las bandas (0 = banda inferior, 1 = banda superior).
   - Bandwidth: Ancho porcentual de las bandas para detectar "Squeezes" de baja volatilidad.
4. Distancia Dinámica a Soportes y Resistencias (en unidades de ATR):
   - Pivotes dinámicos (Rolling 20 y 60 barras) calculados con shift(1) para CERO lookahead bias.
   - Distancia al soporte/resistencia dividida entre el ATR(14) actual.
"""

import pandas as pd
import numpy as np
from typing import Optional


def compute_normalized_atr(df: pd.DataFrame, window_short: int = 14, window_long: int = 100) -> pd.DataFrame:
    """
    Calcula el ATR en unidades absolutas y en porcentaje del precio (estacionario),
    junto con el ratio de régimen de volatilidad (ATR_short / ATR_long).
    """
    df = df.copy()

    # True Range vectorizado
    h = df["high"]
    l = df["low"]
    c_prev = df["close"].shift(1).fillna(df["open"])

    tr1 = h - l
    tr2 = (h - c_prev).abs()
    tr3 = (l - c_prev).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    df["tr"] = tr
    df["atr14"] = tr.rolling(window=window_short, min_periods=5).mean().fillna(tr.mean())
    df["atr100"] = tr.rolling(window=window_long, min_periods=20).mean().fillna(df["atr14"])

    # 1. ATR Normalizado porcentual (estacionario)
    df["atr_pct"] = (df["atr14"] / np.maximum(df["close"], 1e-4)) * 100.0

    # 2. Ratio de Régimen de Volatilidad
    df["vol_regime"] = df["atr14"] / np.maximum(df["atr100"], 1e-4)

    # 3. Banderas de Régimen de Volatilidad
    # Compresión: Volatilidad comprimida (calma antes de la tormenta / potencial breakout)
    df["is_vol_compression"] = (df["vol_regime"] < 0.85).astype(int)
    # Expansión: Volatilidad expandida (movimiento fuerte en curso)
    df["is_vol_expansion"] = (df["vol_regime"] > 1.25).astype(int)

    return df


def compute_bollinger_features(df: pd.DataFrame, window: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    """
    Calcula las métricas derivadas de Bandas de Bollinger:
    - bb_percent_b: Posición del precio dentro del canal de volatilidad.
    - bb_bandwidth: Ancho porcentual del canal ((Upper - Lower) / SMA * 100).
    - bb_squeeze: Bandera cuando el ancho de banda cae al percentil más bajo.
    """
    df = df.copy()

    close_shift = df["close"]
    sma = close_shift.rolling(window=window, min_periods=5).mean().fillna(df["close"])
    std = close_shift.rolling(window=window, min_periods=5).std().fillna(1e-4)

    upper_band = sma + (num_std * std)
    lower_band = sma - (num_std * std)

    band_width_abs = np.maximum(upper_band - lower_band, 1e-4)

    # %B (0.0 = en banda inferior; 1.0 = en banda superior; >1.0 = rotura alcista)
    df["bb_percent_b"] = (df["close"] - lower_band) / band_width_abs
    df["bb_percent_b"] = df["bb_percent_b"].clip(-0.5, 1.5)

    # Bandwidth en porcentaje
    df["bb_bandwidth"] = (band_width_abs / np.maximum(sma, 1e-4)) * 100.0

    # Squeeze: Bandwidth por debajo de su media de 60 barras (compresión extrema)
    bw_rolling_mean = df["bb_bandwidth"].rolling(window=60, min_periods=10).mean()
    df["is_bb_squeeze"] = (df["bb_bandwidth"] < bw_rolling_mean * 0.75).astype(int)

    return df


def compute_support_resistance_distances(
    df: pd.DataFrame,
    window_short: int = 20,
    window_medium: int = 60,
) -> pd.DataFrame:
    """
    Calcula la distancia al soporte y resistencia más cercano en unidades de ATR.
    Utiliza shift(1) para garantizar CERO lookahead bias (el precio actual no define su propio pivote).
    """
    df = df.copy()

    atr = df.get("atr14", df["close"] * 0.015)
    safe_atr = np.maximum(atr, 1e-4)

    # Niveles de Pivotes Previos (shift 1 asegura exclusión de la vela en curso)
    prev_high = df["high"].shift(1)
    prev_low = df["low"].shift(1)

    # Resistencia y Soporte a corto plazo (20 barras ~ 3 días)
    res_20 = prev_high.rolling(window=window_short, min_periods=5).max().fillna(df["high"])
    sup_20 = prev_low.rolling(window=window_short, min_periods=5).min().fillna(df["low"])

    # Resistencia y Soporte a medio plazo (60 barras ~ 9 días)
    res_60 = prev_high.rolling(window=window_medium, min_periods=15).max().fillna(df["high"])
    sup_60 = prev_low.rolling(window=window_medium, min_periods=15).min().fillna(df["low"])

    # Distancias en múltiplos de ATR:
    # Si close está cerca de la resistencia, dist_to_res <= 0.5 (zona de breakout o rechazo)
    df["dist_to_res20_atr"] = ((res_20 - df["close"]) / safe_atr).clip(-3.0, 10.0)
    df["dist_to_sup20_atr"] = ((df["close"] - sup_20) / safe_atr).clip(-3.0, 10.0)

    df["dist_to_res60_atr"] = ((res_60 - df["close"]) / safe_atr).clip(-3.0, 15.0)
    df["dist_to_sup60_atr"] = ((df["close"] - sup_60) / safe_atr).clip(-3.0, 15.0)

    # Banderas de Proximidad a Niveles Clave (a menos de 0.5 ATR del nivel)
    df["near_resistance_breakout"] = (df["dist_to_res20_atr"] <= 0.5).astype(int)
    df["near_support_bounce"] = (df["dist_to_sup20_atr"] <= 0.5).astype(int)

    return df


def compute_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Función de orquestación principal para calcular todas las variables de volatilidad y estructura.
    Maneja automáticamente dataframes de un solo activo o de múltiples activos agrupados por 'ticker'.
    """
    if df.empty:
        return df

    req_cols = {"timestamp", "open", "high", "low", "close"}
    if not req_cols.issubset(df.columns):
        raise ValueError(f"El DataFrame debe contener las columnas: {req_cols}")

    if "ticker" in df.columns and df["ticker"].nunique() > 1:
        processed_groups = []
        for ticker, group in df.groupby("ticker", sort=False):
            group_sorted = group.sort_values(by="timestamp").reset_index(drop=True)
            d1 = compute_normalized_atr(group_sorted)
            d2 = compute_bollinger_features(d1)
            d3 = compute_support_resistance_distances(d2)
            processed_groups.append(d3)
        return pd.concat(processed_groups, ignore_index=True)
    else:
        sorted_df = df.sort_values(by="timestamp").reset_index(drop=True)
        d1 = compute_normalized_atr(sorted_df)
        d2 = compute_bollinger_features(d1)
        d3 = compute_support_resistance_distances(d2)
        return d3
