"""
volume_features.py - TICK-03: Módulo de Volumen Institucional & Detección de Anomalías

Este módulo calcula las métricas de volumen institucional más predictivas para acciones:
1. RVOL (Relative Volume): Ratio del volumen actual vs. la media histórica de esa misma hora del día.
2. Intraday VWAP: Precio medio ponderado por volumen del día y distancia porcentual al VWAP.
3. Volume Z-Score: Puntuación estandarizada de volumen relativo a los últimos 20 periodos.
4. Velas de Absorción y Clímax Institucional: Detección cuantitativa de capitulación con mechas de rechazo.
"""

import pandas as pd
import numpy as np
from typing import Optional


def compute_vwap(df: pd.DataFrame, tz_market: str = "America/New_York") -> pd.DataFrame:
    """
    Calcula el VWAP intradía acumulado por día de mercado.
    El VWAP se reinicia cada nuevo día de negociación (a la apertura de Nueva York).

    Fórmula:
        Typical Price = (High + Low + Close) / 3
        VWAP = sum(Typical Price * Volume) / sum(Volume)
    """
    df = df.copy()

    # Convertir timestamp a la zona horaria del mercado para extraer la fecha exacta de sesión
    ts_local = df["timestamp"].dt.tz_convert(tz_market)
    session_date = ts_local.dt.date

    typical_price = (df["high"] + df["low"] + df["close"]) / 3.0
    pv = typical_price * df["volume"]

    # Calcular sumas acumuladas por sesión diaria
    cum_pv = pv.groupby(session_date).cumsum()
    cum_vol = df["volume"].groupby(session_date).cumsum()

    # Evitar división por cero
    df["vwap"] = cum_pv / np.maximum(cum_vol, 1e-8)

    # Distancia porcentual al VWAP (estacionaria)
    df["vwap_dist_pct"] = ((df["close"] - df["vwap"]) / df["vwap"]) * 100.0
    df["is_above_vwap"] = (df["close"] >= df["vwap"]).astype(int)

    return df


def compute_rvol(df: pd.DataFrame, window: int = 20, tz_market: str = "America/New_York") -> pd.DataFrame:
    """
    Calcula el RVOL (Relative Volume) normalizado por hora del día.
    Compara el volumen de las 10:00 AM con el promedio histórico de las 10:00 AM de los últimos `window` días.

    ¡Importante!: Utiliza shift(1) para garantizar CERO lookahead bias (no incluye la vela actual en su propia media).
    """
    df = df.copy()

    # Extraer la hora local de negociación en Wall Street
    ts_local = df["timestamp"].dt.tz_convert(tz_market)
    hour_bucket = ts_local.dt.hour

    # Para cada hora del día, calcular la media móvil histórica previa
    def calc_hour_rvol(s: pd.Series) -> pd.Series:
        # shift(1) asegura que la vela actual no contamina la media previa
        historical_mean = s.shift(1).rolling(window=window, min_periods=5).mean()
        # Si no hay suficiente historia al inicio, usar la mediana acumulada previa
        fallback_mean = s.shift(1).expanding(min_periods=1).median()
        baseline = historical_mean.fillna(fallback_mean).fillna(s.mean())
        return s / np.maximum(baseline, 1.0)

    # Agrupar por hora de negociación
    df["rvol"] = df.groupby(hour_bucket)["volume"].transform(calc_hour_rvol).fillna(1.0)

    # Indicadores binarios de anomalía
    df["rvol_surge"] = (df["rvol"] >= 2.0).astype(int)
    df["rvol_extreme"] = (df["rvol"] >= 3.5).astype(int)

    return df


def compute_volume_momentum(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    Calcula el momentum y estandarización del volumen:
    - volume_zscore_20: Z-score de volumen respecto a las últimas 20 barras.
    - volume_ratio_ema: Ratio de volumen actual respecto a su EMA de 20 periodos.
    """
    df = df.copy()

    vol_shift = df["volume"].shift(1)
    rolling_mean = vol_shift.rolling(window=window, min_periods=5).mean()
    rolling_std = vol_shift.rolling(window=window, min_periods=5).std().fillna(1.0)

    # Z-Score de volumen
    df["volume_zscore"] = (df["volume"] - rolling_mean) / np.maximum(rolling_std, 1e-4)
    df["volume_zscore"] = df["volume_zscore"].fillna(0.0).clip(-3.0, 5.0)

    # Ratio EMA
    ema_vol = vol_shift.ewm(span=window, min_periods=5).mean()
    df["volume_ratio_ema"] = df["volume"] / np.maximum(ema_vol, 1.0)
    df["volume_ratio_ema"] = df["volume_ratio_ema"].fillna(1.0)

    return df


def compute_absorption_and_climax(df: pd.DataFrame, atr_window: int = 14) -> pd.DataFrame:
    """
    Detecta patrones de huella institucional (Wyckoff / Price Action):
    - is_bullish_absorption: Vela de rango amplio con alto volumen (RVOL >= 2.0) y mecha inferior larga
      (los compradores absorbieron la venta masiva y defendieron el mínimo).
    - is_bearish_exhaustion: Vela de rango amplio con alto volumen (RVOL >= 2.0) y mecha superior larga
      (los vendedores rechazaron el máximo y liquidaron compras).
    """
    df = df.copy()

    # Cálculo vectorizado del True Range y ATR
    h, l, c_prev = df["high"], df["low"], df["close"].shift(1).fillna(df["open"])
    tr = pd.concat([h - l, (h - c_prev).abs(), (l - c_prev).abs()], axis=1).max(axis=1)
    atr = tr.rolling(atr_window, min_periods=5).mean().fillna(tr.mean())

    bar_range = df["high"] - df["low"]
    body_top = np.maximum(df["open"], df["close"])
    body_bottom = np.minimum(df["open"], df["close"])

    upper_wick = df["high"] - body_top
    lower_wick = body_bottom - df["low"]

    # Ratios de mecha respecto al rango total
    safe_range = np.maximum(bar_range, 1e-4)
    lower_wick_ratio = lower_wick / safe_range
    upper_wick_ratio = upper_wick / safe_range

    rvol = df.get("rvol", pd.Series(1.0, index=df.index))

    # Absorción alcista: Alto volumen + rechazo inferior >= 40% + rango >= 1.2x ATR
    df["is_bullish_absorption"] = (
        (rvol >= 2.0) & (lower_wick_ratio >= 0.40) & (bar_range >= 1.2 * atr)
    ).astype(int)

    # Agotamiento bajista: Alto volumen + rechazo superior >= 40% + rango >= 1.2x ATR
    df["is_bearish_exhaustion"] = (
        (rvol >= 2.0) & (upper_wick_ratio >= 0.40) & (bar_range >= 1.2 * atr)
    ).astype(int)

    # Esfuerzo vs Resultado: Volumen por unidad de rango (acumulación oculta)
    df["volume_per_range"] = df["volume"] / safe_range

    return df


def compute_volume_features(df: pd.DataFrame, tz_market: str = "America/New_York") -> pd.DataFrame:
    """
    Función principal que calcula todas las características de volumen institucional.
    Maneja automáticamente dataframes de un solo activo o de múltiples activos agrupados por 'ticker'.
    """
    if df.empty:
        return df

    req_cols = {"timestamp", "open", "high", "low", "close", "volume"}
    if not req_cols.issubset(df.columns):
        raise ValueError(f"El DataFrame debe contener las columnas: {req_cols}")

    # Si hay múltiples tickers, procesar grupo por grupo
    if "ticker" in df.columns and df["ticker"].nunique() > 1:
        processed_groups = []
        for ticker, group in df.groupby("ticker", sort=False):
            group_sorted = group.sort_values(by="timestamp").reset_index(drop=True)
            g1 = compute_vwap(group_sorted, tz_market=tz_market)
            g2 = compute_rvol(g1, tz_market=tz_market)
            g3 = compute_volume_momentum(g2)
            g4 = compute_absorption_and_climax(g3)
            processed_groups.append(g4)
        return pd.concat(processed_groups, ignore_index=True)
    else:
        sorted_df = df.sort_values(by="timestamp").reset_index(drop=True)
        d1 = compute_vwap(sorted_df, tz_market=tz_market)
        d2 = compute_rvol(d1, tz_market=tz_market)
        d3 = compute_volume_momentum(d2)
        d4 = compute_absorption_and_climax(d3)
        return d4
