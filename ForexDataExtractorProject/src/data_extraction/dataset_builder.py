"""
dataset_builder.py - TICK-01: Consolidación del Universo en Formato Parquet Optimizado

Este módulo transforma los archivos mensuales CSV individuales (data/<TICKER>/<CANDLESIZE>/*.csv)
en DataFrames estructurados, limpios y validados, guardados en formato Apache Parquet de alto rendimiento
(data/processed/<TICKER>_<CANDLESIZE>.parquet y data/processed/universe_<CANDLESIZE>.parquet).
"""

import os
import sys
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import pandas as pd
import numpy as np

# Configuración básica de logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("dataset_builder")


def parse_datetime_series(date_series: pd.Series, tz_source: str = "America/New_York") -> pd.Series:
    """
    Convierte una serie de fechas en strings devueltas por IBKR al formato Timestamp UTC estandarizado.
    Maneja formatos:
      - '20261001  08:30:00' (con múltiples espacios)
      - '20261001 08:30:00'
      - '20261001 08:30:00 US/Eastern' (con zona horaria explícita)
      - '20261001' (diario)
    """
    # 1. Normalizar espacios en blanco
    cleaned = date_series.astype(str).str.strip().str.replace(r"\s+", " ", regex=True)

    # 2. Si contiene zona horaria al final (ej. 'US/Eastern' o 'America/New_York'), extraer fecha y hora
    # Tomar los dos primeros componentes (Fecha y Hora)
    parts = cleaned.str.split(" ", expand=True)

    if parts.shape[1] >= 2:
        dt_strings = parts[0] + " " + parts[1]
        parsed = pd.to_datetime(dt_strings, format="%Y%m%d %H:%M:%S", errors="coerce")
    else:
        # Formato diario 'YYYYMMDD'
        parsed = pd.to_datetime(parts[0], format="%Y%m%d", errors="coerce")

    # Si algún elemento falló, aplicar parser genérico
    if parsed.isna().any():
        mask_na = parsed.isna()
        parsed.loc[mask_na] = pd.to_datetime(cleaned.loc[mask_na], errors="coerce")

    # 3. Localizar a zona horaria del mercado (por defecto New York) y convertir a UTC
    if parsed.dt.tz is None:
        parsed = parsed.dt.tz_localize(tz_source, ambiguous="NaT", nonexistent="shift_forward")
    parsed_utc = parsed.dt.tz_convert("UTC")

    return parsed_utc


def validate_price_integrity(df: pd.DataFrame, ticker: str) -> Dict[str, int]:
    """
    Realiza pruebas de cordura (*sanity checks*) sobre los datos de precios y volumen:
    - High >= Low
    - High >= Open y High >= Close
    - Low <= Open y Low <= Close
    - Volume >= 0
    """
    anomalies = {
        "high_less_than_low": int((df["high"] < df["low"]).sum()),
        "high_less_than_open_close": int(((df["high"] < df["open"]) | (df["high"] < df["close"])).sum()),
        "low_greater_than_open_close": int(((df["low"] > df["open"]) | (df["low"] > df["close"])).sum()),
        "negative_volume": int((df["volume"] < 0).sum()),
        "null_values": int(df[["open", "high", "low", "close", "volume"]].isna().sum().sum()),
    }

    total_issues = sum(anomalies.values())
    if total_issues > 0:
        logger.warning(f"[{ticker}] Se detectaron {total_issues} anomalías de integridad de precios: {anomalies}")
    else:
        logger.debug(f"[{ticker}] Integridad de precios 100% verificada.")

    return anomalies


def consolidate_ticker_data(
    ticker: str,
    candlesize: str = "1hour",
    data_dir: str = "data",
    output_dir: str = "data/processed",
) -> Optional[pd.DataFrame]:
    """
    Lee y consolida todos los archivos CSV mensuales de un activo en un único DataFrame ordenado y limpio,
    guardándolo en formato Parquet comprimido con 'snappy'.
    """
    clean_size = candlesize.replace(" ", "")
    ticker_dir = Path(data_dir) / ticker / clean_size

    if not ticker_dir.exists():
        logger.warning(f"El directorio {ticker_dir} no existe. Omitiendo {ticker}.")
        return None

    csv_files = sorted(list(ticker_dir.glob("*.csv")))
    if not csv_files:
        logger.warning(f"No se encontraron archivos CSV en {ticker_dir}.")
        return None

    logger.info(f"[{ticker}] Consolidando {len(csv_files)} archivos CSV mensuales ({csv_files[0].stem} a {csv_files[-1].stem})...")

    dfs = []
    expected_cols = {"date", "open", "high", "low", "close", "volume"}

    for csv_path in csv_files:
        try:
            temp_df = pd.read_csv(csv_path)
            if not expected_cols.issubset(temp_df.columns):
                logger.warning(f"Archivo {csv_path.name} no contiene las columnas requeridas. Omitiendo.")
                continue
            dfs.append(temp_df)
        except Exception as e:
            logger.error(f"Error al leer {csv_path}: {e}")

    if not dfs:
        logger.error(f"[{ticker}] No se pudo cargar ningún archivo CSV válido.")
        return None

    # Concatenar todos los meses
    full_df = pd.concat(dfs, ignore_index=True)

    # 1. Parsear fechas a UTC
    full_df["timestamp"] = parse_datetime_series(full_df["date"])
    full_df = full_df.dropna(subset=["timestamp"])

    # 2. Asegurar tipos numéricos estrictos
    float_cols = ["open", "high", "low", "close"]
    for col in float_cols:
        full_df[col] = pd.to_numeric(full_df[col], errors="coerce")

    full_df["volume"] = pd.to_numeric(full_df["volume"], errors="coerce").fillna(0).astype("float64")
    if "barCount" in full_df.columns:
        full_df["barCount"] = pd.to_numeric(full_df["barCount"], errors="coerce").fillna(0).astype("int64")

    # 3. Eliminar duplicados y ordenar cronológicamente
    full_df = full_df.drop_duplicates(subset=["timestamp"], keep="last")
    full_df = full_df.sort_values(by="timestamp").reset_index(drop=True)

    # 4. Validar integridad de precios
    validate_price_integrity(full_df, ticker)

    # 5. Añadir columna identificadora de ticker
    full_df["ticker"] = ticker.upper()

    # Organizar columnas
    cols_order = ["timestamp", "ticker", "open", "high", "low", "close", "volume"]
    if "barCount" in full_df.columns:
        cols_order.append("barCount")

    final_df = full_df[cols_order].copy()

    # 6. Guardar en Parquet
    out_dir_path = Path(output_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)

    parquet_file = out_dir_path / f"{ticker.upper()}_{clean_size}.parquet"
    final_df.to_parquet(parquet_file, engine="pyarrow", compression="snappy", index=False)

    file_size_kb = parquet_file.stat().st_size / 1024
    logger.info(
        f"[{ticker}] Guardado exitosamente: {len(final_df)} velas "
        f"({final_df['timestamp'].iloc[0].strftime('%Y-%m-%d')} a {final_df['timestamp'].iloc[-1].strftime('%Y-%m-%d')}) "
        f"en {parquet_file.name} ({file_size_kb:.1f} KB)."
    )

    return final_df


def build_universe_dataset(
    tickers: Optional[List[str]] = None,
    candlesize: str = "1hour",
    data_dir: str = "data",
    output_dir: str = "data/processed",
) -> Tuple[pd.DataFrame, Dict[str, Dict]]:
    """
    Consolida todos los tickers del universo, genera sus archivos individuales Parquet
    y construye el archivo maestro unificado `universe_<CANDLESIZE>.parquet`.
    """
    clean_size = candlesize.replace(" ", "")
    data_path = Path(data_dir)

    # Si no se pasan tickers, auto-detectar los disponibles en data/
    if tickers is None:
        all_dirs = [d.name for d in data_path.iterdir() if d.is_dir() and (d / clean_size).exists()]
        # Filtrar Forex histórico antiguo (EURUSD, USDJPY) para enfocarnos en el universo de acciones/ETFs
        tickers = [d for d in all_dirs if d not in ["EURUSD", "USDJPY", "processed", "parquet"]]

    logger.info(f"Iniciando consolidación para {len(tickers)} activos: {tickers}")

    summary_stats = {}
    consolidated_dfs = []

    for sym in tickers:
        df = consolidate_ticker_data(sym, candlesize=candlesize, data_dir=data_dir, output_dir=output_dir)
        if df is not None and not df.empty:
            consolidated_dfs.append(df)
            summary_stats[sym] = {
                "bars": len(df),
                "start": df["timestamp"].iloc[0].strftime("%Y-%m-%d %H:%M"),
                "end": df["timestamp"].iloc[-1].strftime("%Y-%m-%d %H:%M"),
                "min_price": df["low"].min(),
                "max_price": df["high"].max(),
                "avg_hourly_volume": int(df["volume"].mean()),
            }

    if not consolidated_dfs:
        raise RuntimeError("No se pudo consolidar ningún activo. Verifica la ruta de datos.")

    # Crear dataset unificado del universo
    universe_df = pd.concat(consolidated_dfs, ignore_index=True)
    universe_df = universe_df.sort_values(by=["timestamp", "ticker"]).reset_index(drop=True)

    # Guardar universo maestro
    out_dir_path = Path(output_dir)
    universe_file = out_dir_path / f"universe_{clean_size}.parquet"
    universe_df.to_parquet(universe_file, engine="pyarrow", compression="snappy", index=False)

    universe_size_mb = universe_file.stat().st_size / (1024 * 1024)
    logger.info(
        f"\n{'='*70}\n"
        f"UNIVERSO MAESTRO GENERADO: {universe_file}\n"
        f"Total de registros: {len(universe_df):,} velas horarias\n"
        f"Activos consolidados ({len(summary_stats)}): {list(summary_stats.keys())}\n"
        f"Tamaño en disco: {universe_size_mb:.2f} MB\n"
        f"{'='*70}"
    )

    return universe_df, summary_stats


def load_universe_data(
    candlesize: str = "1hour",
    output_dir: str = "data/processed",
) -> pd.DataFrame:
    """
    Función de utilidad de alta velocidad para cargar el universo completo desde Parquet en milisegundos.
    """
    clean_size = candlesize.replace(" ", "")
    universe_file = Path(output_dir) / f"universe_{clean_size}.parquet"
    if not universe_file.exists():
        raise FileNotFoundError(f"El archivo {universe_file} no existe. Ejecuta build_universe_dataset primero.")
    return pd.read_parquet(universe_file)


if __name__ == "__main__":
    # Ejecución directa desde CLI
    print("Iniciando construcción de datasets en formato Parquet...")
    build_universe_dataset(candlesize="1hour")
