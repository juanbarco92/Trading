"""
feature_pipeline.py - TICK-06: Poda de Features, Filtrado de Colinealidad y Pipeline Final

Este módulo orquesta la extracción integral de características, elimina la colinealidad excesiva
y genera la matriz final de 20 a 25 variables ortogonales, estacionarias y listas para Machine Learning.
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import pandas as pd
import numpy as np

# Asegurar que el path raíz del proyecto está en sys.path
root_dir = str(Path(__file__).resolve().parents[2])
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from src.features.volume_features import compute_volume_features
from src.features.market_regime import compute_market_regime_features
from src.features.volatility_features import compute_volatility_features
from src.data_extraction.dataset_builder import load_universe_data

logger = logging.getLogger("feature_pipeline")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# Lista canónica de las 20 variables cuantitativas estrictamente ortogonales (|r| <= 0.75)
CORE_FEATURE_NAMES = [
    # 1. Volumen y Actividad Institucional (6)
    "rvol",
    "rvol_surge",
    "vwap_dist_pct",
    "is_above_vwap",
    "volume_zscore",
    "is_bullish_absorption",
    # 2. Régimen de Mercado y Fuerza Relativa vs SPY (6)
    "is_market_bullish",
    "spy_dist_ema50_pct",
    "alpha_1d",
    "alpha_5d",
    "is_outperforming_spy",
    "rs_divergence_bullish",
    # 3. Volatilidad Dinámica y Estructura S/R (6)
    "atr_pct",
    "vol_regime",
    "is_vol_compression",
    "is_vol_expansion",
    "dist_to_res20_atr",
    "dist_to_sup20_atr",
    # 4. Momentum y Rendimientos Normalizados (2)
    "return_1h",
    "return_4h",
]


def compute_momentum_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula rendimientos porcentuales estacionarios en distintas ventanas horarias."""
    df = df.copy()

    def calc_rets(group: pd.DataFrame) -> pd.DataFrame:
        c = group["close"]
        group["return_1h"] = c.pct_change(1).fillna(0.0) * 100.0
        group["return_4h"] = c.pct_change(4).fillna(0.0) * 100.0
        group["return_24h"] = c.pct_change(24).fillna(0.0) * 100.0
        return group

    if "ticker" in df.columns and df["ticker"].nunique() > 1:
        return df.groupby("ticker", group_keys=False).apply(calc_rets)
    else:
        return calc_rets(df)


def run_full_feature_pipeline(
    df: pd.DataFrame,
    benchmark_symbol: str = "SPY",
) -> pd.DataFrame:
    """
    Ejecuta el pipeline completo de ingeniería de características sobre el DataFrame del universo:
    1. Características de Volumen Institucional (TICK-03)
    2. Régimen de Mercado y Fuerza Relativa vs SPY (TICK-04)
    3. Volatilidad Dinámica y Estructura S/R (TICK-05)
    4. Rendimientos porcentuales de momentum (TICK-06)
    """
    logger.info(f"Iniciando pipeline de características sobre {len(df):,} registros...")

    # Paso 1: Volumen y VWAP
    logger.info("Paso 1/4: Calculando características de volumen institucional...")
    df_vol = compute_volume_features(df)

    # Paso 2: Régimen de Mercado y Fuerza Relativa vs SPY
    logger.info("Paso 2/4: Calculando régimen macro y fuerza relativa (RS vs SPY)...")
    df_regime = compute_market_regime_features(df_vol, benchmark_symbol=benchmark_symbol)

    # Paso 3: Volatilidad y S/R
    logger.info("Paso 3/4: Calculando volatilidad dinámica y distancias a soportes/resistencias...")
    df_volat = compute_volatility_features(df_regime)

    # Paso 4: Rendimientos de Momentum
    logger.info("Paso 4/4: Calculando rendimientos de momentum estacionarios...")
    df_final = compute_momentum_returns(df_volat)

    logger.info("Pipeline de extracción completado con éxito.")
    return df_final


def analyze_collinearity(
    df: pd.DataFrame,
    feature_cols: List[str],
    max_correlation: float = 0.75,
) -> Tuple[pd.DataFrame, List[Tuple[str, str, float]]]:
    """
    Calcula la matriz de correlación de Pearson y detecta pares de variables con colinealidad excesiva (|r| > max_corr).
    """
    corr_matrix = df[feature_cols].corr()

    high_corr_pairs = []
    cols = corr_matrix.columns
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            col_a = cols[i]
            col_b = cols[j]
            r = corr_matrix.loc[col_a, col_b]
            if abs(r) > max_correlation:
                high_corr_pairs.append((col_a, col_b, round(float(r), 3)))

    return corr_matrix, high_corr_pairs


def clean_and_validate_features(
    df: pd.DataFrame,
    feature_cols: List[str],
) -> pd.DataFrame:
    """
    Limpia y valida que la matriz de características esté 100% libre de NaNs, infinitos y valores anómalos:
    - Rellena NaNs iniciales (por ventanas móviles de calentamiento) de forma segura.
    - Acota valores extremos numéricos (clipping).
    """
    df = df.copy()

    # Reemplazar infinitos por NaN
    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)

    # Relleno hacia adelante y luego mediana del activo
    for col in feature_cols:
        if df[col].isna().any():
            # Rellenar con la mediana por ticker
            df[col] = df.groupby("ticker")[col].transform(lambda s: s.fillna(s.median()))
            # Si aún quedan NaNs iniciales, rellenar con 0.0
            df[col] = df[col].fillna(0.0)

    # Asegurar tipos flotantes limpios
    for col in feature_cols:
        if df[col].dtype == "bool":
            df[col] = df[col].astype(int)

    return df


def build_and_save_feature_dataset(
    input_file: str = "data/processed/universe_1hour.parquet",
    output_file: str = "data/processed/features_universe_1hour.parquet",
    metadata_file: str = "data/processed/features_metadata.json",
    max_correlation: float = 0.75,
) -> pd.DataFrame:
    """
    Función orquestadora principal que lee el universo Parquet, ejecuta el pipeline completo,
    analiza colinealidad, valida integridad y guarda el dataset maestro final.
    """
    input_path = Path(input_file)
    if not input_path.exists():
        raise FileNotFoundError(f"No se encontró el archivo {input_file}. Ejecuta TICK-01 primero.")

    logger.info(f"Cargando dataset base desde {input_file}...")
    universe_df = pd.read_parquet(input_path)

    # 1. Ejecutar pipeline
    processed_df = run_full_feature_pipeline(universe_df, benchmark_symbol="SPY")

    # 2. Verificar columnas canónicas disponibles
    available_features = [f for f in CORE_FEATURE_NAMES if f in processed_df.columns]
    logger.info(f"Variables candidatas seleccionadas ({len(available_features)}): {available_features}")

    # 3. Limpieza y validación
    cleaned_df = clean_and_validate_features(processed_df, available_features)

    # 4. Análisis de Colinealidad
    corr_matrix, high_corr_pairs = analyze_collinearity(
        cleaned_df,
        available_features,
        max_correlation=max_correlation,
    )

    if high_corr_pairs:
        logger.warning(f"Se detectaron {len(high_corr_pairs)} pares con correlación > {max_correlation}: {high_corr_pairs}")
    else:
        logger.info(f"CERO colinealidad crítica detectada: todos los pares tienen |r| <= {max_correlation}.")

    # 5. Organizar columnas finales: identificadores + características seleccionadas
    base_cols = ["timestamp", "ticker", "open", "high", "low", "close", "volume"]
    if "barCount" in cleaned_df.columns:
        base_cols.append("barCount")

    final_cols = base_cols + available_features
    final_df = cleaned_df[final_cols].copy()

    # 6. Guardar en Parquet comprimido
    out_path = Path(output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_parquet(out_path, engine="pyarrow", compression="snappy", index=False)

    file_size_mb = out_path.stat().st_size / (1024 * 1024)

    # 7. Guardar metadatos JSON
    metadata = {
        "total_rows": len(final_df),
        "total_tickers": int(final_df["ticker"].nunique()),
        "tickers": final_df["ticker"].unique().tolist(),
        "date_start": str(final_df["timestamp"].min()),
        "date_end": str(final_df["timestamp"].max()),
        "feature_count": len(available_features),
        "features": available_features,
        "high_correlation_pairs": high_corr_pairs,
        "max_correlation_threshold": max_correlation,
        "file_size_mb": round(file_size_mb, 2),
    }

    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=4)

    logger.info(
        f"\n{'='*70}\n"
        f"DATASET DE CARACTERÍSTICAS GENERADO EXITOSAMENTE:\n"
        f"Archivo: {output_file} ({file_size_mb:.2f} MB)\n"
        f"Metadatos: {metadata_file}\n"
        f"Total de Registros: {len(final_df):,} velas horarias\n"
        f"Variables de Señal: {len(available_features)} características ortogonales\n"
        f"{'='*70}"
    )

    return final_df


if __name__ == "__main__":
    build_and_save_feature_dataset()
