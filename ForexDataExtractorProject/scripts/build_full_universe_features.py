"""
build_full_universe_features.py - Generación del Dataset Completo de Features Estructurales.

Aplica a los 21 activos del universo:
1. Triple Barrier institucional con fricción real y límites asimétricos (TP=3.5x ATR, SL=1.8x ATR, horizon=35).
2. Feature engineering del usuario (src/utils/exploration_utils.py):
   - Geometría de velas individuales y agrupadas (calcular_features_vela, create_candle_patterns)
   - Agrupamiento K-Means de patrones de velas (CandleClusterer)
   - Indicadores técnicos contextualizados (RSI, Bollinger, MACD, Estocástico, Parabolic SAR, ADX)
3. Fractal Engineering del usuario (fractal_engineering):
   - Williams Fractals multiventana (5, 13, 34)
   - Líneas de tendencia dinámicas y canales fractales
   - Divergencias precio vs fractal y rupturas de canal
4. Soporte y Resistencia Estructural (src/utils/stock_sr.py):
   - Detección de picos/valles confirmados (scipy.signal.find_peaks, zero lookahead)
   - Distancias relativas en % y ATR al soporte/resistencia más cercano
   - Toques históricos, fuerza del nivel, posición en canal, rebotes y rupturas
5. Filtro de sesión / timing intradiario:
   - Hora del día, día de la semana, indicador de power-hour (cierre de mercado).
"""

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from src.models.labeling import generate_triple_barrier_labels
from src.utils.exploration_utils import engineer_features, fractal_engineering
from src.utils.stock_sr import compute_stock_support_resistance_features


def process_single_ticker(ticker: str, df_ticker: pd.DataFrame) -> pd.DataFrame:
    print(f"\n[{ticker}] Procesando {len(df_ticker)} velas...")
    t0 = time.time()
    
    # 1. Triple Barrier Labels
    df_labeled = generate_triple_barrier_labels(
        df_ticker.copy(),
        k_tp=3.5,
        k_sl=1.8,
        horizon_bars=35,
        friction_pct=0.0005,
        conservative_intrabar=True,
    )
    
    # 2. Features de velas y clustering K-Means del usuario
    df_eng, _ = engineer_features(
        df_labeled,
        n_candles=4,
        n_clusters=6,
        fit_clusterer=True,
        random_state=42,
    )
    
    # 3. Fractal Engineering del usuario
    df_eng = fractal_engineering(df_eng)
    
    # 4. Soporte y Resistencia Estructural (Zero Lookahead)
    df_eng = compute_stock_support_resistance_features(df_eng)
    
    # 5. Features de tiempo y microestructura
    if "datetime" in df_eng.columns:
        dt = pd.to_datetime(df_eng["datetime"])
        df_eng["hour_of_day"] = dt.dt.hour
        df_eng["day_of_week"] = dt.dt.dayofweek
        df_eng["is_power_hour"] = (dt.dt.hour >= 19).astype(int)  # 15:00-16:00 EST / 19:00-20:00 UTC
        df_eng["is_morning_trap"] = ((dt.dt.hour >= 13) & (dt.dt.hour <= 15)).astype(int)
    
    # Asegurar ticker
    df_eng["ticker"] = ticker
    
    elapsed = time.time() - t0
    print(f"[{ticker}] Completado en {elapsed:.2f}s! Total columnas: {len(df_eng.columns)}")
    return df_eng


def main():
    universe_path = PROJECT_ROOT / "data" / "processed" / "universe_1hour.parquet"
    output_path = PROJECT_ROOT / "data" / "processed" / "features_universe_full_structural.parquet"
    
    print(f"1. Cargando universo consolidado desde {universe_path}...")
    df_universe = pd.read_parquet(universe_path)
    
    tickers = sorted(df_universe["ticker"].unique())
    print(f"Universo detectado: {len(tickers)} activos -> {tickers}")
    
    processed_dfs = []
    start_total = time.time()
    
    for ticker in tickers:
        df_t = df_universe[df_universe["ticker"] == ticker].copy()
        time_col = "timestamp" if "timestamp" in df_t.columns else "datetime"
        df_t = df_t.sort_values(time_col).reset_index(drop=True)
        if "datetime" not in df_t.columns and "timestamp" in df_t.columns:
            df_t["datetime"] = pd.to_datetime(df_t["timestamp"])
        try:
            res_t = process_single_ticker(ticker, df_t)
            processed_dfs.append(res_t)
        except Exception as e:
            print(f"ERROR procesando {ticker}: {e}")
            import traceback
            traceback.print_exc()

    print("\n2. Consolidando DataFrame final del universo...")
    final_df = pd.concat(processed_dfs, ignore_index=True)
    
    print(f"Total filas: {len(final_df)}, Total columnas: {len(final_df.columns)}")
    print(f"Guardando en {output_path}...")
    final_df.to_parquet(output_path, compression="zstd", index=False)
    
    total_elapsed = time.time() - start_total
    print(f"\nPROCESO COMPLETADO EXITOSAMENTE en {total_elapsed:.1f}s ({total_elapsed/60:.2f} minutos)!")
    print(f"Archivo generado: {output_path} ({output_path.stat().st_size / (1024*1024):.2f} MB)")


if __name__ == "__main__":
    main()
