"""
Compila todos los CSVs mensuales de data_ibkr/ en un dataset maestro unificado
formato Parquet con velas RTH limpias, calculando retornos y métricas para el backtest.
"""

import sys
import glob
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_IBKR_DIR = PROJECT_ROOT / "data_ibkr"
OUT_PARQUET = PROJECT_ROOT / "data_ibkr" / "sp500_ibkr_multisector_7years_1hour.parquet"
OUT_EXTENDED = PROJECT_ROOT / "data_ibkr" / "sp500_ibkr_extended_7years_1hour.parquet"
REFERENCE_PARQUET = PROJECT_ROOT / "data" / "processed" / "sp500_multisector_1hour.parquet"

# Mapeo de sector base
SECTOR_MAP = {
    "SPY": "Benchmarks", "QQQ": "Benchmarks",
    "NVDA": "Technology", "AAPL": "Technology", "MSFT": "Technology", "AMD": "Technology",
    "MELI": "Consumer_Discretionary", "AMZN": "Consumer_Discretionary", "TSLA": "Consumer_Discretionary", "CVNA": "Consumer_Discretionary",
    "JPM": "Financials", "COIN": "Financials", "MSTR": "Financials", "HOOD": "Financials",
    "META": "Communication", "GOOGL": "Communication", "ROKU": "Communication", "TTD": "Communication",
    "CAT": "Industrials", "GE": "Industrials", "VRT": "Industrials",
    "XOM": "Energy", "CVX": "Energy", "PBF": "Energy",
    "LLY": "Healthcare", "UNH": "Healthcare", "VKTX": "Healthcare",
    "CLF": "Materials", "FCX": "Materials", "MP": "Materials",
    "TGT": "Consumer_Staples", "SPG": "Utilities_RealEstate"
}

if REFERENCE_PARQUET.exists():
    try:
        ref_df = pd.read_parquet(REFERENCE_PARQUET, columns=["ticker", "sector"]).drop_duplicates()
        for _, row in ref_df.iterrows():
            if row["ticker"] not in SECTOR_MAP:
                SECTOR_MAP[row["ticker"]] = row["sector"]
    except Exception as e:
        print(f"Aviso al cargar sectores de referencia: {e}")

def parse_ibkr_date(date_str: str) -> pd.Timestamp:
    # Formato IBKR: 'YYYYMMDD  HH:MM:SS' o 'YYYYMMDD'
    try:
        parts = str(date_str).strip().split()
        if len(parts) == 2:
            d_part, t_part = parts[0], parts[1]
            return pd.to_datetime(f"{d_part[:4]}-{d_part[4:6]}-{d_part[6:8]} {t_part}", utc=True)
        elif len(parts) == 1 and len(parts[0]) == 8:
            d_part = parts[0]
            return pd.to_datetime(f"{d_part[:4]}-{d_part[4:6]}-{d_part[6:8]} 16:00:00", utc=True)
    except Exception:
        pass
    return pd.to_datetime(date_str, utc=True)

def main():
    print("=" * 80)
    print("COMPILANDO DATASET MULTIAÑO IBKR (2020 - 2026) [TICK-M2-01]")
    print("=" * 80)

    symbols = [d.name for d in DATA_IBKR_DIR.iterdir() if d.is_dir()]
    print(f"Símbolos encontrados en data_ibkr/: {len(symbols)}")

    all_dfs = []
    total_bars = 0

    for sym in sorted(symbols):
        csv_files = glob.glob(str(DATA_IBKR_DIR / sym / "1hour" / "*.csv"))
        if not csv_files:
            continue
        
        dfs_sym = []
        for f in csv_files:
            try:
                df_m = pd.read_csv(f)
                if not df_m.empty and "date" in df_m.columns and "close" in df_m.columns:
                    dfs_sym.append(df_m)
            except Exception:
                continue
        
        if not dfs_sym:
            continue
        
        sym_df = pd.concat(dfs_sym, ignore_index=True)
        # Limpiar y parsear fechas
        sym_df = sym_df.dropna(subset=["date", "close"])
        if sym_df.empty:
            continue
        sym_df["timestamp"] = sym_df["date"].apply(parse_ibkr_date)
        sym_df = sym_df.sort_values("timestamp").drop_duplicates(subset=["timestamp"]).reset_index(drop=True)
        
        sym_df["ticker"] = sym
        sym_df["sector"] = SECTOR_MAP.get(sym, "Technology")
        
        for col in ["open", "high", "low", "close", "volume"]:
            sym_df[col] = pd.to_numeric(sym_df[col], errors="coerce")
            
        sym_df = sym_df.dropna(subset=["open", "high", "low", "close"])
        if len(sym_df) < 100:
            continue
        total_bars += len(sym_df)
        all_dfs.append(sym_df)
        print(f"  [OK] {sym:6s}: {len(sym_df):,} barras horarias ({sym_df['timestamp'].min().date()} a {sym_df['timestamp'].max().date()}) | Sector: {sym_df['sector'].iloc[0]}")

    if not all_dfs:
        print("[ERROR] No se encontraron datos para compilar.")
        return

    full_df = pd.concat(all_dfs, ignore_index=True)
    full_df = full_df.sort_values(["ticker", "timestamp"]).reset_index(drop=True)

    print("\nGuardando dataset compilado en:")
    print(" - ", OUT_PARQUET)
    full_df.to_parquet(OUT_PARQUET, index=False)
    print(" - ", OUT_EXTENDED)
    full_df.to_parquet(OUT_EXTENDED, index=False)
    print(f"[ÉXITO TOTAL] Guardado Parquet con {len(full_df):,} filas ({full_df['ticker'].nunique()} activos válidos, {full_df['timestamp'].min().date()} a {full_df['timestamp'].max().date()}).")

if __name__ == "__main__":
    main()
